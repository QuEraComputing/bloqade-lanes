from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

import numpy as np
from kirin import ir
from kirin.analysis import Forward
from kirin.analysis.forward import ForwardFrame
from typing_extensions import Self

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.dialects import move
from bloqade.lanes.utils import no_none_elements_tuple

from . import _shot_remapping
from ._measurement_positions import (
    AtomPosition,
    MeasurementPositions,
    MeasurementSnapshot,
)
from ._post_processing import constructor_function
from .lattice import AtomState, MeasureFuture, MeasureResult, MoveExecution


def _default_best_state_cost(state: AtomState) -> float:
    """Average of move counts plus standard deviation.

    More weight is added to the standard deviation to prefer a balanced number
    of moves across atoms.
    """
    if len(state.data.collision) > 0:
        return float("inf")

    move_counts = np.array(
        [state.data.move_count.get(qubit, 0) for qubit in state.data.qubit_to_locations]
    )
    return 0.1 * np.mean(move_counts).astype(float) + np.std(move_counts).astype(float)


RetType = TypeVar("RetType")


@dataclass
class PostProcessing(Generic[RetType]):
    emit_return: Callable[[Sequence[Sequence[bool]]], Generator[RetType, None, None]]
    emit_detectors: Callable[
        [Sequence[Sequence[bool]]], Generator[list[bool], None, None]
    ]
    emit_observables: Callable[
        [Sequence[Sequence[bool]]], Generator[list[bool], None, None]
    ]


@dataclass
class AtomInterpreter(Forward[MoveExecution]):
    lattice = MoveExecution

    arch_spec: ArchSpec = field(kw_only=True)
    current_state: MoveExecution = field(init=False)
    best_state_cost: Callable[[AtomState], float] = field(
        kw_only=True, default=_default_best_state_cost
    )
    _detectors: list[MoveExecution] = field(init=False, default_factory=list)
    _observables: list[MoveExecution] = field(init=False, default_factory=list)
    final_measurement_count: int = field(init=False, default=0)
    measurement_record_count: int = field(init=False, default=0)
    keys = ("atom",)

    def __post_init__(self):
        super().__post_init__()

    def initialize(self) -> Self:
        self.current_state = AtomState()
        self._detectors.clear()
        self._observables.clear()
        self.final_measurement_count = 0
        self.measurement_record_count = 0
        return super().initialize()

    def method_self(self, method) -> MoveExecution:
        return MoveExecution.bottom()

    def eval_fallback(self, frame: ForwardFrame[MoveExecution], node: ir.Statement):
        return tuple(MoveExecution.bottom() for _ in node.results)

    def get_shot_remapping(
        self, method: ir.Method, *, no_raise: bool = True
    ) -> _shot_remapping.ShotRemappingOk | _shot_remapping.ShotRemappingErr:
        """Run the analysis on ``method`` and return the index list that
        projects a hardware frame onto the per-measurement array
        post-processing consumes, as a ``ShotRemappingOk``. On failure,
        returns ``ShotRemappingErr`` carrying a
        ``ShotRemappingDiagnostic``.

        ``mapping[k]`` is the frame slot of measurement record ``k``, so
        a caller writes ``frame[:, mapping]``. Convenience wrapper
        around the standalone
        ``bloqade.lanes.analysis.atom._shot_remapping.get_shot_remapping``;
        see that function's docstring for the two-step post-processing
        model and the diagnostics emitted on failure.

        The mapping is derived from the analysis's per-measurement
        records, not from ``method``'s return value, so a kernel that
        returns only some of its measurements — or returns them out of
        order — still gets a complete, correctly-ordered mapping
        (issue #967). Callers (typically the compiler service) are
        responsible for surfacing the diagnostic in the failure case; a
        failure here is a compiler-pipeline regression, not a user
        error.

        Args:
            method: kirin method to analyse.
            no_raise: when ``True`` (default), the failure is reported as
                a ``ShotRemappingErr`` so callers see a single failure
                shape. That covers both an analysis crash and an address
                the arch spec cannot resolve, which
                :meth:`get_measurement_positions` surfaces as a
                ``ValueError``. Flip to ``False`` when debugging an
                analysis-side bug to let the original exception
                propagate.
        """
        try:
            positions = self.get_measurement_positions(method, no_raise=no_raise)
        except ValueError as error:
            # ``get_measurement_positions`` resolves every captured location
            # through ``ArchSpec.get_position``, which raises on an address
            # the spec doesn't know. Callers of this method contract for a
            # diagnostic, not an exception — typically an arch spec that
            # doesn't match the one the program was compiled against.
            if not no_raise:
                raise
            return _shot_remapping.ShotRemappingErr(
                diagnostic=_shot_remapping.ShotRemappingDiagnostic(
                    message=(
                        "the analysis captured a location the arch spec "
                        f"cannot resolve: {error}"
                    ),
                ),
            )
        return _shot_remapping.get_shot_remapping(positions)

    def _atom_at(
        self,
        location: LocationAddress,
        qubit_id: int,
        zone_offsets: dict[int, int],
        measurement_id: int | None = None,
    ) -> AtomPosition:
        """Build one :class:`AtomPosition`, resolving its frame index.

        ``zone_offsets`` maps a zone id to where that zone starts in the
        measurement's frame. A location in a zone the measurement
        doesn't cover has no slot in the frame and gets ``None``.
        """
        frame_index = None
        if (offset := zone_offsets.get(location.zone_id)) is not None:
            within_zone = self.arch_spec.get_zone_index(
                location, ZoneAddress(location.zone_id)
            )
            if within_zone is not None:
                frame_index = offset + within_zone
        return AtomPosition(
            qubit_id=qubit_id,
            location_address=location,
            position=self.arch_spec.get_position(location),
            measurement_id=measurement_id,
            frame_index=frame_index,
        )

    def _atoms_by_address(
        self,
        occupancy: dict[LocationAddress, int],
        zone_offsets: dict[int, int],
    ) -> tuple[AtomPosition, ...]:
        return tuple(
            self._atom_at(location, occupancy[location], zone_offsets)
            for location in sorted(occupancy)
        )

    def _check_records_cover_every_measurement(
        self, snapshots: Sequence[MeasurementSnapshot]
    ) -> None:
        """Reject a result whose readouts don't cover ``0..n-1`` exactly.

        One snapshot per measurement *statement* only models a program
        where each statement executes once. Put a measurement in a loop
        and the statement has several executions with several distinct
        frames, which this shape cannot represent — the snapshots would
        overlap.

        That case is detectable here: the interpreter mints a fresh
        ``measurement_id`` per visit, so a second visit to a
        ``GetFutureResult`` yields an id its first visit didn't have. The
        two ``MeasureResult`` values are incomparable, so the lattice
        joins them to ``Unknown`` and the readout drops out of the walk
        entirely; either way the surviving ids stop being a complete
        ``0..n-1`` cover.
        """
        record_ids = sorted(
            atom_position.measurement_id
            for snapshot in snapshots
            for atom_position in snapshot.readout
            if atom_position.measurement_id is not None
        )
        expected = list(range(self.measurement_record_count))
        if record_ids != expected:
            raise ValueError(
                "measurement records do not cover "
                f"0..{self.measurement_record_count - 1} exactly (got "
                f"{record_ids}); a measurement statement most likely "
                "executed more than once, which one-snapshot-per-statement "
                "cannot represent"
            )

    def get_measurement_positions(
        self, method: ir.Method, *, no_raise: bool = False
    ) -> MeasurementPositions:
        """Where every atom sat at each measurement in ``method``.

        Returns one :class:`MeasurementSnapshot` per measurement
        statement, in the order the statements appear in the IR. Each
        snapshot carries three widening scopes — ``readout`` (atoms the
        program actually reads, ordered by ``measurement_id``),
        ``measured_zones`` (every atom the hardware measures), and
        ``processor`` (every atom on the device). See
        ``_measurement_positions`` for the full contract.

        ``MeasurementPositions.readout`` flattens the per-snapshot
        readouts into one ``measurement_id``-ordered tuple, so
        ``result.readout[k].position`` is where the atom in column ``k``
        of the raw per-shot measurement array was sitting.

        Args:
            method: kirin method to analyse.
            no_raise: when ``True``, an analysis crash is swallowed by
                ``Forward.run_no_raise`` and the result is assembled
                from whatever was captured before the failure — which
                may be empty or partial. Defaults to ``False`` so
                analysis bugs surface instead of yielding a silently
                short list of positions.

        Raises:
            ValueError: if a captured location has no position under
                ``arch_spec`` — the analysis and arch spec disagree
                about hardware layout.
        """
        if not no_raise:
            frame, _ = self.run(method)
            return self.collect_measurement_positions(method, frame)

        # Swallow the failure like ``run_no_raise`` does, but remember it: a
        # frame from a crashed run is missing the entries for every statement
        # the analysis never reached, and that is invisible downstream.
        try:
            frame, _ = self.run(method)
        except Exception:  # noqa: BLE001 - matches Forward.run_no_raise
            return MeasurementPositions(measurements=(), analysis_failed=True)
        return self.collect_measurement_positions(method, frame)

    def collect_measurement_positions(
        self,
        method: ir.Method,
        frame: ForwardFrame[MoveExecution],
        *,
        analysis_failed: bool = False,
    ) -> MeasurementPositions:
        """Read the snapshots out of a *converged* analysis frame.

        Deliberately a second pass rather than bookkeeping accumulated
        during interpretation: abstract interpretation may visit a
        statement any number of times before it reaches a fixpoint, so
        anything recorded as a side effect of visiting has to be made
        idempotent by hand. Walking the IR once afterwards reads only
        what the fixpoint settled on, and statement order comes from the
        IR itself rather than from a counter.

        Readouts attach to their measurement through
        ``GetFutureResult.measurement_future.owner`` — the statement that
        produced the future — so nothing depends on two counters agreeing.
        """
        # Every zone spans the same number of addresses: words are global
        # and a zone_id merely tags them, so each zone contributes
        # ``len(words) * sites_per_word`` slots to the frame.
        frame_stride = len(self.arch_spec.words) * self.arch_spec.sites_per_word

        measure_stmts: list[move.EndMeasure | move.Measure] = []
        readouts: dict[ir.Statement, list[MeasureResult]] = {}
        for stmt in method.callable_region.walk():
            if isinstance(stmt, (move.EndMeasure, move.Measure)):
                measure_stmts.append(stmt)
            elif isinstance(stmt, move.GetFutureResult):
                result = frame.get(stmt.result)
                # A GetFutureResult that didn't resolve emits no measurement
                # and so has no place in any frame.
                owner = stmt.measurement_future.owner
                # A future that isn't a statement result (a block argument,
                # say) belongs to no measurement statement in this walk.
                if isinstance(result, MeasureResult) and isinstance(
                    owner, ir.Statement
                ):
                    readouts.setdefault(owner, []).append(result)

        snapshots = []
        for index, stmt in enumerate(measure_stmts):
            future = frame.get(
                stmt.result if isinstance(stmt, move.EndMeasure) else stmt.future
            )
            state = frame.get(stmt.current_state)
            if not isinstance(future, MeasureFuture) or not isinstance(
                state, AtomState
            ):
                # The analysis never resolved this measurement; it has no
                # occupancy to report, so there is no snapshot to build.
                continue

            zone_addresses = tuple(stmt.zone_addresses)
            zone_offsets = {
                zone.zone_id: position * frame_stride
                for position, zone in enumerate(zone_addresses)
            }
            measured_zones: dict[LocationAddress, int] = {}
            for occupancy in future.results.values():
                measured_zones.update(occupancy)
            processor = {
                location: qubit_id
                for qubit_id, location in state.data.qubit_to_locations.items()
            }
            readout = sorted(
                (
                    self._atom_at(
                        result.location_address,
                        result.qubit_id,
                        zone_offsets,
                        result.measurement_id,
                    )
                    for result in readouts.get(stmt, [])
                ),
                key=lambda atom: atom.measurement_id or 0,
            )
            snapshots.append(
                MeasurementSnapshot(
                    index=index,
                    zone_addresses=zone_addresses,
                    frame_size=len(zone_addresses) * frame_stride,
                    readout=tuple(readout),
                    measured_zones=self._atoms_by_address(measured_zones, zone_offsets),
                    processor=self._atoms_by_address(processor, zone_offsets),
                )
            )

        if not analysis_failed:
            self._check_records_cover_every_measurement(snapshots)
        return MeasurementPositions(
            measurements=tuple(snapshots), analysis_failed=analysis_failed
        )

    def get_post_processing(
        self, method: ir.Method[..., RetType]
    ) -> PostProcessing[RetType]:
        """Reconstruct user values, detectors and observables from the
        *lowered move* kernel.

        .. deprecated::
            Use :func:`bloqade.gemini.post_processing.generate_post_processing`
            instead. It abstract-interprets the **user's** kernel rather than
            the lowered one, so it reconstructs whatever the kernel returns
            without depending on the move IR's shape.

            This method reads ``MeasureResult.measurement_id`` off the move
            kernel, which only lines up with the rest of the pipeline while
            lowering preserves the record ordering — an invariant that has to
            be maintained by hand (see
            ``tests/analysis/atom/test_measure_id_invariant.py``). Deriving
            user values from the user's own kernel removes that coupling.

        Callers that still need this, and what they are waiting on:

        - ``emit_detectors`` / ``emit_observables`` have no replacement yet.
          They come from ``annotate.SetDetector`` / ``SetObservable``
          statements collected while walking the lowered kernel, and
          ``generate_post_processing`` only sees a detector that appears
          inside the return value.
        - The simulator paths (``gemini/compile/task.py``,
          ``gemini/device/physical_simulator.py``) consume a per-measurement
          array directly and never project a hardware frame, so nothing
          about the frame-mapping split applies to them.

        No runtime warning is raised for that reason: every in-tree caller
        is on one of those two paths, so a warning would fire on correct
        code with nowhere to migrate.
        """
        _, output = self.run(method)

        func = cast(Callable[[Sequence[bool]], RetType], constructor_function(output))
        if func is None:
            raise ValueError("Unable to infer return result value from method output")

        def post_processing_return(measurement_results: Sequence[Sequence[bool]]):
            yield from map(func, measurement_results)

        detector_funcs: tuple[Callable[[Sequence[bool]], bool] | None, ...] = tuple(
            map(constructor_function, self._detectors)
        )
        if not no_none_elements_tuple(detector_funcs):
            raise ValueError("Unable to infer detector measurement values")

        def detectors(measurement_results: Sequence[Sequence[bool]]):
            yield from (
                [func(measurement_shot) for func in detector_funcs]
                for measurement_shot in measurement_results
            )

        observable_funcs: tuple[Callable[[Sequence[bool]], bool] | None, ...] = tuple(
            map(constructor_function, self._observables)
        )
        if not no_none_elements_tuple(observable_funcs):
            raise ValueError("Unable to infer observable measurement values")

        def observables(measurement_results: Sequence[Sequence[bool]]):
            yield from (
                [func(measurement_shot) for func in observable_funcs]
                for measurement_shot in measurement_results
            )

        return PostProcessing(post_processing_return, detectors, observables)
