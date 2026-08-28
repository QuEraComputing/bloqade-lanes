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
from bloqade.lanes.utils import no_none_elements_tuple

from . import _shot_remapping
from ._measurement_positions import (
    AtomPosition,
    MeasurementPositions,
    MeasurementSnapshot,
)
from ._post_processing import constructor_function
from .lattice import AtomState, MoveExecution


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


@dataclass(frozen=True)
class _MeasureSnapshotRecord:
    """What a measurement statement observed, captured during interpretation.

    Keyed by statement rather than appended to a list so a fixpoint that
    revisits a statement overwrites rather than duplicates.
    """

    order: int
    zone_addresses: tuple[ZoneAddress, ...]
    measured_zones: dict[LocationAddress, int]
    processor: dict[LocationAddress, int]


@dataclass(frozen=True)
class _ReadoutRecord:
    """One resolved ``move.GetFutureResult``, keyed by its statement."""

    measurement_id: int
    qubit_id: int
    location_address: LocationAddress
    measurement_count: int


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
    _measure_snapshots: dict[ir.Statement, _MeasureSnapshotRecord] = field(
        init=False, default_factory=dict
    )
    _readouts: dict[ir.Statement, _ReadoutRecord] = field(
        init=False, default_factory=dict
    )
    keys = ("atom",)

    def __post_init__(self):
        super().__post_init__()

    def initialize(self) -> Self:
        self.current_state = AtomState()
        self._detectors.clear()
        self._observables.clear()
        self.final_measurement_count = 0
        self.measurement_record_count = 0
        self._measure_snapshots.clear()
        self._readouts.clear()
        return super().initialize()

    def method_self(self, method) -> MoveExecution:
        return MoveExecution.bottom()

    def eval_fallback(self, frame: ForwardFrame[MoveExecution], node: ir.Statement):
        return tuple(MoveExecution.bottom() for _ in node.results)

    def get_shot_remapping(
        self, method: ir.Method, *, no_raise: bool = True
    ) -> _shot_remapping.ShotRemappingOk | _shot_remapping.ShotRemappingErr:
        """Run the analysis on ``method`` and return the flat Zone-0
        bitstring index list (in row-major order over the nested
        ``IListResult[IListResult[MeasureResult]]`` return shape) as a
        ``ShotRemappingOk``. On failure, returns ``ShotRemappingErr``
        carrying a ``ShotRemappingDiagnostic``.

        Convenience wrapper around the standalone
        ``bloqade.lanes.analysis.atom._shot_remapping.get_shot_remapping``;
        see that function's docstring for the contract on the analysis
        output shape, the meaning of the returned indices, and the
        diagnostic emitted on failure.

        ``method``'s return value is expected to refine to
        ``IListResult[IListResult[MeasureResult]]`` — the shape produced
        by lowering a logical ``terminal_measure`` (or any kernel that
        returns a nested ilist of measurement results) through the
        atom-analysis chain. Callers (typically the compiler service)
        are responsible for surfacing the diagnostic in the failure
        case; a failure here is a compiler-pipeline regression, not a
        user error.

        Args:
            method: kirin method to analyse.
            no_raise: when ``True`` (default), an analysis crash is
                caught by ``Forward.run_no_raise`` and falls through
                into the standard ``ShotRemappingErr`` path with the
                ``Bottom`` lattice as the offending value, so callers
                see a single failure shape. Flip to ``False`` when
                debugging an analysis-side bug to let the original
                exception propagate.
        """
        run_method = self.run_no_raise if no_raise else self.run
        _, output = run_method(method)
        return _shot_remapping.get_shot_remapping(output, self.arch_spec)

    def record_measure_snapshot(
        self,
        stmt: ir.Statement,
        zone_addresses: tuple[ZoneAddress, ...],
        results: dict[ZoneAddress, dict[LocationAddress, int]],
        state: AtomState,
    ) -> None:
        """Capture what a measurement statement observed.

        Called from the ``move.Measure`` / ``move.EndMeasure`` impls,
        which already have both the per-zone occupancy they hand to
        ``MeasureFuture`` and the incoming ``AtomState``. Keyed by
        statement, so a fixpoint that revisits the statement overwrites
        rather than double-counting.
        """
        measured_zones: dict[LocationAddress, int] = {}
        for occupancy in results.values():
            measured_zones.update(occupancy)
        self._measure_snapshots[stmt] = _MeasureSnapshotRecord(
            order=len(self._measure_snapshots),
            zone_addresses=zone_addresses,
            measured_zones=measured_zones,
            processor={
                location: qubit_id
                for qubit_id, location in state.data.qubit_to_locations.items()
            },
        )

    def record_readout(
        self,
        stmt: ir.Statement,
        *,
        measurement_id: int,
        qubit_id: int,
        location_address: LocationAddress,
        measurement_count: int,
    ) -> None:
        """Capture one resolved ``move.GetFutureResult``.

        ``measurement_count`` is the owning ``MeasureFuture``'s 1-based
        statement index, which is how the readout is attributed back to
        its snapshot.
        """
        self._readouts[stmt] = _ReadoutRecord(
            measurement_id=measurement_id,
            qubit_id=qubit_id,
            location_address=location_address,
            measurement_count=measurement_count,
        )

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
        run_method = self.run_no_raise if no_raise else self.run
        run_method(method)

        # Every zone spans the same number of addresses: words are global
        # and a zone_id merely tags them, so each zone contributes
        # ``len(words) * sites_per_word`` slots to the frame.
        frame_stride = len(self.arch_spec.words) * self.arch_spec.sites_per_word

        # ``MeasureFuture.measurement_count`` is the 1-based index of the
        # measurement statement, which is how a readout finds its snapshot.
        readouts_by_measurement: dict[int, list[_ReadoutRecord]] = {}
        for record in self._readouts.values():
            readouts_by_measurement.setdefault(record.measurement_count, []).append(
                record
            )

        snapshots = []
        for snapshot in sorted(self._measure_snapshots.values(), key=lambda s: s.order):
            zone_offsets = {
                zone.zone_id: position * frame_stride
                for position, zone in enumerate(snapshot.zone_addresses)
            }
            readout = sorted(
                (
                    self._atom_at(
                        record.location_address,
                        record.qubit_id,
                        zone_offsets,
                        record.measurement_id,
                    )
                    for record in readouts_by_measurement.get(snapshot.order + 1, [])
                ),
                key=lambda atom: atom.measurement_id or 0,
            )
            snapshots.append(
                MeasurementSnapshot(
                    index=snapshot.order,
                    zone_addresses=snapshot.zone_addresses,
                    frame_size=len(snapshot.zone_addresses) * frame_stride,
                    readout=tuple(readout),
                    measured_zones=self._atoms_by_address(
                        snapshot.measured_zones, zone_offsets
                    ),
                    processor=self._atoms_by_address(snapshot.processor, zone_offsets),
                )
            )
        return MeasurementPositions(measurements=tuple(snapshots))

    def get_post_processing(
        self, method: ir.Method[..., RetType]
    ) -> PostProcessing[RetType]:
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
