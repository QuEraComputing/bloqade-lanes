"""Shot-remapping helper: hardware frame -> per-measurement array.

Result post-processing runs in two steps, and this module is the join
between them.

**Step 1 — value reconstruction.** ``bloqade.gemini.post_processing``
(built on ``MeasurementIDAnalysis`` from bloqade-circuit) abstract-
interprets the *user's* kernel, tracing values from ``terminal_measure``
through whatever operations construct the return value. It yields a
callable over a flat per-measurement array of shape
``(n_shots, n_measurements)``, indexed by ``RawMeasureId.idx``.

**Step 2 — frame projection.** The machine's API returns a *frame*: one
flat bitstring per shot covering every location in the measurement
mode's zones, including sites no atom is ever moved into. This module
produces the index list that projects a frame down to the array step 1
expects::

    frame --[mapping]--> per-measurement array --[post-processing]--> values

so a caller writes ``per_measurement = frame[:, mapping]``.

The join key is the measurement record index: step 1's
``RawMeasureId.idx`` and step 2's ``MeasureResult.measurement_id`` count
the same measurements in the same order. ``mapping[k]`` is therefore the
frame slot of record ``k``, and ``len(mapping)`` is the total number of
records the program emits.

Records come from the analysis's own per-``GetFutureResult`` bookkeeping
(via ``AtomInterpreter.get_measurement_positions``), *not* from walking
the kernel's return value. Those differ: a kernel is free to return a
subset of its measurements, or return them out of order, while still
emitting every record to hardware. Deriving the mapping from the return
value produced a short or permuted list that silently disagreed with
step 1 — see issue #967.

See: issues #563, #967.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._measurement_positions import AtomPosition, MeasurementPositions


@dataclass(frozen=True)
class ShotRemappingDiagnostic:
    """Compiler-developer-facing diagnostic emitted when
    ``get_shot_remapping`` cannot derive a frame index list.

    A failure here indicates an analysis or pipeline regression rather
    than a user error — the user supplied a kernel, the compiler service
    lowered it, and somewhere along the way the analysis stopped
    producing a coherent set of measurement records.

    Attributes:
        message: human-readable description of the failure.
        offending_value: the value that triggered the failure, when
            there is a single one to point at — the first misattributed
            record, the record with no frame slot, or the record id list
            that failed to cover ``0..n-1``. ``None`` for failures with
            no single culprit: an empty program, or an analysis that
            crashed before producing anything.

            This was ``MoveExecution | LocationAddress`` while the
            mapping was derived by walking the return value's lattice
            values. It now walks measurement records instead, so the
            values that can appear here changed with it.
    """

    message: str
    offending_value: AtomPosition | list[int] | None = None


@dataclass(frozen=True)
class ShotRemappingOk:
    """Successful shot-remapping result.

    Attributes:
        mapping: ``mapping[k]`` is the frame slot holding measurement
            record ``k``. Index a hardware frame with it directly —
            ``frame[:, mapping]`` yields the per-measurement array that
            post-processing consumes.
        frame_size: width of the frame ``mapping`` indexes into. A
            caller should check its hardware rows are this wide before
            projecting.
        measurement_index: IR position of the measurement statement
            whose frame this is.
    """

    mapping: list[int]
    frame_size: int = 0
    measurement_index: int = 0


@dataclass(frozen=True)
class ShotRemappingErr:
    """Failed shot-remapping result.

    ``diagnostic`` carries the contextual message and the offending
    value, aimed at the compiler developer debugging the failed
    lowering.
    """

    diagnostic: ShotRemappingDiagnostic


def get_shot_remapping(
    positions: MeasurementPositions,
) -> ShotRemappingOk | ShotRemappingErr:
    """Derive the frame -> per-measurement projection for ``positions``.

    Args:
        positions: result of
            ``AtomInterpreter.get_measurement_positions``. The *last*
            measurement statement supplies the frame, since that is the
            terminal readout whose bitstring the machine returns.

    Returns:
        ``ShotRemappingOk`` carrying the index list on success, or
        ``ShotRemappingErr`` on failure. Failure modes:

        - the program contains no measurement statement, so there is no
          frame to project;
        - a measurement statement before the last one also produced
          readouts. Post-processing indexes one flat array by record id,
          which cannot span two separately-returned frames, so this is
          rejected rather than silently mapped against the wrong frame;
        - the record ids are not a contiguous ``0..n-1`` cover, which
          would leave holes in ``mapping``;
        - a record has no slot in its own frame, which would mean the
          analysis and arch spec disagree about hardware layout.

        The diagnostic is aimed at compiler developers, not end users; a
        failure here indicates a pipeline regression rather than a
        malformed kernel.
    """
    if not positions.measurements:
        return ShotRemappingErr(
            diagnostic=ShotRemappingDiagnostic(
                message="program contains no measurement statement to project",
            ),
        )

    frame = positions.measurements[-1]
    stale = [
        atom_position
        for snapshot in positions.measurements[:-1]
        for atom_position in snapshot.readout
    ]
    if stale:
        return ShotRemappingErr(
            diagnostic=ShotRemappingDiagnostic(
                message=(
                    f"{len(stale)} measurement record(s) belong to a frame "
                    f"before the last one (measurement {frame.index}); a "
                    "single flat projection cannot span multiple frames"
                ),
                offending_value=stale[0],
            ),
        )

    records = frame.readout
    record_ids = sorted(
        atom_position.measurement_id
        for atom_position in records
        if atom_position.measurement_id is not None
    )
    if record_ids != list(range(len(records))):
        return ShotRemappingErr(
            diagnostic=ShotRemappingDiagnostic(
                message=(
                    "measurement record ids are not a contiguous 0..n-1 "
                    f"cover of {len(records)} record(s); got {record_ids}"
                ),
                offending_value=record_ids,
            ),
        )

    mapping = [0] * len(records)
    for atom_position in records:
        if atom_position.frame_index is None:
            return ShotRemappingErr(
                diagnostic=ShotRemappingDiagnostic(
                    message=(
                        f"record {atom_position.measurement_id} at "
                        f"{atom_position.location_address} has no slot in "
                        "its own measurement frame"
                    ),
                    offending_value=atom_position,
                ),
            )
        # ``measurement_id`` is non-None here: the contiguity check above
        # rejects any record missing one.
        assert atom_position.measurement_id is not None
        mapping[atom_position.measurement_id] = atom_position.frame_index

    return ShotRemappingOk(
        mapping=mapping,
        frame_size=frame.frame_size,
        measurement_index=frame.index,
    )
