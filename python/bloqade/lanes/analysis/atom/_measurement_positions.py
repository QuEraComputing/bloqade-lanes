"""Where the atoms physically are at each measurement.

``AtomInterpreter.get_measurement_positions`` answers "which atom sat
where when this program measured?", at three widening scopes:

``readout``
    One entry per ``move.GetFutureResult`` that resolved to a real
    qubit, ordered by ``measurement_id``. This lines up one-to-one with
    the columns of the raw per-shot measurement array, so
    ``readout[k].position`` is where the atom reported by column ``k``
    was sitting. It is the geometric complement of
    ``get_shot_remapping``, which gives the hardware *bit index* for the
    same ``k``.

``measured_zones``
    Every occupied location in the zones the measurement covers,
    whether or not the program reads its result. This is what the
    hardware physically measures — a zone readout does not skip atoms
    just because the kernel ignores them.

``qpu_state``
    Every atom on the device at that instant, including ones in zones
    the measurement doesn't touch. Useful for reconstructing the full
    machine state at measurement time.

``measured_zones`` is a subset of ``qpu_state``, and the addresses in
``readout`` are a subset of ``measured_zones``. The wider two carry no
``measurement_id`` for atoms that were never read out.

The readout frame
-----------------

``AtomPosition.frame_index`` places an atom in the measurement's
*frame*: the flat list you get by walking that measurement's
``zone_addresses`` in order and, within each zone, enumerating every
address word-major / site-minor::

    frame_index = zone_position * len(words) * sites_per_word
                  + word_id * sites_per_word
                  + site_id

The within-zone part is exactly ``ArchSpec.get_zone_index``; the frame
just concatenates one such block per measured zone. Every zone
contributes the same ``len(words) * sites_per_word`` slots, because
words are global and a ``zone_id`` merely tags them.

Two things follow. The frame is scoped to *this* measurement — an atom
in a zone the measurement doesn't cover has no slot and gets ``None``,
which is why ``qpu_state`` entries outside ``zone_addresses`` carry no
frame index. And ``frame_index`` is an *address* ordering, unrelated to
``measurement_id`` (IR readout order) and to any grid-geometry
ordering; on Gemini a word is not a row, so walking addresses is not
walking the grid.
"""

from __future__ import annotations

from dataclasses import dataclass

from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress


@dataclass(frozen=True)
class AtomPosition:
    """One atom, and where it was when a measurement happened.

    Attributes:
        qubit_id: the physical qubit occupying ``location_address``.
        location_address: the hardware location it occupied.
        position: that location's physical ``(x, y)`` coordinates,
            resolved through ``ArchSpec.get_position``.
        measurement_id: the atom's column in the raw per-shot
            measurement array, or ``None`` when the program never read
            this atom's result. Set on every ``readout`` entry and
            ``None`` everywhere else — it is never invented for atoms
            the program didn't read.
        frame_index: the atom's slot in the measurement's frame (see the
            module docstring). ``None`` *only* for a ``qpu_state`` atom
            sitting in a zone the measurement doesn't cover, which has
            no slot in that frame. Always populated on ``readout`` and
            ``measured_zones``, both of which are built from the
            measurement's own zones.

    Both optional fields are deliberately non-defaulting: every
    construction site states them, so a missing value is a type error
    rather than a silent ``None``.
    """

    qubit_id: int
    location_address: LocationAddress
    position: tuple[float, float]
    measurement_id: int | None
    frame_index: int | None


@dataclass(frozen=True)
class MeasurementSnapshot:
    """The machine at one measurement statement.

    Attributes:
        index: position of the measurement statement in IR order,
            counting from 0.
        zone_addresses: the zones this measurement covers.
        frame_size: number of slots in the readout frame — every
            ``frame_index`` on this snapshot falls in
            ``range(frame_size)``.
        readout: atoms whose results the program reads, ordered by
            ``measurement_id``.
        measured_zones: every atom in ``zone_addresses``, ordered by
            location address.
        qpu_state: every atom on the device, ordered by location
            address.
    """

    index: int
    zone_addresses: tuple[ZoneAddress, ...]
    frame_size: int
    readout: tuple[AtomPosition, ...]
    measured_zones: tuple[AtomPosition, ...]
    qpu_state: tuple[AtomPosition, ...]


@dataclass(frozen=True)
class MeasurementPositions:
    """Result of :meth:`AtomInterpreter.get_measurement_positions`.

    ``measurements`` holds one :class:`MeasurementSnapshot` per
    measurement statement, in the order the statements appear in the IR.
    Today's Gemini specs set ``feed_forward=False`` and validation
    enforces a single terminal measure, so this is usually length 1 —
    but the shape generalises to mid-circuit measurement without a
    breaking change.

    ``analysis_failed`` marks a result assembled from *partial* records:
    the analysis raised and the caller asked for it to be swallowed, so
    whatever was captured before the failure is all there is. The
    records that survive are internally consistent — a truncated prefix
    of measurement ids is still contiguous — so consumers cannot detect
    the truncation by inspecting ``measurements`` and must check this
    flag instead.
    """

    measurements: tuple[MeasurementSnapshot, ...]
    analysis_failed: bool = False

    @property
    def readout(self) -> tuple[AtomPosition, ...]:
        """Every read-out atom across all measurements, ordered by
        ``measurement_id``.

        Index this with a column number from the raw per-shot
        measurement array — the same ``k`` that indexes
        ``get_shot_remapping``'s mapping.
        """
        atoms = [atom for snapshot in self.measurements for atom in snapshot.readout]
        atoms.sort(key=lambda a: a.measurement_id or 0)
        return tuple(atoms)

    @property
    def positions(self) -> tuple[tuple[float, float], ...]:
        """Just the coordinates from :attr:`readout`, in the same order."""
        return tuple(atom.position for atom in self.readout)
