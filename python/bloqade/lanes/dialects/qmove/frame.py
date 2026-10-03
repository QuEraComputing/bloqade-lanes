"""Frames: where a qmove method's arguments must be, and what it may do.

Every qmove method opens its chain with ``qmove.enter(frame)``. The frame is a
``Frame`` (partial: entry slots, scratch and effects), a ``MachineFrame`` (the
whole machine, as for the entry kernel), or ``None`` (a hole for synthesis).

A ``Frame`` is a *shape* (how many slots each qubit parameter takes, plus
scratch) and a *binding* of those slots to concrete locations. Keeping the two apart
leaves room for relocatable frames later without changing the IR; for now the
binding is always concrete.

Exit = entry, for a partial ``Frame``: on return every argument atom is back in
its own entry slot and the scratch slots are empty again, so there is no
separate exit layout. A ``MachineFrame`` has no entry slots, so this does not
apply to it.
"""

from __future__ import annotations

from dataclasses import dataclass

from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress


@dataclass(frozen=True)
class FrameShape:
    param_slots: tuple[tuple[int, int], ...]
    """``(parameter index, slot count)`` for each qubit-typed parameter.

    Indices exclude ``self``. A ``Qubit`` parameter takes one slot and an
    ``IList[Qubit, Literal[N]]`` parameter takes ``N``, in element order.
    """
    scratch_slots: int = 0

    @property
    def total_slots(self) -> int:
        return sum(count for _, count in self.param_slots) + self.scratch_slots


@dataclass(frozen=True)
class Effects:
    """Zone-wide operations a subroutine is permitted to perform."""

    cz_zones: frozenset[ZoneAddress] = frozenset()
    measure_zones: frozenset[ZoneAddress] = frozenset()
    global_pulses: bool = False

    def is_subset_of(self, other: Effects) -> bool:
        return (
            self.cz_zones <= other.cz_zones
            and self.measure_zones <= other.measure_zones
            and (other.global_pulses or not self.global_pulses)
        )


@dataclass(frozen=True)
class MachineFrame:
    """The whole machine: no footprint limit, and every effect is allowed.

    The entry kernel's frame. A method under it may allocate qubits, and only
    another whole-machine method may call it (F5).
    """


@dataclass(frozen=True)
class Frame:
    shape: FrameShape
    binding: tuple[LocationAddress, ...]
    """One location per slot: parameter slots in parameter order, then scratch."""
    effects: Effects = Effects()

    @property
    def footprint(self) -> frozenset[LocationAddress]:
        return frozenset(self.binding)
