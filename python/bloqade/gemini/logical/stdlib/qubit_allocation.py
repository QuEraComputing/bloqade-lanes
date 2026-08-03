from typing import TypeVar

from bloqade.types import Qubit
from kirin import types
from kirin.dialects import ilist

from bloqade import squin
from bloqade.gemini.common.dialects.qubit import new_at

from .. import kernel

N = TypeVar("N")

PositionIndex = types.Union(types.Int, types.NoneType)


@kernel(aggressive_unroll=True, verify=False)
def qalloc_at(
    positions: ilist.IList[PositionIndex, N],  # pyright: ignore[reportInvalidTypeForm]
) -> ilist.IList[Qubit, N]:
    """Allocate logical qubits at optional linear logical positions.

    An integer ``position`` pins the qubit to zone 0 at word ``2 * position``;
    ``None`` allocates an unpinned qubit for the layout heuristic to place.
    The input must be statically known, and the calling kernel must set
    ``aggressive_unroll=True``, so the map can lower into individual
    allocations.
    """

    def position_to_new_at(
        position: PositionIndex,  # pyright: ignore[reportInvalidTypeForm]
    ) -> Qubit:
        if position is None:
            qubit = squin.qalloc(1)[0]
        else:
            qubit = new_at(0, position * 2, 0)
        return qubit

    return ilist.map(position_to_new_at, positions)
