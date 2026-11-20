from typing import Any, TypeVar

from bloqade.types import Qubit
from kirin import lowering
from kirin.dialects import ilist

from .stmts import QallocAt, Register

N = TypeVar("N")


@lowering.wraps(Register)
def register(qubits: ilist.IList[Qubit, Any]) -> None:
    """Register ``qubits`` as one code block, positions in list order.

    Physical pipeline only. Qubits must be either all pinned (with ``new_at``)
    or all unpinned. Pinned blocks preserve arbitrary initial addresses.
    Unpinned blocks must fit in a word and receive contiguous sites, with
    ``qubits[p]`` at site ``offset + p``. Has no runtime effect.
    """
    ...


@lowering.wraps(QallocAt)
def qalloc_at(
    slots: ilist.IList[int, N],
) -> ilist.IList[ilist.IList[Qubit, Any], N]:
    """Allocate physical qubits at zero-based indices into ``PhysicalPipeline.slots``.

    The input must be compile-time constant. Each returned inner IList contains
    the physical qubits of one selected code block, in the configured address
    order. Slot coordinates are supplied to the compiler, not to this kernel.
    """
    ...
