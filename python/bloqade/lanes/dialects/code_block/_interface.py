from typing import Any

from bloqade.types import Qubit
from kirin import lowering
from kirin.dialects import ilist

from .stmts import Register


@lowering.wraps(Register)
def register(qubits: ilist.IList[Qubit, Any]) -> None:
    """Register ``qubits`` as one code block, positions in list order.

    Physical pipeline only. The block's initial layout is a contiguous run of
    sites in one word, with ``qubits[p]`` at site ``offset + p``. A block must
    fit in a word, and its qubits must be either all pinned (with ``new_at``) or
    all unpinned. Has no runtime effect.
    """
    ...
