from typing import Any

from bloqade.types import Qubit
from kirin import lowering
from kirin.dialects import ilist

from .stmts import Register


@lowering.wraps(Register)
def register(qubits: ilist.IList[Qubit, Any]) -> None:
    """Register ``qubits`` as one code block, positions in list order.

    Physical pipeline only. Qubits must be either all pinned (with ``new_at``)
    or all unpinned. Pinned blocks preserve arbitrary initial addresses.
    Unpinned blocks must fit in a word and receive contiguous sites, with
    ``qubits[p]`` at site ``offset + p``. Has no runtime effect.
    """
    ...
