from typing import Any

from kirin import lowering
from kirin.dialects import ilist

from bloqade import types

from ..dialects.operations.stmts import DEFAULT_STEANE_STAR_SUPPORT, StarRz


@lowering.wraps(StarRz)
def star_rz(
    theta: float,
    qubits: ilist.IList[types.Qubit, Any],
    qubit_indices: tuple[int, int, int] = DEFAULT_STEANE_STAR_SUPPORT,
) -> None: ...
