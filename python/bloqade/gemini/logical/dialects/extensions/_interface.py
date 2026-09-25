from typing import Any

from bloqade.types import Qubit
from kirin import lowering
from kirin.dialects import ilist

from bloqade.gemini.star import DEFAULT_STEANE_STAR_SUPPORT

from .stmts import StarRz


@lowering.wraps(StarRz)
def star_rz(
    theta: float,
    qubits: ilist.IList[Qubit, Any],
    qubit_indices: tuple[int, int, int] = DEFAULT_STEANE_STAR_SUPPORT,
) -> None:
    """Apply a STAR/TMR logical-Z rotation injection primitive.

    Args:
        theta: Target logical rotation angle in SQuIn IR turn units.
        qubits: List of logical qubits.
        qubit_indices: Steane weight-3 logical-Z support. Defaults to ``(4, 5, 6)``.
    """
    ...
