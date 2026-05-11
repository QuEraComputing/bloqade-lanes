from typing import Any

from kirin.dialects import ilist

from bloqade import types

from ..dialects import operations
from ..dialects.operations.stmts import DEFAULT_STEANE_STAR_SUPPORT
from ..group import kernel


@kernel(aggressive_unroll=True, verify=False)
def star_rz(
    theta: float,
    qubits: ilist.IList[types.Qubit, Any],
) -> None:
    operations.star_rz(theta, qubits, DEFAULT_STEANE_STAR_SUPPORT)
