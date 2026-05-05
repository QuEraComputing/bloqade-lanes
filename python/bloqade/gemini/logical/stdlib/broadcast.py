from typing import Any

from kirin.dialects import ilist

from bloqade import types

from ..dialects import operations
from ..group import kernel


@kernel(aggressive_unroll=True, verify=False)
def star_rz(
    theta: float,
    qubits: ilist.IList[types.Qubit, Any],
):
    operations.star_rz(theta, qubits)
