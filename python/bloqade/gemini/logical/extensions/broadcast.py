from typing import Any

from bloqade.squin.stdlib.broadcast.gate import _radian_to_turn
from kirin.dialects import ilist

from bloqade import types
from bloqade.gemini.star import DEFAULT_STEANE_STAR_SUPPORT

from ..dialects import extensions
from ..group import kernel


@kernel(aggressive_unroll=True, verify=False)
def star_rz(
    theta: float,
    qubits: ilist.IList[types.Qubit, Any],
) -> None:
    extensions.star_rz(_radian_to_turn(theta), qubits, DEFAULT_STEANE_STAR_SUPPORT)
