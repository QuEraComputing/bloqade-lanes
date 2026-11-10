from kirin.dialects import ilist

from bloqade import types
from bloqade.gemini.logical.extensions import broadcast
from bloqade.gemini.logical.group import kernel


@kernel(aggressive_unroll=True, verify=False)
def star_rz(
    theta: float,
    qubit: types.Qubit,
) -> None:
    broadcast.star_rz(theta, ilist.IList([qubit]))
