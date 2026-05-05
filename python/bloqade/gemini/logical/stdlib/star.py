from kirin.dialects import ilist

from bloqade import types

from ..group import kernel
from . import broadcast


@kernel(aggressive_unroll=True, verify=False)
def star_rz(
    theta: float,
    qubit: types.Qubit,
):
    broadcast.star_rz(theta, ilist.IList([qubit]))
