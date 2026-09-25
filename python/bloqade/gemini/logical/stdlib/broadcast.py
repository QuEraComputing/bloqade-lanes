from typing import Any

from bloqade.squin.stdlib.broadcast import (
    cnot as cnot,
    cx as cx,
    cy as cy,
    cz as cz,
    h as h,
    rx as rx,
    ry as ry,
    rz as rz,
    s as s,
    s_adj as s_adj,
    shift as shift,
    sqrt_x as sqrt_x,
    sqrt_x_adj as sqrt_x_adj,
    sqrt_y as sqrt_y,
    sqrt_y_adj as sqrt_y_adj,
    sqrt_z as sqrt_z,
    sqrt_z_adj as sqrt_z_adj,
    swap as swap,
    t as t,
    t_adj as t_adj,
    u3 as u3,
    x as x,
    y as y,
    z as z,
)
from bloqade.squin.stdlib.broadcast.gate import _radian_to_turn
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
    operations.star_rz(_radian_to_turn(theta), qubits, DEFAULT_STEANE_STAR_SUPPORT)
