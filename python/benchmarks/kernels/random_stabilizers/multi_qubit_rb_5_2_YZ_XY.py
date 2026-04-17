"""Generated from gemini_benchmarking multi_qubit_rb circuits."""

from __future__ import annotations

from bloqade import squin

# pyright: reportCallIssue=false


@squin.kernel(typeinfer=True, fold=True)
def multi_qubit_rb_5_2_YZ_XY():
    q = squin.qalloc(5)
    squin.cz(q[0], q[1])
    squin.sqrt_x(q[0])
    squin.cz(q[0], q[4])
    squin.x(q[2])
    squin.sqrt_y(q[1])
    squin.s(q[2], adjoint=True)
    squin.sqrt_x(q[4])
    squin.sqrt_x(q[1])
    squin.cz(q[0], q[3])
    squin.sqrt_y_adj(q[2])
    squin.sqrt_y_adj(q[1])
    squin.cz(q[0], q[2])
    squin.cz(q[1], q[2])
    squin.s(q[1])
    squin.cz(q[1], q[4])
    squin.s(q[4], adjoint=True)
    squin.sqrt_x(q[0])
    squin.s(q[1])
    squin.sqrt_x_adj(q[0])
    squin.cz(q[1], q[3])
    squin.z(q[0])
    squin.cz(q[2], q[4])
    squin.cz(q[2], q[3])
    squin.y(q[1])
    squin.y(q[3])
    squin.sqrt_x(q[2])
    squin.z(q[2])
    squin.sqrt_y(q[0])
    squin.sqrt_x_adj(q[3])
    squin.cz(q[3], q[4])
    squin.sqrt_x(q[0])
    squin.sqrt_y_adj(q[3])
    squin.sqrt_x(q[4])
