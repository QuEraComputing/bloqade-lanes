"""Generated from gemini_benchmarking multi_qubit_rb circuits."""

from __future__ import annotations

from bloqade import squin

# pyright: reportCallIssue=false


@squin.kernel(typeinfer=True, fold=True)
def multi_qubit_rb_5_8_ZZZZX():
    q = squin.qalloc(5)
    squin.s(q[0])
    squin.sqrt_x_adj(q[4])
    squin.sqrt_y_adj(q[4])
    squin.cz(q[0], q[1])
    squin.sqrt_x_adj(q[1])
    squin.sqrt_x(q[1])
    squin.z(q[3])
    squin.z(q[4])
    squin.cz(q[0], q[4])
    squin.sqrt_x(q[2])
    squin.cz(q[0], q[3])
    squin.cz(q[0], q[2])
    squin.s(q[1])
    squin.y(q[0])
    squin.y(q[2])
    squin.s(q[0], adjoint=True)
    squin.cz(q[1], q[2])
    squin.x(q[2])
    squin.cz(q[1], q[4])
    squin.cz(q[1], q[3])
    squin.s(q[4])
    squin.sqrt_x_adj(q[3])
    squin.s(q[2])
    squin.cz(q[2], q[4])
    squin.cz(q[2], q[3])
    squin.sqrt_x_adj(q[0])
    squin.z(q[3])
    squin.cz(q[3], q[4])
    squin.sqrt_x_adj(q[0])
    squin.y(q[2])
    squin.sqrt_y_adj(q[4])
