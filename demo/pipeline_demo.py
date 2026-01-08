from bloqade.gemini.dialects import logical as gemini_logical
from kirin.dialects import ilist

from bloqade import squin
from bloqade.lanes.logical_mvp import (
    compile_to_physical_stim_program,
)

kernel = squin.kernel.add(gemini_logical.dialect)
kernel.run_pass = squin.kernel.run_pass


@kernel
def main():
    q = squin.qalloc(9)

    # random params
    theta = 0.234
    phi_0 = 0.123
    phi_1 = 0.934
    phi_2 = 0.343

    # initialization (encoding)
    squin.broadcast.rx(2 * theta, ilist.IList([q[3], q[4], q[6], q[7]]))
    squin.broadcast.h(ilist.IList([q[3], q[4], q[6], q[7]]))
    squin.rx(phi_2, q[2])
    squin.rx(phi_1, q[5])
    squin.rx(phi_0, q[8])

    # transversal logic
    squin.broadcast.cx(ilist.IList([q[2], q[3]]), ilist.IList([q[0], q[1]]))
    squin.cx(q[0], q[1])
    squin.cx(q[4], q[1])
    squin.broadcast.cx(ilist.IList([q[5], q[6]]), ilist.IList([q[0], q[1]]))
    squin.cx(q[0], q[1])
    squin.broadcast.cx(ilist.IList([q[7], q[8]]), ilist.IList([q[1], q[0]]))

    gemini_logical.terminal_measure(q)


error_model = compile_to_physical_stim_program(main)
print(error_model)
