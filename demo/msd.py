from bloqade.gemini import logical as gemini_logical
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.impls import generate_arch
from bloqade.lanes.logical_mvp import compile_squin_to_move

kernel = squin.kernel.add(gemini_logical.dialect)
kernel.run_pass = squin.kernel.run_pass


@kernel
def main(cond: bool):
    # see arXiv: 2412.15165v1, Figure 3a
    reg = qubit.qalloc(5)
    squin.broadcast.t(reg)

    squin.broadcast.sqrt_x(ilist.IList([reg[0], reg[1], reg[4]]))
    if cond:
        squin.broadcast.cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
    squin.broadcast.sqrt_y(ilist.IList([reg[0], reg[3]]))
    squin.broadcast.cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[2], reg[4]]))
    squin.sqrt_x_adj(reg[0])
    squin.broadcast.cz(ilist.IList([reg[0], reg[1]]), ilist.IList([reg[4], reg[3]]))
    squin.broadcast.sqrt_y_adj(reg)

    return gemini_logical.terminal_measure(reg)


main = compile_squin_to_move(main, transversal_rewrite=True)

atom_interp = atom.AtomInterpreter(main.dialects, arch_spec=generate_arch())
frame, _ = atom_interp.run(main)
main.print(analysis=frame.entries)

# compile_and_visualize(main)
