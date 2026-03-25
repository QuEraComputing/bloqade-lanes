import math

from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.gemini import compile_to_stim_program, logical as gemini_logical
from bloqade.gemini.logical.stdlib import default_post_processing


@gemini_logical.kernel(aggressive_unroll=True)
def main():
    # see arXiv: 2412.15165v1, Figure 3a
    reg = qubit.qalloc(5)
    squin.broadcast.u3(0.3041 * math.pi, 0.25 * math.pi, 0.0, reg)

    squin.broadcast.sqrt_x(ilist.IList([reg[0], reg[1], reg[4]]))
    squin.broadcast.cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
    squin.broadcast.sqrt_y(ilist.IList([reg[0], reg[3]]))
    squin.broadcast.cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[2], reg[4]]))
    squin.sqrt_x_adj(reg[0])
    squin.broadcast.cz(ilist.IList([reg[0], reg[1]]), ilist.IList([reg[4], reg[3]]))
    squin.broadcast.sqrt_y_adj(reg)

    default_post_processing(reg)


### Visualize ###
# from bloqade.lanes.logical_mvp import compile_squin_to_move_and_visualize
# compile_squin_to_move_and_visualize(main, transversal_rewrite=True)
result = compile_to_stim_program(main)
