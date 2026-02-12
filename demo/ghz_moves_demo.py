from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.lanes.logical_mvp import compile_squin_to_move_and_visualize


@squin.kernel(typeinfer=True, fold=True)
def log_depth_ghz():
    size = 10
    q0 = qubit.new()
    squin.h(q0)
    reg = ilist.IList([q0])
    for i in range(size):
        current = len(reg)
        missing = size - current
        if missing > current:
            num_alloc = current
        else:
            num_alloc = missing

        if num_alloc > 0:
            new_qubits = qubit.qalloc(num_alloc)
            squin.broadcast.cx(reg[-num_alloc:], new_qubits)
            reg = reg + new_qubits


@squin.kernel(typeinfer=True, fold=True)
def ghz_optimal():
    qs = qubit.qalloc(10)
    squin.broadcast.sqrt_y(qs)
    squin.z(qs[0])
    squin.cz(qs[0], qs[5])
    squin.broadcast.cz(qs[:2] + qs[5:7], qs[2:4] + qs[7:9])
    squin.broadcast.cz(ilist.IList([qs[3], qs[8]]), ilist.IList([qs[4], qs[9]]))
    squin.broadcast.sqrt_y(qs)
    squin.sqrt_y_adj(qs[0])


compile_squin_to_move_and_visualize(log_depth_ghz, animated=True)
# compile_squin_to_move_and_visualize(ghz_optimal)
# compile_squin_to_move_and_visualize(
#     ghz_optimal, transversal_rewrite=True, animated=True
# )
