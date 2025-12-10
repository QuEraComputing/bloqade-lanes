from bloqade.gemini.dialects import logical as gemini_logical
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.lanes.logical_mvp import compile_squin

kernel = squin.kernel.add(gemini_logical.dialect)
kernel.run_pass = squin.kernel.run_pass


@kernel(typeinfer=False, fold=True)
def main():
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

    return gemini_logical.terminal_measure(reg)


main = compile_squin(main)
