from typing import Literal

from bloqade.gemini.dialects import logical as gemini_logical
from kirin import rewrite
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.impls import generate_arch
from bloqade.lanes.logical_mvp import compile_squin
from bloqade.lanes.rewrite import move2squin

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


@squin.kernel
def initialize(
    theta: float, phi: float, lam: float, qubits: ilist.IList[qubit.Qubit, Literal[7]]
):
    squin.u3(theta, phi, lam, qubits[6])
    squin.broadcast.sqrt_y_adj(qubits[:6])
    evens = qubits[::2]
    odds = qubits[1::2]

    squin.broadcast.cz(odds, evens[:-1])
    squin.sqrt_y(qubits[6])
    squin.broadcast.cz(evens[:-1], ilist.IList([qubits[3], qubits[5], qubits[6]]))
    squin.broadcast.sqrt_y(qubits[2:])
    squin.broadcast.cz(evens[:-1], odds)
    squin.broadcast.sqrt_y(ilist.IList([qubits[1], qubits[2], qubits[4]]))


main = compile_squin(main, transversal_rewrite=True)

arch_spec = generate_arch(4, 5)
interp = atom.AtomInterpreter(main.dialects, arch_spec=arch_spec)
frame, _ = interp.run(main)

rule = move2squin.InsertQubits()
rewrite.Walk(rule).rewrite(main.code)
rewrite.Walk(
    move2squin.RewriteMoveDialect(
        arch_spec, tuple(rule.physical_ssa_values), frame.atom_state_map, initialize
    )
).rewrite(main.code)
main.print()
