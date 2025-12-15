from typing import Any

from bloqade.gemini.dialects import logical as gemini_logical
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.lanes.arch.gemini.impls import generate_arch
from bloqade.lanes.arch.gemini.logical.upstream import steane7_initialize
from bloqade.lanes.logical_mvp import compile_squin
from bloqade.lanes.transform import MoveToSquinTransformer, SimpleNoiseModel

kernel = squin.kernel.add(gemini_logical.dialect)
kernel.run_pass = squin.kernel.run_pass


@kernel(typeinfer=False, fold=True)
def main():
    size = 8
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
def cz_unpaired_noise(qubits: ilist.IList[qubit.Qubit, Any]):
    squin.broadcast.depolarize(0.001, qubits)
    squin.broadcast.qubit_loss(0.0001, qubits)


@squin.kernel
def lane_noise(qubit: qubit.Qubit):
    squin.depolarize(0.002, qubit)
    squin.qubit_loss(0.0002, qubit)


@squin.kernel
def idle_noise(qubits: ilist.IList[qubit.Qubit, Any]):
    squin.broadcast.depolarize(0.0005, qubits)
    squin.broadcast.qubit_loss(0.00005, qubits)


arch_spec = generate_arch()

main = compile_squin(main, transversal_rewrite=False)
main.print()

transformer = MoveToSquinTransformer(
    arch_spec=arch_spec,
    logical_initialization=steane7_initialize,
    noise_model=SimpleNoiseModel(lane_noise, idle_noise, cz_unpaired_noise),
    aggressive_unroll=False,
)

main = transformer.transform(main)
main.print()
