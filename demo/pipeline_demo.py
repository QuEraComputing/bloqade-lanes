import io
from typing import Any

from bloqade.gemini.dialects import logical as gemini_logical
from bloqade.stim.emit import EmitStimMain
from bloqade.stim.upstream import squin_to_stim
from kirin.dialects import debug, ilist

from bloqade import qubit, squin, stim
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

    gemini_logical.terminal_measure(reg)


# @kernel
# def main():
#     q = squin.qalloc(9)

#     # random params
#     theta = 0.234
#     phi_0 = 0.123
#     phi_1 = 0.934
#     phi_2 = 0.343

#     # initialization (encoding)
#     squin.rx(phi_2, q[2])

#     squin.rx(2 * theta, q[3])
#     squin.h(q[3])

#     squin.rx(2 * theta, q[4])
#     squin.h(q[4])

#     squin.rx(phi_1, q[5])

#     squin.rx(2 * theta, q[6])
#     squin.h(q[6])

#     squin.rx(2 * theta, q[7])
#     squin.h(q[7])

#     squin.rx(phi_0, q[8])

#     # transversal logic
#     squin.cx(q[2], q[0])
#     squin.cx(q[3], q[1])
#     squin.cx(q[0], q[1])
#     squin.cx(q[4], q[1])
#     squin.cx(q[5], q[0])
#     squin.cx(q[6], q[1])
#     squin.cx(q[0], q[1])
#     squin.cx(q[7], q[1])
#     squin.cx(q[8], q[0])

#     gemini_logical.terminal_measure(q)


@squin.kernel
def cz_unpaired_noise(qubits: ilist.IList[qubit.Qubit, Any]):
    debug.info("CZ Unpaired Noise")
    squin.broadcast.depolarize(0.001, qubits)
    squin.broadcast.qubit_loss(0.0001, qubits)


@squin.kernel
def lane_noise(qubit: qubit.Qubit):
    debug.info("Lane Noise")
    squin.depolarize(0.002, qubit)
    squin.qubit_loss(0.0002, qubit)


@squin.kernel
def idle_noise(qubits: ilist.IList[qubit.Qubit, Any]):
    debug.info("Idle Noise")
    squin.broadcast.depolarize(0.0005, qubits)
    squin.broadcast.qubit_loss(0.00005, qubits)


# main.print()
arch_spec = generate_arch()

main = compile_squin(main, transversal_rewrite=True)

transformer = MoveToSquinTransformer(
    arch_spec=arch_spec,
    logical_initialization=steane7_initialize,
    noise_model=SimpleNoiseModel(lane_noise, idle_noise, cz_unpaired_noise),
    aggressive_unroll=False,
)

main = transformer.transform(main)
main.print()
main = squin_to_stim(main)

buf = io.StringIO()
emit = EmitStimMain(dialects=stim.main, io=buf)
emit.initialize()
emit.run(node=main)
result = buf.getvalue().strip()
print(result)
