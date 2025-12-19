import io
from typing import Any

from bloqade.gemini.dialects import logical as gemini_logical
from bloqade.stim.emit import EmitStimMain
from bloqade.stim.upstream import squin_to_stim
from kirin.dialects import debug, ilist

from bloqade import qubit, squin, stim
from bloqade.lanes.logical_mvp import (
    compile_to_physical_squin_noise_model,
)
from bloqade.lanes.transform import SimpleNoiseModel

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

    return gemini_logical.terminal_measure(q)


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


noise_kernel = compile_to_physical_squin_noise_model(
    main,
    SimpleNoiseModel(lane_noise, idle_noise, cz_unpaired_noise),
)
noise_kernel.print()
noise_kernel = squin_to_stim(noise_kernel)

buf = io.StringIO()
emit = EmitStimMain(dialects=stim.main, io=buf)
emit.initialize()
emit.run(node=noise_kernel)
result = buf.getvalue().strip()
print(result)
