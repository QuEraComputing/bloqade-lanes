from typing import Any

from bloqade.types import Qubit as Qubit
from kirin import ir
from kirin.analysis import forward
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.lanes import layout
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.rewrite.move2squin import noise


@squin.kernel
def lane_noise_kernel(qubit: qubit.Qubit):
    return


@squin.kernel
def bus_idle_noise_kernel(qubits: ilist.IList[qubit.Qubit, Any]):
    return


@squin.kernel
def cz_unpaired_noise_kernel(qubits: ilist.IList[qubit.Qubit, Any]):
    return


@squin.kernel
def cz_paired_noise_kernel(
    controls: ilist.IList[qubit.Qubit, Any], targets: ilist.IList[qubit.Qubit, Any]
):
    return


@squin.kernel
def global_rz_noise_kernel(qubits: ilist.IList[qubit.Qubit, Any], angle: float):
    return


@squin.kernel
def local_rz_noise_kernel(qubits: ilist.IList[qubit.Qubit, Any], angle: float):
    return


@squin.kernel
def global_r_noise_kernel(
    qubits: ilist.IList[qubit.Qubit, Any], theta: float, phi: float
):
    return


@squin.kernel
def local_r_noise_kernel(
    qubits: ilist.IList[qubit.Qubit, Any], theta: float, phi: float
):
    return


MODEL = noise.SimpleNoiseModel(
    lane_noise=lane_noise_kernel,
    idle_noise=bus_idle_noise_kernel,
    cz_unpaired_noise=cz_unpaired_noise_kernel,
    cz_paired_noise=cz_paired_noise_kernel,
    global_rz_noise=global_rz_noise_kernel,
    local_rz_noise=local_rz_noise_kernel,
    global_r_noise=global_r_noise_kernel,
    local_r_noise=local_r_noise_kernel,
)


def test_insert_move_noise():
    state = ir.TestValue()
    test_block = ir.Block([node := move.Move(state, lanes=())])

    physical_ssa_values = {
        0: (zero := ir.TestValue()),
        1: (one := ir.TestValue()),
    }

    atom_state: Any = ...

    atom_state_map = forward.ForwardFrame(node, entries={node.result: atom_state})

    rewriter = noise.InsertNoise(
        arch_spec=get_arch_spec(),
        physical_ssa_values=physical_ssa_values,  # type: ignore
        atom_state_map=atom_state_map,
        noise_model=MODEL,
    )

    rewriter.rewrite(test_block)

    expected_block = ir.Block(
        [
            move.Move(state, lanes=()),
        ]
    )
