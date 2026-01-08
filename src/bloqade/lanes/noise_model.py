from typing import Any

from bloqade.cirq_utils.noise.model import (
    GeminiNoiseModelABC,
    GeminiOneZoneNoiseModel,
)
from kirin.dialects import debug, ilist

from bloqade import qubit, squin
from bloqade.lanes.transform import SimpleNoiseModel

PAIRED_KEYS = [
    "IX",
    "IY",
    "IZ",
    "XI",
    "XX",
    "XY",
    "XZ",
    "YI",
    "YX",
    "YY",
    "YZ",
    "ZI",
    "ZX",
    "ZY",
    "ZZ",
]


def generate_simple_noise_model(
    noise_model: GeminiNoiseModelABC | None = None,
) -> SimpleNoiseModel:

    if noise_model is None:
        noise_model = GeminiOneZoneNoiseModel()

    cz_unpaired_loss_prob = noise_model.cz_unpaired_loss_prob
    cz_unpaired_gate_px = noise_model.cz_unpaired_gate_px
    cz_unpaired_gate_py = noise_model.cz_unpaired_gate_py
    cz_unpaired_gate_pz = noise_model.cz_unpaired_gate_pz

    @squin.kernel
    def cz_unpaired_noise(qubits: ilist.IList[qubit.Qubit, Any]):
        debug.info("CZ Unpaired Noise")
        squin.broadcast.single_qubit_pauli_channel(
            cz_unpaired_gate_px, cz_unpaired_gate_py, cz_unpaired_gate_pz, qubits
        )
        squin.broadcast.qubit_loss(cz_unpaired_loss_prob, qubits)

    mover_px = noise_model.mover_px
    mover_py = noise_model.mover_py
    mover_pz = noise_model.mover_pz
    move_lost_prob = noise_model.move_loss_prob

    @squin.kernel
    def lane_noise(qubit: qubit.Qubit):
        debug.info("Lane Noise")
        squin.single_qubit_pauli_channel(mover_px, mover_py, mover_pz, qubit)
        squin.qubit_loss(move_lost_prob, qubit)

    sitter_px = noise_model.sitter_px
    sitter_py = noise_model.sitter_py
    sitter_pz = noise_model.sitter_pz
    sit_loss_prob = noise_model.sit_loss_prob

    @squin.kernel
    def idle_noise(qubits: ilist.IList[qubit.Qubit, Any]):
        debug.info("Idle Noise")
        squin.broadcast.single_qubit_pauli_channel(
            sitter_px, sitter_py, sitter_pz, qubits
        )
        squin.broadcast.qubit_loss(sit_loss_prob, qubits)

    cz_paired_error_dict = noise_model.cz_paired_error_probabilities
    if cz_paired_error_dict is None:
        raise ValueError("CZ paired error probabilities must be provided.")

    cz_paired_error_probabilities = ilist.IList(
        [cz_paired_error_dict[k] for k in PAIRED_KEYS]
    )

    cz_unpaired_loss_prob = noise_model.cz_gate_loss_prob

    @squin.kernel
    def cz_paired_noise(
        controls: ilist.IList[qubit.Qubit, Any], targets: ilist.IList[qubit.Qubit, Any]
    ):
        debug.info("CZ Paired Noise")
        squin.broadcast.two_qubit_pauli_channel(
            cz_paired_error_probabilities, controls, targets
        )

        def pair_qubit(i: int):
            return ilist.IList([controls[i], targets[i]])

        groups = ilist.map(pair_qubit, ilist.range(len(controls)))
        squin.broadcast.correlated_qubit_loss(cz_unpaired_loss_prob, groups)

    local_px = noise_model.local_px
    local_py = noise_model.local_py
    local_pz = noise_model.local_pz
    local_loss_prob = noise_model.local_loss_prob

    @squin.kernel
    def local_r_noise(
        qubits: ilist.IList[qubit.Qubit, Any], axis_angle: float, rotation_angle: float
    ):
        debug.info("Local Gate Noise")
        squin.broadcast.single_qubit_pauli_channel(local_px, local_py, local_pz, qubits)
        squin.broadcast.qubit_loss(local_loss_prob, qubits)

    @squin.kernel
    def local_rz_noise(qubits: ilist.IList[qubit.Qubit, Any], rotation_angle: float):
        debug.info("Local Rz Noise")
        squin.broadcast.single_qubit_pauli_channel(local_px, local_py, local_pz, qubits)
        squin.broadcast.qubit_loss(local_loss_prob, qubits)

    global_px = noise_model.global_px
    global_py = noise_model.global_py
    global_pz = noise_model.global_pz
    global_loss_prob = noise_model.global_loss_prob

    @squin.kernel
    def global_r_noise(
        qubits: ilist.IList[qubit.Qubit, Any], axis_angle: float, rotation_angle: float
    ):
        debug.info("Global Gate Noise")
        squin.broadcast.single_qubit_pauli_channel(
            global_px, global_py, global_pz, qubits
        )
        squin.broadcast.qubit_loss(global_loss_prob, qubits)

    @squin.kernel
    def global_rz_noise(qubits: ilist.IList[qubit.Qubit, Any], rotation_angle: float):
        debug.info("Global Rz Noise")
        squin.broadcast.single_qubit_pauli_channel(
            global_px, global_py, global_pz, qubits
        )
        squin.broadcast.qubit_loss(global_loss_prob, qubits)

    return SimpleNoiseModel(
        lane_noise=lane_noise,
        idle_noise=idle_noise,
        cz_unpaired_noise=cz_unpaired_noise,
        cz_paired_noise=cz_paired_noise,
        global_rz_noise=global_rz_noise,
        local_rz_noise=local_rz_noise,
        global_r_noise=global_r_noise,
        local_r_noise=local_r_noise,
    )
