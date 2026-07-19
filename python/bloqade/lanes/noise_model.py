from typing import TYPE_CHECKING, Any

from kirin.dialects import debug, ilist

from bloqade import qubit, squin
from bloqade.lanes.transform import SimpleLogicalNoiseModel, SimpleNoiseModel

if TYPE_CHECKING:
    from bloqade.cirq_utils.noise.model import (
        GeminiNoiseModelABC,
    )

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
    noise_model: "GeminiNoiseModelABC | None" = None,
    loss: bool = True,
) -> SimpleNoiseModel:
    """Generate a physical noise model from a bloqade-circuit noise model.

    Args:
        noise_model: The bloqade-circuit noise model to use. Defaults to None.
        loss: Whether to include loss in the noise model. Defaults to True.

    Returns:
        A simple noise model for physical gate/move noise insertion.
    """
    from bloqade.cirq_utils.noise.model import GeminiOneZoneNoiseModel

    if noise_model is None:
        noise_model = GeminiOneZoneNoiseModel()

    # Read Pauli rates through the *_pauli_rates / two_qubit_pauli getters so the
    # upstream ``scaling_factor`` is honored (the raw ``*_px`` fields ignore it).
    # Loss probabilities are read raw on purpose: upstream ``scaling_factor``
    # only scales the Pauli rates, not loss. Per-category loss scaling is tracked
    # upstream in QuEraComputing/bloqade-circuit#836.
    cz_unpaired_loss_prob = noise_model.cz_unpaired_loss_prob
    cz_unpaired_gate_px, cz_unpaired_gate_py, cz_unpaired_gate_pz = (
        noise_model.cz_unpaired_pauli_rates
    )

    @squin.kernel
    def cz_unpaired_noise(qubits: ilist.IList[qubit.Qubit, Any]):
        debug.info("CZ Unpaired Noise")
        squin.broadcast.single_qubit_pauli_channel(
            cz_unpaired_gate_px, cz_unpaired_gate_py, cz_unpaired_gate_pz, qubits
        )
        if loss:
            squin.broadcast.qubit_loss(cz_unpaired_loss_prob, qubits)

    mover_px, mover_py, mover_pz = noise_model.mover_pauli_rates
    move_lost_prob = noise_model.move_loss_prob

    @squin.kernel
    def lane_noise(qubit: qubit.Qubit):
        debug.info("Lane Noise")
        squin.single_qubit_pauli_channel(mover_px, mover_py, mover_pz, qubit)
        if loss:
            squin.qubit_loss(move_lost_prob, qubit)

    sitter_px, sitter_py, sitter_pz = noise_model.sitter_pauli_rates
    sit_loss_prob = noise_model.sit_loss_prob

    @squin.kernel
    def idle_noise(qubits: ilist.IList[qubit.Qubit, Any]):
        debug.info("Idle Noise")
        squin.broadcast.single_qubit_pauli_channel(
            sitter_px, sitter_py, sitter_pz, qubits
        )
        if loss:
            squin.broadcast.qubit_loss(sit_loss_prob, qubits)

    if noise_model.cz_paired_error_probabilities is None:
        raise ValueError("CZ paired error probabilities must be provided.")

    # two_qubit_pauli applies scaling_factor to the correlated CZ rates; read the
    # scaled dict back off the channel. Missing keys (dropped when a scaled rate
    # hits 0) default to 0.0 rather than raising.
    cz_paired_error_dict = noise_model.two_qubit_pauli.error_probabilities
    cz_paired_error_probabilities = ilist.IList(
        [cz_paired_error_dict.get(k, 0.0) for k in PAIRED_KEYS]
    )

    cz_paired_loss_prob = noise_model.cz_gate_loss_prob

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

        if loss:
            groups = ilist.map(pair_qubit, ilist.range(len(controls)))
            squin.broadcast.correlated_qubit_loss(cz_paired_loss_prob, groups)

    local_px, local_py, local_pz = noise_model.local_pauli_rates
    local_loss_prob = noise_model.local_loss_prob

    @squin.kernel
    def local_r_noise(
        qubits: ilist.IList[qubit.Qubit, Any], axis_angle: float, rotation_angle: float
    ):
        debug.info("Local Gate Noise")
        squin.broadcast.single_qubit_pauli_channel(local_px, local_py, local_pz, qubits)
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, qubits)

    @squin.kernel
    def local_rz_noise(qubits: ilist.IList[qubit.Qubit, Any], rotation_angle: float):
        debug.info("Local Rz Noise")
        squin.broadcast.single_qubit_pauli_channel(local_px, local_py, local_pz, qubits)
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, qubits)

    global_px, global_py, global_pz = noise_model.global_pauli_rates
    global_loss_prob = noise_model.global_loss_prob

    @squin.kernel
    def global_r_noise(
        qubits: ilist.IList[qubit.Qubit, Any], axis_angle: float, rotation_angle: float
    ):
        debug.info("Global Gate Noise")
        squin.broadcast.single_qubit_pauli_channel(
            global_px, global_py, global_pz, qubits
        )
        if loss:
            squin.broadcast.qubit_loss(global_loss_prob, qubits)

    @squin.kernel
    def global_rz_noise(qubits: ilist.IList[qubit.Qubit, Any], rotation_angle: float):
        debug.info("Global Rz Noise")
        squin.broadcast.single_qubit_pauli_channel(
            global_px, global_py, global_pz, qubits
        )
        if loss:
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


def generate_logical_noise_model(
    noise_model: "GeminiNoiseModelABC | None" = None,
    loss: bool = True,
) -> SimpleLogicalNoiseModel:
    """Generate a logical noise model with initialization kernels.

    Creates a physical noise model and adds Steane [[7,1,3]] clean and noisy
    initialization kernels, all derived from the same source parameters.

    Args:
        noise_model: The bloqade-circuit noise model to use. Defaults to None.
        loss: Whether to include loss channels. Defaults to True.

    Returns:
        A logical noise model with gate/move noise and both initialization kernels.
    """
    from bloqade.cirq_utils.noise.model import GeminiOneZoneNoiseModel

    if noise_model is None:
        noise_model = GeminiOneZoneNoiseModel()

    physical = generate_simple_noise_model(noise_model, loss=loss)

    from bloqade.lanes.arch.gemini.logical.upstream import steane7_initialize_with_noise

    # Pauli rates via the scaled getters; loss probabilities raw (see
    # generate_simple_noise_model for the rationale).
    local_px, local_py, local_pz = noise_model.local_pauli_rates
    mover_px, mover_py, mover_pz = noise_model.mover_pauli_rates
    sitter_px, sitter_py, sitter_pz = noise_model.sitter_pauli_rates
    cz_paired_error_dict = noise_model.two_qubit_pauli.error_probabilities
    cz_paired_error_probabilities = ilist.IList(
        [cz_paired_error_dict.get(k, 0.0) for k in PAIRED_KEYS]
    )

    clean_init, noisy_init = steane7_initialize_with_noise(
        local_px=local_px,
        local_py=local_py,
        local_pz=local_pz,
        local_loss_prob=noise_model.local_loss_prob,
        mover_px=mover_px,
        mover_py=mover_py,
        mover_pz=mover_pz,
        move_loss_prob=noise_model.move_loss_prob,
        sitter_px=sitter_px,
        sitter_py=sitter_py,
        sitter_pz=sitter_pz,
        sit_loss_prob=noise_model.sit_loss_prob,
        cz_errors=cz_paired_error_probabilities,
        cz_paired_loss=noise_model.cz_gate_loss_prob,
        cz_unpaired_loss=noise_model.cz_unpaired_loss_prob,
        loss=loss,
    )

    return SimpleLogicalNoiseModel.from_simple(
        physical,
        logical_initialize_clean=clean_init,
        logical_initialize_noisy=noisy_init,
    )
