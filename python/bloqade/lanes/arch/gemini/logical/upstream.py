from typing import Iterator, Literal, TypeVar

from kirin import ir, rewrite
from kirin.dialects import debug, ilist

from bloqade import qubit, squin
from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress
from bloqade.lanes.rewrite.transversal import (
    RewriteGetItem,
    RewriteLogicalToPhysicalConversion,
    RewriteStarRz,
)

from .rewrite import RewriteFill, RewriteInitialize, RewriteMoves
from .stmts import dialect

AddressType = TypeVar("AddressType", bound=LocationAddress | LaneAddress)


def steane7_transversal_map(address: AddressType) -> Iterator[AddressType] | None:
    """Map logical addresses to physical addresses via site expansion.

    The Steane [[7,1,3]] code encodes one logical qubit into seven physical qubits.
    Each logical site expands to 7 physical sites within the same word:

        Logical site s → Physical sites s*7, s*7+1, ..., s*7+6

    Word ID is preserved. Only expands logical site IDs (0 and 1).
    Returns None for site IDs >= 2 (already physical / not a logical site).
    """
    if address.site_id >= 2:
        return None
    base = address.site_id * 7
    return (address.replace(site_id=base + i) for i in range(7))


@squin.kernel
def steane7_initialize(
    theta: float, phi: float, lam: float, qubits: ilist.IList[qubit.Qubit, Literal[7]]
):
    debug.info("Begin Steane7 Initialize")
    squin.u3(theta, phi, lam, qubits[6])
    squin.broadcast.sqrt_y_adj(qubits[:6])
    evens = qubits[::2]  # [0, 2, 4, 6]
    odds = qubits[1::2]  # [1, 3, 5]

    # Fixed: CZ pairs should be (1,2), (3,4), (5,6) not (1,0), (3,2), (5,4)
    squin.broadcast.cz(odds, evens[1:])
    squin.sqrt_y(qubits[6])
    squin.broadcast.cz(evens[:-1], ilist.IList([qubits[3], qubits[5], qubits[6]]))
    squin.broadcast.sqrt_y(qubits[2:])
    squin.broadcast.cz(evens[:-1], odds)
    squin.broadcast.sqrt_y(ilist.IList([qubits[1], qubits[2], qubits[4]]))
    squin.x(qubits[3])
    squin.broadcast.z(ilist.IList([qubits[1], qubits[5]]))
    debug.info("End Steane7 Initialize")


def steane7_initialize_with_noise(
    local_px: float = 0.0,
    local_py: float = 0.0,
    local_pz: float = 0.0,
    local_loss_prob: float = 0.0,
    mover_px: float = 0.0,
    mover_py: float = 0.0,
    mover_pz: float = 0.0,
    move_loss_prob: float = 0.0,
    sitter_px: float = 0.0,
    sitter_py: float = 0.0,
    sitter_pz: float = 0.0,
    sit_loss_prob: float = 0.0,
    loss: bool = True,
) -> tuple[
    ir.Method[[float, float, float, ilist.IList[qubit.Qubit, Literal[7]]], None],
    ir.Method[[float, float, float, ilist.IList[qubit.Qubit, Literal[7]]], None],
]:
    """Return (clean_kernel, noisy_kernel) for Steane [[7,1,3]] initialization.

    The clean kernel is the ideal initialization circuit. The noisy kernel is
    the same circuit with single-qubit Pauli channels after each gate layer
    and move/sitter noise after each CZ layer, parameterized by the provided
    error probabilities.

    For CZ layers, one side physically moves to the other. The mover/sitter
    assignment alternates across layers:

    - Layer 1 (odds x evens[1:]): targets move, controls sit
    - Layer 2 (evens[:-1] x [3,5,6]): controls move, targets sit
    - Layer 3 (evens[:-1] x odds): targets move, controls sit

    Args:
        local_px: X-error probability for single-qubit gates.
        local_py: Y-error probability for single-qubit gates.
        local_pz: Z-error probability for single-qubit gates.
        local_loss_prob: Loss probability for single-qubit gates.
        mover_px: X-error probability for qubits that move during CZ.
        mover_py: Y-error probability for qubits that move during CZ.
        mover_pz: Z-error probability for qubits that move during CZ.
        move_loss_prob: Loss probability for moving qubits during CZ.
        sitter_px: X-error probability for stationary qubits during CZ.
        sitter_py: Y-error probability for stationary qubits during CZ.
        sitter_pz: Z-error probability for stationary qubits during CZ.
        sit_loss_prob: Loss probability for stationary qubits during CZ.
        loss: Whether to include loss channels.

    Returns:
        A tuple of (clean_kernel, noisy_kernel).
    """

    @squin.kernel
    def noisy_initialize(
        theta: float,
        phi: float,
        lam: float,
        qubits: ilist.IList[qubit.Qubit, Literal[7]],
    ):
        debug.info("Begin Steane7 Noisy Initialize")

        # U3 on qubit 6
        squin.u3(theta, phi, lam, qubits[6])
        squin.single_qubit_pauli_channel(local_px, local_py, local_pz, qubits[6])
        if loss:
            squin.qubit_loss(local_loss_prob, qubits[6])

        # sqrt_y_adj on qubits 0-5
        squin.broadcast.sqrt_y_adj(qubits[:6])
        squin.broadcast.single_qubit_pauli_channel(
            local_px, local_py, local_pz, qubits[:6]
        )
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, qubits[:6])

        evens = qubits[::2]  # [0, 2, 4, 6]
        odds = qubits[1::2]  # [1, 3, 5]

        # CZ layer 1: controls=odds (sitters), targets=evens[1:] (movers)
        squin.broadcast.cz(odds, evens[1:])
        squin.broadcast.single_qubit_pauli_channel(
            sitter_px, sitter_py, sitter_pz, odds
        )
        squin.broadcast.single_qubit_pauli_channel(
            mover_px, mover_py, mover_pz, evens[1:]
        )
        if loss:
            squin.broadcast.qubit_loss(sit_loss_prob, odds)
            squin.broadcast.qubit_loss(move_loss_prob, evens[1:])

        # sqrt_y on qubit 6
        squin.sqrt_y(qubits[6])
        squin.single_qubit_pauli_channel(local_px, local_py, local_pz, qubits[6])
        if loss:
            squin.qubit_loss(local_loss_prob, qubits[6])

        # CZ layer 2: controls=evens[:-1] (movers), targets=[3,5,6] (sitters)
        cz_targets = ilist.IList([qubits[3], qubits[5], qubits[6]])
        squin.broadcast.cz(evens[:-1], cz_targets)
        squin.broadcast.single_qubit_pauli_channel(
            mover_px, mover_py, mover_pz, evens[:-1]
        )
        squin.broadcast.single_qubit_pauli_channel(
            sitter_px, sitter_py, sitter_pz, cz_targets
        )
        if loss:
            squin.broadcast.qubit_loss(move_loss_prob, evens[:-1])
            squin.broadcast.qubit_loss(sit_loss_prob, cz_targets)

        # sqrt_y on qubits 2-6
        squin.broadcast.sqrt_y(qubits[2:])
        squin.broadcast.single_qubit_pauli_channel(
            local_px, local_py, local_pz, qubits[2:]
        )
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, qubits[2:])

        # CZ layer 3: controls=evens[:-1] (sitters), targets=odds (movers)
        squin.broadcast.cz(evens[:-1], odds)
        squin.broadcast.single_qubit_pauli_channel(
            sitter_px, sitter_py, sitter_pz, evens[:-1]
        )
        squin.broadcast.single_qubit_pauli_channel(mover_px, mover_py, mover_pz, odds)
        if loss:
            squin.broadcast.qubit_loss(sit_loss_prob, evens[:-1])
            squin.broadcast.qubit_loss(move_loss_prob, odds)

        # sqrt_y on [1, 2, 4]
        correction_qubits = ilist.IList([qubits[1], qubits[2], qubits[4]])
        squin.broadcast.sqrt_y(correction_qubits)
        squin.broadcast.single_qubit_pauli_channel(
            local_px, local_py, local_pz, correction_qubits
        )
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, correction_qubits)

        # X on qubit 3
        squin.x(qubits[3])
        squin.single_qubit_pauli_channel(local_px, local_py, local_pz, qubits[3])
        if loss:
            squin.qubit_loss(local_loss_prob, qubits[3])

        # Z on [1, 5]
        z_qubits = ilist.IList([qubits[1], qubits[5]])
        squin.broadcast.z(z_qubits)
        squin.broadcast.single_qubit_pauli_channel(
            local_px, local_py, local_pz, z_qubits
        )
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, z_qubits)

        debug.info("End Steane7 Noisy Initialize")

    return steane7_initialize, noisy_initialize


class SpecializeGemini:

    def __init__(self, sites_per_word: int = 16):
        self.sites_per_word = sites_per_word

    def emit(self, mt: ir.Method, no_raise=True) -> ir.Method:
        out = mt.similar(dialects=mt.dialects.add(dialect))

        rewrite.Walk(
            rewrite.Chain(
                RewriteStarRz(steane7_transversal_map),
                RewriteMoves(sites_per_word=self.sites_per_word),
                RewriteFill(),
                RewriteInitialize(),
                RewriteGetItem(steane7_transversal_map),
                RewriteLogicalToPhysicalConversion(),
            )
        ).rewrite(out.code)

        if not no_raise:
            out.verify()

        return out
