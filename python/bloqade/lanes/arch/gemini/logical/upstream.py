from typing import Any, Iterator, Literal, Sequence, TypeVar

from kirin import ir, rewrite
from kirin.dialects import debug, ilist

from bloqade import qubit, squin
from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress
from bloqade.lanes.passes import TransversalRewritePass

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


N = TypeVar("N")

BroadcastInitKernel = ir.Method[
    [
        ilist.IList[float, Any],
        ilist.IList[float, Any],
        ilist.IList[float, Any],
        ilist.IList[ilist.IList[qubit.Qubit, Literal[7]], Any],
    ],
    None,
]
"""Broadcasted Steane [[7,1,3]] initialization kernel.

Runs state-prep in parallel over ``N`` logical qubits: per-qubit ``theta``/
``phi``/``lam`` lists and a list of ``N`` seven-qubit physical registers.
"""


@squin.kernel
def steane7_initialize_broadcast(
    theta: ilist.IList[float, N],
    phi: ilist.IList[float, N],
    lam: ilist.IList[float, N],
    qubits: ilist.IList[ilist.IList[qubit.Qubit, Literal[7]], N],
):

    num_rows = len(qubits)
    num_cols = len(qubits[0])

    def _new_row(j: int):
        def _get(i: int):
            return qubits[i][j]

        return ilist.map(_get, ilist.range(num_rows))

    qubits_t = ilist.map(_new_row, ilist.range(num_cols))

    def get_rows(indices):
        def _inner(cumulant, i):
            return cumulant + qubits_t[i]

        return ilist.foldl(_inner, indices, ilist.IList([]))

    debug.info("Begin Steane7 Initialize")
    for i in range(len(theta)):
        qubit = qubits[i][6]
        squin.u3(theta[i], phi[i], lam[i], qubit)

    evens = [0, 2, 4, 6]
    odds = [1, 3, 5]

    squin.broadcast.sqrt_y_adj(get_rows([0, 1, 2, 3, 4, 5]))
    squin.broadcast.cz(get_rows(odds), get_rows(evens[1:]))
    squin.broadcast.sqrt_y(get_rows([6]))
    squin.broadcast.cz(get_rows(evens[:-1]), get_rows([3, 5, 6]))
    squin.broadcast.sqrt_y(get_rows([2, 3, 4, 5, 6]))
    squin.broadcast.cz(get_rows(evens[:-1]), get_rows(odds))
    squin.broadcast.sqrt_y(get_rows([1, 2, 4]))
    squin.broadcast.x(get_rows([3]))
    squin.broadcast.z(get_rows([1, 5]))

    debug.info("End Steane7 Initialize")


@squin.kernel
def steane7_initialize(
    theta: float, phi: float, lam: float, qubits: ilist.IList[qubit.Qubit, Literal[7]]
):
    return steane7_initialize_broadcast(
        ilist.IList([theta]),
        ilist.IList([phi]),
        ilist.IList([lam]),
        ilist.IList([qubits]),
    )


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
    cz_errors: Sequence[float] | None = None,
    loss: bool = True,
) -> tuple[BroadcastInitKernel, BroadcastInitKernel]:
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
        cz_errors: 2-qubit pauli error probabilities for CZ execution.
        loss: Whether to include loss channels.

    Returns:
        A tuple of (clean_kernel, noisy_kernel).
    """

    if cz_errors is None:
        cz_errors = 15 * [0.0]

    cz_errors_ilist = ilist.IList(cz_errors)

    @squin.kernel
    def noisy_initialize_broadcast(
        theta: ilist.IList[float, N],
        phi: ilist.IList[float, N],
        lam: ilist.IList[float, N],
        qubits: ilist.IList[ilist.IList[qubit.Qubit, Literal[7]], N],
    ):

        num_rows = len(qubits)
        num_cols = len(qubits[0])

        def _new_row(j: int):
            def _get(i: int):
                return qubits[i][j]

            return ilist.map(_get, ilist.range(num_rows))

        qubits_t = ilist.map(_new_row, ilist.range(num_cols))

        def get_rows(indices):
            def _inner(cumulant, i):
                return cumulant + qubits_t[i]

            return ilist.foldl(_inner, indices, ilist.IList([]))

        debug.info("Begin Steane7 Noisy Initialize")

        evens = [0, 2, 4, 6]
        odds = [1, 3, 5]

        # Cache each column selection once; every gate layer applies a Pauli
        # channel (and optional loss) to the same qubits it gated.
        col6 = get_rows([6])
        cols0_5 = get_rows([0, 1, 2, 3, 4, 5])
        cols_odds = get_rows(odds)
        cols_evens_tail = get_rows(evens[1:])  # [2, 4, 6]
        cols_evens_head = get_rows(evens[:-1])  # [0, 2, 4]
        cols_356 = get_rows([3, 5, 6])
        cols2_6 = get_rows([2, 3, 4, 5, 6])
        cols124 = get_rows([1, 2, 4])
        col3 = get_rows([3])
        cols15 = get_rows([1, 5])

        # U3 on column 6
        for i in range(len(theta)):
            squin.u3(theta[i], phi[i], lam[i], qubits[i][6])
        squin.broadcast.single_qubit_pauli_channel(local_px, local_py, local_pz, col6)
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, col6)

        # sqrt_y_adj on columns 0-5
        squin.broadcast.sqrt_y_adj(cols0_5)
        squin.broadcast.single_qubit_pauli_channel(
            local_px, local_py, local_pz, cols0_5
        )
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, cols0_5)

        # CZ layer 1: controls=odds (sitters), targets=evens[1:] (movers)
        squin.broadcast.single_qubit_pauli_channel(
            sitter_px, sitter_py, sitter_pz, cols_odds
        )
        squin.broadcast.single_qubit_pauli_channel(
            mover_px, mover_py, mover_pz, cols_evens_tail
        )
        squin.broadcast.cz(cols_odds, cols_evens_tail)
        squin.broadcast.two_qubit_pauli_channel(
            cz_errors_ilist, cols_odds, cols_evens_tail
        )
        squin.broadcast.single_qubit_pauli_channel(
            sitter_px, sitter_py, sitter_pz, cols_odds
        )
        squin.broadcast.single_qubit_pauli_channel(
            mover_px, mover_py, mover_pz, cols_evens_tail
        )
        if loss:
            squin.broadcast.qubit_loss(sit_loss_prob, cols_odds)
            squin.broadcast.qubit_loss(move_loss_prob, cols_evens_tail)

        # sqrt_y on column 6
        squin.broadcast.sqrt_y(col6)
        squin.broadcast.single_qubit_pauli_channel(local_px, local_py, local_pz, col6)
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, col6)

        # CZ layer 2: controls=evens[:-1] (movers), targets=[3,5,6] (sitters)
        squin.broadcast.single_qubit_pauli_channel(
            mover_px, mover_py, mover_pz, cols_evens_head
        )
        squin.broadcast.single_qubit_pauli_channel(
            sitter_px, sitter_py, sitter_pz, cols_356
        )
        squin.broadcast.cz(cols_evens_head, cols_356)
        squin.broadcast.two_qubit_pauli_channel(
            cz_errors_ilist, cols_evens_head, cols_356
        )
        squin.broadcast.single_qubit_pauli_channel(
            mover_px, mover_py, mover_pz, cols_evens_head
        )
        squin.broadcast.single_qubit_pauli_channel(
            sitter_px, sitter_py, sitter_pz, cols_356
        )
        if loss:
            squin.broadcast.qubit_loss(move_loss_prob, cols_evens_head)
            squin.broadcast.qubit_loss(sit_loss_prob, cols_356)

        # sqrt_y on columns 2-6
        squin.broadcast.sqrt_y(cols2_6)
        squin.broadcast.single_qubit_pauli_channel(
            local_px, local_py, local_pz, cols2_6
        )
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, cols2_6)

        # CZ layer 3: controls=evens[:-1] (sitters), targets=odds (movers)
        squin.broadcast.single_qubit_pauli_channel(
            sitter_px, sitter_py, sitter_pz, cols_evens_head
        )
        squin.broadcast.single_qubit_pauli_channel(
            mover_px, mover_py, mover_pz, cols_odds
        )
        squin.broadcast.cz(cols_evens_head, cols_odds)
        squin.broadcast.two_qubit_pauli_channel(
            cz_errors_ilist, cols_evens_head, cols_odds
        )
        squin.broadcast.single_qubit_pauli_channel(
            sitter_px, sitter_py, sitter_pz, cols_evens_head
        )
        squin.broadcast.single_qubit_pauli_channel(
            mover_px, mover_py, mover_pz, cols_odds
        )
        if loss:
            squin.broadcast.qubit_loss(sit_loss_prob, cols_evens_head)
            squin.broadcast.qubit_loss(move_loss_prob, cols_odds)

        # sqrt_y on [1, 2, 4]
        squin.broadcast.sqrt_y(cols124)
        squin.broadcast.single_qubit_pauli_channel(
            local_px, local_py, local_pz, cols124
        )
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, cols124)

        # X on column 3
        squin.broadcast.x(col3)
        squin.broadcast.single_qubit_pauli_channel(local_px, local_py, local_pz, col3)
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, col3)

        # Z on [1, 5]
        squin.broadcast.z(cols15)
        squin.broadcast.single_qubit_pauli_channel(local_px, local_py, local_pz, cols15)
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, cols15)

        debug.info("End Steane7 Noisy Initialize")

    return steane7_initialize_broadcast, noisy_initialize_broadcast


class SpecializeGemini:
    def __init__(self, sites_per_word: int = 16):
        self.sites_per_word = sites_per_word

    def emit(self, mt: ir.Method, no_raise=True) -> ir.Method:
        out = mt.similar(dialects=mt.dialects.add(dialect))

        TransversalRewritePass(
            out.dialects, transversal_location_map=steane7_transversal_map
        )(out)

        rewrite.Walk(
            rewrite.Chain(
                RewriteMoves(sites_per_word=self.sites_per_word),
                RewriteFill(),
                RewriteInitialize(),
            )
        ).rewrite(out.code)

        if not no_raise:
            out.verify()

        return out
