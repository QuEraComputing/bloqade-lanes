from typing import Any, Iterator, Literal, Sequence, TypeVar

from kirin import ir, rewrite
from kirin.dialects import debug, ilist

from bloqade import qubit, squin
from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress
from bloqade.lanes.passes import TransversalRewritePass
from bloqade.rewrite.passes import AggressiveUnroll
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
    cz_paired_loss: float = 0.0,
    cz_unpaired_loss: float = 0.0,
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
        cz_errors: Sequence of 15 2-qubit Pauli error probabilities for CZ execution
            (excluding II).
        loss: Whether to include loss channels.

    Returns:
        A tuple of (clean_kernel, noisy_kernel).
    """

    if cz_errors is None:
        cz_errors = [0.0] * 15
    elif len(cz_errors) != 15:
        raise ValueError(f"cz_errors must have length 15 (got {len(cz_errors)})")

    cz_errors_ilist = ilist.IList(cz_errors)


    M = TypeVar("M")
    K = TypeVar("K")


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

        def get_rows(indices: ilist.IList[int, M]) -> ilist.IList[qubit.Qubit, Any]:
            def _inner(cumulant: ilist.IList[qubit.Qubit, Any], i: int):
                return cumulant + qubits_t[i]
            
            res = ilist.foldl(_inner, indices, ilist.IList([]))
            return res

        def not_in(max: int, indices: ilist.IList[int, Any]):
            def _in(acc: ilist.IList[int, Any], ele: int):
                if ele not in indices:
                    acc = acc + [ele]
                
                return acc
                
            return ilist.foldl(_in, ilist.range(max), ilist.IList([]))
                

        def cz_w_noise(target_ids: ilist.IList[int, M], control_ids: ilist.IList[int, M]):
            rest = not_in(num_cols, target_ids + control_ids)
            targets = get_rows(target_ids)
            controls = get_rows(control_ids)
            movers = controls
            sitters = get_rows(target_ids + rest)
            others = get_rows(rest)
            squin.broadcast.single_qubit_pauli_channel(
                sitter_px, sitter_py, sitter_pz, sitters
            )
            squin.broadcast.single_qubit_pauli_channel(
                mover_px, mover_py, mover_pz, movers
            )
            
            if loss: 
                squin.broadcast.qubit_loss(sit_loss_prob, sitters)
                squin.broadcast.qubit_loss(move_loss_prob, movers)   
                
            squin.broadcast.cz(controls, targets)
            squin.broadcast.two_qubit_pauli_channel(
                cz_errors_ilist, controls, targets
            )
            squin.broadcast.single_qubit_pauli_channel(
                sitter_px, sitter_py, sitter_pz, sitters
            )
            squin.broadcast.single_qubit_pauli_channel(
                mover_px, mover_py, mover_pz, movers
            )
            if loss:
                squin.broadcast.qubit_loss(cz_paired_loss, targets + controls)
                squin.broadcast.qubit_loss(cz_unpaired_loss, others)
                squin.broadcast.qubit_loss(sit_loss_prob, sitters)
                squin.broadcast.qubit_loss(move_loss_prob, movers)   
                         

        def gate_with_noise(indices: ilist.IList[int, Any], gate):
            qubits = get_rows(indices)
            gate(qubits)
            squin.broadcast.single_qubit_pauli_channel(
                local_px, local_py, local_pz, qubits
            )
            if loss:
                squin.broadcast.qubit_loss(local_loss_prob, qubits)
                

        debug.info("Begin Steane7 Noisy Initialize")


        # U3 on column 6
        col6 = get_rows(ilist.IList([6]))

        for i in range(len(theta)):
            squin.u3(theta[i], phi[i], lam[i], qubits[i][6])
        squin.broadcast.single_qubit_pauli_channel(local_px, local_py, local_pz, col6)
        if loss:
            squin.broadcast.qubit_loss(local_loss_prob, col6)

        # sqrt_y_adj on columns 0-5
        gate_with_noise(ilist.IList([0,1,2,3,4,5]), squin.broadcast.sqrt_y_adj)

        # CZ layer 1: controls=odds (sitters), targets=evens[1:] (movers)
        cz_w_noise(ilist.IList([1,3,5]), ilist.IList([2,4,6]))

        # sqrt_y on column 6
        gate_with_noise(ilist.IList([6]), squin.broadcast.sqrt_y)

        # CZ layer 2: controls=evens[:-1] (movers), targets=[3,5,6] (sitters)
        cz_w_noise(ilist.IList([0,2,4]),ilist.IList([3,5,6]))

        # sqrt_y on columns 2-6
        gate_with_noise(ilist.IList([2,3,4,5,6]), squin.broadcast.sqrt_y)

        # CZ layer 3: controls=evens[:-1] (sitters), targets=odds (movers)
        cz_w_noise(ilist.IList([0,2,4]),ilist.IList([1,3,5]))

        # sqrt_y on [1, 2, 4]
        gate_with_noise(ilist.IList([1,2,4]), squin.broadcast.sqrt_y)

        # X on column 3
        gate_with_noise(ilist.IList([3]), squin.broadcast.x)

        # Z on [1, 5]
        gate_with_noise(ilist.IList([1,5]), squin.broadcast.z)

        debug.info("End Steane7 Noisy Initialize")

    AggressiveUnroll(noisy_initialize_broadcast.dialects).fixpoint(noisy_initialize_broadcast)
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
