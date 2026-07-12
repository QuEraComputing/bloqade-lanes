"""[[8,3,2]] logical-gadget demo for the movement-level primitives.

This demo shows how to build *physically correct* logical gadgets for the
[[8,3,2]] code (the smallest 3D color code, on a single cube) using the
arrange dialect and squin gates, and how the move compiler turns them into
physical atom movement.

[[8,3,2]] conventions used throughout
-------------------------------------
Qubits live on the 8 vertices of a cube, v = (x, y, z) with x,y,z in {0,1},
labelled by the integer n = 4*x + 2*y + z:

    n : (x,y,z) : parity x^y^z
    0 : (0,0,0) : 0 (even)
    1 : (0,0,1) : 1 (odd)
    2 : (0,1,0) : 1 (odd)
    3 : (0,1,1) : 0 (even)
    4 : (1,0,0) : 1 (odd)
    5 : (1,0,1) : 0 (even)
    6 : (1,1,0) : 0 (even)
    7 : (1,1,1) : 1 (odd)

  * even-parity set = {0, 3, 5, 6},  odd-parity set = {1, 2, 4, 7}
    (the cube's bipartite 2-colouring).
  * Stabilisers: one weight-8 X cell, S_X = X^{(x)8}, plus four independent
    weight-4 Z faces.
  * Logical operators: X-bar_i = X on the face perpendicular to axis i
    (weight 4); Z-bar_i = Z on an edge along axis i (weight 2).
  * Logical zero |000>_L is the unique +1 eigenstate of every stabiliser and of
    every Z-bar_i. That leaves only the computational states 00000000 and
    11111111, symmetrised by S_X = X^{(x)8}, so

        |000>_L = (|0>^{(x)8} + |1>^{(x)8}) / sqrt(2)   == the 8-qubit GHZ state.

Gadgets implemented
-------------------
  * allocator + |000>_L init      : qubit.new_at + GHZ preparation.
  * transversal CX (two blocks)   : CX^{(x)8} == logical CX-bar on all 3 logical
                                    pairs (CSS transversality).
  * logical CCZ (one block)       : T on even-parity vertices, T-dagger on odd
                                    == logical CCZ-bar_{123} (the signature
                                    transversal non-Clifford gate of [[8,3,2]]).
  * logical SWAP (one block)      : SWAP of logical qubits 1<->2 via the cube
                                    vertex permutation for the x<->y axis swap,
                                    driven by arrange.permute.

The VIRTUAL flag (below) selects how the logical SWAP's single ``arrange.permute``
is realized, via ``insert_moves = not VIRTUAL``: VIRTUAL = False commits the
physical moves now (the atoms are rearranged in place), while VIRTUAL = True is a
free software relabel with no atom movement (the swap is absorbed lazily by a
later transversal move).
"""

from typing import Any, Literal, TypeVar

from bloqade.types import Qubit
from kirin.dialects import ilist

from bloqade import squin
from bloqade.gemini import physical
from bloqade.gemini.common.dialects import arrange, qubit
from bloqade.lanes.heuristics.physical import make_physical_placement_strategy
from bloqade.lanes.passes import ASAPPlacePass
from bloqade.lanes.pipeline import PhysicalPipeline
from bloqade.lanes.visualize import debugger

# ---------------------------------------------------------------------------
# Toggle: VIRTUAL selects how the logical relabel (arrange.permute) is realized,
# via insert_moves = not VIRTUAL:
#   VIRTUAL = False -> insert_moves=True : the permutation also commits physical
#             moves, so the atoms are rearranged in place now.
#   VIRTUAL = True  -> insert_moves=False: a pure software relabel -- no move
#             instructions are emitted; the move compiler only moves atoms
#             lazily when a later gate needs them adjacent.
# ---------------------------------------------------------------------------
VIRTUAL = False

# arrange.permute's insert_moves attribute. Derived here at module scope (rather
# than as `insert_moves=not VIRTUAL` inside the kernel) because the kernel
# lowering captures a bare global constant but cannot lower a `not` expression.
INSERT_MOVES = not VIRTUAL

N = TypeVar("N")

# An [[8,3,2]] logical block is exactly 8 physical qubits (cube vertices 0..7).
LogicalBlock = ilist.IList[Qubit, Literal[8]]

# even / odd parity 2-colouring of the cube vertices.
EVEN_VERTICES = ilist.IList([0, 3, 5, 6])
ODD_VERTICES = ilist.IList([1, 2, 4, 7])

# Vertex permutation realising the x<->y axis swap, i.e. v=(x,y,z) -> (y,x,z).
# As a arrange.permute argument: qubits[i] moves to the current location of
# qubits[SWAP_XY[i]]. It is an involution (fixes the x=y diagonal, swaps 2<->4
# and 3<->5) and implements logical SWAP-bar between logical qubits 1 and 2.
SWAP_XY = ilist.IList([0, 1, 4, 5, 2, 3, 6, 7])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@physical.kernel(verify=False)
def flat(blocks: ilist.IList[LogicalBlock, Any]) -> ilist.IList[Qubit, Any]:
    """Flatten a list of logical blocks into a single list of physical qubits."""

    def _concat(cumulant, block):
        return cumulant + block

    return ilist.foldl(_concat, blocks, ilist.IList([]))


@physical.kernel(verify=False)
def init_logical_zero(reg: LogicalBlock):
    """Prepare |000>_L = GHZ_8 on a freshly allocated block.

    A Hadamard on vertex 0 followed by a CNOT chain along the vertices yields
    (|0>^{(x)8} + |1>^{(x)8}) / sqrt(2). The move compiler stages the pairs.
    """
    squin.h(reg[0])
    for i in range(7):
        squin.cx(reg[i], reg[i + 1])


# ---------------------------------------------------------------------------
# Allocator (slot-based, mirrors demo/steane_demo.py)
# ---------------------------------------------------------------------------
def eight_three_two_allocator():
    """Build (qalloc, qalloc_slot) for [[8,3,2]] blocks.

    Each canonical slot occupies one zone-0 word, with cube vertex n placed at
    site n of that word; the words are spread out to give the placement engine
    room for parallelism. The concrete word choice is a layout convenience, not
    a correctness concern.
    """
    slot_words = ilist.IList([0, 4, 8, 12, 16, 2, 6, 10])

    @physical.kernel(verify=False)
    def qalloc_slot(slot_index: int) -> LogicalBlock:
        word = slot_words[slot_index]

        def _alloc_vertex(site: int):
            # vertex n -> (zone=0, this slot's word, site=n)
            return qubit.new_at(0, word, site)

        reg = ilist.map(_alloc_vertex, ilist.range(8))
        init_logical_zero(reg)
        return reg

    @physical.kernel(verify=False)
    def qalloc(slot_indices: ilist.IList[int, N]) -> ilist.IList[LogicalBlock, N]:
        def _inner(slot_index: int):
            return qalloc_slot(slot_index)

        return ilist.map(_inner, slot_indices)

    return qalloc, qalloc_slot


qalloc, qalloc_slot = eight_three_two_allocator()


# ---------------------------------------------------------------------------
# Transversal Clifford gadget: CX between two blocks
# ---------------------------------------------------------------------------
@physical.kernel(verify=False)
def transversal_cx(
    controls: ilist.IList[LogicalBlock, N],
    targets: ilist.IList[LogicalBlock, N],
):
    """Transversal CX between two [[8,3,2]] blocks.

    CX^{(x)8} on the 8 aligned vertex pairs implements logical CX-bar on all
    three logical-qubit pairs simultaneously (CSS transversality). The move
    compiler stages each pair adjacent for its entangling layer.
    """
    squin.broadcast.cx(flat(controls), flat(targets))


# ---------------------------------------------------------------------------
# Transversal non-Clifford gadget: logical CCZ (the signature [[8,3,2]] gate)
# ---------------------------------------------------------------------------
@physical.kernel(verify=False)
def logical_ccz(reg: ilist.IList[LogicalBlock, Any]):
    """Logical CCZ-bar_{123} via T on even-parity vertices, T-dagger on odd,
    broadcast across every block in ``reg``.

    This diagonal transversal gate (per block: T on {0,3,5,6}, T-dagger on
    {1,2,4,7}) is the defining transversal non-Clifford gate of the [[8,3,2]]
    code. CCZ is Hermitian, so the even/odd assignment fixes the gate up to its
    (trivial) inverse. Gathering the even/odd vertices of every block and
    issuing a single broadcast applies the gadget to all blocks at once.
    """

    def _even(block: LogicalBlock):
        def _pick(v: int):
            return block[v]

        return ilist.map(_pick, EVEN_VERTICES)

    def _odd(block: LogicalBlock):
        def _pick(v: int):
            return block[v]

        return ilist.map(_pick, ODD_VERTICES)

    squin.broadcast.t(flat(ilist.map(_even, reg)))
    squin.broadcast.t_adj(flat(ilist.map(_odd, reg)))


# ---------------------------------------------------------------------------
# Intra-block logical gadget: SWAP of logical qubits 1<->2 via arrange.permute
# ---------------------------------------------------------------------------
@physical.kernel(verify=False)
def logical_swap(reg: ilist.IList[LogicalBlock, Any]):
    """Logical SWAP-bar_{12} realised by the x<->y cube vertex permutation,
    broadcast across every block in ``reg``.

    Extends SWAP_XY to a permutation over the whole flattened register (each
    block's SWAP_XY offset by its block position) and issues a single
    arrange.permute. The relabel is tracked by the compiler -- afterwards
    reg[b][i] refers to what was reg[b][SWAP_XY[i]] -- so downstream gates see
    the swapped logical qubits with no need to rebuild or return the blocks.
    With insert_moves = not VIRTUAL, VIRTUAL True is a free software relabel (no
    atoms move; a later transversal move absorbs it lazily) while VIRTUAL False
    also commits the physical moves so the code blocks are rearranged in place.
    """
    block_size = len(SWAP_XY)

    def _extend(acc: ilist.IList[int, Any], p: int):
        def _add(ele: int):
            return ele + p * block_size

        return acc + ilist.map(_add, SWAP_XY)

    perm = ilist.foldl(_extend, ilist.range(len(reg)), ilist.IList([]))
    arrange.permute(flat(reg), perm, insert_moves=INSERT_MOVES)


# ---------------------------------------------------------------------------
# Measurement (single terminal measure, regrouped per block)
# ---------------------------------------------------------------------------
@physical.kernel(verify=False)
def measure_logical_block(blocks: ilist.IList[LogicalBlock, Any]):
    """Flatten all blocks, issue the single allowed terminal measure, then
    regroup the bits into per-block (8-qubit) slices."""
    measurements = squin.broadcast.measure(flat(blocks))
    groups = []
    for i in range(len(blocks)):
        groups = groups + [measurements[8 * i : 8 * i + 8]]
    return groups


# ---------------------------------------------------------------------------
# Compose: two blocks, transversal CX, logical CCZ, logical SWAP
# ---------------------------------------------------------------------------
@physical.kernel(aggressive_unroll=True, verify=False)
def main():
    blocks = qalloc(ilist.IList([0, 1]))

    # logical CX-bar(block 0 -> block 1) on all three logical pairs
    transversal_cx(blocks[0:1], blocks[1:2])

    # logical SWAP-bar_{12} on block 0 (passed as a 1-block list; physical moves
    # committed iff VIRTUAL is False). The relabel is tracked by the compiler, so
    # `blocks` keeps referring to the same qubits with block 0's labels swapped.
    logical_swap(blocks[0:1])

    # logical CCZ-bar_{123} on every block (broadcast), acting on block 0's
    # relabelled logical qubits produced by the swap above.
    logical_ccz(blocks)

    return measure_logical_block(blocks)


# ---------------------------------------------------------------------------
# Compile through the physical pipeline and visualise the resulting moves
# ---------------------------------------------------------------------------
strat = make_physical_placement_strategy(
    return_moves=False, move_solutions_per_layer=3, search_budget=None
)
pipeline = PhysicalPipeline(placement_strategy=strat, place_opt_type=ASAPPlacePass)

# no_raise=False so a compilation failure is loud instead of silently emitting
# a degenerate (empty) program.
compiled = pipeline.emit(main, no_raise=False)
debugger(compiled, pipeline.arch_spec)
