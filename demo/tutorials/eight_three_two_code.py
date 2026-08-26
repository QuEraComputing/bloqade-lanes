# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Use case: movement gadgets for the `[[8,3,2]]` code
#
# This tutorial builds the movement-level `[[8,3,2]]` example from physical
# kernels and compiles it onto the Gemini physical architecture. It demonstrates
# four code gadgets:
#
# - allocation and preparation of logical `|000>` blocks,
# - transversal CX between two blocks,
# - transversal logical CCZ within a block, and
# - a logical SWAP implemented by a cube-vertex permutation.
#
# The `[[8,3,2]]` code is the smallest three-dimensional color code. Its eight
# physical qubits occupy the vertices `v = (x, y, z)` of a cube, with each
# coordinate in `{0, 1}`. We label a vertex by
#
# ```text
# n = 4*x + 2*y + z
# ```
#
# | `n` | `(x, y, z)` | parity `x ^ y ^ z` |
# | ---: | :---------: | :----------------: |
# | 0 | `(0, 0, 0)` | even |
# | 1 | `(0, 0, 1)` | odd |
# | 2 | `(0, 1, 0)` | odd |
# | 3 | `(0, 1, 1)` | even |
# | 4 | `(1, 0, 0)` | odd |
# | 5 | `(1, 0, 1)` | even |
# | 6 | `(1, 1, 0)` | even |
# | 7 | `(1, 1, 1)` | odd |
#
# The code has one weight-eight X stabilizer and four independent weight-four Z
# face stabilizers. A logical X is supported on a cube face, while a logical Z
# is supported on an edge. The simultaneous `+1` eigenstate of the stabilizers
# and all three logical Z operators is
#
# ```text
# |000>_L = (|00000000> + |11111111>) / sqrt(2),
# ```
#
# so logical-zero preparation is exactly eight-qubit GHZ preparation.

# %% [markdown]
# ## Imports and code conventions
#
# An encoded block contains exactly eight physical qubits. The even and odd
# vertex sets are the cube's bipartite two-coloring. Exchanging the cube's x and
# y axes gives the vertex permutation used for the logical SWAP.
#
# `VIRTUAL` controls how `arrange.permute` is realized:
#
# - `False`: emit atom moves immediately and rearrange the block in place;
# - `True`: only relabel the qubits in software and defer movement until a later
#   interaction requires it.
#
# `insert_moves` must be a compile-time constant, so it is derived at module
# scope rather than inside a kernel.

# %%
from typing import Any, Literal, TypeVar

from bloqade.types import Qubit
from kirin.dialects import ilist

from bloqade import squin
from bloqade.gemini import physical
from bloqade.gemini.common.dialects import arrange, qubit
from bloqade.lanes.dialects.arch import loc
from bloqade.lanes.heuristics.physical import make_physical_placement_strategy
from bloqade.lanes.passes import ASAPPlacePass
from bloqade.lanes.transform import PhysicalPipeline

VIRTUAL = False
INSERT_MOVES = not VIRTUAL

N = TypeVar("N")
LogicalBlock = ilist.IList[Qubit, Literal[8]]

EVEN_VERTICES = ilist.IList([0, 3, 5, 6])
ODD_VERTICES = ilist.IList([1, 2, 4, 7])

# Vertex permutation v=(x,y,z) -> (y,x,z). It fixes the x=y diagonal and
# exchanges vertices 2 <-> 4 and 3 <-> 5.
SWAP_XY = ilist.IList([0, 1, 4, 5, 2, 3, 6, 7])

# %% [markdown]
# ## Shared helpers and logical-zero preparation
#
# Some transversal gadgets receive a list of blocks. `flat` concatenates those
# blocks into one physical register so a broadcast gate can act on every
# aligned pair.
#
# To prepare a logical-zero block, a Hadamard on vertex 0 is followed by a
# log-depth CNOT tree. Each CNOT layer doubles the entangled prefix, producing
# the eight-qubit GHZ state.


# %%
@physical.kernel(verify=False)
def flat(blocks: ilist.IList[LogicalBlock, Any]) -> ilist.IList[Qubit, Any]:
    """Flatten encoded blocks into a single physical-qubit register."""

    def _concat(cumulant, block):
        return cumulant + block

    return ilist.foldl(_concat, blocks, ilist.IList([]))


@physical.kernel(verify=False)
def initialize_logical_zero(register: LogicalBlock):
    """Prepare `|000>_L`, the eight-qubit GHZ state."""
    squin.h(register[0])
    squin.cx(register[0], register[1])
    squin.broadcast.cx(register[0:2], register[2:4])
    squin.broadcast.cx(register[0:4], register[4:])


# %% [markdown]
# ## Allocate encoded blocks at architecture locations
#
# Each canonical slot occupies a `2 x 4` rectangle in zone 0. Cube vertex `n`
# is placed at
#
# ```text
# row = base_row + n // 4
# column = base_column + 2 * (n % 4)
# ```
#
# The x bit selects the row, while `(y, z)` selects the column. The column stride
# is two because the even grid columns are home positions and the odd columns
# are interstitial CZ-partner positions.
#
# `loc(zone, row, column)` resolves a grid coordinate to a location address.
# Its constant `zone_id`, `word_id`, and `site_id` fields can then be passed to
# `qubit.new_at`.


# %%
def eight_three_two_allocator():
    """Construct allocators for canonical `[[8,3,2]]` block slots."""
    # Slots 0-3 occupy rows 0-1, and slots 4-7 occupy rows 3-4. Adjacent slot
    # origins are separated by eight columns to leave routing space.
    slot_base_rows = ilist.IList([0, 0, 0, 0, 3, 3, 3, 3])
    slot_base_columns = ilist.IList([0, 8, 16, 24, 0, 8, 16, 24])

    @physical.kernel(verify=False)
    def qalloc_slot(slot_index: int) -> LogicalBlock:
        base_row = slot_base_rows[slot_index]
        base_column = slot_base_columns[slot_index]

        def _allocate_vertex(vertex: int):
            address = loc(
                0,
                base_row + vertex // 4,
                base_column + 2 * (vertex % 4),
            )
            return qubit.new_at(
                address.zone_id,
                address.word_id,
                address.site_id,
            )

        register = ilist.map(_allocate_vertex, ilist.range(8))
        initialize_logical_zero(register)
        return register

    @physical.kernel(verify=False)
    def qalloc(
        slot_indices: ilist.IList[int, N],
    ) -> ilist.IList[LogicalBlock, N]:
        def _allocate_slot(slot_index: int):
            return qalloc_slot(slot_index)

        return ilist.map(_allocate_slot, slot_indices)

    return qalloc, qalloc_slot


qalloc, qalloc_slot = eight_three_two_allocator()

# %% [markdown]
# ## Transversal CX
#
# Applying CX to the eight aligned vertex pairs of two blocks implements
# logical CX on all three encoded logical-qubit pairs simultaneously. The move
# compiler is responsible for staging each physical pair at adjacent CZ sites.


# %%
@physical.kernel(verify=False)
def transversal_cx(
    controls: ilist.IList[LogicalBlock, N],
    targets: ilist.IList[LogicalBlock, N],
):
    """Apply aligned physical CX gates between encoded blocks."""
    squin.broadcast.cx(flat(controls), flat(targets))


# %% [markdown]
# ## Transversal logical CCZ
#
# The signature non-Clifford gate of this code is obtained by applying T on the
# even-parity vertices and T-adjoint on the odd-parity vertices. The helper
# gathers the two vertex colors from every supplied block, allowing the gadget
# to be broadcast across multiple blocks.


# %%
@physical.kernel(verify=False)
def logical_ccz(blocks: ilist.IList[LogicalBlock, Any]):
    """Apply logical CCZ to the three encoded qubits in every block."""

    def _even(block: LogicalBlock):
        def _pick(vertex: int):
            return block[vertex]

        return ilist.map(_pick, EVEN_VERTICES)

    def _odd(block: LogicalBlock):
        def _pick(vertex: int):
            return block[vertex]

        return ilist.map(_pick, ODD_VERTICES)

    squin.broadcast.t(flat(ilist.map(_even, blocks)))
    squin.broadcast.t_adj(flat(ilist.map(_odd, blocks)))


# %% [markdown]
# ## Logical SWAP with `arrange.permute`
#
# Exchanging the cube's x and y axes swaps logical qubits 1 and 2 while
# preserving the code space. The experiment below applies that symmetry to one
# selected encoded block.
#
# After `arrange.permute`, the compiler tracks the changed association between
# register entries and atoms. With `INSERT_MOVES=True`, it additionally emits
# the routes that physically commit the permutation.


# %%
@physical.kernel(verify=False)
def logical_swap(block: LogicalBlock):
    """Exchange logical qubits 1 and 2 in one encoded block."""
    arrange.permute(
        block,
        SWAP_XY,
        insert_moves=INSERT_MOVES,
    )


# %% [markdown]
# ## Terminal measurement
#
# A physical kernel has a single terminal measurement. This helper flattens all
# blocks for the measurement and then regroups the results into eight-bit
# slices, one slice per encoded block.


# %%
@physical.kernel(verify=False)
def measure_logical_blocks(blocks: ilist.IList[LogicalBlock, Any]):
    measurements = squin.broadcast.measure(flat(blocks))
    groups = []
    for index in range(len(blocks)):
        groups = groups + [measurements[8 * index : 8 * index + 8]]
    return groups


# %% [markdown]
# ## Compose the experiment
#
# The complete example allocates eight encoded blocks, commits the logical SWAP
# on block 0, and applies two layers of transversal CX:
#
# 1. neighboring blocks: `0 -> 1`, `2 -> 3`, `4 -> 5`, and `6 -> 7`;
# 2. the lower half into the upper half: `0 -> 4`, ..., `3 -> 7`.
#
# `logical_ccz` is defined above as a reusable code gadget but is not needed in
# this particular movement workload.


# %%
@physical.kernel(aggressive_unroll=True, verify=False)
def main():
    blocks = qalloc_slot(ilist.IList([0, 1, 2, 3, 4, 5, 6, 7]))

    logical_swap(blocks[0])
    transversal_cx(blocks[0::2], blocks[1::2])
    transversal_cx(blocks[0:4], blocks[4:])

    return measure_logical_blocks(blocks)


# %% [markdown]
# ## Compile to atom moves
#
# This example uses the entropy placement strategy with return moves disabled.
# Disabling return moves is required because the committed permutation changes
# the block's physical arrangement. `ASAPPlacePass` groups operations into
# parallel placement regions as early as their dependencies allow.

# %%
placement_strategy = make_physical_placement_strategy(
    return_moves=False,
    move_solutions_per_layer=100,
    search_budget=None,
    strategy="entropy",
)
pipeline = PhysicalPipeline(
    placement_strategy=placement_strategy,
    place_opt_type=ASAPPlacePass,
)

# Make compilation failures explicit instead of returning a degenerate program.
compiled = pipeline.emit(main, no_raise=False)

# %% [markdown]
# The compiled method is now movement IR over 64 atoms. A short summary makes
# the size of the generated route visible without printing the entire IR.

# %%
from bloqade.lanes.dialects import move

compiled_statements = list(compiled.callable_region.walk())
move_layers = sum(isinstance(statement, move.Move) for statement in compiled_statements)
cz_pulses = sum(isinstance(statement, move.CZ) for statement in compiled_statements)
print(f"compiled {move_layers} move layers and {cz_pulses} CZ pulses")

# %% [markdown]
# ## Explore the compiled movement interactively
#
# In a local notebook, pass the compiled method and the pipeline architecture to
# the movement debugger:
#
# ```python
# from bloqade.lanes.visualize import debugger
#
# debugger(compiled, pipeline.arch_spec)
# ```
#
# The debugger opens an interactive view in which the slider and arrow keys
# step through the atom configuration. The call is shown rather than executed
# here because documentation builds run without an interactive Qt window.
#
# Set `VIRTUAL = True` near the top of the tutorial and rerun the page to compare
# the committed SWAP with a software-only relabel. In the virtual case, the
# permutation itself emits no moves; later transversal gates absorb the updated
# qubit association during routing.
