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
# # Experimental movement control
#
# Most programs should let the placement strategy route atoms. The physical
# dialect also exposes `arrange.move_to` and `arrange.permute` for experiments
# that require explicit movement or qubit-label control.

# %%
from typing import Literal, TypeVar

from bloqade.types import Qubit
from kirin.dialects import ilist

from bloqade import squin
from bloqade.gemini import physical
from bloqade.gemini.common.dialects import arrange, qubit
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import arch, move
from bloqade.lanes.heuristics.physical import make_physical_placement_strategy
from bloqade.lanes.transform import PhysicalPipeline

N = TypeVar("N")
Logical832Block = ilist.IList[Qubit, Literal[8]]

# Label cube vertex (x, y, z) by n = 4*x + 2*y + z. Exchanging the x and y
# coordinates fixes vertices 0, 1, 6, and 7 and swaps 2 <-> 4 and 3 <-> 5.
SWAP_XY = ilist.IList([0, 1, 4, 5, 2, 3, 6, 7])


# %%
@physical.kernel(verify=False)
def locations(
    rows: ilist.IList[int, N], columns: ilist.IList[int, N]
) -> ilist.IList[LocationAddress, N]:
    def _location(index: int):
        return arch.loc(0, rows[index], columns[index])

    return ilist.map(_location, ilist.range(len(rows)))


@physical.kernel(verify=False)
def allocate_at(
    addresses: ilist.IList[LocationAddress, N],
) -> ilist.IList[Qubit, N]:
    def _allocate(address: LocationAddress):
        return qubit.new_at(address.zone_id, address.word_id, address.site_id)

    return ilist.map(_allocate, addresses)


# %%
@physical.kernel(aggressive_unroll=True, verify=False)
def explicit_move_program():
    static_addresses = locations(ilist.IList([0, 1]), ilist.IList([0, 0]))
    mobile_addresses = locations(ilist.IList([0, 1]), ilist.IList([2, 2]))
    static = allocate_at(static_addresses)
    mobile = allocate_at(mobile_addresses)

    def _partner(index: int):
        return arch.cz_partner(static_addresses[index])

    partner_addresses = ilist.map(_partner, ilist.range(2))
    arrange.move_to(mobile, partner_addresses)
    squin.broadcast.cx(mobile, static)
    arrange.move_to(mobile, mobile_addresses)

    return squin.broadcast.measure(static + mobile)


# %%
no_return_strategy = make_physical_placement_strategy(return_moves=False)
move_pipeline = PhysicalPipeline(placement_strategy=no_return_strategy)
explicit_move_ir = move_pipeline.emit(explicit_move_program, no_raise=False)
explicit_move_ir.print()

# %% [markdown]
# ## Permuting qubits with `arrange.permute`
#
# `arrange.permute(qubits, permutation)` changes the compiler's association
# between the entries of `qubits` and the atoms carrying their quantum states.
# Its index convention is:
#
# ```text
# after the permutation, qubits[i] refers to the old qubits[permutation[i]]
# ```
#
# The permutation must contain every index exactly once and must be known at
# compile time.

# %% [markdown]
# ### Use case: a logical SWAP in the `[[8,3,2]]` code
#
# Gemini's built-in logical dialect targets Steane-encoded qubits, so this
# alternative code is expressed in a physical kernel as a block of eight
# physical qubits whose code-space meaning is supplied by the program.
#
# The `[[8,3,2]]` color code places its eight physical qubits on the vertices
# of a cube. Label vertex `(x, y, z)` by `n = 4*x + 2*y + z`. Exchanging the
# cube's x and y axes maps the vertices as follows:
#
# | new vertex reference | old vertex reference |
# | --- | --- |
# | `0` | `0` |
# | `1` | `1` |
# | `2` | `4` |
# | `3` | `5` |
# | `4` | `2` |
# | `5` | `3` |
# | `6` | `6` |
# | `7` | `7` |
#
# Therefore the permutation is `[0, 1, 4, 5, 2, 3, 6, 7]`. Under the logical
# operator convention used by the `[[8,3,2]]` demo, this cube symmetry swaps
# the x- and y-associated encoded logical qubits while preserving the code
# space.

# %% [markdown]
# #### Lazy logical SWAP: relabel the code block
#
# With the default `insert_moves=False`, the operation is a free logical
# relabel. No move instruction is emitted at the permutation itself; later
# placement and routing absorb the changed association. Subsequent operations
# continue to use `block[0]`, `block[1]`, and so on normally, but the compiler
# tracks their permuted vertex identities.


# %%
@physical.kernel(aggressive_unroll=True, verify=False)
def lazy_832_logical_swap():
    block: Logical832Block = squin.qalloc(8)
    arrange.permute(block, SWAP_XY)
    return squin.broadcast.measure(block)


# %%
lazy_permutation_ir = PhysicalPipeline().emit(lazy_832_logical_swap, no_raise=False)
lazy_moves = [
    statement
    for statement in lazy_permutation_ir.callable_region.walk()
    if isinstance(statement, move.Move)
]
print(f"move layers emitted for the lazy permutation: {len(lazy_moves)}")

# %% [markdown]
# #### Committed logical SWAP: rearrange the code block
#
# Set `insert_moves=True` when the atom arrangement itself must be updated
# immediately. This emits physical moves and therefore requires a no-return
# placement strategy. A palindrome strategy would undo the committed movement
# and is intentionally rejected.


# %%
@physical.kernel(aggressive_unroll=True, verify=False)
def committed_832_logical_swap():
    block: Logical832Block = squin.qalloc(8)
    arrange.permute(
        block,
        SWAP_XY,
        insert_moves=True,
    )
    return squin.broadcast.measure(block)


# %%
committed_swap_ir = move_pipeline.emit(committed_832_logical_swap, no_raise=False)
committed_moves = [
    statement
    for statement in committed_swap_ir.callable_region.walk()
    if isinstance(statement, move.Move)
]
print(f"move layers emitted for the committed swap: {len(committed_moves)}")
for movement in committed_moves:
    movement.print()

# %% [markdown]
# Use the default relabel-only form when only the logical identity mapping
# matters. Use `insert_moves=True` when later physical operations depend on the
# permuted spatial arrangement. The permutation and `insert_moves` flag cannot
# be runtime kernel arguments; both must be compile-time resolvable.
