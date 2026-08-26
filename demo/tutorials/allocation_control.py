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
# # Controlling initial layout in logical and physical kernels
#
# Normal `squin.qalloc(n)` lets the layout heuristic choose every initial
# location. The appropriate placement API depends on the abstraction level:
#
# - In a **Gemini Logical** kernel, use `logical.qalloc_at(...)` to allocate
#   encoded logical qubits at logical position indices. `None` leaves an entry
#   unpinned.
# - In a **Gemini Physical** kernel, use `qubit.new_at(...)` to allocate one
#   physical qubit at an explicit `(zone_id, word_id, site_id)` address.

# %%
from typing import Any

from kirin.dialects import ilist

from bloqade import squin
from bloqade.gemini import logical, physical
from bloqade.gemini.common.dialects import qubit
from bloqade.gemini.device import (
    GeminiLogicalSimulator,
    GeminiLogicalSimulatorTask,
    GeminiPhysicalSimulator,
    PhysicalSimulatorTask,
)
from bloqade.lanes.dialects import move


def logical_home_words(task: GeminiLogicalSimulatorTask[Any]) -> list[int]:
    """Extract one home word from each seven-site Steane block."""
    fill = next(
        statement
        for statement in task.physical_move_kernel.callable_region.walk()
        if isinstance(statement, move.Fill)
    )
    return [address.word_id for address in fill.location_addresses[::7]]


def physical_initial_addresses(
    task: PhysicalSimulatorTask[Any],
) -> list[tuple[int, int, int]]:
    """Return the initial `(zone, word, site)` address of each physical qubit."""
    fill = next(
        statement
        for statement in task.physical_move_kernel.callable_region.walk()
        if isinstance(statement, move.Fill)
    )
    return [
        (address.zone_id, address.word_id, address.site_id)
        for address in fill.location_addresses
    ]


# %% [markdown]
# ## Logical kernels: allocate with `logical.qalloc_at`
#
# Each entry in `qalloc_at` describes one encoded logical qubit. An integer pins
# that logical qubit to a logical position, while `None` asks the layout
# heuristic to choose its position. Because the register size and positions
# must be available during compilation, pass a static `ilist.IList` and enable
# `aggressive_unroll` on the calling logical kernel.


# %%
@logical.kernel(aggressive_unroll=True)
def position_based_allocation():
    # Positions 0 and 3 are pinned. The middle qubit is placed heuristically.
    qubits = logical.qalloc_at(ilist.IList([0, None, 3]))
    squin.cx(qubits[0], qubits[1])
    return logical.terminal_measure(qubits)


# %%
position_task = GeminiLogicalSimulator().task(position_based_allocation)
print(f"selected home words: {logical_home_words(position_task)}")

# %% [markdown]
# A logical position `p` currently maps to the home of an encoded Steane block
# at `(zone=0, word=2*p, site=0)` in the logical architecture. The compiler then
# expands each logical qubit into its seven physical data qubits.

# %% [markdown]
# ## Physical kernels: allocate with `qubit.new_at`
#
# A physical kernel operates on individual atoms rather than encoded logical
# qubits. Here `qubit.new_at(zone_id, word_id, site_id)` pins individual
# physical qubits to concrete addresses. It can be mixed with `squin.qalloc`
# when only some of the physical qubits need fixed starting locations.


# %%
@physical.kernel(aggressive_unroll=True, verify=False)
def address_based_allocation():
    left = qubit.new_at(0, 0, 0)
    right = qubit.new_at(0, 4, 0)
    free = squin.qalloc(1)

    squin.broadcast.h(ilist.IList([left, right]))
    squin.cx(left, free[0])
    return squin.broadcast.measure(ilist.IList([left, right, free[0]]))


# %%
address_task = GeminiPhysicalSimulator().task(address_based_allocation)
print(f"initial physical addresses: {physical_initial_addresses(address_task)}")

# %% [markdown]
# Both APIs constrain only the initial placement. Later two-qubit gates may
# still require routing. Use `logical.qalloc_at` for encoded logical qubits and
# reserve `qubit.new_at` for individual atoms in physical kernels.
