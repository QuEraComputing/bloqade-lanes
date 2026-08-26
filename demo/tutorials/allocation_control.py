# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: bloqade-lanes (3.12.13)
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
# > For additional reference, you can look at the Gemini Logical circuit builder here: https://bloqade.quera.com/studio/gemini/
# - In a **Gemini Physical** kernel, use `qubit.new_at(...)` to allocate one
#   physical qubit at an explicit `(zone_id, word_id, site_id)` address.

# %%
from kirin.dialects import ilist

from bloqade import squin
from bloqade.gemini import logical, physical
from bloqade.gemini.common.dialects import qubit
from bloqade.gemini.device import (
    GeminiLogicalSimulator,
    GeminiPhysicalSimulator,
)

# %% [markdown]
# ## Logical kernels: allocate with `logical.qalloc_at`
#
# Each entry in `qalloc_at` describes one encoded logical qubit. An integer pins
# that logical qubit to a logical position, while `None` asks the layout
# heuristic to choose its position. Because the register size and positions
# must be available during compilation, pass a static `ilist.IList` and enable
# `aggressive_unroll` on the calling logical kernel.


# %% [markdown]
# Here, we show the slot ID's for the different locations that logical qubits can be allocated on the processor, indexed from 0-9.
#
# <img src="./star_demo_imgs/logical_slot_ids.png" style="height: 300px; width: auto;">


# %%
@logical.kernel(aggressive_unroll=True)
def position_based_allocation():
    # Allocate qubits at positions 0 and 1.
    qubits = logical.qalloc_at(ilist.IList([0, 1]))
    squin.cx(qubits[0], qubits[1])
    return logical.terminal_measure(qubits)


# %%
position_task = GeminiLogicalSimulator().task(position_based_allocation)

# %%
# position_task.visualize()

# %% [markdown]
# <img src="star_demo_imgs/log_qubit_placement_2.png" width=500>
#
# Notice that the physical qubits for each logical qubit are interleaved: at even columns are the physical qubits corresponding to logical qubit 0, and at odd columns are the physical qubits corresponding to logical qubit 1.

# %% [markdown]
# ## Physical kernels: allocate with `qubit.new_at`
#
# A physical kernel operates on individual atoms rather than encoded logical
# qubits. Here `qubit.new_at(zone_id, word_id, site_id)` pins individual
# physical qubits to concrete addresses.


# %%
@physical.kernel(aggressive_unroll=True, verify=False)
def address_based_allocation():
    left = qubit.new_at(0, 0, 0)
    right = qubit.new_at(0, 4, 0)

    squin.broadcast.h(ilist.IList([left, right]))
    squin.cx(left, right)
    return squin.broadcast.measure(ilist.IList([left, right]))


# %%
address_task = GeminiPhysicalSimulator().task(address_based_allocation)

# %%
# address_task.visualize()

# %% [markdown]
# <img src="star_demo_imgs/phys_qubit_placement_2.png" width=500>
#
# We place both atoms at site index 0, hence why they are in the leftmost columns. Words 0 and 4 refer to the leftmost sites in the bottom and second-to-bottom row, respectively.
# > For more details on the Gemini physical architecture and the layout of sites/words, refer to [Physical architecture customization](../../physical_arch_customization/).
