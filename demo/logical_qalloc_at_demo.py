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
# # Demo for using qalloc_at for Gemini Logical dialect
#
# In this notebook, we give a short demo for using the `logical.qalloc_at` statement in the Gemini Logical dialect to allocate qubits at particular positions.

# %% [markdown]
# Here, we show the slot ID's for the different locations that logical qubits can be allocated on the processor, indexed from 0-9.
#
# <img src="./demo_imgs/logical_slot_ids.png" height=200>


# %%
# Define dialects to program in
from kirin.dialects import ilist

from bloqade.gemini import logical

# Define Gemini simulator device
from bloqade.gemini.device import GeminiLogicalSimulator

# %% [markdown]
# We now go through an example of allocating logical qubits at specific "positions".


# %%
@logical.kernel(aggressive_unroll=True)
def logical_qubit_allocation():
    qubits = logical.qalloc_at(ilist.IList([1, 2, 3, 4]))

    return logical.terminal_measure(qubits)


# %%
allocated_task = GeminiLogicalSimulator().task(logical_qubit_allocation)

# %%
# %matplotlib qt

# %%
# We see that the four qubits are allocated at positions 1-4, skipping position 0.
allocated_task.visualize()


# %%
# We can alternative allocate logical qubits in different orders, at different "positions".
@logical.kernel(aggressive_unroll=True)
def logical_qubit_allocation_alt():
    qubits = logical.qalloc_at(ilist.IList([3, 1, 2, 4]))

    return logical.terminal_measure(qubits)


# %%
allocated_task_alt = GeminiLogicalSimulator().task(logical_qubit_allocation_alt)

# %%
allocated_task_alt.visualize()
