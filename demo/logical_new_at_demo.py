# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: bloqade-lanes (3.12.13.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Demonstration of Customizing Logical Layout
# In this short notebook, we demonstrate how you can use the "new_at" statement in the Gemini Logical dialect to control the initial layout of the atoms in your program.

# %% [markdown]
# Below is the architecture specification used by the logical compiler. The index of the locations that a logical qubit can occupy in the logical architecture is defined below.
# > A logical qubit can only initially occupy *left* sites of each column (corresponding to the blue circles in the below diagram).

# %% [markdown]
# <img src="./star_demo_imgs/gemini_logical_words.png" width="400">

# %% [markdown]
# ## Example of Allocating Qubits at Specific Locations
# Here, we give an example of a kernel written in the Gemini Logical dialect that specifies the initial layouts of the logical qubits.

# %%
# Define dialects to program the kernel
from kirin.dialects import ilist

from bloqade import squin
from bloqade.gemini import logical as gemini_logical
from bloqade.gemini.common.dialects.qubit import new_at
from bloqade.gemini.device import GeminiLogicalSimulator


# %%
@gemini_logical.kernel(aggressive_unroll=True)
def main():
    # Pinned qubits at explicit physical addresses.
    a = new_at(0, 0, 0)
    b = new_at(0, 4, 0)
    # Un-pinned qubits — the layout heuristic chooses their home sites.
    reg = squin.qalloc(2)
    # CZ between pinned and un-pinned qubits.
    squin.cz(a, reg[0])
    squin.cz(b, reg[1])
    gemini_logical.terminal_measure(ilist.IList([a, b, reg[0], reg[1]]))


# %%
task = GeminiLogicalSimulator().task(main)

# %%
# %matplotlib qt

# %%
task.visualize()


# %% [markdown]
# We can also experiment with an alternative layout where we instead use the qubits in the top right.


# %%
@gemini_logical.kernel(aggressive_unroll=True)
def main_alt_layout():
    # Pinned qubits at explicit physical addresses.
    a = new_at(0, 14, 0)
    b = new_at(0, 18, 0)
    # Un-pinned qubits — the layout heuristic chooses their home sites.
    # The default layout heuristic will try to make CZ move patterns similar, but for
    # this case, it will choose lower word ID's (words 0 and 4).
    reg = squin.qalloc(2)
    # CZ between pinned and un-pinned qubits.
    squin.cz(a, reg[0])
    squin.cz(b, reg[1])
    gemini_logical.terminal_measure(ilist.IList([a, b, reg[0], reg[1]]))


# %%
task_alt_layout = GeminiLogicalSimulator().task(main_alt_layout)

# %%
task_alt_layout.visualize()
