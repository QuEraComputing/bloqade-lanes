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
# # Introduction to bloqade-lanes
#
# In this notebook, we provide an introduction to `bloqade-lanes`, what tools are provided, and what these tools might be useful for.
#
# > This notebook assumes prior high-level familiarity with quantum computing and neutral atom quantum computing concepts.

# %% [markdown]
# # Install Dependencies
# To run this notebook, you need to run `pip install bloqade-lanes[visualization]`.

# %% [markdown]
# <img src="star_demo_imgs/comp_workflow.png" height=500>
#
# `bloqade-lanes` sits in the "middle" of our compiler stack and primarily serves the purpose of compiling a user's quantum circuit down to atom moves.

# %% [markdown]
# # Example Usage
# You can first define a quantum program using our `squin` dialect, providing qubit allocation and gate statements. We prepare a Bell state below for the sake of demonstration.

# %%
from bloqade import squin
from bloqade.gemini.device import GeminiPhysicalSimulator


# %%
@squin.kernel
def bell_state():
    qubits = squin.qalloc(2)
    squin.h(qubits[0])
    squin.cx(qubits[0], qubits[1])
    return squin.broadcast.measure(qubits)


# %%
# Run the compilation on the bell_state kernel
bell_task = GeminiPhysicalSimulator().task(bell_state)

# %%
# Visualize the compiled atom move program
bell_task.visualize()

# %% [markdown]
# From visualizing the program, we see the steps of the atom program: applying a series of local gates on the first atom (Hadamard), then moving the atom to its CZ partner, applying the Rydberg beam, and then moving the atom back.

# %% [markdown]
# This is the indended paradigm for using `bloqade-lanes`: the user provides their program at the circuit level, and our compiler will figure out the schedule of atom moves and pulses needed to realize that program.
