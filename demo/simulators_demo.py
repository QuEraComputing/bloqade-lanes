# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: bloqade-lanes (3.12.13.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Using Logical and Physical Simulators
#
# In this notebook, we provide a short demo for using the logical and physical simulators, as well as using different simulation backends for the logical/physical simulators.

# %% [markdown]
# ## Setup
# To run this notebook with the appropriate dependencies, you can run
#
# `pip install "bloqade-lanes[sim, clifft]"`

# %% [markdown]
# ## Constructing Simulators
# We have two different simulators representing two different parts of the compilation pipeline: the `GeminiLogicalSimulator` and the `GeminiPhysicalSimulator`. These two simulators take in different inputs (a logical circuit versus a physical circuit, respectively), and therefore have slightly different compilations.

# %%
# For postprocessing
import numpy as np

# Defining dialects to program logical or physical kernels in
from bloqade import squin
from bloqade.gemini import logical

# Constructing a logical or physical simulator and alternative simulation backend
from bloqade.gemini.device import (
    CliffTSimulatorBackend,
    GeminiLogicalSimulator,
    GeminiPhysicalSimulator,
)

# %% [markdown]
# ## Basic Path for Constructing and Using a Logical Simulator
# Here, we provide a basic usage path of constructing and using a logical simulator. By default, the simulation backend is `tsim`.

# %%
# Construct a logical simulator
logical_sim = GeminiLogicalSimulator()


# %%
# Define a logical program to run on Gemini
@logical.kernel(aggressive_unroll=True)
def test_logical_program():
    reg = squin.qalloc(5)
    squin.broadcast.x(reg)
    squin.cx(reg[0], reg[1])
    return logical.terminal_measure(reg)


# %%
# Compile the logical program to a task
logical_task = logical_sim.task(test_logical_program)

# %%
logical_result = logical_task.run(shots=1000)

# %%
print(np.asarray(logical_result.measurements).shape)

# %% [markdown]
# ## Specifying an Alternative Simulation Backend
# You can also specify an alternative simulation backend when constructing the simulator. For example, you can use the `CliffTSimulatorBackend` for CliffT integration.
# > To use `CliffTSimulatorBackend`, you can run `pip install "bloqade-lanes[clifft]"`.

# %%
# Create a logical simulator that uses CliffT as a simulator backend
logical_sim_clifft = GeminiLogicalSimulator(backend=CliffTSimulatorBackend())

# %%
# Compile the program to a task and use CliffT to sample the results.
logical_task_clifft = logical_sim_clifft.task(test_logical_program)

# %%
logical_result_clifft = logical_task_clifft.run(shots=1000)

# %%
print(np.asarray(logical_result_clifft.measurements).shape)

# %% [markdown]
# ## Using Physical Simulator and Configuring Simulator Backend
# You can analogously construct a `GeminiPhysicalSimulator` that compiles a physical squin kernel, and executes the task with a specified simulation backend.

# %%
# Construct a GeminiPhysicalSimulator that compiles a physical squin kernel and executes it.
physical_sim = GeminiPhysicalSimulator()


# %%
@squin.kernel()
def test_physical_program():
    reg = squin.qalloc(5)
    squin.broadcast.x(reg)
    squin.cx(reg[0], reg[1])
    return squin.broadcast.measure(reg)


# %%
physical_task = physical_sim.task(test_physical_program)

# %%
physical_result = physical_task.run(shots=1000)

# %%
print(np.asarray(physical_result.measurements).shape)

# %%
# You can alternatively construct a GeminiPhysicalSimulator with a different simulator backend, like CliffT.
physical_sim_clifft = GeminiPhysicalSimulator(backend=CliffTSimulatorBackend())

# %%
physical_task_clifft = physical_sim_clifft.task(test_physical_program)

# %%
physical_result_clifft = physical_task_clifft.run(shots=1000)

# %%
print(np.asarray(physical_result_clifft.measurements).shape)

# %%
