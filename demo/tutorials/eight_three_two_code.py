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
# # Use case: the `[[8,3,2]]` code
#
# The smallest three-dimensional color code uses eight physical qubits at the
# vertices of a cube and encodes three logical qubits. This tutorial adapts the
# former movement demo into a compact physical-kernel example.
#
# We label a cube vertex `(x, y, z)` by `4*x + 2*y + z`. The logical
# `|000>` state is an eight-qubit GHZ state. The code's transversal logical CCZ
# can be expressed with T gates on the even-parity vertices and T-adjoint gates
# on the odd-parity vertices.

# %%
from typing import Literal

from bloqade.types import Qubit
from kirin.dialects import ilist

from bloqade import squin
from bloqade.gemini import physical
from bloqade.gemini.device import GeminiPhysicalSimulator
from bloqade.lanes.dialects import move

LogicalBlock = ilist.IList[Qubit, Literal[8]]
EVEN_VERTICES = ilist.IList([0, 3, 5, 6])
ODD_VERTICES = ilist.IList([1, 2, 4, 7])


# %%
@physical.kernel(verify=False)
def initialize_logical_zero(block: LogicalBlock) -> None:
    """Prepare `(|00000000> + |11111111>) / sqrt(2)`."""
    squin.h(block[0])
    squin.cx(block[0], block[1])
    squin.broadcast.cx(block[0:2], block[2:4])
    squin.broadcast.cx(block[0:4], block[4:8])


# %%
@physical.kernel(aggressive_unroll=True, verify=False)
def eight_three_two_program():
    block = squin.qalloc(8)
    initialize_logical_zero(block)

    # Transversal logical CCZ on the three encoded qubits.
    squin.broadcast.t(ilist.IList([block[0], block[3], block[5], block[6]]))
    squin.broadcast.t_adj(ilist.IList([block[1], block[2], block[4], block[7]]))

    return squin.broadcast.measure(block)


# %% [markdown]
# Passing the physical kernel to the simulator compiles its two-qubit gates into
# moves on the Gemini physical architecture. The compiled move IR is available
# on the task for inspection.

# %%
code_task = GeminiPhysicalSimulator().task(eight_three_two_program)
compiled_statements = list(code_task.physical_move_kernel.callable_region.walk())
print(
    "compiled operations: "
    f"{sum(isinstance(statement, move.Move) for statement in compiled_statements)} "
    "move layers, "
    f"{sum(isinstance(statement, move.CZ) for statement in compiled_statements)} "
    "CZ pulses"
)

# %% [markdown]
# For larger experiments, the allocator from the former `move_demo.py` can be
# generalized to place each eight-qubit block in a separate architecture slot.
# The important separation is that the code gadget describes *which qubits must
# interact*, while the architecture and placement strategy determine *how atoms
# travel* to realize those interactions.
