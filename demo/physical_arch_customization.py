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
# # Demo of Physical Compiler and Architecture Customization
#
# What if you wanted to program beyond just the Gemini MVP Hardware specifications, and wanted lower-level control over programming on physical qubits in your program?
#
# In this notebook, we go over some features that allow you to customize the physical architecture, explore options with our physical compiler, and finally, discuss an experimental feature that allows you to program intermediate atom positions.

# %% [markdown]
# ## Terminology

# %% [markdown]
# Before we get started, it's useful to define how we address our architecture. We have three "levels" to addressing atoms in our architecture: zones, words, and sites. A concrete depiction of the architecture for Gemini physical is shown below:

# %% [markdown]
# <img src="./star_demo_imgs/zone_address_diagram.png" width="800">

# %% [markdown]
# A "zone" is the top-level collection of words; a "word" is a collection of "sites", and a "site" contains one atom.
#
# An address for a particular atom is of the form `(zone_id, word_id, site_id)`.

# %% [markdown]
# ## Customizing Physical Layout
#
# One way that you can tune the performance of your program is to customize the physical layout of your atoms. You can achieve this with the "new_at" statement exposed in the "qubit" dialect.
#
# > The "new_at" statement is supported for both the logical and physical compilers.

# %% [markdown]
# To showcase how you might use the "new_at" statement for allocating logical qubits, we provide a following example helper function for the Steane code using the Gemini physical architecture specification:
# > Having such a helper function is not strictly necessary for writing your program; it helps with the syntax so you don't have to write `(zone_id, word_id, site_id)` for each qubit in your program.
# > This is not the only way to write such a helper function: you can customize how you want to map your logical qubits to physical qubits on the architecture and the state preparation kernel. This, combined with the ability to customize the architecture itself (both atom layout and the buses), gives you flexibility to explore different codes with different layouts and atom moves.

# %%
from typing import Any, Literal, TypeVar

from bloqade.types import Qubit
from kirin.dialects import ilist
from kirin.dialects.ilist import IList

from bloqade import squin
from bloqade.gemini.common.dialects import qubit
from bloqade.lanes.arch.gemini.logical import steane7_initialize

kernel = squin.kernel.add(qubit)
kernel.run_pass = squin.kernel.run_pass

# %%
LogicalQubit = IList[Qubit, Literal[7]]


# %%
def steane_slot_allocator():
    """Generates a qubit allocator for logical qubits.
    Tries to allocate logical qubits into the architecture in an efficient way to
    make parallelism in the logical gagdets as easily as possible in the move compiler.

    """
    # Ordering of words on Gemini Physical Architecture. Word ID's 0, 4, 8, 12, 16 are on the "left side";
    # Word ID's 2, 6, 10, 14, 18 are on the "right side".
    slot_words = ilist.IList([0, 4, 8, 12, 16, 2, 6, 10, 14, 18])

    # Creates "slots" to allocate logical qubits. Logical qubits are all allocated within one word
    # for this particular gadget.
    slots = IList(
        [
            IList([(0, word_id, site_id) for site_id in range(7)])
            for word_id in slot_words
        ]
    )

    # Define a kernel for allocating at a particular slot index (shorthand)
    @kernel
    def qalloc_slot(
        slot_index: int, theta: float, phi: float, lam: float
    ) -> LogicalQubit:
        def allocate_at(address: tuple[int, int, int]):
            return qubit.new_at(address[0], address[1], address[2])

        addresses = slots[slot_index]

        reg = ilist.map(allocate_at, addresses)

        # Apply the state prep kernel
        steane7_initialize(theta, phi, lam, reg)

        return reg

    # Define a kernel for allocating at a list of slot indices (shorthand)
    @kernel
    def qalloc(
        slot_indices: list[int] | IList[int, Any],
        theta: float = 0.0,
        phi: float = 0.0,
        lam: float = 0.0,
    ) -> IList[LogicalQubit, Any]:

        def _inner(slot_index: int):
            return qalloc_slot(slot_index, theta, phi, lam)

        return ilist.map(_inner, slot_indices)

    return qalloc, qalloc_slot


# %%
# Create these gadgets that allow you to allocate qubits at particular words on the Gemini Physical Architecture
qalloc, qalloc_slot = steane_slot_allocator()

N = TypeVar("N")


# %%
# Define a helper function for flattening a nested list of logical qubits (for syntax, because)
# we are programming kernels that act on physical qubits
@kernel
def flat(
    reg: ilist.IList[LogicalQubit, Any],
) -> ilist.IList[Qubit, Any]:
    """Flatten a logical register into a single list of physical qubits"""

    def _inner(cumulant, ele):
        return cumulant + ele

    return ilist.foldl(_inner, reg, ilist.IList([]))


# %% [markdown]
# ## Defining Gadgets
#
# You can define gadgets for your logical program by defining kernels that act on the physical qubits.
# > This can allow for you to customize for different gadgets with different `broadcast` semantics as well as explore non-transversal implementations of gates.


# %%
@kernel
def cx(controls: ilist.IList[LogicalQubit, N], targets: ilist.IList[LogicalQubit, N]):
    """Efficient broadcasted cx gate over steane logical qubits"""
    squin.broadcast.cx(flat(controls), flat(targets))


# %%
@kernel
def measure_logical_reg(logical_reg: ilist.IList[LogicalQubit, Any]):
    """Helper function to get around single measurement restriction of kernel.
    first flatten the logical register into physical qubits, then reconstruct
    the groups of physical measurements into groups related to logical qubits.

    """
    # measurements must be flattened, only one measurement is allowed!
    measurements = squin.broadcast.measure(flat(logical_reg))
    logical_groups = []
    for i in range(len(logical_reg)):
        logical_groups = logical_groups + [measurements[7 * i : 7 * i + 7]]

    return logical_groups


# %%
