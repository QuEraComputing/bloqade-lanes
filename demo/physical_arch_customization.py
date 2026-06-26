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
# In this notebook, we go over some features that allow you to explore options with our physical compiler.

# %% [markdown]
# ## Setup
# To run this notebook with the appropriate dependencies, you can run
#
# `pip install "bloqade-lanes[sim, visualization]"`

# %% [markdown]
# ## Terminology

# %% [markdown]
# Before we get started, it's useful to define how we address our architecture. We have three "levels" to addressing atoms in our architecture: zones, words, and sites. A concrete depiction of the architecture for Gemini physical is shown below:

# Import utilities to define the SQuIN kernel dialect that we will be writing our programs in.
from typing import Any, Literal, TypeVar

# %%
import matplotlib.pyplot as plt
from bloqade.types import Qubit
from kirin.dialects import ilist
from kirin.dialects.ilist import IList
from matplotlib.patches import Rectangle

from bloqade import squin
from bloqade.gemini.common.dialects import qubit

# Define simulator and compilation passes
from bloqade.gemini.device import PhysicalSimulator
from bloqade.lanes.arch.gemini.logical import steane7_initialize

# Define physical architecture
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.bytecode.encoding import ZoneAddress
from bloqade.lanes.heuristics.physical.movement import make_physical_placement_strategy
from bloqade.lanes.passes import ALAPPlacePass, ASAPPlacePass


# We define a helper function that visualizes an architecture specification.
def plot_labeled_arch(
    arch_spec,
    zone_id: int = 0,
    *,
    title: str = "Architecture Words",
    label_sites: bool = False,
    show_pair_boxes: bool = True,
    row_label: str | None = None,
    marker_size: float | None = None,
    font_size: float | None = None,
):
    locations = []
    for loc in arch_spec.yield_zone_locations(ZoneAddress(zone_id)):
        pos = arch_spec.try_get_position(loc)
        if pos is not None:
            locations.append((loc, pos))

    x_values = sorted({pos[0] for _, pos in locations})
    y_values = sorted({pos[1] for _, pos in locations})
    dx = min((b - a for a, b in zip(x_values, x_values[1:])), default=2.0)
    dy = min((b - a for a, b in zip(y_values, y_values[1:])), default=10.0)

    fig_width = max(6, 0.75 * len(x_values))
    fig_height = max(5, 0.95 * len(y_values) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.1)
    ax.set_facecolor("#fbfbfb")

    for row, y in enumerate(y_values):
        ax.add_patch(
            Rectangle(
                (x_values[0] - 2 * dx, y - 0.28 * dy),
                (x_values[-1] - x_values[0]) + 4 * dx,
                0.56 * dy,
                facecolor=("#f5f9ff" if row % 2 == 0 else "#fff8f0"),
                edgecolor="none",
                zorder=0,
            )
        )
        if row_label is not None:
            ax.text(
                -0.055,
                y,
                f"{row_label} {row}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=8,
                color="#666666",
                fontweight="bold",
                clip_on=False,
            )

    if show_pair_boxes:
        for pair_index in range(len(x_values) // 2):
            left_x = x_values[2 * pair_index]
            right_x = x_values[2 * pair_index + 1]
            ax.add_patch(
                Rectangle(
                    (left_x - 0.8 * dx, y_values[0] - 0.45 * dy),
                    (right_x - left_x) + 1.6 * dx,
                    (y_values[-1] - y_values[0]) + 0.9 * dy,
                    facecolor="none",
                    edgecolor="#d7d7d7",
                    linewidth=0.8,
                    zorder=1,
                )
            )
            ax.text(
                (left_x + right_x) / 2,
                y_values[0] - 0.75 * dy,
                f"pair {pair_index}",
                ha="center",
                va="top",
                fontsize=8,
                color="#777777",
            )

    if marker_size is None:
        marker_size = 355 if not label_sites else 170
    if font_size is None:
        font_size = 6.2 if not label_sites else 4.6

    for loc, (x, y) in locations:
        color = "#2878d4" if loc.word_id % 2 == 0 else "#f58518"
        edge = "#1c5fa8" if loc.word_id % 2 == 0 else "#bd6412"
        label = f"w{loc.word_id}"
        if label_sites:
            label = f"w{loc.word_id}\ns{loc.site_id}"
        ax.scatter(
            [x],
            [y],
            s=marker_size,
            color=color,
            edgecolor=edge,
            linewidth=1.1,
            zorder=3,
        )
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=font_size,
            color="white",
            fontweight="bold",
            family="DejaVu Sans Mono",
            linespacing=0.86,
            zorder=4,
        )

    ax.set_title(title, fontsize=18, pad=18, fontweight="bold")
    ax.text(
        0.5,
        0.965,
        f"Zone {zone_id}. Blue words are even word IDs; orange words are odd word IDs.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    ax.set_xlabel("physical x position (um)")
    ax.set_ylabel("physical y position (um)")
    ax.set_xlim(x_values[0] - 3 * dx, x_values[-1] + 2 * dx)
    ax.set_ylim(y_values[0] - dy, y_values[-1] + 0.75 * dy)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e9e9e9", linewidth=0.5, zorder=0)
    return fig, ax


physical_arch_spec = get_physical_arch_spec()
plot_labeled_arch(
    physical_arch_spec,
    title="Gemini Physical Architecture Word/Site Labels",
    label_sites=True,
    show_pair_boxes=False,
)


# %% [markdown]
# A "zone" is the top-level collection of words; a "word" is a collection of "sites", and a "site" contains one atom.
#
# An address for a particular atom is of the form `(zone_id, word_id, site_id)`.

# %% [markdown]
# Each column has the word ID's that are used for the logical architecture. We basically duplicate the logical architecture 8 times to obtain our physical architecture, and you can use the "site_id" to index which "box" to be in.
# > The diagram only plots the (word_id, site_id) for each site; the "zone_id" is omitted as it is always 0 (for the Gemini MVP architecture, we only have one zone).

# %% [markdown]
# ## Customizing Physical Layout
#
# One way that you can tune the performance of your program is to customize the physical layout of your atoms. You can achieve this with the "new_at" statement exposed in the "qubit" dialect.
#
# > The "new_at" statement is supported for both the logical and physical compilers. For the "new_at" demonstration for the logical compiler, see "demo/logical_new_at_demo.py".

# %% [markdown]
# ## Programming for Gemini Logical MVP, but at the physical level
# To give some intuition behind programming at the physical level, we showcase how you can write effectively the same program as written using the Gemini Logical dialect in terms of gates and atom moves, but by programming at the physical instead of the logical level.
# > Although this might seem initially redundant, programming at the physical level gives you flexibility to customize logical-to-physical implementations based on your use case.

# %%
kernel = squin.kernel.add(qubit)
kernel.run_pass = squin.kernel.run_pass

# %%
# We define a LogicalQubit which is a list of 7 qubits that we can manipulate.
LogicalQubit = IList[Qubit, Literal[7]]


# %%
def steane_slot_allocator():
    """Generates a qubit allocator for logical qubits.
    Tries to allocate logical qubits into the architecture in an efficient way to
    make parallelism in the logical gadgets as easily as possible in the move compiler.

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
# Create these gadgets that allow you to allocate qubits at particular words on the Gemini Physical architecture
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
# Define the four qubit GHZ state kernel like we had before
@kernel(typeinfer=True)
def main():
    reg = qalloc([0, 1, 2, 3], 0.0, 0.0, 0.0)

    # squin.broadcast.h(reg[0])
    # cx(reg[:1], reg[1:2])
    # cx(reg[:2], reg[2:])

    return measure_logical_reg(reg)


# %% [markdown]
# Now, we can use our PhysicalSimulator device to visualize the atom moves. As our program is on the physical level now, we can visualize the atom moves for the state preparation kernel as well.

# %%
simulator = PhysicalSimulator()

# %%
physical_sim_task = simulator.task(main)

# %%
# %matplotlib qt

# %%
physical_sim_task.visualize()

# %% [markdown]
# ## Compiler Feature: ASAP/ALAP Scheduling
# We have implemented some basic circuit optimization through our ASAP and ALAP gate scheduling. These passes are classes that you can tell the compiler to use.

# %%
physical_sim_task_asap = simulator.task(main, place_opt_type=ASAPPlacePass)

# %%
physical_sim_task_asap.visualize()

# %%
physical_sim_task_alap = simulator.task(main, place_opt_type=ALAPPlacePass)

# %%
# For this use case, doesn't appear to produce a "nice" program. ASAP is what we want for this program.
physical_sim_task_alap.visualize()


# %% [markdown]
# ## Compiler Feature: Tune Compiler Search Parameters
# You can also provide a custom `placement_strategy` that defines an alternative `move_solutions_per_layer`, `search_budget`, and `strategy`.
#
# The compiler will run a graph-based search algorithm to compile your circuit to atom moves.


# %%
@kernel
def cz(controls: ilist.IList[LogicalQubit, N], targets: ilist.IList[LogicalQubit, N]):
    """Efficient broadcasted cz gate over steane logical qubits"""
    squin.broadcast.cz(flat(controls), flat(targets))


# %%
@kernel(typeinfer=True)
def allocate_five_qubits():
    reg = qalloc([0, 1, 2, 3, 4], 0.0, 0.0, 0.0)

    # squin.broadcast.sqrt_x(flat(ilist.IList([reg[0], reg[1], reg[4]])))
    # cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
    # squin.broadcast.sqrt_y(flat(ilist.IList([reg[1], reg[3]])))
    # cz(ilist.IList([reg[1], reg[3]]), ilist.IList([reg[2], reg[4]]))
    # squin.broadcast.sqrt_x_adj(flat(ilist.IList([reg[1]])))
    # cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[1], reg[4]]))
    # squin.broadcast.sqrt_x_adj(flat(reg))
    return measure_logical_reg(reg)


# %%
placement_strategy = make_physical_placement_strategy(
    move_solutions_per_layer=10000, search_budget=None, strategy="ids"
)

# %%
physical_msd_task = simulator.task(
    allocate_five_qubits,
    place_opt_type=ASAPPlacePass,
    placement_strategy=placement_strategy,
)

# %%
physical_msd_task.visualize()

# %%
physical_msd_task.physical_move_kernel.print()

# %%
