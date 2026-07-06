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
# ## Setup
# To run this notebook with the appropriate dependencies, you can run
#
# `pip install "bloqade-lanes[sim, visualization]"`

# %% [markdown]
# Below is the architecture specification used by the logical compiler. The index of the locations that a logical qubit can occupy in the logical architecture is defined below.
# > A logical qubit can only initially occupy *left* sites of each column (corresponding to the blue circles in the below diagram).

# %%
import matplotlib.pyplot as plt

# Define dialects to program the kernel
from kirin.dialects import ilist
from matplotlib.patches import Rectangle

from bloqade import squin
from bloqade.gemini import logical as gemini_logical
from bloqade.gemini.common.dialects.qubit import new_at
from bloqade.gemini.device import GeminiLogicalSimulator

# Define logical arch spec
from bloqade.lanes.arch.gemini.logical import get_arch_spec as get_logical_arch_spec
from bloqade.lanes.bytecode.encoding import ZoneAddress


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


logical_arch_spec = get_logical_arch_spec()
plot_labeled_arch(
    logical_arch_spec,
    title="Gemini Logical Architecture Words",
    label_sites=False,
    show_pair_boxes=True,
)


# %% [markdown]
# For reference, we display again the connectivity on the logical architecture.
#
# <img src="./star_demo_imgs/gemini_logical_buses.png" height=500>

# %% [markdown]
# # Example Using Default Allocation
# Below, we give an example of a kernel where we don't specify the location of the qubits during allocation.


# %%
@gemini_logical.kernel(aggressive_unroll=True)
def default_allocation():
    reg = squin.qalloc(4)
    squin.broadcast.h(ilist.IList([reg[0], reg[2]]))
    squin.broadcast.cx(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
    gemini_logical.terminal_measure(reg)


# %%
default_alloc_task = GeminiLogicalSimulator().task(default_allocation)

# %%
# %matplotlib qt

# %%
default_alloc_task.visualize()

# %% [markdown]
# ## Example of Allocating Qubits at Specific Locations
# Here, we give an example of a kernel written in the Gemini Logical dialect that specifies the initial layouts of the logical qubits.
# With more control over the layout, we can better enforce parallelism.


# %%
@gemini_logical.kernel(aggressive_unroll=True)
def explicit_allocation():
    # Pinned qubits at explicit physical addresses.
    a = new_at(0, 0, 0)
    b = new_at(0, 8, 0)
    c = new_at(0, 4, 0)
    d = new_at(0, 12, 0)
    squin.broadcast.h(ilist.IList([a, b]))
    squin.broadcast.cx(ilist.IList([a, b]), ilist.IList([c, d]))
    gemini_logical.terminal_measure(ilist.IList([a, b, c, d]))


# %%
explicit_alloc_task = GeminiLogicalSimulator().task(explicit_allocation)

# %%
explicit_alloc_task.visualize()

# %%
