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
# # Creating and visualizing custom architectures
#
# An architecture describes where atoms may be placed, which sites may interact,
# and the buses and lanes along which atoms may move. `ZoneBuilder` provides
# explicit control over the geometry and buses inside one zone, while
# `ArchBuilder` combines zones into a validated architecture.

# %%
from IPython.display import HTML, display

from bloqade.lanes.arch import ArchBuilder, ZoneBuilder
from bloqade.lanes.visualize.arch import ArchVisualizer

# %% [markdown]
# This example creates a gate zone with four rows and two interleaved words per
# row. Each word contains four sites. The coordinate arrays define the physical
# grid in micrometers; words refer to positions in that grid by index.
#
# Words are added from left to right and bottom to top, so the bottom row
# contains words 0 and 1, the next row contains words 2 and 3, and so on.

# %%
gate_zone = ZoneBuilder.from_positions(
    "gate",
    x_coordinates=[0.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0],
    y_coordinates=[0.0, 55.0, 110.0, 165.0],
    word_shape=(4, 1),
    x_clearance=3.0,
    y_clearance=3.0,
)

for row_index in range(4):
    gate_zone.add_word([0, 2, 4, 6], [row_index])
    gate_zone.add_word([1, 3, 5, 7], [row_index])

# %% [markdown]
# Site buses move the same site indices within every participating word. These
# two buses implement the two dimensions of a four-site hypercube.

# %%
gate_zone.add_site_bus(src=[0, 2], dst=[1, 3])
gate_zone.add_site_bus(src=[0, 1], dst=[2, 3])

# %% [markdown]
# Word buses move the same site ID between corresponding source and destination
# words. The first two buses connect rows; the third connects the two
# interleaved word columns. Each source/destination collection must satisfy the
# AOD Cartesian-product constraint.

# %%
gate_zone.add_word_bus(src=[0, 1, 4, 5], dst=[2, 3, 6, 7])
gate_zone.add_word_bus(src=[0, 1, 2, 3], dst=[4, 5, 6, 7])
gate_zone.add_word_bus(src=[0, 2, 4, 6], dst=[1, 3, 5, 7])

# Pair adjacent words for transversal CZ operations.
gate_zone.add_entangling_pairs(
    words_a=[0, 2, 4, 6],
    words_b=[1, 3, 5, 7],
)

arch_builder = ArchBuilder()
arch_builder.add_zone(gate_zone)
arch_builder.add_mode("all", ["gate"])
arch_spec = arch_builder.build()

built_zone = arch_spec.zones[0]

print(f"words: {len(arch_spec.words)}")
print(f"sites per word: {arch_spec.sites_per_word}")
print(f"word buses: {len(built_zone.word_buses)}")
print(f"site buses: {len(built_zone.site_buses)}")
print(f"entangling pairs: {len(built_zone.entangling_pairs)}")

# %% [markdown]
# `plot_interactive` returns a Plotly figure. Hover over a site to inspect its
# `(zone_id, word_id, site_id)`. Hover over a bus to see its source and
# destination site addresses and emphasize its path above any overlapping
# buses. Use the mouse wheel or trackpad to zoom, and use the buttons to show
# labels, switch between exact and schematic paths, or reveal every bus.
# Individual buses can also be selected from the legend.

# %%
figure = ArchVisualizer(arch_spec).plot_interactive(
    theme="light",
    path_style="cartoon",
    show_site_ids=False,
    show_all_buses=False,
    width=1100,
    height=650,
)
# mkdocs-jupyter's nbconvert template does not render Plotly's vendor-specific
# MIME bundle. Emitting the equivalent HTML makes the same interactive figure
# work both in Jupyter and on the generated documentation page.
display(HTML(figure.to_html(full_html=False, include_plotlyjs="cdn")))

# %% [markdown]
# For a static Matplotlib figure, use `ArchVisualizer(arch_spec).plot(...)`.
# The interactive version is usually easier to read once an architecture has
# more than a few buses.
