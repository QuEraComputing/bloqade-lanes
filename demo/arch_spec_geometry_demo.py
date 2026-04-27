# %% [markdown]
# # Build an ArchSpec from scratch and query its geometry
# ---
# This demo walks through the **low-level** ``ArchSpec`` builder API
# (``ZoneBuilder`` + ``ArchBuilder``) and the geometry accessors exposed on
# the resulting :class:`bloqade.lanes.layout.ArchSpec`.  It is meant as
# a reference for someone new to the bytecode/arch-spec surface — the
# focus is on *what's already there*, not on introducing new API.
#
# For the higher-level blueprint flow (``ArchBlueprint`` + ``ZoneSpec`` +
# ``build_arch``) see ``demo/zone_based_arch_demo.py``.

# %% [markdown]
# ## 1. Build a small two-zone architecture by hand
#
# We construct a tiny architecture with:
#
# * a **gate zone** of 4 words (1 row × 4 cols), 2 sites per word, with a
#   site bus, a word bus, and two CZ entangling pairs (word 0 ↔ 1,
#   word 2 ↔ 3);
# * a **memory zone** of 2 storage words with a site-swap bus and no
#   word bus or entangling pairs;
# * a single inter-zone bus connecting gate word 0 to mem word 0.
#
# All buses are validated for AOD Cartesian-product geometry; the builder
# will refuse layouts that can't be realized as uniform AOD shifts.

# %%
from bloqade.lanes.arch import ArchBuilder, ZoneBuilder
from bloqade.lanes.bytecode._native import Grid as RustGrid
from bloqade.lanes.layout.encoding import LocationAddress, ZoneAddress

# --- Gate zone -------------------------------------------------------------
# 8 evenly-spaced x positions, 1 y position. Each word occupies 2
# consecutive x positions, so word_shape = (2, 1) and sites_per_word = 2.
gate_grid = RustGrid.from_positions(
    [0.0, 10.0, 30.0, 40.0, 60.0, 70.0, 90.0, 100.0],
    [0.0],
)
gate = ZoneBuilder(
    name="gate",
    grid=gate_grid,
    word_shape=(2, 1),
    x_clearance=3.0,
    y_clearance=3.0,
)
# Place 4 words side-by-side. ``add_word`` returns the zone-local word id.
w0 = gate.add_word(x_sites=[0, 1], y_sites=[0])
w1 = gate.add_word(x_sites=[2, 3], y_sites=[0])
w2 = gate.add_word(x_sites=[4, 5], y_sites=[0])
w3 = gate.add_word(x_sites=[6, 7], y_sites=[0])

# Site bus: swap site 0 ↔ site 1 inside every word that opted in.
gate.add_site_bus(src=[0], dst=[1])

# Word bus: shuttle the left pair (words 0/1) into the right pair's
# physical slots (words 2/3). All four word origins line up on a uniform
# shift of +60 µm in x, so this is a valid AOD bus.
gate.add_word_bus(src=[w0, w1], dst=[w2, w3])

# CZ entangling pairs.
gate.add_entangling_pairs(words_a=[w0, w2], words_b=[w1, w3])

# --- Memory zone -----------------------------------------------------------
# A small zone holding two storage words. ``ArchSpec`` validation
# requires every zone to share the same grid dimensions and
# ``sites_per_word``, so the mem grid mirrors the gate grid's shape;
# we just place fewer words on it.
mem_grid = RustGrid.from_positions(
    [0.0, 10.0, 30.0, 40.0, 60.0, 70.0, 90.0, 100.0],
    [40.0],
)
mem = ZoneBuilder(
    name="mem",
    grid=mem_grid,
    word_shape=(2, 1),
    x_clearance=3.0,
    y_clearance=3.0,
)
mem.add_word(x_sites=[0, 1], y_sites=[0])
mem.add_word(x_sites=[2, 3], y_sites=[0])
# Without a bus or entangling pair referencing them, zone words are
# treated as belonging to zone 0 by ``word_zone_map``. Add a site bus
# so the mem words are properly owned by the mem zone.
mem.add_site_bus(src=[0], dst=[1])

# --- Compose ---------------------------------------------------------------
arch_builder = ArchBuilder()
gate_id = arch_builder.add_zone(gate)
mem_id = arch_builder.add_zone(mem)

# Inter-zone bus: ``zone[region]`` returns a name-qualified
# ``(zone_name, [word_ids])`` tuple ready to feed ``connect``.
arch_builder.connect(
    src=gate[0:1, 0:1],  # gate word 0
    dst=mem[0:1, 0:1],  # mem word 0
)

# A mode picks which zones contribute to the hardware-shot bitstring.
arch_builder.add_mode(name="all", zones=["gate", "mem"])

arch = arch_builder.build()

# %% [markdown]
# ## 2. Query the geometry
#
# Once built, the :class:`ArchSpec` exposes a handful of accessors that
# resolve word/site/zone identifiers to physical coordinates and
# connectivity. The cells below exercise each one.

# %%
print(f"sites_per_word: {arch.sites_per_word}")
print(f"max_qubits:     {arch.max_qubits}")
print(f"len(words):     {len(arch.words)}")
print(f"len(zones):     {len(arch.zones)}")
print(f"len(modes):     {len(arch.modes)}")

# Direct indexing replaces ``word_by_id`` / ``zone_by_id``: word ids and
# zone ids are positions in the ``arch.words`` / ``arch.zones`` tuples.
print(f"arch.words[0].site_indices: {arch.words[0].site_indices}")
print(f"arch.zones[gate_id].name:   {arch.zones[gate_id].name}")
print(f"arch.zones[mem_id].name:    {arch.zones[mem_id].name}")

# %% [markdown]
# ### Iterating the zone bitstring
#
# ``yield_zone_locations(ZoneAddress(z))`` walks every ``LocationAddress``
# in canonical word/site order *tagged with zone z* — the layout
# hardware shots are reported against under that mode's zone view.
# Note that the iterator does not filter words to the requested zone:
# every word in the architecture is visited, with its zone field set
# to ``z``.

# %%
print("\nGate-mode bitstring iteration (every word tagged with gate_id):")
for loc in arch.yield_zone_locations(ZoneAddress(gate_id)):
    print(f"  word={loc.word_id} site={loc.site_id} zone={loc.zone_id}")

# %% [markdown]
# ### Resolving physical positions
#
# To get the physical (x, y) of a word, build the address with the
# word's *home* zone (``word_zone_map[word_id]``) so ``get_position``
# resolves against the right grid.

# %%
word_zone = arch.word_zone_map
print("\nPhysical (word, site) layout:")
for word_id in range(len(arch.words)):
    home = word_zone[word_id]
    for site_id in range(arch.sites_per_word):
        loc = LocationAddress(word_id=word_id, site_id=site_id, zone_id=home)
        x, y = arch.get_position(loc)
        print(
            f"  zone={home} word={word_id} site={site_id}" f" → ({x:5.1f}, {y:5.1f}) µm"
        )

# %% [markdown]
# ### Entangling-pair lookups
#
# ``get_cz_partner`` returns the matching-site partner for a location, or
# ``None`` if the location isn't in any entangling pair.

# %%
print("\nCZ partners (each pair shown once):")
for word_id in range(len(arch.words)):
    home = word_zone[word_id]
    for site_id in range(arch.sites_per_word):
        loc = LocationAddress(word_id=word_id, site_id=site_id, zone_id=home)
        partner = arch.get_cz_partner(loc)
        if partner is None:
            continue
        if (loc.word_id, loc.site_id) > (partner.word_id, partner.site_id):
            continue  # already printed under the lower endpoint
        print(
            f"  word={loc.word_id} site={loc.site_id}"
            f"  ↔  word={partner.word_id} site={partner.site_id}"
        )

# %% [markdown]
# ### Bus-level connectivity (``resolve_forward`` / ``resolve_backward``)
#
# Each zone's ``word_buses`` and ``site_buses`` are the Rust-backed
# bus objects the decoder uses for ``LaneAddress`` resolution.
# ``resolve_forward(src)`` returns the destination word/site for a given
# source on the bus (or ``None`` if the source isn't covered).

# %%
print("\nGate zone word bus 0:")
gate_zone = arch.zones[gate_id]
word_bus = gate_zone.word_buses[0]
print(f"  src={list(word_bus.src)} dst={list(word_bus.dst)}")
print(f"  resolve_forward(0)  = {word_bus.resolve_forward(0)}  (word 0 → word 2)")
print(f"  resolve_backward(2) = {word_bus.resolve_backward(2)}  (word 2 ← word 0)")

print("\nGate zone site bus 0:")
site_bus = gate_zone.site_buses[0]
print(f"  src={list(site_bus.src)} dst={list(site_bus.dst)}")
print(f"  resolve_forward(0)  = {site_bus.resolve_forward(0)}  (site 0 → site 1)")
print(f"  resolve_backward(1) = {site_bus.resolve_backward(1)}  (site 1 ← site 0)")

# %% [markdown]
# ### Lane endpoints
#
# A ``LaneAddress`` identifies a single src/dst pair on a bus.
# ``get_lane_address`` recovers the lane from a (src, dst) location pair;
# ``get_endpoints`` is the inverse, useful when iterating over
# ``arch.paths``.

# %%
src = LocationAddress(word_id=w0, site_id=0, zone_id=gate_id)
dst = LocationAddress(word_id=w2, site_id=0, zone_id=gate_id)
lane = arch.get_lane_address(src, dst)
assert lane is not None
print(f"\nLane for {src} → {dst}:")
print(f"  {lane}")
recovered_src, recovered_dst = arch.get_endpoints(lane)
print(f"  endpoints: {recovered_src}, {recovered_dst}")
print(f"  waypoints: {arch.get_path(lane)}")

# %% [markdown]
# ## 3. Render the layout
#
# A quick scatter plot of every site, coloured by zone, with CZ pairs drawn
# as red edges. ``ArchSpec.x_bounds`` / ``y_bounds`` are convenient when
# composing your own figures.

# %%
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(8, 3))
zone_colors = ["tab:blue", "tab:orange"]

# Group sites by their physical home zone via ``word_zone_map``;
# build each ``LocationAddress`` using the word's home zone so
# ``get_position`` resolves to the right grid.
for zid, zone in enumerate(arch.zones):
    xs, ys = [], []
    for word_id in range(len(arch.words)):
        if word_zone[word_id] != zid:
            continue
        for site_id in range(arch.sites_per_word):
            loc = LocationAddress(word_id=word_id, site_id=site_id, zone_id=zid)
            x, y = arch.get_position(loc)
            xs.append(x)
            ys.append(y)
    ax.scatter(xs, ys, c=zone_colors[zid], label=zone.name, s=80, zorder=2)

# CZ pairs as red edges (each drawn once).
for word_id in range(len(arch.words)):
    home = word_zone[word_id]
    for site_id in range(arch.sites_per_word):
        loc = LocationAddress(word_id=word_id, site_id=site_id, zone_id=home)
        partner = arch.get_cz_partner(loc)
        if partner is None:
            continue
        if (loc.word_id, loc.site_id) > (partner.word_id, partner.site_id):
            continue
        x0, y0 = arch.get_position(loc)
        x1, y1 = arch.get_position(partner)
        ax.plot([x0, x1], [y0, y1], color="tab:red", lw=1.0, zorder=1)

ax.set_xlabel("x (µm)")
ax.set_ylabel("y (µm)")
ax.set_title("ArchSpec geometry (sites coloured by zone, CZ pairs in red)")
ax.legend(loc="upper right")
ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout()
plt.show()
