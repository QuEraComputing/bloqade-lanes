# %% Section 1: Building a Custom Architecture
#
# The zone-based API defines architectures using ArchBlueprint + ZoneSpec.
# Each zone is a grid of words with configurable topology:
#
#   - word_topology: how words connect to each other (word buses)
#   - site_topology: how sites connect within a word (site buses)
#   - entangling: whether CZ pairs are formed between adjacent columns
#
# Words are arranged in a (num_rows x num_cols) grid. When entangling=True,
# columns are paired for CZ gates: (col 0, col 1), (col 2, col 3), etc.
#
# Example: 4x2 grid with 4 sites per word:
#
#   row 0:  o o o o | o o o o   <- 1 CZ pair (word 0 + word 1)
#   row 1:  o o o o | o o o o   <- 1 CZ pair (word 2 + word 3)
#   row 2:  o o o o | o o o o   <- 1 CZ pair (word 4 + word 5)
#   row 3:  o o o o | o o o o   <- 1 CZ pair (word 6 + word 7)

from bloqade.lanes.arch import (
    ArchBlueprint,
    DeviceLayout,
    HypercubeSiteTopology,
    HypercubeWordTopology,
    MatchingTopology,
    ZoneSpec,
    build_arch,
)

# A single-zone architecture with hypercube connectivity.
# 4 rows x 2 columns = 8 words, 4 sites each = 32 total sites.
# HypercubeWordTopology on (4 rows x 2 cols) gives 3 word buses:
#   1 col dim (dim 0):  CZ pair connectivity within each row
#   2 row dims (dim 1-2): hypercube connectivity between rows
# HypercubeSiteTopology on 4 sites gives 2 site buses.
blueprint = ArchBlueprint(
    zones={
        "gate": ZoneSpec(
            num_rows=4,
            num_cols=2,
            entangling=True,
            word_topology=HypercubeWordTopology(),
            site_topology=HypercubeSiteTopology(),
        ),
    },
    layout=DeviceLayout(sites_per_word=4),
)
result = build_arch(blueprint)
arch = result.arch

sites_per_word = len(arch.words[0].site_indices)

print("=== Section 1: Single-Zone Architecture ===")
print()
print(f"  Words:          {len(arch.words)}")
print(f"  Sites per word: {sites_per_word}")
print(f"  Total sites:    {len(arch.words) * sites_per_word}")
print(f"  Word buses:     {len(arch.word_buses)}")
print(f"  Site buses:     {len(arch.site_buses)}")
print(
    f"  CZ pairs:       {len(arch.entangling_zones[0]) if arch.entangling_zones else 0}"
)
print()

# %% Section 2: Multi-Zone Architecture (Processing + Memory)
#
# The key use case for zone-based architectures: distinct regions with
# different connectivity.
#
#   Processing zone (entangling=True):
#     - HypercubeWordTopology: 3 word buses (2 row dims + 1 col dim)
#     - HypercubeSiteTopology: 2 site buses (intra-word shuffling)
#     - CZ pairs enabled for entangling gates
#
#   Memory zone (entangling=False):
#     - No word topology: atoms cannot move between words within the zone
#     - No site topology: atoms cannot shuffle within a word
#     - Used for storage only
#
# Physical layout (4x2 grid per zone, 4 sites per word):
#
#   proc zone:              mem zone:
#     row 0: o o o o o o o o   row 0: o o o o o o o o
#     row 1: o o o o o o o o   row 1: o o o o o o o o
#     row 2: o o o o o o o o   row 2: o o o o o o o o
#     row 3: o o o o o o o o   row 3: o o o o o o o o
#
# Inter-zone: MatchingTopology connects each proc word to its
# corresponding mem word by grid position (1:1 mapping).
# This allows atoms to be shuttled between processing and memory.

multi_blueprint = ArchBlueprint(
    zones={
        "proc": ZoneSpec(
            num_rows=4,
            num_cols=2,
            entangling=True,
            word_topology=HypercubeWordTopology(),
            site_topology=HypercubeSiteTopology(),
        ),
        "mem": ZoneSpec(
            num_rows=4,
            num_cols=2,
            entangling=False,
            # No word_topology: no intra-zone word movement
            # No site_topology: no intra-word site shuffling
        ),
    },
    layout=DeviceLayout(
        sites_per_word=4,
        site_spacing=10.0,
        pair_spacing=20.0,
        row_spacing=50.0,
    ),
)

# Connect the two zones: each proc word maps 1:1 to its mem counterpart
multi_result = build_arch(
    multi_blueprint,
    connections={
        ("proc", "mem"): MatchingTopology(),
    },
)

multi_arch = multi_result.arch

print()
print("=== Section 2: Multi-Zone Architecture (Processing + Memory) ===")
print()

# Zone summary
print("Zones:")
for name, spec in multi_blueprint.zones.items():
    print(
        f"  {name}: {spec.num_rows}x{spec.num_cols} grid = {spec.num_words} words, "
        f"entangling={spec.entangling}"
    )
print()

# Bus summary — the key difference between zones
proc_site_bus_count = sum(
    1
    for bus in multi_arch.site_buses
    if bus.words is not None and 0 in bus.words  # word 0 is in proc
)
mem_site_bus_count = sum(
    1
    for bus in multi_arch.site_buses
    if bus.words is not None
    and multi_result.zone_grids["mem"].word_id_offset in bus.words
)

print("Bus summary:")
print(f"  Total word buses:  {len(multi_arch.word_buses)}")
print(f"    Intra-zone (proc hypercube): {len(multi_arch.word_buses) - 1}")
print("    Inter-zone (matching):       1")
print(f"  Total site buses:  {len(multi_arch.site_buses)}")
print(
    f"    proc zone: {proc_site_bus_count} site buses (HypercubeSite, 4 sites -> 2 buses)"
)
print(f"    mem zone:  {mem_site_bus_count} site buses (no site topology)")
print()

# This is the punchline: proc has rich connectivity, mem has none
print("Per-zone site bus scoping:")
print(
    f"  proc words with site buses: {sorted(multi_arch.has_site_buses & set(multi_result.zone_grids['proc'].all_word_ids))}"
)
print(
    f"  mem words with site buses:  {sorted(multi_arch.has_site_buses & set(multi_result.zone_grids['mem'].all_word_ids))}"
)
