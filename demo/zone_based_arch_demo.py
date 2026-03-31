# %% Section 1: Gemini-Equivalent Architecture
#
# The existing Gemini architecture uses generate_arch_hypercube() which
# creates a 1D array of 2-column words with hypercube word buses.
#
# The new zone-based API can express the same architecture using a single
# entangling zone with a HypercubeWordTopology. Each old 2-column word
# becomes a CZ pair of two single-row words with interleaved sites:
#
#   Old (1 word, 2 cols x 4 sites):     New (2 words, 4 sites each, interleaved):
#     col0  col1                           w0.s0 w1.s0 w0.s1 w1.s1 w0.s2 ...
#       o     o                              o     o     o     o     o   ...
#       o     o
#       o     o
#       o     o
#
# We use a 4x2 grid (4 rows, 1 CZ pair per row) with word_size_y=4:
#
#   row 0:  o o o o o o o o   <- 1 CZ pair (word 0 + word 1)
#   row 1:  o o o o o o o o   <- 1 CZ pair (word 2 + word 3)
#   row 2:  o o o o o o o o   <- 1 CZ pair (word 4 + word 5)
#   row 3:  o o o o o o o o   <- 1 CZ pair (word 6 + word 7)

from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
from bloqade.lanes.arch import (
    ArchBlueprint,
    DeviceLayout,
    HypercubeSiteTopology,
    HypercubeWordTopology,
    MatchingTopology,
    ZoneSpec,
    build_arch,
)

HYPERCUBE_DIMS = 2
WORD_SIZE_Y = 4

# Build using the old API
old_arch = generate_arch_hypercube(hypercube_dims=HYPERCUBE_DIMS, word_size_y=WORD_SIZE_Y)

# Build the same architecture using the new zone-based API.
# 2^2 = 4 old words -> 4 CZ pairs -> 4 rows x 2 columns.
#
# HypercubeWordTopology on (4 rows x 2 cols) gives 3 word buses:
#   1 col dim (dim 0):  CZ pair connectivity within each row
#                        (replaces old intra-word site buses for left<->right)
#   2 row dims (dim 1-2): hypercube connectivity between rows
#                          (equivalent to old inter-word hypercube buses)
#
# HypercubeSiteTopology on 4 sites gives 2 site buses:
#   Moves atoms within a single word (row of sites)
new_blueprint = ArchBlueprint(
    zones={
        "gate": ZoneSpec(
            num_rows=2**HYPERCUBE_DIMS,
            num_cols=2,
            entangling=True,
            word_topology=HypercubeWordTopology(),
            site_topology=HypercubeSiteTopology(),
        ),
    },
    layout=DeviceLayout(sites_per_word=WORD_SIZE_Y),
)
new_result = build_arch(new_blueprint)
new_arch = new_result.arch

# Compare the two architectures
old_sites_per_word = len(old_arch.words[0].site_indices)
new_sites_per_word = len(new_arch.words[0].site_indices)

print("=== Section 1: Gemini-Equivalent Architecture ===")
print()
print(f"{'':20s} {'Old API':>10s}  {'New API':>10s}")
print(f"{'-'*20} {'-'*10}  {'-'*10}")
print(f"{'Words':20s} {len(old_arch.words):>10d}  {len(new_arch.words):>10d}")
print(f"{'Sites per word':20s} {old_sites_per_word:>10d}  {new_sites_per_word:>10d}")
print(f"{'Total sites':20s} {len(old_arch.words) * old_sites_per_word:>10d}  {len(new_arch.words) * new_sites_per_word:>10d}")
print(f"{'Word buses':20s} {len(old_arch.word_buses):>10d}  {len(new_arch.word_buses):>10d}")
print(f"{'Site buses':20s} {len(old_arch.site_buses):>10d}  {len(new_arch.site_buses):>10d}")
print()
print("Note: The old API uses 2-column words (8 sites each), while the new API")
print("uses single-row words (4 sites each) with interleaved CZ pairs.")
print(f"Total sites are equal: {len(old_arch.words) * old_sites_per_word} = {len(new_arch.words) * new_sites_per_word}")
print()
print("New API uses HypercubeSiteTopology (log2(4) = 2 site buses) for intra-word")
print("movement and HypercubeWordTopology (2 row dims + 1 col dim = 3 word buses)")
print("for inter-word movement.")

# %% Section 2: Multi-Zone Architecture (Processing + Memory)
#
# This is the key use case for #367: defining architectures with distinct
# processing and memory regions that have different connectivity.
#
# Both zones use the same 4x2 word grid (4 rows, 1 CZ pair per row),
# but with different capabilities:
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

blueprint = ArchBlueprint(
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
result = build_arch(blueprint, connections={
    ("proc", "mem"): MatchingTopology(),
})

arch = result.arch

print()
print()
print("=== Section 2: Multi-Zone Architecture (Processing + Memory) ===")
print()

# Zone summary
print("Zones:")
for name, spec in blueprint.zones.items():
    print(f"  {name}: {spec.num_rows}x{spec.num_cols} grid = {spec.num_words} words, "
          f"entangling={spec.entangling}")
print()

# Bus summary — the key difference between zones
proc_site_bus_count = sum(
    1 for bus in arch.site_buses
    if bus.words is not None and 0 in bus.words  # word 0 is in proc
)
mem_site_bus_count = sum(
    1 for bus in arch.site_buses
    if bus.words is not None and result.zone_grids["mem"].word_id_offset in bus.words
)

print("Bus summary:")
print(f"  Total word buses:  {len(arch.word_buses)}")
print(f"    Intra-zone (proc hypercube): {len(arch.word_buses) - 1}")
print(f"    Inter-zone (matching):       1")
print(f"  Total site buses:  {len(arch.site_buses)}")
print(f"    proc zone: {proc_site_bus_count} site buses (HypercubeSite, 4 sites -> 2 buses)")
print(f"    mem zone:  {mem_site_bus_count} site buses (no site topology)")
print()

# This is the punchline: proc has rich connectivity, mem has none
print("Per-zone site bus scoping (the key feature of #367):")
print(f"  proc words with site buses: {sorted(arch.has_site_buses & set(result.zone_grids['proc'].all_word_ids))}")
print(f"  mem words with site buses:  {sorted(arch.has_site_buses & set(result.zone_grids['mem'].all_word_ids))}")
