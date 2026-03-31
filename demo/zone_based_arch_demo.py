# %% Section 1: Gemini-Equivalent Architecture
#
# The existing Gemini architecture uses generate_arch_hypercube() which
# creates a 1D array of 2-column words with hypercube word buses.
#
# The new zone-based API can express the same architecture using a single
# entangling zone with a HypercubeWordTopology. Each old 2-column word
# becomes a CZ pair of two single-row words with interleaved sites:
#
#   Old (1 word, 2 cols x 5 sites):     New (2 words, 5 sites each, interleaved):
#     col0  col1                           w0.s0 w1.s0 w0.s1 w1.s1 w0.s2 ...
#       o     o                              o     o     o     o     o   ...
#       o     o
#       o     o
#       o     o
#       o     o
#
# We use hypercube_dims=2 (4 old words -> 8 new words) for readability.

from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
from bloqade.lanes.arch.gemini_adapter import build_gemini_arch

HYPERCUBE_DIMS = 2
WORD_SIZE_Y = 5

# Build using the old API
old_arch = generate_arch_hypercube(hypercube_dims=HYPERCUBE_DIMS, word_size_y=WORD_SIZE_Y)

# Build the same architecture using the new zone-based API
new_result = build_gemini_arch(hypercube_dims=HYPERCUBE_DIMS, word_size_y=WORD_SIZE_Y)
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
print("Note: The old API uses 2-column words (10 sites each), while the new API")
print("uses single-row words (5 sites each) with interleaved CZ pairs.")
print(f"Total sites are equal: {len(old_arch.words) * old_sites_per_word} = {len(new_arch.words) * new_sites_per_word}")

# %% Section 2: Multi-Zone Architecture (Processing + Memory)
#
# This is the key use case for #367: defining architectures with distinct
# processing and memory regions that have different connectivity.
#
# Physical layout (2x4 grid per zone, 4 sites per word):
#
#   Processing zone (entangling, hypercube word + site topology):
#     row 0: o o o o o o o o  gap  o o o o o o o o   <- 2 CZ pairs, 4 words
#     row 1: o o o o o o o o  gap  o o o o o o o o   <- 2 CZ pairs, 4 words
#
#   Memory zone (no entangling, no intra-zone connectivity):
#     row 0: o o o o o o o o  gap  o o o o o o o o
#     row 1: o o o o o o o o  gap  o o o o o o o o
#
#   Inter-zone: MatchingTopology connects each proc word to its
#   corresponding mem word by grid position (1:1 mapping).

from bloqade.lanes.arch import (
    ArchBlueprint,
    DeviceLayout,
    HypercubeSiteTopology,
    HypercubeWordTopology,
    MatchingTopology,
    ZoneSpec,
    build_arch,
)

blueprint = ArchBlueprint(
    zones={
        "proc": ZoneSpec(
            num_rows=2,
            num_cols=4,
            entangling=True,
            word_topology=HypercubeWordTopology(),
            site_topology=HypercubeSiteTopology(),
        ),
        "mem": ZoneSpec(
            num_rows=2,
            num_cols=4,
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
