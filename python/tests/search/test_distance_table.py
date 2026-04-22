"""Unit tests for DistanceTable — mirrors Rust heuristic.rs DistanceTable.

Uses the logical Gemini arch (same fixture pattern as test_tree.py) so we
have a real lane graph without needing a hand-crafted ArchSpec.

Topology quick-reference (logical Gemini):
  - Word buses move atoms between words: word 0 ↔ word 1 ↔ … (forward/backward).
  - Site buses move atoms within a word between sites.
  - LocationAddress(word_id, site_id, zone_id=0).

Reachability facts used in tests:
  - (0,0) → (1,0): 1 hop via word bus forward.
  - (0,0) → (0,0): 0 hops (self).
  - LocationAddress(word=999, site=999): not in the graph → unreachable.

NOTE: The logical Gemini arch has NO transport-path timing data (no "paths"
key in the JSON).  Rust's LaneIndex only populates lane_durations from
arch_spec.paths and returns fastest_lane_duration_us() == None when empty.
Python mirrors this: with_time_distances() checks arch_spec.paths and
returns early (no time data) for arches without explicit paths.  Tests that
exercise time-distance behaviour therefore use a minimal arch with inline
path waypoints so that both Rust and Python compute matching durations.
"""

from __future__ import annotations

import json

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import LocationAddress
from bloqade.lanes.search.distance_table import DistanceTable
from bloqade.lanes.search.tree import ConfigurationTree

# ── Fixture ──────────────────────────────────────────────────────────


def _make_tree() -> ConfigurationTree:
    """Minimal tree — only the arch_spec and lane index matter for DistanceTable."""
    arch_spec = logical.get_arch_spec()
    # Placement is arbitrary; DistanceTable only uses the lane graph.
    placement = {0: LocationAddress(0, 0)}
    return ConfigurationTree.from_initial_placement(arch_spec, placement)


LOC_A = LocationAddress(0, 0)  # word 0, site 0
LOC_B = LocationAddress(1, 0)  # word 1, site 0  — 1 hop from A via word bus
LOC_UNREACHABLE = LocationAddress(999, 999)  # not in arch


# ── Arch fixture with explicit transport paths ────────────────────────
#
# Provides a minimal two-word arch where site bus 0→5 has an explicit path
# so that Python's with_time_distances (which only uses arch_spec.paths)
# finds timing data.  The path is a straight segment for simplicity.
#
# Lane encoding for site bus forward (site 0 → site 5, word 0, zone 0):
#   LaneAddr { move_type: SiteBus(0), zone_id: 0, word_id: 0, site_id: 0,
#               bus_id: 0, direction: Forward }
# Encoded via LaneAddr::encode_u64() — matches what Python LaneAddress hashes to.
# We inject a path using the lane's encoded integer (obtained at runtime below).
_ARCH_WITH_PATHS_JSON_TEMPLATE = """{{
    "version": "2.0",
    "words": [
        {{ "sites": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
                    [0, 1], [1, 1], [2, 1], [3, 1], [4, 1]] }},
        {{ "sites": [[0, 2], [1, 2], [2, 2], [3, 2], [4, 2],
                    [0, 3], [1, 3], [2, 3], [3, 3], [4, 3]] }}
    ],
    "zones": [
        {{
            "grid": {{
                "x_start": 1.0, "y_start": 2.5,
                "x_spacing": [2.0, 2.0, 2.0, 2.0],
                "y_spacing": [2.5, 7.5, 2.5]
            }},
            "site_buses": [
                {{ "src": [0, 1, 2, 3, 4], "dst": [5, 6, 7, 8, 9] }}
            ],
            "word_buses": [
                {{ "src": [0], "dst": [1] }}
            ],
            "words_with_site_buses": [0, 1],
            "sites_with_word_buses": [5, 6, 7, 8, 9],
            "entangling_pairs": [[0, 1]]
        }}
    ],
    "zone_buses": [],
    "modes": [
        {{ "name": "default", "zones": [0], "bitstring_order": [] }}
    ],
    "paths": [
        {{
            "lane": {lane_enc},
            "waypoints": [[1.0, 2.5], [1.0, 5.0]]
        }}
    ]
}}"""


def _make_tree_with_paths():
    """Create a ConfigurationTree from an arch that has explicit transport paths.

    Returns (tree, loc_src, loc_dst) where src→dst is the lane with a path.

    Two-pass construction: first build arch without paths to discover the u64
    lane encoding (needed for the JSON "paths" key), then rebuild with the
    path injected.  The lane encoding is serialized as a hex string because
    Rust ``TransportPath.lane`` uses hex serde.
    """
    from bloqade.lanes.bytecode._native import ArchSpec as _RustArchSpec
    from bloqade.lanes.layout.arch import ArchSpec

    # First build tree without paths to discover the lane encoding.
    no_path_json = json.loads(_ARCH_WITH_PATHS_JSON_TEMPLATE.format(lane_enc='"0x0"'))
    no_path_json.pop("paths", None)
    arch_spec_no_path = ArchSpec(_RustArchSpec.from_json(json.dumps(no_path_json)))
    tmp_placement = {0: LocationAddress(0, 0)}
    tmp_tree = ConfigurationTree.from_initial_placement(
        arch_spec_no_path, tmp_placement
    )

    # Find the site-bus forward lane from site 0 to site 5 in word 0.
    src_loc = LocationAddress(0, 0, 0)  # word=0, site=0, zone=0
    dst_loc = LocationAddress(0, 5, 0)  # word=0, site=5, zone=0
    target_lane = None
    for lane in tmp_tree.outgoing_lanes(src_loc):
        _, endpoint = arch_spec_no_path.get_endpoints(lane)
        if endpoint == dst_loc:
            target_lane = lane
            break

    assert target_lane is not None, "Could not find src→dst lane"
    # Get the Rust encoded u64 for this lane — serialised as hex for JSON.
    lane_enc_int: int = target_lane._inner.encode()
    lane_enc_hex = f'"0x{lane_enc_int:016x}"'

    # Now build the real arch with path injected.
    arch_json = _ARCH_WITH_PATHS_JSON_TEMPLATE.format(lane_enc=lane_enc_hex)
    arch_spec = ArchSpec(_RustArchSpec.from_json(arch_json))
    placement = {0: src_loc}
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    return tree, src_loc, dst_loc


# ── hop_distance ─────────────────────────────────────────────────────


def test_hop_distance_self_is_zero():
    """hop_distance(X, X) == 0 when X is a target."""
    tree = _make_tree()
    table = DistanceTable.build(tree, [LOC_A])
    assert table.hop_distance(LOC_A, LOC_A) == 0.0


def test_hop_distance_one_hop():
    """hop_distance(A, B) == 1 for directly-connected neighbours."""
    tree = _make_tree()
    table = DistanceTable.build(tree, [LOC_B])
    assert table.hop_distance(LOC_A, LOC_B) == 1.0


def test_hop_distance_unreachable_returns_inf():
    """hop_distance returns float('inf') when target is not in the graph."""
    tree = _make_tree()
    table = DistanceTable.build(tree, [LOC_UNREACHABLE])
    result = table.hop_distance(LOC_A, LOC_UNREACHABLE)
    assert result == float("inf")


def test_hop_distance_unknown_target_returns_inf():
    """hop_distance returns float('inf') when queried target was never registered."""
    tree = _make_tree()
    table = DistanceTable.build(tree, [LOC_A])
    # LOC_B was never added as a target
    result = table.hop_distance(LOC_A, LOC_B)
    assert result == float("inf")


# ── with_time_distances / time_distance / fastest_lane_us ────────────


def test_fastest_lane_us_is_none_before_time_distances():
    """fastest_lane_us() returns None before with_time_distances is called."""
    tree = _make_tree()
    table = DistanceTable.build(tree, [LOC_B])
    assert table.fastest_lane_us() is None


def test_fastest_lane_us_is_none_for_arch_without_paths():
    """fastest_lane_us() returns None when the arch has no transport-path timing data.

    The logical Gemini arch has no "paths" key in the JSON, so Python's
    with_time_distances() (Rust-aligned) skips time distances entirely.
    """
    tree = _make_tree()
    table = DistanceTable.build(tree, [LOC_B]).with_time_distances(tree)
    fastest = table.fastest_lane_us()
    assert fastest is None


def test_fastest_lane_us_is_positive_after_time_distances():
    """fastest_lane_us() returns a positive float for an arch with explicit paths."""
    tree, src_loc, dst_loc = _make_tree_with_paths()
    table = DistanceTable.build(tree, [dst_loc]).with_time_distances(tree)
    fastest = table.fastest_lane_us()
    assert fastest is not None
    assert fastest > 0.0


def test_time_distance_self_is_zero():
    """time_distance(X, X) == 0.0 when X is a target (arch with paths)."""
    tree, src_loc, dst_loc = _make_tree_with_paths()
    table = DistanceTable.build(tree, [src_loc]).with_time_distances(tree)
    assert table.time_distance(src_loc, src_loc) == 0.0


def test_time_distance_one_hop_is_finite_positive():
    """time_distance for a reachable location is finite and > 0 (arch with paths)."""
    tree, src_loc, dst_loc = _make_tree_with_paths()
    table = DistanceTable.build(tree, [dst_loc]).with_time_distances(tree)
    td = table.time_distance(src_loc, dst_loc)
    assert td > 0.0
    assert td != float("inf")


def test_time_distance_unreachable_returns_inf():
    """time_distance returns float('inf') for unreachable targets."""
    tree = _make_tree()
    table = DistanceTable.build(tree, [LOC_UNREACHABLE]).with_time_distances(tree)
    result = table.time_distance(LOC_A, LOC_UNREACHABLE)
    assert result == float("inf")


# ── distance (blended) ───────────────────────────────────────────────


def test_blended_distance_wt_zero_equals_hop():
    """With w_t=0, blended distance equals hop distance."""
    tree = _make_tree()
    table = DistanceTable.build(tree, [LOC_B]).with_time_distances(tree)
    hop = table.hop_distance(LOC_A, LOC_B)
    blended = table.distance(LOC_A, LOC_B, w_t=0.0)
    assert blended == hop


def test_blended_distance_formula():
    """Blended distance follows (1-w_t)*hop + w_t*(time/fastest) for arch with paths."""
    tree, src_loc, dst_loc = _make_tree_with_paths()
    table = DistanceTable.build(tree, [dst_loc]).with_time_distances(tree)

    hop = table.hop_distance(src_loc, dst_loc)
    time_us = table.time_distance(src_loc, dst_loc)
    fastest = table.fastest_lane_us()
    assert fastest is not None

    w_t = 0.5
    expected = (1.0 - w_t) * hop + w_t * (time_us / fastest)
    result = table.distance(src_loc, dst_loc, w_t=w_t)
    assert abs(result - expected) < 1e-12


def test_blended_distance_self_is_zero():
    """Blended distance to self is 0 regardless of w_t."""
    tree = _make_tree()
    table = DistanceTable.build(tree, [LOC_A]).with_time_distances(tree)
    assert table.distance(LOC_A, LOC_A, w_t=0.5) == 0.0


def test_blended_distance_unreachable_returns_inf():
    """Blended distance returns float('inf') for unreachable targets."""
    tree = _make_tree()
    table = DistanceTable.build(tree, [LOC_UNREACHABLE]).with_time_distances(tree)
    result = table.distance(LOC_A, LOC_UNREACHABLE, w_t=0.5)
    assert result == float("inf")


# ── ConfigurationTree caching ───────────────────────────────────────


def test_tree_caches_distance_table():
    """ConfigurationTree.distance_table() returns the same instance on repeat lookups."""
    tree = _make_tree()
    targets = frozenset({LOC_A})
    w_t = 0.5

    # First lookup
    t1 = tree.distance_table(targets=targets, w_t=w_t)
    # Second lookup with identical args
    t2 = tree.distance_table(targets=targets, w_t=w_t)

    # Must be the same object (identity, not just equality)
    assert t1 is t2


def test_tree_distance_table_different_targets_different_cache():
    """Different target sets get separate cached instances."""
    tree = _make_tree()

    t1 = tree.distance_table(targets=frozenset({LOC_A}), w_t=0.5)
    t2 = tree.distance_table(targets=frozenset({LOC_B}), w_t=0.5)

    # Different targets → different instances
    assert t1 is not t2


def test_tree_distance_table_different_wt_different_cache():
    """Different w_t values get separate cached instances."""
    tree = _make_tree()
    targets = frozenset({LOC_A})

    t1 = tree.distance_table(targets=targets, w_t=0.0)
    t2 = tree.distance_table(targets=targets, w_t=0.5)

    # Different w_t → different instances (one has time data, one doesn't)
    assert t1 is not t2
