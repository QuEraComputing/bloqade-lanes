from typing import Callable

import pytest

from bloqade.lanes.arch import (
    AllToAllSiteTopology,
    ArchBlueprint,
    DeviceLayout,
    HypercubeWordTopology,
    ZoneSpec,
    build_arch,
)
from bloqade.lanes.layout.encoding import LaneAddress, LocationAddress
from bloqade.lanes.layout.path import PathFinder


def _build_pathfinder() -> PathFinder:
    # 8 words with 3-dimensional hypercube word buses gives multiple route choices.
    bp = ArchBlueprint(
        zones={
            "gate": ZoneSpec(
                num_rows=1,
                num_cols=8,
                entangling=True,
                word_topology=HypercubeWordTopology(),
                site_topology=AllToAllSiteTopology(),
            )
        },
        layout=DeviceLayout(sites_per_word=10),
    )
    return PathFinder(build_arch(bp).arch)


def _path_duration(
    path_finder: PathFinder, locations: tuple[LocationAddress, ...]
) -> float:
    duration = 0.0
    for src, dst in zip(locations, locations[1:]):
        lane = path_finder.spec.get_lane_address(src, dst)
        assert lane is not None
        duration += path_finder.metrics.get_lane_duration_us(lane)
    return duration


def _path_weight(
    path_finder: PathFinder,
    locations: tuple[LocationAddress, ...],
    edge_weight: Callable[[LaneAddress], float],
) -> float:
    weight = 0.0
    for src, dst in zip(locations, locations[1:]):
        lane = path_finder.spec.get_lane_address(src, dst)
        assert lane is not None
        weight += edge_weight(lane)
    return weight


def test_find_path_defaults_to_duration_shortest_paths():
    path_finder = _build_pathfinder()
    # Word 0 → word 6 (0b000 → 0b110): two hops via word 2 or word 4
    start = LocationAddress(0, 5)
    end = LocationAddress(6, 5)

    candidate_paths = (
        (LocationAddress(0, 5), LocationAddress(2, 5), LocationAddress(6, 5)),
        (LocationAddress(0, 5), LocationAddress(4, 5), LocationAddress(6, 5)),
    )
    durations = [_path_duration(path_finder, path) for path in candidate_paths]
    min_duration = min(durations)
    expected_shortest_paths = {
        path
        for path, duration in zip(candidate_paths, durations)
        if duration == pytest.approx(min_duration)
    }

    result = path_finder.find_path(start, end, edge_weight=None)
    assert result is not None
    lanes, locations = result
    assert len(lanes) > 0
    assert locations in expected_shortest_paths


def test_find_path_uses_custom_edge_weight_shortest_path():
    path_finder = _build_pathfinder()
    # Word 0 → word 6: can go via word 2 or word 4
    start = LocationAddress(0, 5)
    end = LocationAddress(6, 5)

    def custom_edge_weight(lane_address: LaneAddress) -> float:
        src, dst = path_finder.get_endpoints(lane_address)
        assert src is not None and dst is not None
        # Penalize routes through word 2 to force the 0→4→6 route.
        if src.word_id == 2 or dst.word_id == 2:
            return 100.0
        return 1.0

    path_via_word2 = (
        LocationAddress(0, 5),
        LocationAddress(2, 5),
        LocationAddress(6, 5),
    )
    path_via_word4 = (
        LocationAddress(0, 5),
        LocationAddress(4, 5),
        LocationAddress(6, 5),
    )
    assert _path_weight(path_finder, path_via_word4, custom_edge_weight) < _path_weight(
        path_finder, path_via_word2, custom_edge_weight
    )

    result = path_finder.find_path(start, end, edge_weight=custom_edge_weight)
    assert result is not None
    _, locations = result
    assert locations == path_via_word4


def test_find_path_tie_breaks_with_path_heuristic():
    """Tests the path heuristic as a tie-breaker for shortest paths"""
    path_finder = _build_pathfinder()
    start = LocationAddress(0, 5)
    end = LocationAddress(6, 5)

    def constant_weight(_lane_address: LaneAddress) -> float:
        return 1.0

    def prefer_word4(
        _lanes: tuple[LaneAddress, ...], locations: tuple[LocationAddress, ...]
    ) -> float:
        return 0.0 if LocationAddress(4, 5) in locations else 1.0

    result = path_finder.find_path(
        start,
        end,
        edge_weight=constant_weight,
        path_heuristic=prefer_word4,
    )
    assert result is not None
    _, locations = result
    assert locations == (
        LocationAddress(0, 5),
        LocationAddress(4, 5),
        LocationAddress(6, 5),
    )


@pytest.mark.parametrize(
    "occupied",
    [
        frozenset({LocationAddress(0, 5)}),
        frozenset({LocationAddress(6, 5)}),
    ],
)
def test_find_path_returns_none_when_start_or_end_is_occupied(
    occupied: frozenset[LocationAddress],
):
    path_finder = _build_pathfinder()
    start = LocationAddress(0, 5)
    end = LocationAddress(6, 5)

    result = path_finder.find_path(start, end, occupied=occupied)
    assert result is None


def test_find_path_returns_none_when_intermediate_nodes_block_all_routes():
    """Tests that find_path returns None when intermediate nodes block all possible paths"""
    path_finder = _build_pathfinder()
    start = LocationAddress(0, 5)
    end = LocationAddress(6, 5)
    word_size = len(path_finder.spec.words[0].site_indices)
    num_words = len(path_finder.spec.words)
    # Block every word except start (0) and end (6) to ensure no path exists
    occupied = frozenset(
        {
            LocationAddress(word_id, site_id)
            for word_id in range(num_words)
            if word_id not in (0, 6)
            for site_id in range(word_size)
        }
    )

    result = path_finder.find_path(start, end, occupied=occupied)
    assert result is None


def test_find_path_respects_per_bus_word_scoping():
    """Site buses scoped to specific words should not be used by other words."""
    # Build a 2-zone arch: zone A has site buses, zone B has none.
    bp = ArchBlueprint(
        zones={
            "a": ZoneSpec(
                num_rows=1,
                num_cols=2,
                entangling=True,
                word_topology=HypercubeWordTopology(),
                site_topology=AllToAllSiteTopology(),
            ),
            "b": ZoneSpec(
                num_rows=1,
                num_cols=2,
                entangling=True,
                word_topology=HypercubeWordTopology(),
                # No site_topology: zone B has no site buses
            ),
        },
        layout=DeviceLayout(sites_per_word=4),
    )
    from bloqade.lanes.arch.topology import MatchingTopology

    result = build_arch(bp, connections={("a", "b"): MatchingTopology()})
    arch = result.arch

    # Zone B words should not have site bus moves
    zone_b_words = set(result.zone_grids["b"].all_word_ids)
    for word_id in zone_b_words:
        assert word_id not in arch.has_site_buses

    # Attempting an intra-word site move in zone B should fail (no site bus)
    b_word = min(zone_b_words)
    start = LocationAddress(b_word, 0)
    end = LocationAddress(b_word, 1)
    assert arch.get_lane_address(start, end) is None
