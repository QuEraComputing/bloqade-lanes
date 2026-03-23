from typing import Callable

import pytest

from bloqade.gemini.arch.impls import generate_arch_hypercube
from bloqade.lanes.layout.encoding import LaneAddress, LocationAddress
from bloqade.lanes.layout.path import PathFinder


def _build_pathfinder() -> PathFinder:
    # A 2D hypercube (4 words) gives multiple route choices across word buses.
    arch_spec = generate_arch_hypercube(hypercube_dims=2, word_size_y=5)
    return PathFinder(arch_spec)


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
    start = LocationAddress(0, 5)
    end = LocationAddress(3, 5)

    candidate_paths = (
        (LocationAddress(0, 5), LocationAddress(1, 5), LocationAddress(3, 5)),
        (LocationAddress(0, 5), LocationAddress(2, 5), LocationAddress(3, 5)),
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
    start = LocationAddress(0, 5)
    end = LocationAddress(3, 5)

    def custom_edge_weight(lane_address: LaneAddress) -> float:
        src, dst = path_finder.get_endpoints(lane_address)
        assert src is not None and dst is not None
        # Penalize routes that go through word 1 to force the 0->2->3 route.
        if src.word_id == 1 or dst.word_id == 1:
            return 100.0
        return 1.0

    path_via_word1 = (
        LocationAddress(0, 5),
        LocationAddress(1, 5),
        LocationAddress(3, 5),
    )
    path_via_word2 = (
        LocationAddress(0, 5),
        LocationAddress(2, 5),
        LocationAddress(3, 5),
    )
    assert _path_weight(path_finder, path_via_word2, custom_edge_weight) < _path_weight(
        path_finder, path_via_word1, custom_edge_weight
    )

    result = path_finder.find_path(start, end, edge_weight=custom_edge_weight)
    assert result is not None
    _, locations = result
    assert locations == path_via_word2


def test_find_path_tie_breaks_with_path_heuristic():
    """Tests the path heuristic as a tie-breaker for shortest paths"""
    path_finder = _build_pathfinder()
    start = LocationAddress(0, 5)
    end = LocationAddress(3, 5)

    def constant_weight(_lane_address: LaneAddress) -> float:
        return 1.0

    def prefer_word2(
        _lanes: tuple[LaneAddress, ...], locations: tuple[LocationAddress, ...]
    ) -> float:
        return 0.0 if LocationAddress(2, 5) in locations else 1.0

    result = path_finder.find_path(
        start,
        end,
        edge_weight=constant_weight,
        path_heuristic=prefer_word2,
    )
    assert result is not None
    _, locations = result
    assert locations == (
        LocationAddress(0, 5),
        LocationAddress(2, 5),
        LocationAddress(3, 5),
    )


@pytest.mark.parametrize(
    "occupied",
    [
        frozenset({LocationAddress(0, 5)}),
        frozenset({LocationAddress(3, 5)}),
    ],
)
def test_find_path_returns_none_when_start_or_end_is_occupied(
    occupied: frozenset[LocationAddress],
):
    path_finder = _build_pathfinder()
    start = LocationAddress(0, 5)
    end = LocationAddress(3, 5)

    result = path_finder.find_path(start, end, occupied=occupied)
    assert result is None


def test_find_path_returns_none_when_intermediate_nodes_block_all_routes():
    """Tests that find_path returns None when intermediate nodes block all possible paths"""
    path_finder = _build_pathfinder()
    start = LocationAddress(0, 5)
    end = LocationAddress(3, 5)
    word_size = len(path_finder.spec.words[0].site_indices)
    occupied = frozenset(
        {
            LocationAddress(word_id, site_id)
            for word_id in (1, 2)
            for site_id in range(word_size)
        }
    )

    result = path_finder.find_path(start, end, occupied=occupied)
    assert result is None
