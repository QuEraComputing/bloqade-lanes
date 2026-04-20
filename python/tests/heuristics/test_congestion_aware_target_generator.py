from __future__ import annotations

import pytest

from bloqade.lanes import layout
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.heuristics.physical.target_generator import (
    _lane_key,
    _LaneKey,
    _sum_base,
    _sum_weighted,
)
from bloqade.lanes.layout import Direction, LaneAddress, MoveType, PathFinder


@pytest.fixture(scope="module")
def arch() -> layout.ArchSpec:
    return get_arch_spec()


def _pick_cz_pair(
    arch: layout.ArchSpec,
) -> tuple[layout.LocationAddress, layout.LocationAddress]:
    """Return the first CZ-partnered (loc, partner) pair from arch.home_sites."""
    for s in arch.home_sites:
        p = arch.get_cz_partner(s)
        if p is not None and p != s:
            return s, p
    raise AssertionError(
        "fixture prerequisite failed: arch has no CZ-partnered home site"
    )


def test_lane_key_strips_direction():
    forward = LaneAddress(MoveType.SITE, 1, 2, 3, Direction.FORWARD, 4)
    backward = LaneAddress(MoveType.SITE, 1, 2, 3, Direction.BACKWARD, 4)
    assert _lane_key(forward) == _lane_key(backward)


def test_lane_key_distinguishes_different_lanes():
    a = LaneAddress(MoveType.SITE, 1, 2, 3, Direction.FORWARD, 4)
    b = LaneAddress(MoveType.SITE, 1, 2, 3, Direction.FORWARD, 5)  # different zone
    assert _lane_key(a) != _lane_key(b)


def test_lane_key_tuple_shape():
    lane = LaneAddress(MoveType.SITE, 1, 2, 3, Direction.FORWARD, 4)
    key: _LaneKey = _lane_key(lane)
    assert isinstance(key, tuple)
    assert len(key) == 5
    assert key == (MoveType.SITE, 1, 2, 3, 4)


def test_sum_base_empty_path(arch):
    pf = PathFinder(arch)
    loc, _ = _pick_cz_pair(arch)
    path = pf.find_path(loc, loc)
    assert path is not None
    assert _sum_base(path, pf) == 0.0


def test_sum_weighted_empty_path(arch):
    pf = PathFinder(arch)
    loc, _ = _pick_cz_pair(arch)
    path = pf.find_path(loc, loc)
    assert path is not None
    assert _sum_weighted(path, lambda lane: 42.0) == 0.0


def test_sum_weighted_sums_per_lane(arch):
    pf = PathFinder(arch)
    src, dst = _pick_cz_pair(arch)
    path = pf.find_path(src, dst)
    assert path is not None
    lane_count = len(path[0])
    assert _sum_weighted(path, lambda lane: 1.0) == float(lane_count)
