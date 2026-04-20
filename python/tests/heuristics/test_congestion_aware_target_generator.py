from __future__ import annotations

import math

import pytest

from bloqade.lanes import layout
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.heuristics.physical.target_generator import (
    _choose_control,
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


def test_choose_control_lower_cost_wins():
    assert _choose_control(cost_c=1.0, cost_t=2.0, len_c=10, len_t=1) is True
    assert _choose_control(cost_c=2.0, cost_t=1.0, len_c=1, len_t=10) is False


def test_choose_control_cost_tie_uses_length():
    # Equal cost → shorter path wins
    assert _choose_control(cost_c=1.0, cost_t=1.0, len_c=2, len_t=5) is True
    assert _choose_control(cost_c=1.0, cost_t=1.0, len_c=5, len_t=2) is False


def test_choose_control_all_tied_prefers_control():
    assert _choose_control(cost_c=1.0, cost_t=1.0, len_c=3, len_t=3) is True


def test_choose_control_inf_handled():
    # Target infeasible → control wins
    assert _choose_control(cost_c=5.0, cost_t=math.inf, len_c=1, len_t=0) is True
    # Control infeasible → target wins
    assert _choose_control(cost_c=math.inf, cost_t=5.0, len_c=0, len_t=1) is False
