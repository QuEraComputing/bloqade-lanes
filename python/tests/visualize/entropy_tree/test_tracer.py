from __future__ import annotations

from pathlib import Path

from bloqade.lanes.bytecode.encoding import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
)
from bloqade.lanes.visualize.entropy_tree.tracer import (
    _decode_config,
    _decode_lane,
    build_entropy_trace,
    load_kernel_from_file,
)


def test_decode_lane_forward_site():
    lane = _decode_lane((0, 0, 3, 5, 7, 2))
    assert isinstance(lane, LaneAddress)
    assert lane.direction == Direction.FORWARD
    assert lane.move_type == MoveType.SITE
    assert lane.zone_id == 3
    assert lane.word_id == 5
    assert lane.site_id == 7
    assert lane.bus_id == 2


def test_decode_lane_backward_zone():
    lane = _decode_lane((1, 2, 0, 0, 0, 0))
    assert lane.direction == Direction.BACKWARD
    assert lane.move_type == MoveType.ZONE


def test_decode_config_returns_qid_mapping():
    entries = [(0, 1, 2, 3), (1, 0, 4, 5)]
    cfg = _decode_config(entries)
    assert cfg == {
        0: LocationAddress(2, 3, 1),
        1: LocationAddress(4, 5, 0),
    }


def test_build_entropy_trace_exposes_non_layer_qubits_as_blocked_locations():
    kernel_path = (
        Path(__file__).parents[3] / "benchmarks/kernels/large/trotter_rand_35.py"
    )
    kernel = load_kernel_from_file(kernel_path, "trotter_rand_35")

    bundle = build_entropy_trace(
        kernel=kernel,
        kernel_name="trotter_rand_35",
        layer_index=0,
        max_expansions=5,
        max_goal_candidates=3,
    )

    assert len(bundle.traced_target) == 2
    assert len(bundle.blocked_locations) == 33
    assert set(bundle.blocked_locations).isdisjoint(bundle.traced_target.values())
    assert all(loc in bundle.location_to_global_qid for loc in bundle.blocked_locations)
