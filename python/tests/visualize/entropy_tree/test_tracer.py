from __future__ import annotations

import pytest

from bloqade import squin
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


@pytest.mark.slow
def test_build_entropy_trace_targets_all_qubits_without_spectator_blocking():
    """With ``block_spectators`` off (the default), every qubit in the block is
    part of the CZ solve, so the traced target covers all qubits and no
    spectators are exposed as blocked locations.

    Spectator scoping (participants-only target, spectators blocked) is opt-in
    via ``RustPlacementTraversal.block_spectators``; ``build_entropy_trace`` does
    not enable it, so the trace reflects the unscoped default.
    """

    @squin.kernel(typeinfer=True, fold=True)
    def spectator_kernel():
        q = squin.qalloc(4)
        squin.u3(0.1, 0.2, 0.3, q[2])
        squin.u3(0.1, 0.2, 0.3, q[3])
        squin.cz(q[0], q[1])

    bundle = build_entropy_trace(
        kernel=spectator_kernel,
        kernel_name="spectator_kernel",
        layer_index=0,
        max_expansions=5,
        max_goal_candidates=3,
    )

    # All four qubits are routed (the two CZ participants plus the two
    # spectators), and nothing is blocked under the default.
    assert len(bundle.traced_target) == 4
    assert len(bundle.blocked_locations) == 0
