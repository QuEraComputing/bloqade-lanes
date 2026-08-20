from __future__ import annotations

import math
from collections import Counter

import pytest

from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.arch.path import PathFinder
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
)
from bloqade.lanes.heuristics.physical.target_generator import (
    CongestionAwareTargetGenerator,
    DefaultTargetGenerator,
    TargetContext,
    _choose_control,
    _lane_key,
    _LaneKey,
    _make_weight_fn,
    _sum_base,
    _sum_weighted,
)


@pytest.fixture(scope="module")
def arch() -> ArchSpec:
    return get_arch_spec()


def _pick_cz_pair(
    arch: ArchSpec,
) -> tuple[LocationAddress, LocationAddress]:
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


class _WeightCtx:
    """Minimal stand-in for the generator's factor fields."""

    def __init__(self, direction: float, site: float) -> None:
        self.direction_factor = direction
        self.shared_site_factor = site


def _first_lane(pf: PathFinder) -> LaneAddress:
    # Pick any lane in the physical graph for tests.
    return next(iter(pf.end_points_cache))


def _counts(*entries: tuple[Direction, int]) -> Counter[Direction]:
    c: Counter[Direction] = Counter()
    for d, n in entries:
        c[d] += n
    return c


def test_weight_fn_no_congestion_returns_base(arch):
    pf = PathFinder(arch)
    weight = _make_weight_fn(pf, {}, set(), _WeightCtx(0.5, 1.1))
    lane = _first_lane(pf)
    base = pf.metrics.get_lane_duration_cost(lane)
    assert weight(lane) == base


def test_weight_fn_single_same_direction_reward(arch):
    """One prior same-direction commit: N=1, M=0, factor = df ** 1."""
    pf = PathFinder(arch)
    lane = _first_lane(pf)
    committed_lanes = {_lane_key(lane): _counts((lane.direction, 1))}
    weight = _make_weight_fn(pf, committed_lanes, set(), _WeightCtx(0.5, 1.1))
    base = pf.metrics.get_lane_duration_cost(lane)
    assert weight(lane) == base * 0.5


def test_weight_fn_same_direction_reward_compounds_with_count(arch):
    """Multiple same-direction priors: factor = df ** N."""
    pf = PathFinder(arch)
    lane = _first_lane(pf)
    committed_lanes = {_lane_key(lane): _counts((lane.direction, 3))}
    weight = _make_weight_fn(pf, committed_lanes, set(), _WeightCtx(0.5, 1.1))
    base = pf.metrics.get_lane_duration_cost(lane)
    assert weight(lane) == pytest.approx(base * 0.5**3)


def test_weight_fn_single_opposite_direction_penalty(arch):
    """One prior opposite-direction commit: N=0, M=1, factor = df ** -1."""
    pf = PathFinder(arch)
    lane = _first_lane(pf)
    committed_lanes = {_lane_key(lane): _counts((lane.reverse().direction, 1))}
    weight = _make_weight_fn(pf, committed_lanes, set(), _WeightCtx(0.5, 1.1))
    base = pf.metrics.get_lane_duration_cost(lane)
    assert weight(lane) == base * 2.0  # 0.5 ** -1


def test_weight_fn_opposite_penalty_compounds_with_count(arch):
    """Multiple opposite priors: factor = df ** -M."""
    pf = PathFinder(arch)
    lane = _first_lane(pf)
    committed_lanes = {_lane_key(lane): _counts((lane.reverse().direction, 3))}
    weight = _make_weight_fn(pf, committed_lanes, set(), _WeightCtx(0.5, 1.1))
    base = pf.metrics.get_lane_duration_cost(lane)
    assert weight(lane) == pytest.approx(base * 0.5**-3)


def test_weight_fn_balanced_traffic_is_neutral(arch):
    """Equal same and opposite priors cancel: N == M ⇒ factor = 1."""
    pf = PathFinder(arch)
    lane = _first_lane(pf)
    committed_lanes = {
        _lane_key(lane): _counts(
            (lane.direction, 2),
            (lane.reverse().direction, 2),
        )
    }
    weight = _make_weight_fn(pf, committed_lanes, set(), _WeightCtx(0.5, 1.1))
    base = pf.metrics.get_lane_duration_cost(lane)
    assert weight(lane) == base


def test_weight_fn_mixed_traffic_uses_signed_net(arch):
    """N=3, M=1 ⇒ factor = df ** 2."""
    pf = PathFinder(arch)
    lane = _first_lane(pf)
    committed_lanes = {
        _lane_key(lane): _counts(
            (lane.direction, 3),
            (lane.reverse().direction, 1),
        )
    }
    weight = _make_weight_fn(pf, committed_lanes, set(), _WeightCtx(0.5, 1.1))
    base = pf.metrics.get_lane_duration_cost(lane)
    assert weight(lane) == pytest.approx(base * 0.5**2)


def test_weight_fn_shared_site_without_lane_reuse(arch):
    pf = PathFinder(arch)
    lane = _first_lane(pf)
    src, dst = pf.get_endpoints(lane)
    assert src is not None and dst is not None
    weight = _make_weight_fn(pf, {}, {src}, _WeightCtx(0.5, 1.1))
    base = pf.metrics.get_lane_duration_cost(lane)
    assert weight(lane) == base * 1.1


def test_weight_fn_direction_and_shared_site_stack(arch):
    """Direction factor and shared-site factor are orthogonal signals
    and compose multiplicatively. A lane that is both same-direction
    reused AND has an endpoint in committed_sites pays both.
    """
    pf = PathFinder(arch)
    lane = _first_lane(pf)
    src, _dst = pf.get_endpoints(lane)
    assert src is not None
    committed_lanes = {_lane_key(lane): _counts((lane.direction, 1))}
    weight = _make_weight_fn(pf, committed_lanes, {src}, _WeightCtx(0.5, 1.1))
    base = pf.metrics.get_lane_duration_cost(lane)
    assert weight(lane) == pytest.approx(base * 0.5 * 1.1)


def test_weight_fn_balanced_traffic_still_applies_shared_site(arch):
    """Balanced direction traffic (N == M) zeroes only the direction
    exponent. A coincident shared-site crossing still contributes.
    """
    pf = PathFinder(arch)
    lane = _first_lane(pf)
    src, _ = pf.get_endpoints(lane)
    assert src is not None
    committed_lanes = {
        _lane_key(lane): _counts(
            (lane.direction, 1),
            (lane.reverse().direction, 1),
        )
    }
    weight = _make_weight_fn(pf, committed_lanes, {src}, _WeightCtx(0.5, 1.1))
    base = pf.metrics.get_lane_duration_cost(lane)
    assert weight(lane) == pytest.approx(base * 1.1)


def _ctx(
    arch: ArchSpec,
    layout_tup: tuple[LocationAddress, ...],
    controls: tuple[int, ...],
    targets: tuple[int, ...],
) -> TargetContext:
    state = ConcreteState(
        occupied=frozenset(),
        layout=layout_tup,
        move_count=(0,) * len(layout_tup),
    )
    return TargetContext(
        arch_spec=arch,
        state=state,
        controls=controls,
        targets=targets,
        lookahead_cz_layers=(),
        cz_stage_index=0,
    )


def test_generate_empty_stage_returns_current_placement(arch):
    loc0, loc1 = _pick_cz_pair(arch)
    ctx = _ctx(arch, (loc0, loc1), controls=(), targets=())
    out = CongestionAwareTargetGenerator().generate(ctx)
    assert out == [{0: loc0, 1: loc1}]


def test_generate_already_partnered_pair_is_noop(arch):
    loc0, loc1 = _pick_cz_pair(arch)
    ctx = _ctx(arch, (loc0, loc1), controls=(0,), targets=(1,))
    out = CongestionAwareTargetGenerator().generate(ctx)
    assert out == [{0: loc0, 1: loc1}]


def test_sort_longest_first_orders_by_descending_uncongested_min_cost(gate_arch):
    """Longest-first orders pairs by descending minimal move cost.

    The pair whose cheapest CZ move (control- or target-direction) is the
    most expensive must be committed first. This requires two *non*-
    partnered pairs: an already-partnered placement has a zero-length
    move in both directions, so its score would be ``0`` and the sort
    could never distinguish it.
    """
    arch = gate_arch(num_cols=8)
    # qids 0/1 are the "short" pair (single-hop CZ move: w0<->w2, w3<->w1);
    # qids 2/3 the "long" pair (two-hop move: w4->w7, w6->w5). Distinct hop
    # counts guarantee distinct move costs, and the two pairs occupy
    # disjoint word regions so their paths do not interfere.
    layout = (
        LocationAddress(0, 0),
        LocationAddress(3, 0),
        LocationAddress(4, 0),
        LocationAddress(6, 0),
    )
    ctx = _ctx(arch, layout, controls=(0, 2), targets=(1, 3))
    pf = PathFinder(arch)

    # Fixture precondition, stated structurally (lane/hop count) rather than
    # by re-deriving the generator's own min-cost score: the long pair's CZ
    # move must span strictly more lanes than the short pair's, so it has the
    # larger uncongested cost.
    def move_hop_count(ctrl_loc: LocationAddress, tgt_loc: LocationAddress) -> int:
        partner = arch.get_cz_partner(tgt_loc)
        assert partner is not None
        path = pf.find_path(ctrl_loc, partner)
        assert path is not None
        return len(path[0])

    short_hops = move_hop_count(LocationAddress(0, 0), LocationAddress(3, 0))
    long_hops = move_hop_count(LocationAddress(4, 0), LocationAddress(6, 0))
    assert long_hops > short_hops, (
        f"fixture prerequisite: the long pair must span more lanes than the "
        f"short pair (short={short_hops}, long={long_hops})"
    )

    sorted_pairs = CongestionAwareTargetGenerator()._sort_pairs_longest_first(ctx, pf)
    assert sorted_pairs[0] == (
        2,
        3,
    ), f"longest-first expected (2,3) first, got {sorted_pairs}"


def test_target_direction_chosen_when_target_path_cheaper(arch_builder):
    """When moving the target is cheaper than moving the control, the
    target-direction candidate wins.

    Synthetic arch: the control move (``w0 -> w3``) has a cheap two-hop
    route through hub ``w4`` and an expensive detour ``w0-w5-w6-w3``. The
    target move (``w2 -> w1``) is a fixed two-hop route through ``w7`` that
    the hub does not touch. With the hub free the control route is
    cheapest, so the control moves; once a blocker occupies the hub, the
    control atom is forced onto the detour, which costs more than the
    target route, so the target-direction wins.
    """
    #   w0 --w4-- w3        (control short route via hub w4)
    #   |          |
    #   w5 ------- w6       (control detour)
    #   CZ pairs (w0,w1) and (w2,w3); target route w2 - w7 - w1
    positions = [
        (0.0, 0.0),  # 0: w0  control start (X)
        (30.0, 200.0),  # 1: w1  partner(X)
        (0.0, 200.0),  # 2: w2  target start (Y)
        (20.0, 0.0),  # 3: w3  partner(Y)
        (10.0, 0.0),  # 4: w4  hub (blocker site)
        (0.0, 100.0),  # 5: w5  detour waypoint
        (20.0, 100.0),  # 6: w6  detour waypoint
        (15.0, 200.0),  # 7: w7  target-route midpoint
    ]
    buses = [(0, 4), (4, 3), (0, 5), (5, 6), (6, 3), (2, 7), (7, 1)]
    entangling = [(0, 1), (2, 3)]
    arch = arch_builder(positions, buses, entangling)

    x = LocationAddress(0, 0, 0)
    y = LocationAddress(2, 0, 0)
    hub = LocationAddress(4, 0, 0)
    partner_x = arch.get_cz_partner(x)  # w1
    partner_y = arch.get_cz_partner(y)  # w3
    gen = CongestionAwareTargetGenerator()

    # Hub free: the control route (via the hub) is cheapest -> control moves.
    ctx_free = _ctx(arch, (x, y), controls=(0,), targets=(1,))
    plan_free = gen.generate(ctx_free)[0]
    assert plan_free[0] == partner_y and plan_free[1] == y, (
        f"with the hub free the control should move to {partner_y}; " f"got {plan_free}"
    )

    # Blocker on the hub forces the control detour; the target route is
    # now cheaper -> target moves, control stays put.
    ctx_blocked = _ctx(arch, (x, y, hub), controls=(0,), targets=(1,))
    plan_blocked = gen.generate(ctx_blocked)[0]
    assert plan_blocked[0] == x
    assert plan_blocked[1] == partner_x, (
        f"with the hub blocked the target should move to {partner_x}; "
        f"got {plan_blocked}"
    )
    assert plan_blocked[2] == hub


def test_neutral_factors_reproduce_default_on_symmetric_stage(gate_arch):
    """With ``direction_factor = shared_site_factor = 1.0`` (neutral
    multipliers) and a single-pair symmetric stage, the congestion-aware
    heuristic reduces to the default (control-moves) by symmetry +
    tiebreak. Sanity check the reduction.
    """
    arch = gate_arch(num_cols=8)
    # A single non-partnered pair whose control- and target-direction
    # moves are symmetric (equal cost and length), so the tiebreak in
    # ``_choose_control`` prefers control -- matching DefaultTargetGenerator.
    layout = (LocationAddress(0, 0), LocationAddress(2, 0))
    ctx = _ctx(arch, layout, controls=(0,), targets=(1,))
    out_neutral = CongestionAwareTargetGenerator(1.0, 1.0).generate(ctx)
    out_default = DefaultTargetGenerator().generate(ctx)
    assert out_neutral == out_default, (
        f"neutral factors (1.0) should match DefaultTargetGenerator "
        f"on symmetric stages.\nneutral: {out_neutral}\ndefault: {out_default}"
    )


def test_multi_pair_avoids_opposite_direction_reuse(arch_builder):
    """A small ``direction_factor`` makes the second-committed pair avoid
    reusing a lane in the opposite direction.

    Bridge arch: lane ``L = (w1, w2)`` is the only connection between the
    left cluster (``w0``, ``w5``, ...) and the right cluster (``w3``,
    ``w4``, ...). Pair A (the longer pair, committed first) crosses ``L``
    forward (``w0 -> w1 -> w2 -> w3``). Pair B's cheaper control direction
    would cross ``L`` backward (``w4 -> w2 -> w1 -> w5``); its target
    direction (``w8 -> w10 -> w11 -> w9``) avoids ``L`` but costs slightly
    more.

    With ``direction_factor = 1`` (neutral) B takes the cheaper
    L-crossing control direction. With the default ``direction_factor =
    0.5`` the opposite-direction penalty on ``L`` makes B's control
    direction more expensive than its target direction, so B flips to the
    L-avoiding target direction.
    """
    positions = [
        (0.0, 70.0),  # 0  ctrl A (left)
        (0.0, 0.0),  # 1  L0  (bridge left end)
        (60.0, 0.0),  # 2  R0  (bridge right end); L = (w1, w2)
        (60.0, 70.0),  # 3  partner(tgtA); A-ctrl destination (right)
        (60.0, -60.0),  # 4  ctrl B (right)
        (0.0, -60.0),  # 5  partner(tgtB); B-ctrl destination (left)
        (0.0, 140.0),  # 6  tgt A (partner of w3)
        (60.0, 140.0),  # 7  partner(ctrlA); A-tgt dest (isolated)
        (0.0, -140.0),  # 8  tgt B (partner of w5)
        (195.0, -140.0),  # 9  partner(ctrlB); B-tgt destination
        (65.0, -140.0),  # 10 B-tgt route waypoint
        (130.0, -140.0),  # 11 B-tgt route waypoint
    ]
    buses = [
        (0, 1),
        (1, 2),
        (2, 3),
        (2, 4),
        (1, 5),
        (8, 10),
        (10, 11),
        (11, 9),
    ]
    entangling = [(0, 7), (3, 6), (5, 8), (4, 9)]
    arch = arch_builder(positions, buses, entangling)

    layout = (
        LocationAddress(0, 0, 0),  # q0 ctrl A
        LocationAddress(6, 0, 0),  # q1 tgt A
        LocationAddress(4, 0, 0),  # q2 ctrl B
        LocationAddress(8, 0, 0),  # q3 tgt B
    )
    ctx = _ctx(arch, layout, controls=(0, 2), targets=(1, 3))

    def b_direction(direction_factor: float) -> str:
        plan = CongestionAwareTargetGenerator(
            direction_factor=direction_factor
        ).generate(ctx)[0]
        # Pair B is (ctrl=q2 @ w4, tgt=q3 @ w8). The control direction moves
        # q2 to partner(w8)=w5 (crossing L); the target direction moves q3
        # to partner(w4)=w9 (avoiding L).
        ctrl_side = plan[2] == LocationAddress(5, 0, 0) and plan[3] == LocationAddress(
            8, 0, 0
        )
        tgt_side = plan[3] == LocationAddress(9, 0, 0) and plan[2] == LocationAddress(
            4, 0, 0
        )
        if ctrl_side:
            return "ctrl"
        if tgt_side:
            return "tgt"
        raise AssertionError(f"unexpected plan for pair B: {plan}")

    assert (
        b_direction(1.0) == "ctrl"
    ), "neutral direction_factor: B should reuse lane L (control direction)"
    assert (
        b_direction(0.5) == "tgt"
    ), "small direction_factor: B should avoid opposite-direction reuse of L"


def test_direction_factor_zero_or_negative_raises():
    """``direction_factor <= 0`` breaks the opposite-direction branch
    (``0 ** -M`` is undefined; negative bases with non-integer exponents
    leave Dijkstra's invariants)."""
    with pytest.raises(ValueError, match="must be strictly positive"):
        CongestionAwareTargetGenerator(direction_factor=0.0)
    with pytest.raises(ValueError, match="must be strictly positive"):
        CongestionAwareTargetGenerator(direction_factor=-0.1)


def test_shared_site_factor_negative_raises():
    with pytest.raises(ValueError, match="must be non-negative"):
        CongestionAwareTargetGenerator(shared_site_factor=-0.1)


def test_generate_is_deterministic_across_calls(arch):
    loc0, loc1 = _pick_cz_pair(arch)
    ctx = _ctx(arch, (loc0, loc1), controls=(0,), targets=(1,))
    gen = CongestionAwareTargetGenerator()
    assert gen.generate(ctx) == gen.generate(ctx)


def test_both_directions_infeasible_returns_empty_list(gate_arch):
    """When neither CZ-move direction is feasible, generate() returns []."""
    arch = gate_arch(num_cols=4)
    ctrl = LocationAddress(0, 0, 0)
    tgt = LocationAddress(2, 0, 0)
    # Occupy both partner destinations so neither the control move
    # (ctrl -> partner(tgt)) nor the target move (tgt -> partner(ctrl))
    # has a free endpoint to route to.
    blocker_ctrl = arch.get_cz_partner(tgt)
    blocker_tgt = arch.get_cz_partner(ctrl)
    ctx = _ctx(
        arch,
        (ctrl, tgt, blocker_ctrl, blocker_tgt),
        controls=(0,),
        targets=(1,),
    )
    assert CongestionAwareTargetGenerator().generate(ctx) == []
