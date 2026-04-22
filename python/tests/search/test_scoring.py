"""Tests for CandidateScorer."""

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import Direction, LocationAddress, MoveType
from bloqade.lanes.search.scoring import CandidateScorer
from bloqade.lanes.search.search_params import SearchParams
from bloqade.lanes.search.tree import ConfigurationTree


def _make_scorer_and_tree(
    target: dict[int, LocationAddress] | None = None,
):
    """Create a scorer and tree with the logical Gemini arch spec."""
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0),
        1: LocationAddress(4, 0),
    }
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    if target is None:
        target = {}
    params = SearchParams()
    scorer = CandidateScorer(params=params, target=target)
    return scorer, tree


def test_distance_to_target_zero_when_at_target():
    scorer, tree = _make_scorer_and_tree()
    dist = scorer._distance_to_target(
        LocationAddress(0, 0), LocationAddress(0, 0), tree
    )
    assert dist == 0.0


def test_distance_to_target_positive_when_not_at_target():
    scorer, tree = _make_scorer_and_tree()
    dist = scorer._distance_to_target(
        LocationAddress(0, 0), LocationAddress(1, 0), tree
    )
    assert dist > 0.0


def test_mobility_at_position():
    scorer, tree = _make_scorer_and_tree()
    occupied = tree.root.occupied_locations
    mob = scorer._mobility_at(
        LocationAddress(0, 0),
        LocationAddress(7, 0),
        occupied,
        tree,
    )
    assert isinstance(mob, float)
    assert mob >= 0


def test_mobility_at_weights_legal_lanes_by_post_lane_distance():
    # adjusted in Phase 1 distance-table switch: _mobility_at now calls the
    # DistanceTable directly instead of delegating to _distance_to_target, so
    # monkeypatching is no longer appropriate. Expected value is computed from
    # the real DistanceTable to verify the 1/(1+d) weighting logic is intact.
    scorer, tree = _make_scorer_and_tree()
    position = LocationAddress(0, 0)
    target = LocationAddress(7, 0)
    occupied = tree.root.occupied_locations

    legal_dsts: list[LocationAddress] = []
    for lane in tree.outgoing_lanes(position):
        _, dst = tree.arch_spec.get_endpoints(lane)
        if dst not in occupied:
            legal_dsts.append(dst)

    assert len(legal_dsts) >= 2

    mobility = scorer._mobility_at(position, target, occupied, tree)
    # Compute expected directly from the real distance table (same path the
    # implementation now uses) to verify the 1/(1+d) weighting formula.
    expected = 0.0
    for dst in legal_dsts:
        d = scorer._distance_to_target(dst, target, tree)
        if d != float("inf"):
            expected += 1.0 / (1.0 + d)

    import pytest as _pytest

    assert mobility == _pytest.approx(expected, abs=1e-12)


def test_score_all_qubit_bus_pairs_returns_dict():
    target = {0: LocationAddress(1, 0), 1: LocationAddress(5, 0)}
    scorer, tree = _make_scorer_and_tree(target=target)
    scores = scorer.score_all_qubit_bus_pairs(tree.root, entropy=1, tree=tree)
    assert isinstance(scores, dict)
    assert len(scores) > 0
    for key, val in scores.items():
        assert len(key) == 4  # (qubit_id, move_type, bus_id, direction)
        assert isinstance(val, float)


def test_score_all_qubit_bus_pairs_skips_resolved_qubits():
    target = {0: LocationAddress(0, 0), 1: LocationAddress(5, 0)}
    scorer, tree = _make_scorer_and_tree(target=target)
    scores = scorer.score_all_qubit_bus_pairs(tree.root, entropy=1, tree=tree)
    qubit_ids_in_scores = {k[0] for k in scores}
    assert 0 not in qubit_ids_in_scores


def test_score_all_entropy_shifts_weights():
    target = {0: LocationAddress(1, 0), 1: LocationAddress(5, 0)}
    scorer, tree = _make_scorer_and_tree(target=target)
    scores_e1 = scorer.score_all_qubit_bus_pairs(tree.root, entropy=1, tree=tree)
    scores_e3 = scorer.score_all_qubit_bus_pairs(tree.root, entropy=3, tree=tree)
    common_keys = set(scores_e1) & set(scores_e3)
    assert len(common_keys) > 0
    assert any(scores_e1[k] != scores_e3[k] for k in common_keys)


def test_score_moveset_positive_for_good_move():
    target = {0: LocationAddress(1, 0), 1: LocationAddress(4, 0)}
    scorer, tree = _make_scorer_and_tree(target=target)
    from bloqade.lanes.layout import WordLaneAddress

    lane = WordLaneAddress(0, 0, 0)
    moveset = frozenset({lane})
    score = scorer.score_moveset(moveset, tree.root, tree)
    assert score > 0.0


def test_score_moveset_arrived_gain():
    target = {0: LocationAddress(1, 0)}
    scorer, tree = _make_scorer_and_tree(target=target)
    from bloqade.lanes.layout import WordLaneAddress

    lane = WordLaneAddress(0, 0, 0)
    moveset = frozenset({lane})
    score_with_arrival = scorer.score_moveset(moveset, tree.root, tree)
    scorer_no_arrival = CandidateScorer(
        params=SearchParams(), target={0: LocationAddress(7, 0)}
    )
    score_no_arrival = scorer_no_arrival.score_moveset(moveset, tree.root, tree)
    assert score_with_arrival > score_no_arrival


def test_score_all_qubit_bus_pairs_excludes_blocked_destinations():
    target = {0: LocationAddress(1, 0)}
    scorer, base_tree = _make_scorer_and_tree(target=target)
    blocked = frozenset({LocationAddress(1, 0)})
    tree = ConfigurationTree(
        arch_spec=base_tree.arch_spec,
        root=base_tree.root,
        blocked_locations=blocked,
    )
    scores = scorer.score_all_qubit_bus_pairs(tree.root, entropy=1, tree=tree)
    for qid, mt, bus_id, direction in scores:
        loc = tree.root.configuration[qid]
        lane = next(
            (
                la
                for la in tree.outgoing_lanes(loc)
                if la.move_type == mt
                and la.bus_id == bus_id
                and la.direction == direction
            ),
            None,
        )
        assert lane is not None
        _, dst = tree.arch_spec.get_endpoints(lane)
        assert dst not in blocked


def test_score_rectangle_bus_candidates_marks_non_movers_invalid():
    target = {0: LocationAddress(5, 0)}
    scorer, tree = _make_scorer_and_tree(target=target)
    buckets = scorer.score_rectangle_bus_candidates(tree.root, entropy=1, tree=tree)
    assert buckets

    non_mover_src = tree.root.configuration[1]
    assert any(non_mover_src in bucket.invalid_sources for bucket in buckets.values())


def test_score_rectangle_bus_candidates_marks_collision_sources_invalid():
    target = {0: LocationAddress(5, 0)}
    scorer, base_tree = _make_scorer_and_tree(target=target)
    blocked = frozenset({LocationAddress(5, 0)})
    tree = ConfigurationTree(
        arch_spec=base_tree.arch_spec,
        root=base_tree.root,
        blocked_locations=blocked,
    )
    buckets = scorer.score_rectangle_bus_candidates(tree.root, entropy=1, tree=tree)

    mover_src = tree.root.configuration[0]
    assert any(mover_src in bucket.invalid_sources for bucket in buckets.values())


def test_score_rectangle_bus_candidates_negative_scores_become_invalid(monkeypatch):
    target = {0: LocationAddress(0, 5)}
    scorer, tree = _make_scorer_and_tree(target=target)
    mover_loc = tree.root.configuration[0]

    chosen_key: tuple[int, MoveType, int, Direction] | None = None
    for mt in (MoveType.SITE, MoveType.WORD):
        buses = (
            range(len(tree.arch_spec.site_buses))
            if mt == MoveType.SITE
            else range(len(tree.arch_spec.word_buses))
        )
        for bus_id in buses:
            for direction in (Direction.FORWARD, Direction.BACKWARD):
                lane = tree.lane_for_source(mt, bus_id, direction, mover_loc)
                if lane is not None:
                    chosen_key = (0, mt, bus_id, direction)
                    break
            if chosen_key is not None:
                break
        if chosen_key is not None:
            break

    assert chosen_key is not None

    monkeypatch.setattr(
        scorer,
        "score_all_qubit_bus_pairs",
        lambda _node, _entropy, _tree: {chosen_key: -1.0},
    )
    buckets = scorer.score_rectangle_bus_candidates(tree.root, entropy=1, tree=tree)
    assert any(mover_loc in bucket.invalid_sources for bucket in buckets.values())
