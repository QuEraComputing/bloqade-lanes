"""Tests for CandidateScorer."""

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import LocationAddress
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
        1: LocationAddress(1, 0),
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
        LocationAddress(0, 0), LocationAddress(0, 5), tree
    )
    assert dist > 0.0


def test_mobility_at_position():
    scorer, tree = _make_scorer_and_tree()
    occupied = tree.root.occupied_locations
    mob = scorer._mobility_at(LocationAddress(0, 0), occupied, tree)
    assert isinstance(mob, int)
    assert mob >= 0


def test_score_all_qubit_bus_pairs_returns_dict():
    target = {0: LocationAddress(0, 5), 1: LocationAddress(1, 5)}
    scorer, tree = _make_scorer_and_tree(target=target)
    scores = scorer.score_all_qubit_bus_pairs(tree.root, entropy=1, tree=tree)
    assert isinstance(scores, dict)
    assert len(scores) > 0
    for key, val in scores.items():
        assert len(key) == 4  # (qubit_id, move_type, bus_id, direction)
        assert isinstance(val, float)


def test_score_all_qubit_bus_pairs_skips_resolved_qubits():
    target = {0: LocationAddress(0, 0), 1: LocationAddress(1, 5)}
    scorer, tree = _make_scorer_and_tree(target=target)
    scores = scorer.score_all_qubit_bus_pairs(tree.root, entropy=1, tree=tree)
    qubit_ids_in_scores = {k[0] for k in scores}
    assert 0 not in qubit_ids_in_scores


def test_score_all_entropy_shifts_weights():
    target = {0: LocationAddress(0, 5), 1: LocationAddress(1, 5)}
    scorer, tree = _make_scorer_and_tree(target=target)
    scores_e1 = scorer.score_all_qubit_bus_pairs(tree.root, entropy=1, tree=tree)
    scores_e3 = scorer.score_all_qubit_bus_pairs(tree.root, entropy=3, tree=tree)
    common_keys = set(scores_e1) & set(scores_e3)
    assert len(common_keys) > 0
    assert any(scores_e1[k] != scores_e3[k] for k in common_keys)


def test_score_moveset_positive_for_good_move():
    target = {0: LocationAddress(0, 5), 1: LocationAddress(1, 0)}
    scorer, tree = _make_scorer_and_tree(target=target)
    from bloqade.lanes.layout import SiteLaneAddress

    lane = SiteLaneAddress(0, 0, 0)
    moveset = frozenset({lane})
    score = scorer.score_moveset(moveset, tree.root, tree)
    assert score > 0.0


def test_score_moveset_arrived_gain():
    target = {0: LocationAddress(0, 5)}
    scorer, tree = _make_scorer_and_tree(target=target)
    from bloqade.lanes.layout import SiteLaneAddress

    lane = SiteLaneAddress(0, 0, 0)
    moveset = frozenset({lane})
    score_with_arrival = scorer.score_moveset(moveset, tree.root, tree)
    scorer_no_arrival = CandidateScorer(
        params=SearchParams(), target={0: LocationAddress(0, 9)}
    )
    score_no_arrival = scorer_no_arrival.score_moveset(moveset, tree.root, tree)
    assert score_with_arrival > score_no_arrival


def test_score_all_qubit_bus_pairs_excludes_blocked_destinations():
    target = {0: LocationAddress(0, 5)}
    scorer, base_tree = _make_scorer_and_tree(target=target)
    blocked = frozenset({LocationAddress(0, 5)})
    tree = ConfigurationTree(
        arch_spec=base_tree.arch_spec,
        root=base_tree.root,
        blocked_locations=blocked,
    )
    scores = scorer.score_all_qubit_bus_pairs(tree.root, entropy=1, tree=tree)
    for qid, mt, bus_id, direction in scores:
        loc = tree.root.configuration[qid]
        lane = tree.lane_for_source(mt, bus_id, direction, loc)
        assert lane is not None
        _, dst = tree.arch_spec.get_endpoints(lane)
        assert dst not in blocked
