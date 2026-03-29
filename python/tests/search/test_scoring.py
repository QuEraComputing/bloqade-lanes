"""Tests for CandidateScorer."""

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import LocationAddress
from bloqade.lanes.search.scoring import CandidateScorer
from bloqade.lanes.search.search_params import SearchParams
from bloqade.lanes.search.tree import ConfigurationTree


def _make_scorer_and_tree():
    """Create a scorer and tree with the logical Gemini arch spec."""
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0),
        1: LocationAddress(1, 0),
    }
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    params = SearchParams()
    scorer = CandidateScorer(params=params)
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
    scorer, tree = _make_scorer_and_tree()
    target = {0: LocationAddress(0, 5), 1: LocationAddress(1, 5)}
    scores = scorer.score_all_qubit_bus_pairs(tree.root, target, entropy=1, tree=tree)
    assert isinstance(scores, dict)
    assert len(scores) > 0
    for key, val in scores.items():
        assert len(key) == 4  # (qubit_id, move_type, bus_id, direction)
        assert isinstance(val, float)


def test_score_all_qubit_bus_pairs_skips_resolved_qubits():
    scorer, tree = _make_scorer_and_tree()
    target = {0: LocationAddress(0, 0), 1: LocationAddress(1, 5)}
    scores = scorer.score_all_qubit_bus_pairs(tree.root, target, entropy=1, tree=tree)
    qubit_ids_in_scores = {k[0] for k in scores}
    assert 0 not in qubit_ids_in_scores


def test_score_all_entropy_shifts_weights():
    scorer, tree = _make_scorer_and_tree()
    target = {0: LocationAddress(0, 5), 1: LocationAddress(1, 5)}
    scores_e1 = scorer.score_all_qubit_bus_pairs(
        tree.root, target, entropy=1, tree=tree
    )
    scores_e3 = scorer.score_all_qubit_bus_pairs(
        tree.root, target, entropy=3, tree=tree
    )
    common_keys = set(scores_e1) & set(scores_e3)
    assert len(common_keys) > 0
    assert any(scores_e1[k] != scores_e3[k] for k in common_keys)


def test_score_moveset_positive_for_good_move():
    scorer, tree = _make_scorer_and_tree()
    target = {0: LocationAddress(0, 5), 1: LocationAddress(1, 0)}
    from bloqade.lanes.layout import SiteLaneAddress

    lane = SiteLaneAddress(0, 0, 0)
    moveset = frozenset({lane})
    score = scorer.score_moveset(moveset, tree.root, target, tree)
    assert score > 0.0


def test_score_moveset_arrived_gain():
    scorer, tree = _make_scorer_and_tree()
    target = {0: LocationAddress(0, 5)}
    from bloqade.lanes.layout import SiteLaneAddress

    lane = SiteLaneAddress(0, 0, 0)
    moveset = frozenset({lane})
    score_with_arrival = scorer.score_moveset(moveset, tree.root, target, tree)
    target2 = {0: LocationAddress(0, 9)}
    score_no_arrival = scorer.score_moveset(moveset, tree.root, target2, tree)
    assert score_with_arrival > score_no_arrival
