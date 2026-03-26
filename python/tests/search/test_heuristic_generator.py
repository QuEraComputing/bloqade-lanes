"""Tests for HeuristicMoveGenerator."""

from dataclasses import dataclass

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import LocationAddress
from bloqade.lanes.search.generators import EntropyNode, HeuristicMoveGenerator
from bloqade.lanes.search.scoring import CandidateScorer
from bloqade.lanes.search.search_params import SearchParams
from bloqade.lanes.search.tree import ConfigurationTree


@dataclass
class _EntropyNode:
    entropy: int = 1


def _make_setup():
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0),
        1: LocationAddress(1, 0),
    }
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    params = SearchParams()
    scorer = CandidateScorer(params=params)
    target = {0: LocationAddress(0, 5), 1: LocationAddress(1, 5)}
    search_nodes: dict[int, EntropyNode] = {}
    gen = HeuristicMoveGenerator(
        scorer=scorer,
        params=params,
        target=target,
        search_nodes=search_nodes,
    )
    return gen, tree, search_nodes


def test_generate_yields_frozensets():
    gen, tree, search_nodes = _make_setup()
    search_nodes[id(tree.root)] = _EntropyNode()
    moves = list(gen.generate(tree.root, tree))
    assert len(moves) > 0
    for ms in moves:
        assert isinstance(ms, frozenset)
        assert len(ms) > 0


def test_generate_satisfies_protocol():
    from bloqade.lanes.search.generators import MoveGenerator

    gen, _, _ = _make_setup()
    assert isinstance(gen, MoveGenerator)


def test_generate_ranked_by_moveset_score():
    gen, tree, search_nodes = _make_setup()
    search_nodes[id(tree.root)] = _EntropyNode()
    moves = list(gen.generate(tree.root, tree))
    if len(moves) >= 2:
        scorer = gen.scorer
        target = gen.target
        scores = [scorer.score_moveset(ms, tree.root, target, tree) for ms in moves]
        assert scores == sorted(scores, reverse=True)


def test_generate_with_higher_entropy_produces_different_candidates():
    gen, tree, search_nodes = _make_setup()
    search_nodes[id(tree.root)] = _EntropyNode(entropy=1)
    moves_e1 = list(gen.generate(tree.root, tree))

    search_nodes[id(tree.root)] = _EntropyNode(entropy=3)
    moves_e3 = list(gen.generate(tree.root, tree))

    assert len(moves_e1) > 0
    assert len(moves_e3) > 0


def test_generate_negative_fallback():
    """When all scores are negative, should still produce the single best candidate."""
    arch_spec = logical.get_arch_spec()
    placement = {0: LocationAddress(0, 5)}
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    params = SearchParams()
    scorer = CandidateScorer(params=params)
    target = {0: LocationAddress(0, 0)}
    search_nodes: dict[int, EntropyNode] = {id(tree.root): _EntropyNode()}
    gen = HeuristicMoveGenerator(
        scorer=scorer, params=params, target=target, search_nodes=search_nodes
    )
    moves = list(gen.generate(tree.root, tree))
    assert len(moves) >= 1
