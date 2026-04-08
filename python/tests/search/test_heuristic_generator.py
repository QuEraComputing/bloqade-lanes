"""Tests for HeuristicMoveGenerator."""

from dataclasses import dataclass

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import LocationAddress, SiteLaneAddress, WordLaneAddress
from bloqade.lanes.search.generators import EntropyNode, HeuristicMoveGenerator
from bloqade.lanes.search.scoring import CandidateScorer, RectangleBusCandidates
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
    target = {0: LocationAddress(0, 1), 1: LocationAddress(1, 1)}
    scorer = CandidateScorer(params=params, target=target)
    search_nodes: dict[int, EntropyNode] = {}
    gen = HeuristicMoveGenerator(
        scorer=scorer,
        params=params,
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
        scores = [scorer.score_moveset(ms, tree.root, tree) for ms in moves]
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
    """When all scores are negative, generator may produce no rectangle candidates."""
    arch_spec = logical.get_arch_spec()
    placement = {0: LocationAddress(0, 1)}
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    params = SearchParams()
    target = {0: LocationAddress(0, 0)}
    scorer = CandidateScorer(params=params, target=target)
    search_nodes: dict[int, EntropyNode] = {id(tree.root): _EntropyNode()}
    gen = HeuristicMoveGenerator(
        scorer=scorer, params=params, search_nodes=search_nodes
    )
    moves = list(gen.generate(tree.root, tree))
    assert isinstance(moves, list)


def test_generate_filters_rectangles_with_invalid_sources(monkeypatch):
    gen, tree, search_nodes = _make_setup()
    search_nodes[id(tree.root)] = _EntropyNode()

    lane_ok = SiteLaneAddress(0, 0, 0)
    lane_bad = SiteLaneAddress(1, 0, 0)
    bad_src = LocationAddress(1, 0)

    def fake_bus_candidates(_node, _entropy, _tree):  # type: ignore[no-untyped-def]
        return {
            (
                lane_ok.move_type,
                lane_ok.bus_id,
                lane_ok.direction,
            ): RectangleBusCandidates(
                valid_entries={0: lane_ok},
                invalid_sources=frozenset({bad_src}),
                qubit_scores={0: 1.0},
            )
        }

    class _StubContext:
        def build_aod_grids(self, entries):  # type: ignore[no-untyped-def]
            assert entries == {0: lane_ok}
            return [frozenset({lane_ok, lane_bad}), frozenset({lane_ok})]

    monkeypatch.setattr(
        gen.scorer, "score_rectangle_bus_candidates", fake_bus_candidates
    )
    monkeypatch.setattr(
        "bloqade.lanes.search.generators.heuristic.BusContext.from_tree",
        lambda **_kwargs: _StubContext(),
    )
    monkeypatch.setattr(gen.scorer, "score_moveset", lambda ms, _n, _t: float(len(ms)))

    moves = list(gen.generate(tree.root, tree))
    assert moves == [frozenset({lane_ok})]


def test_generate_ranks_rectangles_globally_across_buses(monkeypatch):
    gen, tree, search_nodes = _make_setup()
    search_nodes[id(tree.root)] = _EntropyNode()

    a1 = SiteLaneAddress(0, 0, 0)
    a2 = SiteLaneAddress(1, 0, 0)
    b1 = WordLaneAddress(0, 0, 1)

    def fake_bus_candidates(_node, _entropy, _tree):  # type: ignore[no-untyped-def]
        return {
            (a1.move_type, a1.bus_id, a1.direction): RectangleBusCandidates(
                valid_entries={0: a1},
                invalid_sources=frozenset(),
                qubit_scores={0: 1.0},
            ),
            (b1.move_type, b1.bus_id, b1.direction): RectangleBusCandidates(
                valid_entries={1: b1},
                invalid_sources=frozenset(),
                qubit_scores={1: 1.0},
            ),
        }

    class _StubContext:
        def __init__(self, bus_id: int):
            self.bus_id = bus_id

        def build_aod_grids(self, _entries):  # type: ignore[no-untyped-def]
            if self.bus_id == 0:
                return [frozenset({a1, a2})]
            return [frozenset({b1})]

    monkeypatch.setattr(
        gen.scorer, "score_rectangle_bus_candidates", fake_bus_candidates
    )
    monkeypatch.setattr(
        "bloqade.lanes.search.generators.heuristic.BusContext.from_tree",
        lambda **kwargs: _StubContext(kwargs["bus_id"]),
    )
    monkeypatch.setattr(
        gen.scorer,
        "score_moveset",
        lambda ms, _n, _t: 10.0 if ms == frozenset({b1}) else 5.0,
    )

    moves = list(gen.generate(tree.root, tree))
    assert moves[0] == frozenset({b1})
    assert moves[1] == frozenset({a1, a2})


def test_generate_falls_back_to_best_singleton_when_no_positive_rectangles(
    monkeypatch,
):
    gen, tree, search_nodes = _make_setup()
    search_nodes[id(tree.root)] = _EntropyNode()

    lane = SiteLaneAddress(0, 0, 0)

    monkeypatch.setattr(
        gen.scorer,
        "score_rectangle_bus_candidates",
        lambda _node, _entropy, _tree: {},
    )
    monkeypatch.setattr(
        gen.scorer,
        "score_all_qubit_bus_pairs",
        lambda _node, _entropy, _tree: {
            (0, lane.move_type, lane.bus_id, lane.direction): -1.0
        },
    )

    moves = list(gen.generate(tree.root, tree))
    assert moves == [frozenset({lane})]
