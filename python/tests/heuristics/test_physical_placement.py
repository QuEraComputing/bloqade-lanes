from __future__ import annotations

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import AtomState, ConcreteState, ExecuteCZ
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.heuristics.physical_placement import PhysicalPlacementStrategy
from bloqade.lanes.search.traversal.goal import SearchResult


def _make_state() -> ConcreteState:
    return ConcreteState(
        occupied=frozenset(),
        layout=(
            layout.LocationAddress(0, 0),
            layout.LocationAddress(1, 0),
        ),
        move_count=(0, 0),
    )


def test_default_traversal_is_entropy():
    strategy = PhysicalPlacementStrategy()
    assert strategy.traversal == "entropy"


def test_traversal_selection_calls_selected_backend(monkeypatch):
    strategy = PhysicalPlacementStrategy(arch_spec=logical.get_arch_spec())
    state = _make_state()
    calls: list[str] = []

    def fake_entropy(**kwargs):
        calls.append("entropy")
        return SearchResult(
            goal_node=kwargs["tree"].root, nodes_expanded=0, max_depth_reached=0
        )

    def fake_greedy(**kwargs):
        calls.append("greedy")
        return SearchResult(
            goal_node=kwargs["tree"].root, nodes_expanded=0, max_depth_reached=0
        )

    def fake_bfs(**kwargs):
        calls.append("bfs")
        return SearchResult(
            goal_node=kwargs["tree"].root, nodes_expanded=0, max_depth_reached=0
        )

    monkeypatch.setattr(
        "bloqade.lanes.heuristics.physical_movement.entropy_guided_search", fake_entropy
    )
    monkeypatch.setattr(
        "bloqade.lanes.heuristics.physical_movement.greedy_best_first", fake_greedy
    )
    monkeypatch.setattr("bloqade.lanes.heuristics.physical_movement.bfs", fake_bfs)

    for traversal in ("entropy", "greedy", "bfs"):
        strategy.traversal = traversal
        _ = strategy.cz_placements(state, controls=(0,), targets=(1,))

    assert calls == ["entropy", "greedy", "bfs"]


def test_cz_placements_returns_bottom_when_search_fails(monkeypatch):
    strategy = PhysicalPlacementStrategy(arch_spec=logical.get_arch_spec())
    state = _make_state()

    def fake_run_search(self, tree, target, callback):
        _ = self, tree, target, callback
        return SearchResult(goal_node=None, nodes_expanded=1, max_depth_reached=0)

    monkeypatch.setattr(PhysicalPlacementStrategy, "_run_search", fake_run_search)
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert out == AtomState.bottom()


def test_cz_placements_populates_move_layers_from_goal_node(monkeypatch):
    strategy = PhysicalPlacementStrategy(arch_spec=logical.get_arch_spec())
    state = _make_state()

    def fake_run_search(self, tree, target, callback):
        _ = self, target, callback
        lane = next(tree.valid_lanes(tree.root))
        goal = tree.apply_move_set(tree.root, frozenset([lane]), strict=False)
        assert goal is not None
        return SearchResult(
            goal_node=goal, nodes_expanded=1, max_depth_reached=goal.depth
        )

    monkeypatch.setattr(PhysicalPlacementStrategy, "_run_search", fake_run_search)
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    assert len(out.move_layers) >= 1
