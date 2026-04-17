from __future__ import annotations

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import AtomState, ConcreteState, ExecuteCZ
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.heuristics.physical_placement import (
    BFSPlacementTraversal,
    EntropyPlacementTraversal,
    GreedyPlacementTraversal,
    PhysicalPlacementStrategy,
    RustPlacementTraversal,
)
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
    assert isinstance(strategy.traversal, EntropyPlacementTraversal)


def test_rejects_string_traversal():
    try:
        PhysicalPlacementStrategy(traversal="entropy")  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        assert False, "expected TypeError for non-object traversal"


def test_traversal_selection_calls_selected_backend(monkeypatch):
    strategy = PhysicalPlacementStrategy(arch_spec=logical.get_arch_spec())
    state = _make_state()
    calls: list[str] = []

    def fake_entropy(self, **kwargs):
        _ = self
        calls.append("entropy")
        return SearchResult(
            goal_node=kwargs["tree"].root, nodes_expanded=0, max_depth_reached=0
        )

    def fake_greedy(self, **kwargs):
        _ = self
        calls.append("greedy")
        return SearchResult(
            goal_node=kwargs["tree"].root, nodes_expanded=0, max_depth_reached=0
        )

    def fake_bfs(self, **kwargs):
        _ = self
        calls.append("bfs")
        return SearchResult(
            goal_node=kwargs["tree"].root, nodes_expanded=0, max_depth_reached=0
        )

    monkeypatch.setattr(
        "bloqade.lanes.heuristics.physical_movement.EntropyPlacementTraversal.path_to_target_config",
        fake_entropy,
    )
    monkeypatch.setattr(
        "bloqade.lanes.heuristics.physical_movement.GreedyPlacementTraversal.path_to_target_config",
        fake_greedy,
    )
    monkeypatch.setattr(
        "bloqade.lanes.heuristics.physical_movement.BFSPlacementTraversal.path_to_target_config",
        fake_bfs,
    )

    for traversal in (
        EntropyPlacementTraversal(),
        GreedyPlacementTraversal(),
        BFSPlacementTraversal(),
    ):
        strategy.traversal = traversal
        _ = strategy.cz_placements(state, controls=(0,), targets=(1,))

    assert calls == ["entropy", "greedy", "bfs"]


def test_cz_placements_returns_bottom_when_search_fails(monkeypatch):
    strategy = PhysicalPlacementStrategy(arch_spec=logical.get_arch_spec())
    state = _make_state()

    def fake_run_search(self, tree, target, traversal=None):
        _ = self, tree, target, traversal
        return SearchResult(goal_node=None, nodes_expanded=1, max_depth_reached=0)

    monkeypatch.setattr(PhysicalPlacementStrategy, "_run_search", fake_run_search)
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert out == AtomState.bottom()


def test_cz_placements_populates_move_layers_from_goal_node(monkeypatch):
    strategy = PhysicalPlacementStrategy(arch_spec=logical.get_arch_spec())
    state = _make_state()

    def fake_run_search(self, tree, target, traversal=None):
        _ = self, target, traversal
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


def test_cz_placements_passes_idle_occupied_as_blockers(monkeypatch):
    strategy = PhysicalPlacementStrategy(arch_spec=logical.get_arch_spec())
    state = ConcreteState(
        occupied=frozenset({layout.LocationAddress(0, 5)}),
        layout=(
            layout.LocationAddress(0, 0),
            layout.LocationAddress(1, 0),
        ),
        move_count=(0, 0),
    )
    seen_blocked_locations: list[frozenset[layout.LocationAddress]] = []

    def fake_run_search(self, tree, target, traversal=None):
        _ = self, target, traversal
        seen_blocked_locations.append(tree.blocked_locations)
        return SearchResult(
            goal_node=tree.root, nodes_expanded=0, max_depth_reached=tree.root.depth
        )

    monkeypatch.setattr(PhysicalPlacementStrategy, "_run_search", fake_run_search)
    _ = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert seen_blocked_locations == [state.occupied]


# ---------------------------------------------------------------------------
# Rust MoveSolver integration tests
# ---------------------------------------------------------------------------


def test_rust_traversal_default_params():
    # TODO: add assertions for weight, restarts, lookahead,
    # deadlock_policy, w_t once they are threaded through to RustPlacementTraversal.
    t = RustPlacementTraversal()
    assert t.strategy == "astar"
    assert t.top_c == 3
    assert t.max_movesets_per_group == 3
    assert t.max_expansions == 300


def test_rust_traversal_dispatches_to_rust_path(monkeypatch):
    strategy = PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(), traversal=RustPlacementTraversal()
    )
    state = _make_state()
    calls: list[str] = []

    def fake_cz_placements_rust(self, state, controls, targets):
        _ = self, state, controls, targets
        calls.append("rust")
        return AtomState.bottom()

    monkeypatch.setattr(
        PhysicalPlacementStrategy, "_cz_placements_rust", fake_cz_placements_rust
    )
    strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert calls == ["rust"]


def test_rust_solver_is_cached():
    strategy = PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(), traversal=RustPlacementTraversal()
    )
    solver_a = strategy._get_rust_solver()
    solver_b = strategy._get_rust_solver()
    assert solver_a is solver_b


def test_cz_placements_rust_returns_execute_cz():
    strategy = PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(), traversal=RustPlacementTraversal()
    )
    state = _make_state()
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    assert len(out.layout) == len(state.layout)


def test_cz_placements_rust_returns_bottom_on_failure(monkeypatch):
    strategy = PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(), traversal=RustPlacementTraversal()
    )
    state = _make_state()

    class _FakeResult:
        status = "unsolvable"
        nodes_expanded = 0

    class _FakeSolver:
        def solve(self, *_args, **_kwargs):
            return _FakeResult()

    monkeypatch.setattr(
        PhysicalPlacementStrategy,
        "_get_rust_solver",
        lambda _self: _FakeSolver(),
    )
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert out == AtomState.bottom()


def test_cz_placements_rust_with_blocked_locations():
    strategy = PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(), traversal=RustPlacementTraversal()
    )
    state = ConcreteState(
        occupied=frozenset({layout.LocationAddress(0, 5)}),
        layout=(
            layout.LocationAddress(0, 0),
            layout.LocationAddress(1, 0),
        ),
        move_count=(0, 0),
    )
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)


def test_cz_placements_rust_handles_zone_move_type(monkeypatch):
    """Regression test for #510: _MT_MAP must include MoveType.ZONE (2)."""
    from bloqade.lanes.bytecode import Direction as BytecodeDirection
    from bloqade.lanes.bytecode import MoveType
    from bloqade.lanes.layout.encoding import LaneAddress

    strategy = PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(), traversal=RustPlacementTraversal()
    )
    state = _make_state()

    class _FakeResult:
        status = "solved"
        # move_layers format: list[list[tuple[dir, move_type, zone, word, site, bus]]]
        # move_type=2 is MoveType.ZONE — the variant that was missing from _MT_MAP
        move_layers = [[(0, 2, 0, 0, 0, 0)]]
        goal_config = [(0, 0, 0, 0), (1, 0, 1, 0)]

    class _FakeSolver:
        def solve(self, *_args, **_kwargs):
            return _FakeResult()

    monkeypatch.setattr(
        PhysicalPlacementStrategy,
        "_get_rust_solver",
        lambda _self: _FakeSolver(),
    )
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    assert len(out.move_layers) == 1
    lane = out.move_layers[0][0]
    assert isinstance(lane, LaneAddress)
    assert lane.move_type == MoveType.ZONE
    assert lane.direction == BytecodeDirection.FORWARD
