from __future__ import annotations

import pytest

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import AtomState, ConcreteState, ExecuteCZ
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.heuristics.physical.placement import (
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
        "bloqade.lanes.heuristics.physical.movement.EntropyPlacementTraversal.path_to_target_config",
        fake_entropy,
    )
    monkeypatch.setattr(
        "bloqade.lanes.heuristics.physical.movement.GreedyPlacementTraversal.path_to_target_config",
        fake_greedy,
    )
    monkeypatch.setattr(
        "bloqade.lanes.heuristics.physical.movement.BFSPlacementTraversal.path_to_target_config",
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
    assert t.max_movesets_per_group == 3
    assert t.max_expansions == 300


def test_rust_traversal_dispatches_to_rust_path(monkeypatch):
    strategy = PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(), traversal=RustPlacementTraversal()
    )
    state = _make_state()
    calls: list[str] = []

    def fake_cz_placements_rust(self, state, controls, targets, lookahead_cz_layers=()):
        _ = self, state, controls, targets, lookahead_cz_layers
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
    from bloqade.lanes.bytecode import Direction as BytecodeDirection, MoveType
    from bloqade.lanes.layout.encoding import LaneAddress

    strategy = PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(), traversal=RustPlacementTraversal()
    )
    state = _make_state()

    from bloqade.lanes.bytecode._native import (
        LaneAddress as NativeLane,
        LocationAddress as NativeLoc,
    )

    class _FakeResult:
        status = "solved"
        nodes_expanded = 1
        # move_layers: list[list[LaneAddress]] — MoveType.ZONE variant
        move_layers = [
            [NativeLane(MoveType.ZONE, 0, 0, 0, 0, BytecodeDirection.FORWARD)]
        ]
        goal_config = {0: NativeLoc(0, 0, 0), 1: NativeLoc(0, 1, 0)}

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


# ---------------------------------------------------------------------------
# target_generator plugin-path tests (shared-budget candidate loop)
# ---------------------------------------------------------------------------


def test_target_generator_none_matches_today_behavior(monkeypatch):
    """Regression guard: None plugin path is functionally identical to today."""
    strategy = PhysicalPlacementStrategy(arch_spec=logical.get_arch_spec())
    state = _make_state()
    seen_targets: list[dict] = []

    def fake_run_search(self, tree, target, traversal=None):
        _ = self, tree, traversal
        seen_targets.append(dict(target))
        return SearchResult(goal_node=tree.root, nodes_expanded=0, max_depth_reached=0)

    monkeypatch.setattr(PhysicalPlacementStrategy, "_run_search", fake_run_search)
    strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert len(seen_targets) == 1


def test_target_generator_empty_plugin_behaves_like_none(monkeypatch):
    strategy = PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        target_generator=lambda ctx: [],
    )
    state = _make_state()
    count = 0

    def fake_run_search(self, tree, target, traversal=None):
        nonlocal count
        count += 1
        _ = self, target, traversal
        return SearchResult(goal_node=tree.root, nodes_expanded=0, max_depth_reached=0)

    monkeypatch.setattr(PhysicalPlacementStrategy, "_run_search", fake_run_search)
    strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert count == 1  # only the default candidate runs


def test_target_generator_cheaper_candidate_wins(monkeypatch):
    """Plugin returns a candidate that solves on first attempt; default never runs."""
    arch_spec = logical.get_arch_spec()
    state = _make_state()
    default_target = {
        0: arch_spec.get_cz_partner(state.layout[1]),
        1: state.layout[1],
    }
    # Alt candidate swaps the roles: target moves to control's partner.
    alt_target = {
        0: state.layout[0],
        1: arch_spec.get_cz_partner(state.layout[0]),
    }
    strategy = PhysicalPlacementStrategy(
        arch_spec=arch_spec,
        target_generator=lambda ctx: [alt_target],
    )
    targets_tried: list[dict] = []

    def fake_run_search(self, tree, target, traversal=None):
        _ = self, traversal
        targets_tried.append(dict(target))
        # First attempt succeeds
        return SearchResult(goal_node=tree.root, nodes_expanded=0, max_depth_reached=0)

    monkeypatch.setattr(PhysicalPlacementStrategy, "_run_search", fake_run_search)
    strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert targets_tried == [alt_target]  # default not tried
    _ = default_target


def test_target_generator_shared_budget(monkeypatch):
    """Sum of per-candidate nodes_expanded cannot exceed configured max."""
    arch_spec = logical.get_arch_spec()
    # Use a state where alt and default targets are distinct: with layout
    # (loc0, loc2), default = {0: partner(loc2)=loc3, 1: loc2}, while the
    # alt below = {0: loc0, 1: partner(loc0)=loc1}.
    state = ConcreteState(
        occupied=frozenset(),
        layout=(
            layout.LocationAddress(0, 0),
            layout.LocationAddress(2, 0),
        ),
        move_count=(0, 0),
    )
    alt_target = {
        0: state.layout[0],
        1: arch_spec.get_cz_partner(state.layout[0]),
    }
    strategy = PhysicalPlacementStrategy(
        arch_spec=arch_spec,
        traversal=EntropyPlacementTraversal(max_expansions=10),
        target_generator=lambda ctx: [alt_target],
    )
    budgets_seen: list[int | None] = []
    consumed_per_call = 4

    def fake_path_to_target_config(self, **kwargs):
        _ = kwargs
        budgets_seen.append(self.max_expansions)
        return SearchResult(
            goal_node=None,
            nodes_expanded=consumed_per_call,
            max_depth_reached=0,
        )

    monkeypatch.setattr(
        EntropyPlacementTraversal,
        "path_to_target_config",
        fake_path_to_target_config,
    )
    strategy.cz_placements(state, controls=(0,), targets=(1,))
    # First call uses full 10; second call uses 10 - 4 = 6.
    assert budgets_seen == [10, 6]


def test_target_generator_raises_propagates():
    def boom(ctx):
        raise RuntimeError("plugin exploded")

    strategy = PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        target_generator=boom,
    )
    state = _make_state()
    with pytest.raises(RuntimeError, match="plugin exploded"):
        strategy.cz_placements(state, controls=(0,), targets=(1,))


def test_rust_path_target_generator_shared_budget(monkeypatch):
    arch_spec = logical.get_arch_spec()
    # Use state where default and alt are distinct (layout[0]=(0,0), layout[1]=(2,0))
    state = ConcreteState(
        occupied=frozenset(),
        layout=(
            layout.LocationAddress(0, 0),
            layout.LocationAddress(2, 0),
        ),
        move_count=(0, 0),
    )
    # A plausible alt — swap control's destination
    alt_target = {
        0: state.layout[0],
        1: arch_spec.get_cz_partner(state.layout[0]),
    }
    strategy = PhysicalPlacementStrategy(
        arch_spec=arch_spec,
        traversal=RustPlacementTraversal(max_expansions=10),
        target_generator=lambda ctx: [alt_target],
    )
    budgets_seen: list[int | None] = []
    consumed = 4

    class _FakeResult:
        def __init__(self):
            self.status = "unsolvable"
            self.nodes_expanded = consumed

    class _FakeSolver:
        def solve(self, *args, **kwargs):
            _ = args
            budgets_seen.append(kwargs.get("max_expansions"))
            return _FakeResult()

    monkeypatch.setattr(
        PhysicalPlacementStrategy,
        "_get_rust_solver",
        lambda _self: _FakeSolver(),
    )
    strategy.cz_placements(state, controls=(0,), targets=(1,))
    # alt candidate first with full 10; default candidate second with 6.
    assert budgets_seen == [10, 6]


def test_rust_path_cz_counter_increments():
    """Parity fix: _cz_counter must increment on the Rust path too."""
    strategy = PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        traversal=RustPlacementTraversal(),
    )
    state = _make_state()
    strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert strategy._cz_counter == 1


# ---------------------------------------------------------------------------
# CongestionAwareTargetGenerator integration tests
# ---------------------------------------------------------------------------


def _pick_cz_pair_integration(arch):
    for s in arch.home_sites:
        p = arch.get_cz_partner(s)
        if p is not None and p != s:
            return s, p
    raise AssertionError("arch has no CZ-partnered home site")


def test_cz_placements_with_congestion_aware_generator_produces_execute_cz():
    """Smoke test: PhysicalPlacementStrategy wired with
    CongestionAwareTargetGenerator completes cz_placements and returns
    an ExecuteCZ result for a simple stage.
    """
    from bloqade.lanes.arch.gemini.physical import get_arch_spec
    from bloqade.lanes.heuristics.physical import (
        CongestionAwareTargetGenerator,
    )

    arch = get_arch_spec()
    loc0, loc1 = _pick_cz_pair_integration(arch)

    strategy = PhysicalPlacementStrategy(
        arch_spec=arch,
        target_generator=CongestionAwareTargetGenerator(),
    )
    state = ConcreteState(occupied=frozenset(), layout=(loc0, loc1), move_count=(0, 0))
    result = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(
        result, ExecuteCZ
    ), f"expected ExecuteCZ, got {type(result).__name__}"


def test_congestion_aware_dedups_with_default_on_symmetric_stage():
    """For a stage where the congestion-aware candidate equals what
    DefaultTargetGenerator would produce, the framework's dedup drops
    the default and search runs exactly once.

    Rust path: assert on strategy.rust_nodes_expanded_total.
    Python path: instrument by counting calls to
                 traversal.path_to_target_config.

    NOTE: when the pair is already partnered (loc0, loc1) = (s, partner(s)),
    CongestionAwareTargetGenerator produces the same target config as the
    default generator.  The framework deduplicates the candidate list before
    the loop, so only one call to path_to_target_config is made.

    If the strategy detects a trivially-solved stage and returns an ExecuteCZ
    without any search (call_count == 0), that is also acceptable — it means
    zero paths were explored, which is at most one.  The assertion is therefore
    call_count <= 1.
    """
    from unittest.mock import patch

    from bloqade.lanes.analysis.placement import ConcreteState
    from bloqade.lanes.arch.gemini.physical import get_arch_spec
    from bloqade.lanes.heuristics.physical import (
        CongestionAwareTargetGenerator,
        EntropyPlacementTraversal,
        PhysicalPlacementStrategy,
    )

    arch = get_arch_spec()
    # Single-pair stage: both generators produce the same candidate.
    loc0, loc1 = _pick_cz_pair_integration(arch)

    strategy = PhysicalPlacementStrategy(
        arch_spec=arch,
        traversal=EntropyPlacementTraversal(),
        target_generator=CongestionAwareTargetGenerator(),
    )
    state = ConcreteState(occupied=frozenset(), layout=(loc0, loc1), move_count=(0, 0))

    call_count = {"n": 0}
    real = EntropyPlacementTraversal.path_to_target_config

    def counting(self, *args, **kwargs):
        call_count["n"] += 1
        return real(self, *args, **kwargs)

    with patch.object(EntropyPlacementTraversal, "path_to_target_config", counting):
        _ = strategy.cz_placements(state, controls=(0,), targets=(1,))

    assert (
        call_count["n"] <= 1
    ), f"expected at most 1 search call after dedup, got {call_count['n']}"


def test_congestion_aware_applies_to_rust_traversal():
    """Plugin applies identically under RustPlacementTraversal; both
    traversals complete with ExecuteCZ for a simple stage.
    """
    from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ
    from bloqade.lanes.arch.gemini.physical import get_arch_spec
    from bloqade.lanes.heuristics.physical import (
        CongestionAwareTargetGenerator,
        EntropyPlacementTraversal,
        PhysicalPlacementStrategy,
        RustPlacementTraversal,
    )

    arch = get_arch_spec()
    loc0, loc1 = _pick_cz_pair_integration(arch)

    for traversal in (
        EntropyPlacementTraversal(),
        RustPlacementTraversal(),
    ):
        strategy = PhysicalPlacementStrategy(
            arch_spec=arch,
            traversal=traversal,
            target_generator=CongestionAwareTargetGenerator(),
        )
        state = ConcreteState(
            occupied=frozenset(), layout=(loc0, loc1), move_count=(0, 0)
        )
        result = strategy.cz_placements(state, controls=(0,), targets=(1,))
        assert isinstance(
            result, ExecuteCZ
        ), f"{type(traversal).__name__} did not produce ExecuteCZ"
