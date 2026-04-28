from __future__ import annotations

import pytest

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import AtomState, ConcreteState, ExecuteCZ
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.heuristics.physical.placement import (
    PhysicalPlacementStrategy,
    RustPlacementTraversal,
)


def _make_state() -> ConcreteState:
    return ConcreteState(
        occupied=frozenset(),
        layout=(
            layout.LocationAddress(0, 0),
            layout.LocationAddress(1, 0),
        ),
        move_count=(0, 0),
    )


def test_default_traversal_is_rust_entropy():
    strategy = PhysicalPlacementStrategy()
    assert isinstance(strategy.traversal, RustPlacementTraversal)
    assert strategy.traversal.strategy == "entropy"


def test_rejects_non_rust_traversal():
    with pytest.raises(TypeError):
        PhysicalPlacementStrategy(traversal="entropy")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Rust MoveSolver integration tests
# ---------------------------------------------------------------------------


def test_rust_traversal_default_params():
    # TODO: add assertions for weight, restarts, lookahead,
    # deadlock_policy, w_t once they are threaded through to RustPlacementTraversal.
    t = RustPlacementTraversal()
    assert t.strategy == "entropy"
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

    from bloqade.lanes.bytecode._native import LocationAddress as NativeLoc

    class _FakeResult:
        status = "solved"
        nodes_expanded = 1
        # move_layers format: list[list[tuple[dir, move_type, zone, word, site, bus]]]
        # move_type=2 is MoveType.ZONE — the variant that was missing from _MT_MAP
        move_layers = [[(0, 2, 0, 0, 0, 0)]]
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


def test_congestion_aware_applies_to_rust_traversal():
    """Plugin applies identically under RustPlacementTraversal; traversal
    completes with ExecuteCZ for a simple stage.
    """
    from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ
    from bloqade.lanes.arch.gemini.physical import get_arch_spec
    from bloqade.lanes.heuristics.physical import (
        CongestionAwareTargetGenerator,
        PhysicalPlacementStrategy,
        RustPlacementTraversal,
    )

    arch = get_arch_spec()
    loc0, loc1 = _pick_cz_pair_integration(arch)

    strategy = PhysicalPlacementStrategy(
        arch_spec=arch,
        traversal=RustPlacementTraversal(),
        target_generator=CongestionAwareTargetGenerator(),
    )
    state = ConcreteState(occupied=frozenset(), layout=(loc0, loc1), move_count=(0, 0))
    result = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(
        result, ExecuteCZ
    ), "RustPlacementTraversal did not produce ExecuteCZ"
