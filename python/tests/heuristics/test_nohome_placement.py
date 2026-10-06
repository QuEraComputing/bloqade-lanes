from __future__ import annotations

import pytest

from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.bytecode import _native
from bloqade.lanes.bytecode._native import SearchStrategy
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.physical.movement import RustPlacementTraversal
from bloqade.lanes.heuristics.physical.nohome import NoHomePlacementStrategy


def _make_state() -> ConcreteState:
    return ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(1, 0),
        ),
        move_count=(0, 0),
    )


def _make_unaligned_state() -> ConcreteState:
    """A state where q0 and q1 sit on non-CZ-partner words, so forming a
    CZ pair requires at least one atom move and one solver node
    expansion."""
    return ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(3, 0),
        ),
        move_count=(0, 0),
    )


def test_nohome_default_construction():
    strategy = NoHomePlacementStrategy(arch_spec=logical.get_arch_spec())
    assert strategy.strategy == SearchStrategy.IDS
    assert strategy.max_expansions == 100
    assert strategy.restarts == 20
    assert strategy.gamma == 0.85
    assert strategy.lambda_lookahead == 0.5
    assert strategy.k_candidates == 8


def test_nohome_completion_bound_reaches_cz_search(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}
    original_entropy_options = _native.EntropyOptions

    def entropy_options(*args, **kwargs):
        seen.update(kwargs)
        return original_entropy_options(*args, **kwargs)

    monkeypatch.setattr(_native, "EntropyOptions", entropy_options)
    strategy = NoHomePlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        strategy=SearchStrategy.ENTROPY,
        completion_bound="weighted_distance",
    )
    strategy._build_move_search()
    assert seen["completion_bound"] == "weighted_distance"


def test_nohome_completion_bound_reaches_move_to_search(
    monkeypatch: pytest.MonkeyPatch,
):
    from bloqade.lanes.heuristics import move_synthesis

    seen: dict[str, object] = {}

    def compute_move_layers(arch_spec, state_before, state_after, *, engine, traversal):
        seen["traversal"] = traversal
        return ()

    monkeypatch.setattr(move_synthesis, "compute_move_layers", compute_move_layers)
    strategy = NoHomePlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        strategy=SearchStrategy.ENTROPY,
        completion_bound="weighted_distance",
    )
    state = _make_state()
    assert strategy.compute_moves(state, state) == ()
    traversal = seen["traversal"]
    assert isinstance(traversal, RustPlacementTraversal)
    assert traversal.completion_bound == "weighted_distance"


def test_nohome_cz_placements_with_completion_bound():
    strategy = NoHomePlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        strategy=SearchStrategy.ENTROPY,
        completion_bound="weighted_distance",
        max_expansions=2000,
        restarts=1,
    )
    state = _make_unaligned_state()
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)


def test_nohome_move_to_with_completion_bound():
    strategy = NoHomePlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        strategy=SearchStrategy.ENTROPY,
        completion_bound="weighted_distance",
    )
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(0, 0), LocationAddress(2, 0)),
        move_count=(0, 0),
    )
    state_after = ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(1, 0), LocationAddress(2, 0)),
        move_count=(1, 0),
    )
    layers = strategy.compute_moves(state_before, state_after)
    assert layers
    assert all(
        not strategy.arch_spec.check_lane_group([lane])
        for layer in layers
        for lane in layer
    )


def test_nohome_cz_placements_smoke():
    """End-to-end: NoHomeCzPlacement dispatches and returns an ExecuteCZ result.

    Exercises the two-phase return + entangling dispatch path. Move-count
    is not asserted because the initial state may already be in a valid
    entangling configuration.
    """
    strategy = NoHomePlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        max_expansions=300,
    )
    state = _make_state()
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    assert len(out.layout) == len(state.layout)


def test_nohome_exposes_rust_nodes_expanded():
    """The shared `rust_nodes_expanded_total` counter accumulates
    `SolveResult.nodes_expanded` per ``cz_placements`` call.

    Uses an unaligned initial state so the solver must expand at least
    one node — see the no-return placement variant for the rationale.
    """
    strategy = NoHomePlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        max_expansions=2000,
        restarts=1,
    )
    state = _make_unaligned_state()
    before = strategy.rust_nodes_expanded_total
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    assert strategy.rust_nodes_expanded_total > before


def test_mover_selection_defaults_to_ranked_and_validates():
    import pytest

    from bloqade.lanes.bytecode import MoverSelection, NoHomeOptions

    default = NoHomeOptions()
    assert default.mover_selection == MoverSelection.RANKED
    assert default.max_mover_candidates == 64
    assert "mover_selection=MoverSelection.RANKED" in repr(default)

    chosen = NoHomeOptions(
        mover_selection=MoverSelection.ROUTE_ALL, max_mover_candidates=8
    )
    assert chosen.mover_selection == MoverSelection.ROUTE_ALL
    assert chosen.max_mover_candidates == 8

    with pytest.raises(ValueError, match="max_mover_candidates"):
        NoHomeOptions(max_mover_candidates=0)


def test_mover_selection_reaches_the_native_options():
    """The strategy field, and the pipeline factory's knob, both reach the
    ``NoHomeOptions`` the Rust placement is built with; ``None`` keeps the
    native default."""
    from bloqade.lanes.analysis.placement import PalindromePlacementStrategy
    from bloqade.lanes.bytecode import MoverSelection
    from bloqade.lanes.heuristics.physical import make_physical_placement_strategy

    arch = logical.get_arch_spec()
    for selection in (MoverSelection.RULE, MoverSelection.ROUTE_ALL):
        strategy = NoHomePlacementStrategy(arch_spec=arch, mover_selection=selection)
        assert strategy._build_nohome_options().mover_selection == selection
    assert (
        NoHomePlacementStrategy(arch_spec=arch)._build_nohome_options().mover_selection
        == MoverSelection.RANKED
    )

    built = make_physical_placement_strategy(
        arch_spec=arch, mover_selection=MoverSelection.RULE
    )
    assert isinstance(built, PalindromePlacementStrategy)
    inner = built.inner
    assert isinstance(inner, NoHomePlacementStrategy)
    assert inner.mover_selection == MoverSelection.RULE
