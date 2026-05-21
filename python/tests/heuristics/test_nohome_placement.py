from __future__ import annotations

from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.bytecode._native import SearchStrategy
from bloqade.lanes.bytecode.encoding import LocationAddress
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


def test_nohome_cz_placements_smoke():
    """End-to-end: solve_nohome dispatches and returns an ExecuteCZ result.

    Exercises the two-phase return + entangling dispatch path through
    ``MoveSolver.solve_nohome``. Move-count is not asserted because the
    initial state may already be in a valid entangling configuration.
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
