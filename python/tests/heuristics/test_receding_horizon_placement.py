from __future__ import annotations

from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ
from bloqade.lanes.arch.gemini import logical, physical
from bloqade.lanes.bytecode._native import SearchStrategy
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.physical.receding_horizon import (
    RecedingHorizonNoReturnPlacementStrategy,
)


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


def test_receding_horizon_default_construction():
    strategy = RecedingHorizonNoReturnPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
    )
    assert strategy.strategy == SearchStrategy.IDS
    assert strategy.k_candidates == 5
    assert strategy.rollout_horizon == 5
    assert strategy.commit_depth == 3
    assert strategy.tier0_next_h_weight == 0.5
    assert strategy.restarts == 1
    assert strategy.max_expansions == 5000


def test_receding_horizon_cz_placements_smoke():
    """End-to-end: RecedingHorizonCzPlacement dispatches and returns ExecuteCZ.

    Initial state is already entangling-feasible (the two qubits sit on a
    valid CZ pair location), so the trajectory terminates immediately with
    no committed move layers — but the full dispatch path still runs.
    """
    strategy = RecedingHorizonNoReturnPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        max_expansions=300,
        k_candidates=3,
        rollout_horizon=3,
        commit_depth=1,
    )
    state = _make_state()
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    assert len(out.layout) == len(state.layout)


def test_receding_horizon_with_multiple_restarts():
    """With ``restarts=2``, the rayon wrapper engages and `pick_best`
    selects across two independent trajectories."""
    strategy = RecedingHorizonNoReturnPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        max_expansions=300,
        restarts=2,
        k_candidates=3,
        rollout_horizon=3,
        commit_depth=1,
        branch_parallel=False,  # leave cores for restart parallelism
    )
    state = _make_state()
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    assert len(out.layout) == len(state.layout)


def test_receding_horizon_exposes_rust_nodes_expanded():
    """The shared `rust_nodes_expanded_total` counter accumulates
    `SolveResult.nodes_expanded` per ``cz_placements`` call.

    Uses an unaligned initial state so the solver must expand at least
    one node — see the no-return placement variant for the rationale.
    """
    strategy = RecedingHorizonNoReturnPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        max_expansions=2000,
        k_candidates=3,
        rollout_horizon=3,
        commit_depth=1,
    )
    state = _make_unaligned_state()
    before = strategy.rust_nodes_expanded_total
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    assert strategy.rust_nodes_expanded_total > before


def test_receding_horizon_defaults_budget_allows_more_than_one_stage():
    """Default construction solves a layer that needs two stages.

    This is the fifth CZ layer of the ``adder_4`` physical benchmark. Its
    first stage costs 261 expansions (two of the five rollouts fall back from
    the beam to IDS) and commits three tier-1 layers without reaching the
    goal; the second stage finishes in six more. The budget is checked
    between stages, so the ``max_expansions=100`` the strategy used to inherit
    from ``NoReturnStrategyBase`` returned ``budget_exceeded`` just before
    that second stage, and the benchmark failed.
    """
    strategy = RecedingHorizonNoReturnPlacementStrategy(
        arch_spec=physical.get_arch_spec(),
    )
    state = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 3),
            LocationAddress(1, 3),
            LocationAddress(3, 2),
            LocationAddress(2, 2),
        ),
        move_count=(0, 0, 0, 0),
    )
    controls, targets = (3, 1), (0, 2)
    out = strategy.cz_placements(
        state,
        controls=controls,
        targets=targets,
        lookahead_cz_layers=((controls, targets), ((3, 1), (0, 2)), ((3,), (2,))),
    )
    assert isinstance(out, ExecuteCZ)
    assert strategy.rust_nodes_expanded_total > 100
