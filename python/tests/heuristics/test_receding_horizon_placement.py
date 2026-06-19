from __future__ import annotations

from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ
from bloqade.lanes.arch.gemini import logical
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
