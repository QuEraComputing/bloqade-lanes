from __future__ import annotations

from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ
from bloqade.lanes.arch.gemini import logical
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


def test_nohome_default_construction():
    strategy = NoHomePlacementStrategy(arch_spec=logical.get_arch_spec())
    assert strategy.strategy == "ids"
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
