from __future__ import annotations

from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.physical.loose_goal import LooseGoalPlacementStrategy


def _make_state() -> ConcreteState:
    return ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(1, 0),
        ),
        move_count=(0, 0),
    )


def test_loose_goal_default_construction():
    strategy = LooseGoalPlacementStrategy(arch_spec=logical.get_arch_spec())
    assert strategy.strategy == "ids"
    assert strategy.max_expansions == 100
    assert strategy.restarts == 20
    assert strategy.hungarian_horizon == 4
    assert strategy.top_c == 3


def test_loose_goal_cz_placements_smoke():
    """End-to-end: solve_entangling dispatches and returns an ExecuteCZ result.

    Initial state already places the two qubits in a valid entangling
    configuration, so ``move_layers`` is empty — this still exercises the
    full dispatch path (build_candidates → solve_entangling → result
    decoding) and catches import / default-Options regressions.
    """
    strategy = LooseGoalPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        max_expansions=300,
    )
    state = _make_state()
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    assert len(out.layout) == len(state.layout)
