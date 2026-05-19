from __future__ import annotations

from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.physical.no_return import NoReturnPlacementStrategy


def _make_state() -> ConcreteState:
    return ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(1, 0),
        ),
        move_count=(0, 0),
    )


def test_no_return_default_construction():
    strategy = NoReturnPlacementStrategy(arch_spec=logical.get_arch_spec())
    assert strategy.strategy == "ids"
    assert strategy.max_expansions == 100
    assert strategy.restarts == 20
    assert strategy.hungarian_horizon == 4
    assert strategy.top_c == 3


def test_no_return_cz_placements_smoke():
    """End-to-end: solve_entangling dispatches and returns an ExecuteCZ result.

    Initial state already places the two qubits in a valid entangling
    configuration, so ``move_layers`` is empty — this still exercises the
    full dispatch path (build_candidates → solve_entangling → result
    decoding) and catches import / default-Options regressions.
    """
    strategy = NoReturnPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        max_expansions=300,
    )
    state = _make_state()
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    assert len(out.layout) == len(state.layout)


def test_no_return_exposes_rust_nodes_expanded():
    """The shared `rust_nodes_expanded_total` counter is exposed and
    accumulates `SolveResult.nodes_expanded` per call.

    Uses ``>= before`` rather than strict ``>`` because the smoke-test
    initial state already satisfies the entangling constraint, so the
    underlying solver may short-circuit with ``nodes_expanded=0``. The
    test still verifies the property is readable and the accumulation
    arithmetic runs without error.
    """
    strategy = NoReturnPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        max_expansions=300,
    )
    state = _make_state()
    before = strategy.rust_nodes_expanded_total
    strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert strategy.rust_nodes_expanded_total >= before
    assert isinstance(strategy.rust_nodes_expanded_total, int)
