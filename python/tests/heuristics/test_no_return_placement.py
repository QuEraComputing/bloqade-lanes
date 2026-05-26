from __future__ import annotations

from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.bytecode._native import SearchStrategy
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


def _make_unaligned_state() -> ConcreteState:
    """A state where q0 and q1 sit on non-CZ-partner words (word 0's
    partner is word 1; word 3's partner is word 2), so forming a CZ pair
    requires at least one atom move and one solver node expansion."""
    return ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(3, 0),
        ),
        move_count=(0, 0),
    )


def test_no_return_default_construction():
    strategy = NoReturnPlacementStrategy(arch_spec=logical.get_arch_spec())
    assert strategy.strategy == SearchStrategy.IDS
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


def test_no_return_chains_two_cz_layers_without_returning_home():
    """The defining property of no-return placement is that layer N's
    output layout flows directly into layer N+1 with no palindrome return.

    Setup: q0 at (0,0), q1 at (3,0) — non-CZ-partner words, layer 1 must
    move at least one atom. Layer 2 repeats the same CZ; given the
    chained input, atoms are already at a valid CZ pair and no further
    movement is required. The total accumulated move_count therefore
    reflects ONLY layer 1's forward moves (with no return-to-home moves
    between layers, which would otherwise double the count).
    """
    strategy = NoReturnPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        max_expansions=5000,
        restarts=2,
    )
    state = _make_unaligned_state()

    layer1 = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(layer1, ExecuteCZ)
    layer1_moves = sum(layer1.move_count)
    assert layer1_moves >= 1, "layer 1 must require at least one forward move"

    # Chain: layer 1's output becomes layer 2's input.
    layer2 = strategy.cz_placements(layer1, controls=(0,), targets=(1,))
    assert isinstance(layer2, ExecuteCZ)
    layer2_moves = sum(layer2.move_count)

    # No-return guarantee: layer 2 inherits layer 1's CZ-aligned layout
    # so no further movement is needed; the accumulated counter equals
    # layer 1's. A palindrome-style strategy would have moved atoms back
    # home between layers and counted those moves too.
    assert layer2_moves == layer1_moves
    assert layer2.layout == layer1.layout


def test_no_return_exposes_rust_nodes_expanded():
    """The shared `rust_nodes_expanded_total` counter accumulates
    `SolveResult.nodes_expanded` per ``cz_placements`` call.

    Uses an unaligned initial state so the Rust solver must expand at
    least one node — guarantees the strict ``>`` assertion is meaningful
    (a previous version of this test used a trivially-valid state and
    fell back to ``>=``, which was unfalsifiable for ``nodes_expanded == 0``).
    """
    strategy = NoReturnPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        max_expansions=2000,
        restarts=1,
    )
    state = _make_unaligned_state()
    before = strategy.rust_nodes_expanded_total
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    assert strategy.rust_nodes_expanded_total > before
