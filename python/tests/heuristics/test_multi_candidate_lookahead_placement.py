from __future__ import annotations

from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.physical.multi_candidate_lookahead import (
    MultiCandidateLookaheadPlacementStrategy,
)
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
    """q0 at word 0, q1 at word 3: not a CZ partner pair (the partner of
    word 0 is word 1). Forming a CZ requires at least one atom move.
    """
    return ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(3, 0),
        ),
        move_count=(0, 0),
    )


def test_default_construction():
    strategy = MultiCandidateLookaheadPlacementStrategy(
        arch_spec=logical.get_arch_spec()
    )
    assert strategy.beam_width == 8
    assert strategy.hungarian_candidates == 20
    assert strategy.lookahead_window == 24
    assert strategy.lookahead_weight == 2.0
    assert strategy.nr_max_expansions == 300
    assert strategy.nr_restarts == 20


def test_cz_placements_smoke():
    """End-to-end smoke: a single trivially-valid CZ placement returns
    an :class:`ExecuteCZ` with the original layout and no moves.
    """
    strategy = MultiCandidateLookaheadPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        beam_width=2,
        hungarian_candidates=4,
        lookahead_window=0,
    )
    state = _make_state()
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    assert len(out.layout) == len(state.layout)


def test_subsumes_no_return_on_single_layer():
    """Safety-net invariant: A.lanes <= NR.lanes on the same circuit.

    Uses an unaligned initial state so NR must produce a non-trivial
    move plan. The beam includes NR as an unconditional candidate, so
    its committed first-layer cost cannot exceed NR's.
    """
    arch = logical.get_arch_spec()
    state = _make_unaligned_state()

    nr = NoReturnPlacementStrategy(arch_spec=arch, max_expansions=2000, restarts=2)
    beam = MultiCandidateLookaheadPlacementStrategy(
        arch_spec=arch,
        beam_width=4,
        hungarian_candidates=8,
        lookahead_window=0,
        nr_max_expansions=2000,
        nr_restarts=2,
    )

    nr_out = nr.cz_placements(state, controls=(0,), targets=(1,))
    beam_out = beam.cz_placements(state, controls=(0,), targets=(1,))

    assert isinstance(nr_out, ExecuteCZ)
    assert isinstance(beam_out, ExecuteCZ)

    nr_lanes = sum(len(layer) for layer in nr_out.move_layers)
    beam_lanes = sum(len(layer) for layer in beam_out.move_layers)
    assert (
        beam_lanes <= nr_lanes
    ), f"safety-net invariant violated: beam={beam_lanes} > nr={nr_lanes}"


def test_chains_two_cz_layers_without_returning_home():
    """Two consecutive CZ layers on the same pair: layer 2's input is
    layer 1's output, and no extra movement is required because the
    layout is already CZ-aligned.

    Mirrors ``test_no_return_chains_two_cz_layers_without_returning_home``
    for the NR strategy.
    """
    strategy = MultiCandidateLookaheadPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        beam_width=2,
        hungarian_candidates=4,
        lookahead_window=0,
        nr_max_expansions=5000,
        nr_restarts=2,
    )
    state = _make_unaligned_state()

    layer1 = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(layer1, ExecuteCZ)
    layer1_moves = sum(layer1.move_count)
    assert layer1_moves >= 1

    layer2 = strategy.cz_placements(layer1, controls=(0,), targets=(1,))
    assert isinstance(layer2, ExecuteCZ)
    layer2_moves = sum(layer2.move_count)
    # No-return: layer 2 inherits layer 1's aligned layout.
    assert layer2_moves == layer1_moves
    assert layer2.layout == layer1.layout


def test_lookahead_window_zero_matches_single_stage_decision():
    """``lookahead_window=0`` means only the current stage is scored;
    confirm this still succeeds and matches NR on the trivially-aligned
    case (where both pick a zero-move layout).
    """
    arch = logical.get_arch_spec()
    state = _make_state()  # trivially-valid

    beam = MultiCandidateLookaheadPlacementStrategy(
        arch_spec=arch,
        beam_width=2,
        hungarian_candidates=2,
        lookahead_window=0,
    )
    out = beam.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    assert sum(len(layer) for layer in out.move_layers) == 0


def test_sq_placements_passthrough():
    strategy = MultiCandidateLookaheadPlacementStrategy(
        arch_spec=logical.get_arch_spec()
    )
    state = _make_state()
    out = strategy.sq_placements(state, qubits=(0, 1))
    assert isinstance(out, ConcreteState)
    assert out.layout == state.layout
    assert out.move_count == state.move_count


def test_measure_placements_emits_zone_maps():
    from bloqade.lanes.analysis.placement import ExecuteMeasure

    strategy = MultiCandidateLookaheadPlacementStrategy(
        arch_spec=logical.get_arch_spec()
    )
    state = _make_state()
    out = strategy.measure_placements(state, qubits=(0, 1))
    assert isinstance(out, ExecuteMeasure)
    assert len(out.zone_maps) == len(state.layout)


def test_validate_initial_layout_noop():
    strategy = MultiCandidateLookaheadPlacementStrategy(
        arch_spec=logical.get_arch_spec()
    )
    state = _make_state()
    # Must not raise on any plausible initial layout.
    strategy.validate_initial_layout(state.layout)
