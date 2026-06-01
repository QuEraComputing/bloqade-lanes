"""Targeted coverage for MultiCandidateLookaheadPlacementStrategy paths
that the smoke/adversarial suites don't exercise: the multi-stage
lookahead scorer, the bystander-collision relocation in the Hungarian
assignment, the oracle move-layer cache, and the early-return guards.
"""

from __future__ import annotations

from bloqade.lanes.analysis.placement import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
)
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.physical.multi_candidate_lookahead import (
    MultiCandidateLookaheadPlacementStrategy,
    _entangling_pair_slots,
    _hungarian_assignment,
    _hungarian_candidates,
)


def _arch():
    return logical.get_arch_spec()


def _strategy(
    *,
    beam_width: int = 4,
    hungarian_candidates: int = 6,
    lookahead_window: int = 4,
) -> MultiCandidateLookaheadPlacementStrategy:
    return MultiCandidateLookaheadPlacementStrategy(
        arch_spec=_arch(),
        beam_width=beam_width,
        hungarian_candidates=hungarian_candidates,
        lookahead_window=lookahead_window,
        nr_max_expansions=2000,
        nr_restarts=2,
    )


# --- Hungarian assignment edge cases ------------------------------------


def test_hungarian_assignment_rejects_more_pairs_than_columns():
    """n_pairs > 2 * n_slot_pairs has no valid assignment -> None."""
    arch = _arch()
    slots = _entangling_pair_slots(arch)[:1]  # only 1 slot pair -> 2 columns
    layout = tuple(LocationAddress(i, 0) for i in range(6))
    pairs = [(0, 1), (2, 3), (4, 5)]  # 3 pairs > 2 columns
    assert _hungarian_assignment(pairs, layout, slots, perturbation=None) is None


def test_hungarian_candidates_empty_pairs_returns_layout():
    """No pairs -> the single trivial candidate is the layout itself."""
    arch = _arch()
    slots = _entangling_pair_slots(arch)
    layout = (LocationAddress(0, 0), LocationAddress(1, 0))
    cands = _hungarian_candidates([], layout, slots, n_candidates=4)
    assert cands == (layout,)


def test_hungarian_assignment_relocates_colliding_bystander():
    """A non-participating qubit sitting on a slot the pair is assigned to
    must be relocated to a free slot so the output has no duplicates.

    q0@w0, q1@w3 form the CZ; Hungarian lands them on slot (0,1), so q1
    moves onto w1 where bystander q2 sits. The relocation loop must move
    q2 to the nearest free slot.
    """
    arch = _arch()
    slots = _entangling_pair_slots(arch)
    layout = (
        LocationAddress(0, 0),
        LocationAddress(3, 0),
        LocationAddress(1, 0),  # bystander on slot-0 right
    )
    out = _hungarian_assignment([(0, 1)], layout, slots, perturbation=None)
    assert out is not None
    keys = [(loc.word_id, loc.site_id) for loc in out]
    assert len(set(keys)) == len(keys), f"duplicate locations: {keys}"
    # The CZ pair must still land on CZ-partner slots.
    partner = arch.get_cz_partner(out[0])
    assert partner == out[1]


# --- Strategy-level guards and multi-stage lookahead --------------------


def test_cz_placements_mismatched_controls_targets_returns_bottom():
    strategy = _strategy()
    state = ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(0, 0), LocationAddress(1, 0)),
        move_count=(0, 0),
    )
    out = strategy.cz_placements(state, controls=(0, 1), targets=(1,))
    assert out == AtomState.bottom()


def test_cz_placements_non_concrete_state_returns_top():
    strategy = _strategy()
    out = strategy.cz_placements(AtomState.top(), controls=(0,), targets=(1,))
    assert out == AtomState.top()


def test_multi_stage_lookahead_scores_future_layer():
    """Supplying lookahead_cz_layers with a *distinct* next stage drives
    the lookahead scorer (_lookahead_cost) and the skip-repeated-stages
    loop. A 4-qubit state with stage0 = CZ(0,1) and stage1 = CZ(2,3).
    """
    strategy = _strategy()
    state = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(3, 0),
            LocationAddress(5, 0),
            LocationAddress(6, 0),
        ),
        move_count=(0, 0, 0, 0),
    )
    lookahead = (((0,), (1,)), ((2,), (3,)))
    out = strategy.cz_placements(
        state, controls=(0,), targets=(1,), lookahead_cz_layers=lookahead
    )
    assert isinstance(out, ExecuteCZ)
    # q0 and q1 must end on CZ-partner slots for this committed layer.
    partner = strategy.arch_spec.get_cz_partner(out.layout[0])
    assert partner == out.layout[1]


def test_oracle_move_layers_cache_hit_on_repeated_transition():
    """The second call with an identical (from, to) layout transition
    returns the cached result rather than recomputing.
    """
    strategy = _strategy()
    layout_from = (
        LocationAddress(0, 0),
        LocationAddress(3, 0),
    )
    layout_to = (
        LocationAddress(0, 0),
        LocationAddress(1, 0),
    )
    first = strategy._oracle_move_layers(layout_from, layout_to)
    key = (
        tuple((loc.word_id, loc.site_id) for loc in layout_from),
        tuple((loc.word_id, loc.site_id) for loc in layout_to),
    )
    assert key in strategy._oracle_cache
    second = strategy._oracle_move_layers(layout_from, layout_to)
    assert first == second


def test_oracle_move_layers_rejects_duplicate_layout():
    """A layout with two qubits at the same location is rejected (None)
    without raising, and the rejection is cached.
    """
    strategy = _strategy()
    layout_from = (LocationAddress(0, 0), LocationAddress(0, 0))  # duplicate
    layout_to = (LocationAddress(0, 0), LocationAddress(1, 0))
    assert strategy._oracle_move_layers(layout_from, layout_to) is None
