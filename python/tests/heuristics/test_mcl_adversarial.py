"""Adversarial bug tests for MultiCandidateLookaheadPlacementStrategy.

These tests cover edge cases that aren't in the main test file:
- Validity check on output layouts (CZ pairs at CZ-partner positions)
- Empty lookahead window
- Multi-stage chaining preserves no-return property
- W=1 degenerates to greedy-with-NR-fallback
- All-zero lookahead_window
- Result type sanity
"""

from __future__ import annotations

from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.physical.multi_candidate_lookahead import (
    MultiCandidateLookaheadPlacementStrategy,
)
from bloqade.lanes.heuristics.physical.no_return import NoReturnPlacementStrategy


def _aligned() -> ConcreteState:
    return ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(0, 0), LocationAddress(1, 0)),
        move_count=(0, 0),
    )


def _unaligned() -> ConcreteState:
    return ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(0, 0), LocationAddress(3, 0)),
        move_count=(0, 0),
    )


def test_output_layout_satisfies_cz_pairs():
    """Every committed stage's output layout MUST place CZ pairs at
    CZ-partner positions. Otherwise the CZ gate cannot execute.
    """
    arch = logical.get_arch_spec()
    strategy = MultiCandidateLookaheadPlacementStrategy(
        arch_spec=arch,
        beam_width=2,
        hungarian_candidates=4,
        lookahead_window=0,
        nr_max_expansions=2000,
        nr_restarts=2,
    )
    state = _unaligned()
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    # The output layout must have q0 and q1 at CZ-partner positions.
    partner = arch.get_cz_partner(out.layout[0])
    assert partner is not None
    assert partner == out.layout[1], (
        f"CZ pair (0,1) not at CZ-partner positions: q0={out.layout[0]}, "
        f"q1={out.layout[1]}, partner_of_q0={partner}"
    )


def test_no_duplicate_locations_in_committed_layout():
    """The committed layout must not have two qubits at the same location."""
    arch = logical.get_arch_spec()
    strategy = MultiCandidateLookaheadPlacementStrategy(
        arch_spec=arch,
        beam_width=4,
        hungarian_candidates=8,
        lookahead_window=0,
    )
    state = _unaligned()
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    keys = [(loc.word_id, loc.site_id) for loc in out.layout]
    assert len(set(keys)) == len(
        keys
    ), f"Duplicate locations in output layout: {out.layout}"


def test_empty_lookahead_runs_clean():
    """Empty lookahead_cz_layers should still produce a valid commit."""
    strategy = MultiCandidateLookaheadPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        beam_width=2,
        hungarian_candidates=4,
        lookahead_window=24,  # but no lookahead actually supplied
    )
    state = _aligned()
    out = strategy.cz_placements(
        state, controls=(0,), targets=(1,), lookahead_cz_layers=()
    )
    assert isinstance(out, ExecuteCZ)


def test_w1_degenerates_to_safe_fallback():
    """At W=1 with no lookahead, the beam has a single trajectory.
    Should never be worse than NR on the single-layer test.
    """
    arch = logical.get_arch_spec()
    state = _unaligned()
    nr = NoReturnPlacementStrategy(arch_spec=arch, max_expansions=2000, restarts=2)
    mcl = MultiCandidateLookaheadPlacementStrategy(
        arch_spec=arch,
        beam_width=1,
        hungarian_candidates=4,
        lookahead_window=0,
        nr_max_expansions=2000,
        nr_restarts=2,
    )
    nr_out = nr.cz_placements(state, controls=(0,), targets=(1,))
    mcl_out = mcl.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(nr_out, ExecuteCZ)
    assert isinstance(mcl_out, ExecuteCZ)
    nr_lanes = sum(len(L) for L in nr_out.move_layers)
    mcl_lanes = sum(len(L) for L in mcl_out.move_layers)
    # With single-stage and lookahead_window=0, the score function = cost.
    # NR is a candidate; MCL should pick min cost ≤ NR.
    assert mcl_lanes <= nr_lanes


def test_layout_type_is_tuple_of_locationaddress():
    """ExecuteCZ.layout should be a tuple of LocationAddress objects."""
    strategy = MultiCandidateLookaheadPlacementStrategy(
        arch_spec=logical.get_arch_spec(), beam_width=2, hungarian_candidates=2
    )
    state = _aligned()
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    assert isinstance(out.layout, tuple)
    for loc in out.layout:
        assert isinstance(loc, LocationAddress)


def test_move_count_matches_layout_change():
    """move_count[i] should increment if and only if qubit i moved
    from its previous location to a different location during this stage.
    """
    arch = logical.get_arch_spec()
    strategy = MultiCandidateLookaheadPlacementStrategy(
        arch_spec=arch,
        beam_width=4,
        hungarian_candidates=8,
        lookahead_window=0,
        nr_max_expansions=2000,
        nr_restarts=2,
    )
    state = _unaligned()
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    for i in range(len(state.layout)):
        moved = state.layout[i] != out.layout[i]
        expected_count = state.move_count[i] + (1 if moved else 0)
        assert out.move_count[i] == expected_count, (
            f"q{i}: before={state.layout[i]}, after={out.layout[i]}, "
            f"moved={moved}, move_count={out.move_count[i]} (expected {expected_count})"
        )


def test_lookahead_window_24_still_succeeds_on_aligned_state():
    """Large lookahead_window with empty lookahead_cz_layers should
    still work (just runs one stage)."""
    strategy = MultiCandidateLookaheadPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        beam_width=8,
        hungarian_candidates=20,
        lookahead_window=24,  # default
    )
    state = _aligned()
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    # Aligned state: no movement needed
    assert sum(len(L) for L in out.move_layers) == 0
