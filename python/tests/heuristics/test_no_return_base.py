"""Edge-path coverage for ``NoReturnStrategyBase``.

The happy paths through ``cz_placements`` / ``sq_placements`` /
``measure_placements`` are exercised by the per-strategy smoke tests
(``test_no_return_placement.py``, ``test_nohome_placement.py``,
``test_receding_horizon_placement.py``). This file fills in the
short-circuit / error branches that those smokes don't hit, using
:class:`NoReturnPlacementStrategy` as a concrete driver.
"""

from __future__ import annotations

from types import SimpleNamespace

from bloqade.lanes.analysis.placement import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    ExecuteMeasure,
    MoveToPlacementStrategyABC,
    UserMoved,
)
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.physical.no_return import NoReturnPlacementStrategy


def _make_strategy() -> NoReturnPlacementStrategy:
    return NoReturnPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        max_expansions=300,
    )


def _make_state() -> ConcreteState:
    # An *unpaired* layout: (2, 0) is not the CZ partner of (0, 0), so
    # cz_placements must run the inner solver (rather than short-circuiting
    # the already-paired no-op). The already-paired no-op has its own test.
    return ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(0, 0), LocationAddress(2, 0)),
        move_count=(0, 0),
    )


# ── cz_placements short-circuits ────────────────────────────────────


def test_cz_placements_returns_bottom_for_mismatched_controls_targets():
    """``len(controls) != len(targets)`` short-circuits to bottom."""
    strategy = _make_strategy()
    out = strategy.cz_placements(_make_state(), controls=(0,), targets=(0, 1))
    assert out == AtomState.bottom()


def test_cz_placements_returns_bottom_for_bottom_input():
    """``state == AtomState.bottom()`` short-circuits to bottom."""
    strategy = _make_strategy()
    out = strategy.cz_placements(AtomState.bottom(), controls=(0,), targets=(1,))
    assert out == AtomState.bottom()


def test_cz_placements_returns_top_for_non_concrete_input():
    """A non-:class:`ConcreteState` input (here: top) returns top."""
    strategy = _make_strategy()
    out = strategy.cz_placements(AtomState.top(), controls=(0,), targets=(1,))
    assert out == AtomState.top()


def test_cz_placements_returns_bottom_when_inner_solver_fails(monkeypatch):
    """``result.status != "solved"`` returns bottom while still accumulating
    nodes_expanded into the observability counter."""
    strategy = _make_strategy()
    fake_result = SimpleNamespace(
        status="budget_exceeded",
        nodes_expanded=7,
        move_layers=[],
        goal_config={},
    )
    monkeypatch.setattr(
        strategy,
        "_invoke_placement",
        lambda *args, **kwargs: fake_result,
    )
    before = strategy.rust_nodes_expanded_total
    out = strategy.cz_placements(_make_state(), controls=(0,), targets=(1,))
    assert out == AtomState.bottom()
    assert strategy.rust_nodes_expanded_total == before + 7


def test_cz_placements_returns_execute_cz_for_concrete_input():
    """Happy path returns an :class:`ExecuteCZ` (sanity-check that the
    short-circuit tests above are testing the right preconditions)."""
    strategy = _make_strategy()
    out = strategy.cz_placements(_make_state(), controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)


def test_cz_placements_no_op_when_already_paired():
    """When the input layout already places every CZ pair at valid entangling
    partner sites (e.g. staged there by move_to / permute), cz_placements
    emits the CZ in place with no moves instead of relocating the qubits.

    Uses the no-home physical strategy + physical arch, which is the
    configuration that exhibited the relocation bug.
    """
    from bloqade.lanes.heuristics.physical import make_physical_placement_strategy

    strategy = make_physical_placement_strategy(return_moves=False)
    arch = strategy.arch_spec

    # Find a (location, CZ partner) pair in the arch.
    target_loc = partner_loc = None
    for w in range(len(arch.words)):
        for s in range(len(arch.words[w].site_indices)):
            loc = LocationAddress(w, s)
            p = arch.get_cz_partner(loc)
            if p is not None:
                target_loc, partner_loc = loc, p
                break
        if target_loc is not None:
            break
    assert (
        target_loc is not None and partner_loc is not None
    ), "test arch has no CZ partner pair"

    # qubit 0 (control) is already at the partner of qubit 1 (target): paired.
    state = ConcreteState(
        occupied=frozenset(),
        layout=(partner_loc, target_loc),
        move_count=(0, 0),
    )
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)
    assert out.move_layers == ()
    assert out.layout == (partner_loc, target_loc)
    assert out.move_count == (0, 0)


# ── sq_placements branches ─────────────────────────────────────────


def test_sq_placements_passthrough_for_non_concrete():
    """Non-concrete inputs (top / bottom) pass through unchanged."""
    strategy = _make_strategy()
    assert (
        strategy.sq_placements(AtomState.bottom(), qubits=(0, 1)) == AtomState.bottom()
    )
    assert strategy.sq_placements(AtomState.top(), qubits=(0, 1)) == AtomState.top()


def test_sq_placements_strips_cz_metadata_for_concrete():
    """For a :class:`ConcreteState` input, sq_placements returns a fresh
    :class:`ConcreteState` (stripping any CZ-specific metadata that an
    :class:`ExecuteCZ` would carry)."""
    strategy = _make_strategy()
    state = _make_state()
    out = strategy.sq_placements(state, qubits=(0, 1))
    assert isinstance(out, ConcreteState)
    assert type(out) is ConcreteState  # not ExecuteCZ / ExecuteMeasure
    assert out.layout == state.layout
    assert out.move_count == state.move_count


# ── measure_placements branches ────────────────────────────────────


def test_measure_placements_passthrough_for_non_concrete():
    """Non-concrete inputs (top / bottom) pass through unchanged."""
    strategy = _make_strategy()
    assert (
        strategy.measure_placements(AtomState.bottom(), qubits=(0, 1))
        == AtomState.bottom()
    )
    assert (
        strategy.measure_placements(AtomState.top(), qubits=(0, 1)) == AtomState.top()
    )


def test_measure_placements_returns_bottom_for_partial_qubit_set():
    """A measure of a strict subset of the laid-out qubits returns bottom
    (measurements must cover the full configuration)."""
    strategy = _make_strategy()
    out = strategy.measure_placements(_make_state(), qubits=(0,))
    assert out == AtomState.bottom()


def test_measure_placements_returns_execute_measure_for_full_qubit_set():
    """Measuring every qubit returns an :class:`ExecuteMeasure`."""
    strategy = _make_strategy()
    out = strategy.measure_placements(_make_state(), qubits=(0, 1))
    assert isinstance(out, ExecuteMeasure)


# ── validate_initial_layout ────────────────────────────────────────


def test_validate_initial_layout_is_noop():
    """Base implementation accepts any layout without raising."""
    strategy = _make_strategy()
    strategy.validate_initial_layout((LocationAddress(0, 0), LocationAddress(1, 0)))


# ── MoveTo support (MoveToPlacementStrategyABC) ─────────────────────


def test_is_move_to_placement_strategy():
    """No-return strategies support user-directed movement, so the MoveTo
    placement interpreter (which gates on ``MoveToPlacementStrategyABC``)
    routes through them instead of returning bottom."""
    assert isinstance(_make_strategy(), MoveToPlacementStrategyABC)


def test_compute_moves_routes_between_layouts():
    """``compute_moves`` synthesizes AOD move layers routing atoms from one
    concrete layout to another (qubit 0: word 0 -> word 1 via word bus, with
    qubit 1 parked at word 2 so the destination is free)."""
    strategy = _make_strategy()
    arch_spec = strategy.arch_spec
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(0, 0), LocationAddress(2, 0)),
        move_count=(0, 0),
    )
    state_after = ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(1, 0), LocationAddress(2, 0)),
        move_count=(1, 0),
    )
    layers = strategy.compute_moves(state_before, state_after)
    assert len(layers) > 0
    for layer in layers:
        for lane in layer:
            assert not arch_spec.check_lane_group([lane])


def test_compute_moves_no_diff_is_empty():
    """Routing a layout to itself yields no move layers."""
    strategy = _make_strategy()
    state = _make_state()
    assert strategy.compute_moves(state, state) == ()


def test_move_to_placements_produces_user_moved():
    """``move_to_placements`` (inherited from ``MoveToPlacementStrategyABC``)
    places a qubit at a user-directed location, producing a ``UserMoved``
    state whose move history is populated by ``compute_moves``."""
    strategy = _make_strategy()
    state = ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(0, 0), LocationAddress(2, 0)),
        move_count=(0, 0),
    )
    out = strategy.move_to_placements(
        state, qubits=(0,), locations=(LocationAddress(1, 0),)
    )
    assert isinstance(out, UserMoved)
    assert out.layout == (LocationAddress(1, 0), LocationAddress(2, 0))
    assert out.accumulated_move_layers == out.move_layers
    assert len(out.move_layers) > 0
