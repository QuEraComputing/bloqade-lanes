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
    return ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(0, 0), LocationAddress(1, 0)),
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
