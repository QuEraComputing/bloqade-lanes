"""Tests for the shared two-word move synthesis (compute_move_layers)."""

from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.analysis.placement.move_synthesis import compute_move_layers
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.layout.encoding import (
    Direction,
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
)


def test_compute_move_layers_word_0_to_1():
    """Same (state_before, state_after) as first CZ case in test_fixed: qubits 0,1 move from word 0 to word 1."""
    arch_spec = get_arch_spec()
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 1),
            LocationAddress(1, 0),
            LocationAddress(1, 1),
        ),
        move_count=(0, 0, 0, 0),
    )
    state_after = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(1, 5),
            LocationAddress(1, 6),
            LocationAddress(1, 0),
            LocationAddress(1, 1),
        ),
        move_count=(1, 1, 0, 0),
    )
    result = compute_move_layers(arch_spec, state_before, state_after)
    expected = (
        (
            SiteLaneAddress(
                word_id=0, site_id=0, bus_id=0, direction=Direction.FORWARD
            ),
            SiteLaneAddress(
                word_id=0, site_id=1, bus_id=0, direction=Direction.FORWARD
            ),
        ),
        (
            WordLaneAddress(
                word_id=0, site_id=5, bus_id=0, direction=Direction.FORWARD
            ),
            WordLaneAddress(
                word_id=0, site_id=6, bus_id=0, direction=Direction.FORWARD
            ),
        ),
    )
    assert result == expected


def test_compute_move_layers_word_1_to_0():
    """Qubits move from word 1 to word 0 (backward word bus)."""
    arch_spec = get_arch_spec()
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 1),
            LocationAddress(1, 0),
            LocationAddress(1, 1),
        ),
        move_count=(1, 1, 0, 0),
    )
    state_after = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 1),
            LocationAddress(0, 5),
            LocationAddress(0, 6),
        ),
        move_count=(1, 1, 1, 1),
    )
    result = compute_move_layers(arch_spec, state_before, state_after)
    expected = (
        (
            SiteLaneAddress(
                word_id=1, site_id=0, bus_id=0, direction=Direction.FORWARD
            ),
            SiteLaneAddress(
                word_id=1, site_id=1, bus_id=0, direction=Direction.FORWARD
            ),
        ),
        (
            WordLaneAddress(
                word_id=0, site_id=5, bus_id=0, direction=Direction.BACKWARD
            ),
            WordLaneAddress(
                word_id=0, site_id=6, bus_id=0, direction=Direction.BACKWARD
            ),
        ),
    )
    assert result == expected


def test_compute_move_layers_no_diffs():
    """When layout is unchanged, move layers are empty."""
    arch_spec = get_arch_spec()
    state = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(1, 0),
        ),
        move_count=(0, 0),
    )
    result = compute_move_layers(arch_spec, state, state)
    assert result == ()
