"""Tests for the shared two-word move synthesis (compute_move_layers)."""

from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.heuristics.move_synthesis import (
    compute_move_layers,
    move_to_entangle,
    move_to_left,
)
from bloqade.lanes.layout.encoding import (
    Direction,
    LocationAddress,
    MoveType,
)


def test_compute_move_layers_word_0_to_1():
    """Cross-word move: qubit moves from word 0 to word 1 via word bus."""
    arch_spec = get_arch_spec()
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 1),
        ),
        move_count=(0, 0),
    )
    state_after = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(1, 0),
            LocationAddress(0, 1),
        ),
        move_count=(1, 0),
    )
    result = compute_move_layers(arch_spec, state_before, state_after)
    # Direct word bus move: (0,0) → (1,0), no site adjustment needed
    assert len(result) > 0
    for layer in result:
        for lane in layer:
            assert not arch_spec.check_lane_group([lane])


def test_compute_move_layers_cross_word_exact():
    """Anchor test: verify exact move layer for a direct word bus hop."""
    arch_spec = get_arch_spec()
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
    result = compute_move_layers(arch_spec, state_before, state_after)
    # Single word bus layer: word 0 → word 1 at site 0
    assert len(result) == 1
    assert len(result[0]) == 1
    lane = result[0][0]
    assert lane.move_type == MoveType.WORD
    assert lane.word_id == 0
    assert lane.site_id == 0
    assert lane.direction == Direction.FORWARD


def test_compute_move_layers_word_bus_exact():
    """Anchor test: verify exact move layer for a word bus move."""
    arch_spec = get_arch_spec()
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(0, 0),),
        move_count=(0,),
    )
    state_after = ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(1, 0),),
        move_count=(1,),
    )
    result = compute_move_layers(arch_spec, state_before, state_after)
    # Single word bus layer: word 0 → word 1
    assert len(result) == 1
    assert len(result[0]) == 1
    lane = result[0][0]
    assert lane.move_type == MoveType.WORD
    assert lane.word_id == 0
    assert lane.site_id == 0
    assert lane.direction == Direction.FORWARD


def test_compute_move_layers_word_1_to_0():
    """Cross-word move: qubit moves from word 1 to word 0 (backward word bus)."""
    arch_spec = get_arch_spec()
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(1, 0),
        ),
        move_count=(0, 0),
    )
    state_after = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(5, 0),
        ),
        move_count=(0, 1),
    )
    result = compute_move_layers(arch_spec, state_before, state_after)
    assert len(result) > 0
    for layer in result:
        for lane in layer:
            assert not arch_spec.check_lane_group([lane])


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


def test_compute_move_layers_same_word():
    """Word bus move (single step)."""
    arch_spec = get_arch_spec()
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(0, 0),),
        move_count=(0,),
    )
    state_after = ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(1, 0),),
        move_count=(1,),
    )
    result = compute_move_layers(arch_spec, state_before, state_after)
    assert len(result) > 0
    for layer in result:
        for lane in layer:
            assert not arch_spec.check_lane_group([lane])


def test_move_to_entangle_wrapper():
    arch_spec = get_arch_spec()
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

    out_state, layers = move_to_entangle(arch_spec, state_before, state_after)
    assert out_state == state_after
    assert layers == compute_move_layers(arch_spec, state_before, state_after)


def test_move_to_left_wrapper():
    arch_spec = get_arch_spec()
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(1, 0), LocationAddress(2, 0)),
        move_count=(1, 0),
    )
    state_after = ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(0, 0), LocationAddress(2, 0)),
        move_count=(2, 0),
    )

    out_state, layers = move_to_left(arch_spec, state_before, state_after)
    forward_layers = compute_move_layers(arch_spec, state_after, state_before)
    expected_layers = tuple(
        tuple(lane.reverse() for lane in move_lanes[::-1])
        for move_lanes in forward_layers[::-1]
    )
    assert out_state == state_after
    assert layers == expected_layers
