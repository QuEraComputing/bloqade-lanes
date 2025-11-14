import pytest

from bloqade.lanes.analysis.placement import AtomState, ConcreteState
from bloqade.lanes.heuristics import fixed
from bloqade.lanes.layout.encoding import (
    Direction,
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
)


def cz_placement_cases():

    yield (
        AtomState.top(),
        (0, 1),
        (2, 3),
        AtomState.top(),
    )

    yield (
        AtomState.bottom(),
        (0, 1),
        (2, 3),
        AtomState.bottom(),
    )

    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 2),
            LocationAddress(1, 0),
            LocationAddress(1, 2),
        ),
        move_count=(0, 0, 0, 0),
    )
    state_after = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(1, 1),
            LocationAddress(1, 3),
            LocationAddress(1, 0),
            LocationAddress(1, 2),
        ),
        move_count=(1, 1, 0, 0),
    )

    yield (
        state_before,
        (0, 1),
        (2, 3),
        state_after,
    )

    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 2),
            LocationAddress(1, 0),
            LocationAddress(1, 2),
        ),
        move_count=(1, 1, 0, 0),
    )
    state_after = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 2),
            LocationAddress(0, 1),
            LocationAddress(0, 3),
        ),
        move_count=(1, 1, 1, 1),
    )
    yield (
        state_before,
        (0, 1),
        (2, 3),
        state_after,
    )

    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 2),
            LocationAddress(0, 4),
            LocationAddress(0, 6),
        ),
        move_count=(1, 1, 0, 0),
    )
    state_after = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 2),
            LocationAddress(0, 1),
            LocationAddress(0, 3),
        ),
        move_count=(1, 1, 1, 1),
    )
    yield (
        state_before,
        (0, 1),
        (2, 3),
        state_after,
    )

    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 2),
            LocationAddress(0, 4),
            LocationAddress(0, 6),
        ),
        move_count=(0, 0, 1, 1),
    )
    state_after = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 5),
            LocationAddress(0, 7),
            LocationAddress(0, 4),
            LocationAddress(0, 6),
        ),
        move_count=(1, 1, 1, 1),
    )
    yield (
        state_before,
        (0, 1),
        (2, 3),
        state_after,
    )

    yield (
        state_before,
        (0, 1, 4),
        (2, 3),
        AtomState.top(),
    )


@pytest.mark.parametrize(
    "state_before, targets, controls, state_after", cz_placement_cases()
)
def test_fixed_cz_placement(
    state_before: AtomState,
    targets: tuple[int, ...],
    controls: tuple[int, ...],
    state_after: AtomState,
):
    placement_strategy = fixed.LogicalPlacementStrategy()
    state_result = placement_strategy.cz_placements(state_before, controls, targets)

    assert state_result == state_after


def test_fixed_sq_placement():
    placement_strategy = fixed.LogicalPlacementStrategy()
    assert AtomState.top() == placement_strategy.sq_placements(
        AtomState.top(), (0, 1, 2)
    )
    assert AtomState.bottom() == placement_strategy.sq_placements(
        AtomState.bottom(), (0, 1, 2)
    )
    state = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 2),
            LocationAddress(1, 0),
            LocationAddress(1, 2),
        ),
        move_count=(0, 0, 0, 0),
    )
    assert state == placement_strategy.sq_placements(state, (0, 1, 2))


def test_fixed_invalid_initial_layout_1():
    placement_strategy = fixed.LogicalPlacementStrategy()
    layout = (
        LocationAddress(0, 0),
        LocationAddress(0, 1),
        LocationAddress(0, 2),
        LocationAddress(0, 3),
    )
    with pytest.raises(ValueError):
        placement_strategy.validate_initial_layout(layout)


def test_fixed_invalid_initial_layout_2():
    placement_strategy = fixed.LogicalPlacementStrategy()
    layout = (
        LocationAddress(0, 0),
        LocationAddress(1, 0),
        LocationAddress(2, 0),
        LocationAddress(3, 0),
    )
    with pytest.raises(ValueError):
        placement_strategy.validate_initial_layout(layout)


def test_move_scheduler_get_direction():
    move_scheduler = fixed.LogicalMoveScheduler()
    assert move_scheduler.get_direction(1) == Direction.FORWARD
    assert move_scheduler.get_direction(-1) == Direction.BACKWARD


def test_move_scheduler_get_y_position():
    move_scheduler = fixed.LogicalMoveScheduler()
    loc1 = LocationAddress(word_id=0, site_id=3)
    loc2 = LocationAddress(word_id=1, site_id=4)
    assert move_scheduler.get_site_y(loc1) == 1
    assert move_scheduler.get_site_y(loc2) == 2


def test_move_scheduler_get_site_bus_id():
    move_scheduler = fixed.LogicalMoveScheduler()
    loc1 = LocationAddress(word_id=0, site_id=2)
    loc2 = LocationAddress(word_id=1, site_id=6)
    bus_id, direction = move_scheduler.get_site_bus_id(loc1, loc2)
    assert bus_id == 2
    assert direction == Direction.FORWARD

    loc3 = LocationAddress(word_id=0, site_id=6)
    loc4 = LocationAddress(word_id=1, site_id=2)
    bus_id, direction = move_scheduler.get_site_bus_id(loc3, loc4)
    assert bus_id == 2
    assert direction == Direction.BACKWARD


def test_move_scheduler_compute_moves():
    move_scheduler = fixed.LogicalMoveScheduler()
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 2),
            LocationAddress(1, 0),
            LocationAddress(1, 2),
        ),
        move_count=(0, 0, 0, 0),
    )
    state_after = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(1, 1),
            LocationAddress(1, 3),
            LocationAddress(1, 0),
            LocationAddress(1, 2),
        ),
        move_count=(1, 1, 0, 0),
    )

    moves = move_scheduler.compute_moves(state_before, state_after)
    assert moves == [
        (
            SiteLaneAddress(
                direction=Direction.FORWARD, word_id=0, site_id=0, lane_id=0
            ),
            SiteLaneAddress(
                direction=Direction.FORWARD, word_id=0, site_id=2, lane_id=0
            ),
        ),
        (
            WordLaneAddress(
                direction=Direction.FORWARD, word_id=0, site_id=2, bus_id=0
            ),
            WordLaneAddress(
                direction=Direction.FORWARD, word_id=0, site_id=4, bus_id=0
            ),
        ),
    ]


def test_move_scheduler_compute_moves_same_word():
    move_scheduler = fixed.LogicalMoveScheduler()
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 2),
            LocationAddress(1, 0),
            LocationAddress(1, 2),
        ),
        move_count=(0, 0, 0, 0),
    )
    state_after = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 3),
            LocationAddress(0, 2),
            LocationAddress(1, 3),
            LocationAddress(1, 2),
        ),
        move_count=(1, 1, 0, 0),
    )

    moves = move_scheduler.compute_moves(state_before, state_after)
    assert moves == [
        (
            SiteLaneAddress(
                direction=Direction.FORWARD, word_id=0, site_id=0, lane_id=1
            ),
        ),
        (
            SiteLaneAddress(
                direction=Direction.FORWARD, word_id=1, site_id=0, lane_id=1
            ),
        ),
    ]
