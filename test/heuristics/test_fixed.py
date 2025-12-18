import pytest

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import AtomState, ConcreteState
from bloqade.lanes.analysis.placement.lattice import ExecuteCZ
from bloqade.lanes.heuristics import fixed
from bloqade.lanes.layout.encoding import (
    Direction,
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
)


def cz_placement_cases():

    all_zones = frozenset([layout.ZoneAddress(0)])

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
            LocationAddress(0, 1),
            LocationAddress(1, 0),
            LocationAddress(1, 1),
        ),
        move_count=(0, 0, 0, 0),
    )
    state_after = ExecuteCZ(
        occupied=frozenset(),
        layout=(
            LocationAddress(1, 5),
            LocationAddress(1, 6),
            LocationAddress(1, 0),
            LocationAddress(1, 1),
        ),
        move_count=(1, 1, 0, 0),
        active_cz_zones=all_zones,
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
            LocationAddress(0, 1),
            LocationAddress(1, 0),
            LocationAddress(1, 1),
        ),
        move_count=(1, 1, 0, 0),
    )
    state_after = ExecuteCZ(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 1),
            LocationAddress(0, 5),
            LocationAddress(0, 6),
        ),
        move_count=(1, 1, 1, 1),
        active_cz_zones=all_zones,
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
            LocationAddress(0, 1),
            LocationAddress(0, 2),
            LocationAddress(0, 3),
        ),
        move_count=(1, 1, 0, 0),
    )
    state_after = ExecuteCZ(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 1),
            LocationAddress(0, 5),
            LocationAddress(0, 6),
        ),
        move_count=(1, 1, 1, 1),
        active_cz_zones=all_zones,
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
            LocationAddress(0, 1),
            LocationAddress(0, 2),
            LocationAddress(0, 3),
        ),
        move_count=(0, 0, 1, 1),
    )
    state_after = ExecuteCZ(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 7),
            LocationAddress(0, 8),
            LocationAddress(0, 2),
            LocationAddress(0, 3),
        ),
        move_count=(1, 1, 1, 1),
        active_cz_zones=all_zones,
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
            LocationAddress(0, 1),
            LocationAddress(0, 2),
            LocationAddress(0, 3),
        ),
        move_count=(1, 0, 0, 0),
    )
    state_after = ExecuteCZ(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 1),
            LocationAddress(0, 5),
            LocationAddress(0, 6),
        ),
        move_count=(1, 0, 1, 1),
        active_cz_zones=all_zones,
    )
    yield (
        state_before,
        (1, 0),
        (3, 2),
        state_after,
    )

    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 1),
            LocationAddress(0, 2),
            LocationAddress(0, 3),
        ),
        move_count=(0, 0, 0, 1),
    )
    state_after = ExecuteCZ(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 7),
            LocationAddress(0, 8),
            LocationAddress(0, 2),
            LocationAddress(0, 3),
        ),
        move_count=(1, 1, 0, 1),
        active_cz_zones=all_zones,
    )
    yield (
        state_before,
        (1, 0),
        (3, 2),
        state_after,
    )

    yield (
        state_before,
        (0, 1, 4),
        (2, 3),
        AtomState.bottom(),
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
        LocationAddress(0, 5),
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


def test_initial_layout():
    layout_heuristic = fixed.LogicalLayoutHeuristic()
    edges = {(i, j): 1 for i in range(10) for j in range(i + 1, 10, 1)}

    edges[(0, 1)] = 10

    edges = sum((weight * (edge,) for edge, weight in edges.items()), ())

    layout = layout_heuristic.compute_layout(tuple(range(10)), [edges])
    print(layout)
    assert layout == (
        LocationAddress(word_id=0, site_id=0),
        LocationAddress(word_id=0, site_id=1),
        LocationAddress(word_id=0, site_id=2),
        LocationAddress(word_id=0, site_id=3),
        LocationAddress(word_id=0, site_id=4),
        LocationAddress(word_id=1, site_id=0),
        LocationAddress(word_id=1, site_id=1),
        LocationAddress(word_id=1, site_id=2),
        LocationAddress(word_id=1, site_id=3),
        LocationAddress(word_id=1, site_id=4),
    )


def test_move_scheduler_cz():

    initial_state = ConcreteState(
        frozenset(),
        tuple(
            LocationAddress(word_id, site_id)
            for word_id in range(2)
            for site_id in range(5)
        ),
        tuple(0 for _ in range(10)),
    )

    placement = fixed.LogicalPlacementStrategy()
    controls = (0, 1, 4)
    targets = (5, 6, 7)

    final_state = placement.cz_placements(
        initial_state,
        controls,
        targets,
    )

    moves = fixed.LogicalMoveScheduler().compute_moves(initial_state, final_state)

    assert moves == [
        (
            SiteLaneAddress(
                direction=Direction.FORWARD, word_id=0, site_id=0, bus_id=0
            ),
            SiteLaneAddress(
                direction=Direction.FORWARD, word_id=0, site_id=1, bus_id=0
            ),
        ),
        (SiteLaneAddress(direction=Direction.FORWARD, word_id=0, site_id=4, bus_id=7),),
        (
            WordLaneAddress(
                direction=Direction.FORWARD, word_id=0, site_id=5, bus_id=0
            ),
            WordLaneAddress(
                direction=Direction.FORWARD, word_id=0, site_id=6, bus_id=0
            ),
            WordLaneAddress(
                direction=Direction.FORWARD, word_id=0, site_id=7, bus_id=0
            ),
        ),
    ]
