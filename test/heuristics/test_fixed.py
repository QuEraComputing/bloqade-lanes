import pytest

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import AtomState, ConcreteState
from bloqade.lanes.analysis.placement.lattice import ExecuteCZ
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.heuristics import fixed
from bloqade.lanes.heuristics.logical_placement import (
    LogicalPlacementStrategy,
    LogicalPlacementStrategyNoHome,
)
from bloqade.lanes.heuristics.move_synthesis import compute_move_layers, move_to_left
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
        move_layers=(
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
        ),
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
        move_layers=(
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
        ),
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
        move_layers=(
            (
                SiteLaneAddress(
                    word_id=0, site_id=2, bus_id=7, direction=Direction.FORWARD
                ),
                SiteLaneAddress(
                    word_id=0, site_id=3, bus_id=7, direction=Direction.FORWARD
                ),
            ),
        ),
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
        move_layers=(
            (
                SiteLaneAddress(
                    word_id=0, site_id=0, bus_id=2, direction=Direction.FORWARD
                ),
                SiteLaneAddress(
                    word_id=0, site_id=1, bus_id=2, direction=Direction.FORWARD
                ),
            ),
        ),
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
        move_layers=(
            (
                SiteLaneAddress(
                    word_id=0, site_id=2, bus_id=7, direction=Direction.FORWARD
                ),
                SiteLaneAddress(
                    word_id=0, site_id=3, bus_id=7, direction=Direction.FORWARD
                ),
            ),
        ),
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
        move_layers=(
            (
                SiteLaneAddress(
                    word_id=0, site_id=0, bus_id=2, direction=Direction.FORWARD
                ),
                SiteLaneAddress(
                    word_id=0, site_id=1, bus_id=2, direction=Direction.FORWARD
                ),
            ),
        ),
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
    placement_strategy = LogicalPlacementStrategy()
    state_result = placement_strategy.cz_placements(state_before, controls, targets)
    if not isinstance(state_before, ConcreteState) or not isinstance(
        state_after, ExecuteCZ
    ):
        assert state_result == state_after
        return

    assert isinstance(state_result, ExecuteCZ)
    assert state_result.active_cz_zones == state_after.active_cz_zones
    assert state_result.layout == state_after.layout
    assert state_result.get_move_layers() == state_after.move_layers
    assert state_result.move_count == state_after.move_count


def test_fixed_sq_placement():
    placement_strategy = LogicalPlacementStrategy()
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
    placement_strategy = LogicalPlacementStrategy()
    layout = (
        LocationAddress(0, 0),
        LocationAddress(0, 1),
        LocationAddress(0, 2),
        LocationAddress(0, 5),
    )
    with pytest.raises(ValueError):
        placement_strategy.validate_initial_layout(layout)


def test_fixed_invalid_initial_layout_2():
    placement_strategy = LogicalPlacementStrategy()
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

    placement = LogicalPlacementStrategy()
    controls = (0, 1, 4)
    targets = (5, 6, 7)

    final_state = placement.cz_placements(
        initial_state,
        controls,
        targets,
    )

    assert final_state.get_move_layers() == (
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
    )


def test_nohome_choose_return_layout():
    placement = LogicalPlacementStrategyNoHome()
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(1, 5),
            LocationAddress(0, 1),
        ),
        move_count=(3, 4),
    )
    mid_state, left_move_layers = placement.choose_return_layout(
        state_before, controls=(0,), targets=(1,)
    )
    assert mid_state.layout == (
        LocationAddress(1, 0),
        LocationAddress(0, 1),
    )
    assert mid_state.move_count == (4, 4)
    _, expected_left_move_layers = move_to_left(
        get_arch_spec(),
        state_before,
        mid_state,
    )
    assert left_move_layers == expected_left_move_layers


def test_nohome_choose_return_layout_duplicate_collision():
    placement = LogicalPlacementStrategyNoHome()
    state_before = ConcreteState(
        occupied=frozenset(
            {
                LocationAddress(0, 1),
                LocationAddress(0, 2),
                LocationAddress(0, 3),
                LocationAddress(0, 4),
                LocationAddress(1, 0),
                LocationAddress(1, 1),
                LocationAddress(1, 2),
                LocationAddress(1, 3),
                LocationAddress(1, 4),
            }
        ),
        layout=(
            LocationAddress(1, 5),
            LocationAddress(0, 0),
        ),
        move_count=(0, 0),
    )
    with pytest.raises(ValueError, match="No empty left-column site"):
        placement.choose_return_layout(state_before, controls=(0,), targets=(1,))


def test_nohome_choose_return_layout_sequential_no_conflicts():
    placement = LogicalPlacementStrategyNoHome()
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 5),
            LocationAddress(0, 6),
            LocationAddress(1, 7),
            LocationAddress(1, 8),
        ),
        move_count=(0, 0, 0, 0),
    )
    mid_state, _ = placement.choose_return_layout(
        state_before, controls=(0, 1), targets=(2, 3)
    )
    assert mid_state.layout == (
        LocationAddress(0, 0),
        LocationAddress(0, 1),
        LocationAddress(1, 2),
        LocationAddress(1, 3),
    )


def test_nohome_cz_placements_combines_return_and_entangle_layers():
    placement = LogicalPlacementStrategyNoHome()
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(1, 5),
            LocationAddress(0, 1),
        ),
        move_count=(0, 0),
    )
    result = placement.cz_placements(state_before, controls=(0,), targets=(1,))
    assert isinstance(result, ExecuteCZ)
    mid_state, left_move_layers = placement.choose_return_layout(
        state_before, controls=(0,), targets=(1,)
    )
    entangle_move_layers = compute_move_layers(
        get_arch_spec(),
        mid_state,
        ConcreteState(
            occupied=mid_state.occupied,
            layout=result.layout,
            move_count=result.move_count,
        ),
    )
    assert result.get_move_layers() == left_move_layers + entangle_move_layers
