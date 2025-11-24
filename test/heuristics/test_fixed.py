import pytest

from bloqade.lanes.analysis.placement import AtomState, ConcreteState
from bloqade.lanes.heuristics import fixed
from bloqade.lanes.layout.encoding import (
    LocationAddress,
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

    gates = [(0, 2), (1, 3)]
    ctrls, trgts = list(zip(*gates))

    yield (
        state_before,
        ctrls,
        trgts,
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
        move_count=(1, 0, 1, 0),
    )

    gates = [(0, 1), (2, 3)]
    ctrls, trgts = list(zip(*gates))

    yield (
        state_before,
        ctrls,
        trgts,
        state_after,
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


def test_fixed_initial_layout():
    layout_strategy = fixed.LogicalLayoutHeuristic()

    edges = {
        (1, 2): 3,
        (2, 3): 4,
        (0, 1): 5,
        (0, 3): 5,
        (0, 2): 1,
    }

    stages = [(key,) * weight for key, weight in edges.items()]
    layout = layout_strategy.compute_layout(tuple(range(4)), stages)

    assert layout == (
        LocationAddress(word_id=1, site_id=8),
        LocationAddress(word_id=1, site_id=4),
        LocationAddress(word_id=1, site_id=2),
        LocationAddress(word_id=1, site_id=6),
    )
