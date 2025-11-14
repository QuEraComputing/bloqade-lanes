import pytest

from bloqade.lanes.analysis.placement import AtomState, ConcreteState
from bloqade.lanes.heuristics import fixed
from bloqade.lanes.layout.encoding import LocationAddress


def fixed_placement_cases():

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


@pytest.mark.parametrize(
    "state_before, targets, controls, state_after", fixed_placement_cases()
)
def test_fixed_placement(
    state_before: AtomState,
    targets: tuple[int, ...],
    controls: tuple[int, ...],
    state_after: AtomState,
):
    placement_strategy = fixed.LogicalPlacementStrategy()
    state_result = placement_strategy.cz_placements(state_before, controls, targets)

    assert state_result == state_after
