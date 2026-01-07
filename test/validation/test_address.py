import pytest
from kirin import ir
from kirin.ir.exception import ValidationErrorGroup
from kirin.validation import ValidationSuite

from bloqade import lanes
from bloqade.lanes.arch.gemini.logical import validation
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import (
    Direction,
    LocationAddress,
    SiteLaneAddress,
    ZoneAddress,
)
from bloqade.lanes.types import State


def invalid_methods():
    @lanes.kernel
    def invalid_location(state: State):

        return move.fill(state, location_addresses=(LocationAddress(2, 1),))

    yield invalid_location

    @lanes.kernel
    def invalid_move_lane(state: State):
        return move.move(state, lanes=(SiteLaneAddress(0, 0, 10, Direction.FORWARD),))

    yield invalid_move_lane

    @lanes.kernel
    def incompatible_move_lane(state: State):
        return move.move(
            state,
            lanes=(
                SiteLaneAddress(0, 0, 1, Direction.FORWARD),
                SiteLaneAddress(0, 1, 0, Direction.FORWARD),
            ),
        )

    yield incompatible_move_lane

    @lanes.kernel
    def invalid_index(future: move.MeasurementFuture):
        return move.get_future_result(
            future, zone_address=ZoneAddress(0), location_address=LocationAddress(5, 5)
        )

    yield invalid_index


@pytest.mark.parametrize("mt", invalid_methods())
def test_address_invalid_location(mt: ir.Method):

    validator = ValidationSuite([validation.AddressValidation])
    result = validator.validate(mt)

    with pytest.raises(ValidationErrorGroup):
        result.raise_if_invalid()
