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


def invalid_methods():
    @lanes.kernel
    def invalid_location():
        move.fill(location_addresses=(LocationAddress(2, 1),))

    yield invalid_location

    @lanes.kernel
    def invalid_move_lane():
        move.move(lanes=(SiteLaneAddress(Direction.FORWARD, 0, 0, 10),))

    yield invalid_move_lane

    @lanes.kernel
    def incompatible_move_lane():
        move.move(
            lanes=(
                SiteLaneAddress(Direction.FORWARD, 0, 0, 1),
                SiteLaneAddress(Direction.FORWARD, 0, 1, 0),
            )
        )

    yield incompatible_move_lane

    @lanes.kernel
    def invalid_measurement():
        future = move.end_measure(zone_address=ZoneAddress(0))
        move.get_measurement_result(future, location_address=LocationAddress(5, 5))

    yield invalid_measurement


@pytest.mark.parametrize("mt", invalid_methods())
def test_invalid_location(mt: ir.Method):

    valdiator = ValidationSuite([validation.AddressValidation])
    result = valdiator.validate(mt)

    with pytest.raises(ValidationErrorGroup):
        result.raise_if_invalid()
