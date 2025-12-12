from typing import TypeVar

import pytest
from bloqade.test_utils import assert_nodes
from kirin import ir, rewrite
from kirin.dialects import ilist, py

from bloqade.lanes.dialects import move, place
from bloqade.lanes.layout.encoding import (
    Direction,
    LaneAddress,
    LocationAddress,
    SiteLaneAddress,
)
from bloqade.lanes.rewrite import transversal

AddressType = TypeVar("AddressType", bound=LocationAddress | LaneAddress)


def trivial_map(address: AddressType) -> tuple[AddressType, ...] | None:
    if address.word_id < 1:
        return (address,)
    return None


def cases():

    node = move.Initialize(
        thetas := (ir.TestValue(),),
        phis := (ir.TestValue(),),
        lams := (ir.TestValue(),),
        location_addresses=(LocationAddress(0, 1), LocationAddress(1, 1)),
    )

    expected_node = move.Initialize(
        thetas,
        phis,
        lams,
        location_addresses=(LocationAddress(0, 1), LocationAddress(1, 1)),
    )

    yield node, expected_node, False

    node = move.Initialize(
        thetas := (ir.TestValue(),),
        phis := (ir.TestValue(),),
        lams := (ir.TestValue(),),
        location_addresses=(LocationAddress(0, 1), LocationAddress(0, 1)),
    )

    expected_node = move.Initialize(
        thetas,
        phis,
        lams,
        location_addresses=(LocationAddress(0, 1), LocationAddress(0, 1)),
    )

    yield node, expected_node, True

    node = move.Move(
        lanes=(
            SiteLaneAddress(Direction.FORWARD, 0, 1, 0),
            SiteLaneAddress(Direction.FORWARD, 1, 1, 0),
        ),
    )

    expected_node = move.Move(
        lanes=(
            SiteLaneAddress(Direction.FORWARD, 0, 1, 0),
            SiteLaneAddress(Direction.FORWARD, 1, 1, 0),
        ),
    )

    yield node, expected_node, False

    node = move.Move(
        lanes=(
            SiteLaneAddress(Direction.FORWARD, 0, 1, 0),
            SiteLaneAddress(Direction.FORWARD, 0, 1, 0),
        ),
    )

    expected_node = move.Move(
        lanes=(
            SiteLaneAddress(Direction.FORWARD, 0, 1, 0),
            SiteLaneAddress(Direction.FORWARD, 0, 1, 0),
        ),
    )

    yield node, expected_node, True


@pytest.mark.parametrize("node, expected_node, has_done_something", cases())
def test_simple_rewrite(
    node: ir.Statement, expected_node: ir.Statement, has_done_something: bool
):
    test_block = ir.Block()
    test_block.stmts.append(py.Constant(10))
    test_block.stmts.append(node)

    expected_block = ir.Block()
    expected_block.stmts.append(py.Constant(10))
    expected_block.stmts.append(expected_node)

    rule = rewrite.Walk(
        rewrite.Chain(
            transversal.RewriteLocations(trivial_map),
            transversal.RewriteMoves(trivial_map),
        )
    )

    result = rule.rewrite(test_block)

    assert_nodes(test_block, expected_block)
    assert result.has_done_something is has_done_something


def test_get_measurement_result():

    measurement_future = ir.TestValue()
    test_block = ir.Block()
    test_block.stmts.append(
        move.GetMeasurementResult(
            measurement_future, location_address=LocationAddress(0, 1)
        )
    )

    expected_block = ir.Block()
    expected_block.stmts.append(
        measure := move.GetMeasurementResult(
            measurement_future, location_address=LocationAddress(0, 1)
        )
    )
    expected_block.stmts.append(ilist.New((measure.result,)))

    rule = rewrite.Walk(transversal.RewriteGetMeasurementResult(trivial_map))

    result = rule.rewrite(test_block)

    assert result.has_done_something
    assert_nodes(test_block, expected_block)


def test_get_measurement_result_no_op():

    measurement_future = ir.TestValue()
    test_block = ir.Block()
    test_block.stmts.append(py.Constant(10))
    test_block.stmts.append(
        move.GetMeasurementResult(
            measurement_future, location_address=LocationAddress(1, 1)
        )
    )

    expected_block = ir.Block()
    expected_block.stmts.append(py.Constant(10))
    expected_block.stmts.append(
        move.GetMeasurementResult(
            measurement_future, location_address=LocationAddress(1, 1)
        )
    )

    rule = rewrite.Walk(transversal.RewriteGetMeasurementResult(trivial_map))

    result = rule.rewrite(test_block)

    assert not result.has_done_something
    assert_nodes(test_block, expected_block)


def test_rewrite_conversion():
    measure_1 = ir.TestValue()
    measure_2 = ir.TestValue()
    test_block = ir.Block()
    test_block.stmts.append(py.Constant(10))
    test_block.stmts.append(place.ConvertToPhysicalMeasurements((measure_1, measure_2)))

    expected_block = ir.Block()
    expected_block.stmts.append(py.Constant(10))
    expected_block.stmts.append(ilist.New((measure_1, measure_2)))

    result = rewrite.Walk(transversal.RewriteLogicalToPhysicalConversion()).rewrite(
        test_block
    )
    assert result.has_done_something
    assert_nodes(test_block, expected_block)
