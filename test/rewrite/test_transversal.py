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

    node = move.Move(
        current_state := ir.TestValue(),
        lanes=(
            SiteLaneAddress(0, 1, 0, Direction.FORWARD),
            SiteLaneAddress(1, 1, 0, Direction.FORWARD),
        ),
    )

    expected_node = move.Move(
        current_state,
        lanes=(
            SiteLaneAddress(0, 1, 0, Direction.FORWARD),
            SiteLaneAddress(1, 1, 0, Direction.FORWARD),
        ),
    )

    yield node, expected_node, False

    node = move.Move(
        current_state := ir.TestValue(),
        lanes=(
            SiteLaneAddress(0, 1, 0, Direction.FORWARD),
            SiteLaneAddress(0, 1, 0, Direction.FORWARD),
        ),
    )

    expected_node = move.Move(
        current_state,
        lanes=(
            SiteLaneAddress(0, 1, 0, Direction.FORWARD),
            SiteLaneAddress(0, 1, 0, Direction.FORWARD),
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
