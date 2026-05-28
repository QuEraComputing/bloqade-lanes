from typing import TypeVar

import pytest
from bloqade.test_utils import assert_nodes
from kirin import ir, rewrite, types
from kirin.dialects import func, ilist, py

from bloqade.lanes.bytecode.encoding import (
    Direction,
    LaneAddress,
    LocationAddress,
    SiteLaneAddress,
)
from bloqade.lanes.dialects import move, place
from bloqade.lanes.passes import TransversalRewritePass
from bloqade.lanes.rewrite import transversal

AddressType = TypeVar("AddressType", bound=LocationAddress | LaneAddress)


def trivial_map(address: AddressType) -> tuple[AddressType, ...] | None:
    if address.word_id < 1:
        return (address,)
    return None


def star_map(address: LocationAddress) -> tuple[LocationAddress, ...] | None:
    if address.word_id < 1 and address.site_id < 2:
        return tuple(address.replace(site_id=10 + i) for i in range(7))
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


def test_rewrite_star_rz_to_physical_local_rz():
    test_block = ir.Block()
    test_block.stmts.append(rotation_angle := py.Constant(0.5))
    test_block.stmts.append(
        move.StarRz(
            current_state := ir.TestValue(),
            rotation_angle.result,
            location_addresses=(LocationAddress(0, 0),),
            qubit_indices=(4, 5, 6),
        )
    )

    expected_block = ir.Block()
    expected_block.stmts.append(rotation_angle := py.Constant(0.5))
    expected_block.stmts.append(
        theta_star := func.Invoke(
            (rotation_angle.result,), callee=transversal.steane_star_theta
        )
    )
    expected_block.stmts.append(
        move.LocalRz(
            current_state,
            theta_star.result,
            location_addresses=(
                LocationAddress(0, 14),
                LocationAddress(0, 15),
                LocationAddress(0, 16),
            ),
        )
    )

    result = rewrite.Walk(transversal.RewriteStarRz(star_map)).rewrite(test_block)

    assert result.has_done_something
    assert_nodes(test_block, expected_block)


def test_rewrite_star_rz_noop_for_already_physical_location():
    test_block = ir.Block()
    test_block.stmts.append(rotation_angle := py.Constant(0.5))
    test_block.stmts.append(
        move.StarRz(
            current_state := ir.TestValue(),
            rotation_angle.result,
            location_addresses=(LocationAddress(0, 4),),
            qubit_indices=(4, 5, 6),
        )
    )

    expected_block = ir.Block()
    expected_block.stmts.append(rotation_angle := py.Constant(0.5))
    expected_block.stmts.append(
        move.StarRz(
            current_state,
            rotation_angle.result,
            location_addresses=(LocationAddress(0, 4),),
            qubit_indices=(4, 5, 6),
        )
    )

    result = rewrite.Walk(transversal.RewriteStarRz(star_map)).rewrite(test_block)

    assert not result.has_done_something
    assert_nodes(test_block, expected_block)


# --- TransversalRewritePass tests ---

_PASS_DIALECTS = ir.DialectGroup(
    [move.dialect, place.dialect, func.dialect, ilist.dialect, py.dialect]
)


def _make_method(*stmts) -> ir.Method:
    block = ir.Block(argtypes=(types.MethodType,))
    for s in stmts:
        block.stmts.append(s)
    region = ir.Region(blocks=block)
    function = func.Function(
        sym_name="test",
        signature=func.Signature((), types.NoneType),
        slots=(),
        body=region,
    )
    return ir.Method(
        dialects=_PASS_DIALECTS,
        code=function,
        sym_name="test",
        arg_names=[],
    )


def test_pass_rewrite_conversion():
    measure_1 = ir.TestValue()
    measure_2 = ir.TestValue()
    method = _make_method(
        py.Constant(10),
        place.ConvertToPhysicalMeasurements((measure_1, measure_2)),
    )

    expected_block = ir.Block(argtypes=(types.MethodType,))
    expected_block.stmts.append(py.Constant(10))
    expected_block.stmts.append(ilist.New((measure_1, measure_2)))

    result = TransversalRewritePass(
        _PASS_DIALECTS, transversal_location_map=trivial_map
    ).unsafe_run(method)

    assert result.has_done_something
    assert_nodes(method.callable_region.blocks[0], expected_block)


def test_pass_rewrite_star_rz():
    rotation_angle = py.Constant(0.5)
    current_state = ir.TestValue()
    method = _make_method(
        rotation_angle,
        move.StarRz(
            current_state,
            rotation_angle.result,
            location_addresses=(LocationAddress(0, 0),),
            qubit_indices=(4, 5, 6),
        ),
    )

    expected_block = ir.Block(argtypes=(types.MethodType,))
    expected_block.stmts.append(exp_angle := py.Constant(0.5))
    expected_block.stmts.append(
        theta_star := func.Invoke(
            (exp_angle.result,), callee=transversal.steane_star_theta
        )
    )
    expected_block.stmts.append(
        move.LocalRz(
            current_state,
            theta_star.result,
            location_addresses=(
                LocationAddress(0, 14),
                LocationAddress(0, 15),
                LocationAddress(0, 16),
            ),
        )
    )

    result = TransversalRewritePass(
        _PASS_DIALECTS, transversal_location_map=star_map
    ).unsafe_run(method)

    assert result.has_done_something
    assert_nodes(method.callable_region.blocks[0], expected_block)
