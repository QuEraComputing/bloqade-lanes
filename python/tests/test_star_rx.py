import bloqade.squin as squin
import pytest
from bloqade.types import QubitType
from kirin import ir, types
from kirin.dialects import func, ilist, py
from kirin.exception import StaticCheckError

from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import move, place, steane713
from bloqade.lanes.dialects.steane713 import extensions
from bloqade.lanes.dialects.steane713.extensions.stmts import StarRx
from bloqade.lanes.dialects.steane713.star import VALID_STEANE_STAR_SUPPORTS
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.rewrite.transversal import steane_star_theta
from bloqade.lanes.transform import LogicalPipeline
from bloqade.lanes.types import StateType

from .validation._steane_kernel import kernel as steane_kernel


def test_star_rx_dialect_uses_turns_and_default_support():
    theta = 1 / 32

    @steane_kernel
    def kernel():
        reg = squin.qalloc(1)
        extensions.star_rx(theta, reg)
        steane713.terminal_measure(reg)

    star_nodes = [
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarRx)
    ]
    assert len(star_nodes) == 1
    assert star_nodes[0].qubit_indices == (4, 5, 6)
    assert isinstance(star_nodes[0].rotation_angle.owner, py.Constant)
    assert star_nodes[0].rotation_angle.owner.value.unwrap() == pytest.approx(theta)


def test_star_rx_dialect_applies_to_multiple_logical_qubits():
    @steane_kernel
    def kernel():
        reg = squin.qalloc(2)
        extensions.star_rx(1 / 32, reg)
        steane713.terminal_measure(reg)

    star_nodes = [
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarRx)
    ]
    assert len(star_nodes) == 1
    assert isinstance(star_nodes[0].qubits.owner, ilist.New)
    assert len(star_nodes[0].qubits.owner.values) == 2


@pytest.mark.parametrize("support", sorted(VALID_STEANE_STAR_SUPPORTS))
def test_star_rx_accepts_steane_weight_three_support(support):
    @steane_kernel
    def kernel():
        reg = squin.qalloc(1)
        extensions.star_rx(0.125, reg, qubit_indices=support)
        steane713.terminal_measure(reg)

    star = next(
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarRx)
    )
    assert star.qubit_indices == support


def test_star_rx_rejects_invalid_support():
    angle = ir.TestValue(type=types.Float)
    reg = ir.TestValue(type=ilist.IListType[QubitType, types.Any])
    stmt = StarRx(angle, reg, qubit_indices=(0, 1, 2))
    with pytest.raises(StaticCheckError, match="qubit_indices"):
        stmt.check()


def test_star_rx_physical_lowering():
    theta = 1 / 32

    @steane_kernel
    def kernel():
        reg = squin.qalloc(1)
        extensions.star_rx(theta, reg)
        steane713.terminal_measure(reg)

    physical_move = LogicalPipeline(
        transversal_rewrite=True,
        placement_strategy=LogicalPlacementStrategyNoHome(),
    ).emit(kernel, no_raise=False)

    assert not any(
        isinstance(stmt, move.StarRx) for stmt in physical_move.callable_region.walk()
    )
    local_rx = [
        stmt
        for stmt in physical_move.callable_region.walk()
        if isinstance(stmt, move.LocalR)
        and stmt.location_addresses
        == (LocationAddress(0, 4), LocationAddress(0, 5), LocationAddress(0, 6))
    ]
    assert len(local_rx) == 1
    assert isinstance(local_rx[0].axis_angle.owner, py.Constant)
    assert local_rx[0].axis_angle.owner.value.unwrap() == 0.0
    angle_owner = local_rx[0].rotation_angle.owner
    assert isinstance(angle_owner, func.Invoke)
    assert angle_owner.callee is steane_star_theta
    assert isinstance(angle_owner.inputs[0].owner, py.Constant)
    assert angle_owner.inputs[0].owner.value.unwrap() == pytest.approx(theta)


def test_star_rx_custom_support_survives_prior_gate_and_lowering():
    @steane_kernel
    def kernel():
        reg = squin.qalloc(1)
        squin.x(reg[0])
        extensions.star_rx(0.125, reg, qubit_indices=(0, 2, 4))
        steane713.terminal_measure(reg)

    physical_move = LogicalPipeline(
        transversal_rewrite=True,
        placement_strategy=LogicalPlacementStrategyNoHome(),
    ).emit(kernel, no_raise=False)
    assert any(
        isinstance(stmt, move.LocalR)
        and stmt.location_addresses
        == (LocationAddress(0, 0), LocationAddress(0, 2), LocationAddress(0, 4))
        and isinstance(stmt.axis_angle.owner, py.Constant)
        and stmt.axis_angle.owner.value.unwrap() == 0.0
        for stmt in physical_move.callable_region.walk()
    )


def test_star_rx_parameterized_angle_remains_symbolic():
    @steane_kernel
    def kernel(theta: float):
        reg = squin.qalloc(1)
        extensions.star_rx(theta, reg)
        steane713.terminal_measure(reg)

    physical_move = LogicalPipeline(
        transversal_rewrite=True,
        placement_strategy=LogicalPlacementStrategyNoHome(),
    ).emit(kernel, no_raise=False)
    assert any(
        isinstance(stmt, move.LocalR)
        and isinstance(stmt.rotation_angle.owner, func.Invoke)
        and stmt.rotation_angle.owner.callee is steane_star_theta
        for stmt in physical_move.callable_region.walk()
    )


def test_star_rx_without_transversal_rewrite_remains_logical_move_statement():
    @steane_kernel
    def kernel():
        reg = squin.qalloc(1)
        extensions.star_rx(1 / 32, reg)
        steane713.terminal_measure(reg)

    logical_move = LogicalPipeline(
        transversal_rewrite=False,
        placement_strategy=LogicalPlacementStrategyNoHome(),
    ).emit(kernel, no_raise=False)
    star_nodes = [
        stmt
        for stmt in logical_move.callable_region.walk()
        if isinstance(stmt, move.StarRx)
    ]
    assert len(star_nodes) == 1
    assert star_nodes[0].qubit_indices == (4, 5, 6)
    assert star_nodes[0].location_addresses == (LocationAddress(0, 0, 0),)


@pytest.mark.parametrize("stage", ["place", "move"])
@pytest.mark.parametrize("support", sorted(VALID_STEANE_STAR_SUPPORTS))
def test_star_rx_lowered_ir_accepts_valid_support(stage, support):
    state = ir.TestValue(type=StateType)
    angle = ir.TestValue(type=types.Float)
    if stage == "place":
        stmt = place.StarRx(state, angle, qubits=(0,), qubit_indices=support)
    else:
        stmt = move.StarRx(
            state,
            angle,
            location_addresses=(LocationAddress(0, 0),),
            qubit_indices=support,
        )
    stmt.check()
    assert stmt.qubit_indices == support


@pytest.mark.parametrize("stage", ["place", "move"])
@pytest.mark.parametrize("support", [(0, 1, 2), (4, 4, 6), (4, 5, 7)])
def test_star_rx_lowered_ir_rejects_invalid_support(stage, support):
    state = ir.TestValue(type=StateType)
    angle = ir.TestValue(type=types.Float)
    if stage == "place":
        stmt = place.StarRx(state, angle, qubits=(0,), qubit_indices=support)
    else:
        stmt = move.StarRx(
            state,
            angle,
            location_addresses=(LocationAddress(0, 0),),
            qubit_indices=support,
        )
    with pytest.raises(StaticCheckError, match="qubit_indices"):
        stmt.check()
