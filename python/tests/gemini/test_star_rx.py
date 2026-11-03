import math

import bloqade.squin as squin
import pytest
from kirin.dialects import func, ilist, py
from kirin.ir.exception import ValidationError

import bloqade.gemini as gemini
from bloqade.gemini import GeminiLogicalDevice
from bloqade.gemini.logical.dialects import extensions
from bloqade.gemini.logical.dialects.extensions.stmts import StarRx
from bloqade.gemini.star import VALID_STEANE_STAR_SUPPORTS
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.rewrite.transversal import steane_star_theta
from bloqade.lanes.transform import LogicalPipeline


def test_star_rx_public_api_uses_radians_and_default_support():
    theta = math.pi / 16

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(1)
        gemini.logical.extensions.star_rx(theta, reg[0])
        gemini.logical.terminal_measure(reg)

    star_nodes = [
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarRx)
    ]
    assert len(star_nodes) == 1
    assert star_nodes[0].qubit_indices == (4, 5, 6)
    assert isinstance(star_nodes[0].rotation_angle.owner, py.Constant)
    assert star_nodes[0].rotation_angle.owner.value.unwrap() == pytest.approx(
        theta / math.tau
    )


def test_star_rx_broadcast_applies_to_multiple_logical_qubits():
    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(2)
        gemini.logical.extensions.broadcast.star_rx(math.pi / 16, reg)
        gemini.logical.terminal_measure(reg)

    star_nodes = [
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarRx)
    ]
    assert len(star_nodes) == 1
    assert isinstance(star_nodes[0].qubits.owner, ilist.New)
    assert len(star_nodes[0].qubits.owner.values) == 2


@pytest.mark.parametrize("support", sorted(VALID_STEANE_STAR_SUPPORTS))
def test_star_rx_accepts_steane_weight_three_support(support):
    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(1)
        extensions.star_rx(0.125, reg, qubit_indices=support)
        gemini.logical.terminal_measure(reg)

    star = next(
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarRx)
    )
    assert star.qubit_indices == support


def test_star_rx_rejects_invalid_support():
    with pytest.raises(ValidationError, match="qubit_indices"):

        @gemini.logical.kernel(aggressive_unroll=True, no_raise=False)
        def kernel():
            reg = squin.qalloc(1)
            extensions.star_rx(0.125, reg, qubit_indices=(0, 1, 2))
            gemini.logical.terminal_measure(reg)


def test_star_rx_public_api_and_physical_lowering():
    theta = math.pi / 16

    @gemini.logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = squin.qalloc(1)
        gemini.logical.extensions.star_rx(theta, reg[0])
        gemini.logical.terminal_measure(reg)

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
    assert angle_owner.inputs[0].owner.value.unwrap() == pytest.approx(theta / math.tau)


def test_star_rx_custom_support_survives_prior_gate_and_lowering():
    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(1)
        squin.x(reg[0])
        extensions.star_rx(0.125, reg, qubit_indices=(0, 2, 4))
        gemini.logical.terminal_measure(reg)

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
    @gemini.logical.kernel(aggressive_unroll=True, verify=False)
    def kernel(theta: float):
        reg = squin.qalloc(1)
        gemini.logical.extensions.star_rx(theta, reg[0])
        gemini.logical.terminal_measure(reg)

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
    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(1)
        gemini.logical.extensions.star_rx(math.pi / 16, reg[0])
        gemini.logical.terminal_measure(reg)

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


def test_star_rx_is_rejected_by_hardware_validation():
    @gemini.logical.kernel()
    def kernel():
        reg = squin.qalloc(1)
        gemini.logical.extensions.star_rx(math.pi / 16, reg[0])
        gemini.logical.terminal_measure(reg)

    suite = GeminiLogicalDevice().validation_suite
    assert suite is not None
    result = suite.validate(kernel)
    assert not result.is_valid
    assert any(
        "gemini.logical.extensions.star_rx" in error.args[0]
        for errors in result.errors.values()
        for error in errors
    )
