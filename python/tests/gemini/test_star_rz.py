import math
from typing import Any, cast

import bloqade.squin as squin
import pytest
from kirin.dialects import py
from kirin.ir.exception import ValidationError

import bloqade.gemini as gemini
from bloqade.gemini.logical.dialects.operations.stmts import (
    StarRz,
    validate_steane_star_support,
)
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import move
from bloqade.lanes.logical_mvp import compile_squin_to_move

VALID_STEANE_STAR_SUPPORTS = {
    (0, 1, 5),
    (0, 2, 4),
    (0, 3, 6),
    (1, 2, 6),
    (1, 3, 4),
    (2, 3, 5),
    (4, 5, 6),
}


def _star_theta(theta: float) -> float:
    magnitude = 2 * math.atan(abs(math.tan(theta / 2)) ** (1 / 3))
    return -math.copysign(magnitude, theta)


def test_star_rz_public_api_single_qubit_lowers_to_statement():
    @gemini.logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = squin.qalloc(1)
        gemini.logical.star_rz(math.pi / 16, reg[0])
        gemini.logical.terminal_measure(reg)

    star_nodes = [
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarRz)
    ]
    assert len(star_nodes) == 1
    assert star_nodes[0].qubit_indices == (4, 5, 6)


def test_star_rz_broadcast_lowers_to_statement():
    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(1)
        gemini.logical.broadcast.star_rz(math.pi / 16, reg)
        gemini.logical.terminal_measure(reg)

    star_nodes = [
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarRz)
    ]
    assert len(star_nodes) == 1
    assert star_nodes[0].qubit_indices == (4, 5, 6)


@pytest.mark.parametrize("support", sorted(VALID_STEANE_STAR_SUPPORTS))
def test_star_rz_accepts_all_steane_weight_three_z_lines(support):
    @gemini.logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = squin.qalloc(1)
        gemini.logical.broadcast.star_rz(math.pi / 16, reg, qubit_indices=support)
        gemini.logical.terminal_measure(reg)

    star_node = next(
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarRz)
    )
    assert star_node.qubit_indices == support


def test_star_rz_accepts_positional_support():
    @gemini.logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = squin.qalloc(1)
        gemini.logical.broadcast.star_rz(math.pi / 16, reg, (1, 2, 6))
        gemini.logical.terminal_measure(reg)

    star_node = next(
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarRz)
    )
    assert star_node.qubit_indices == (1, 2, 6)


@pytest.mark.parametrize(
    "support",
    [(0, 1, 2), (4, 4, 6), (-1, 5, 6), (4, 5, 7), (4.0, 5, 6)],
)
def test_star_rz_rejects_invalid_steane_supports(support):
    with pytest.raises(ValueError, match="qubit_indices"):
        validate_steane_star_support(support)


def test_star_rz_statement_check_rejects_invalid_support():
    with pytest.raises(ValidationError, match="qubit_indices"):

        @gemini.logical.kernel(aggressive_unroll=True, no_raise=False)
        def kernel():
            reg = squin.qalloc(1)
            gemini.logical.star_rz(
                math.pi / 16,
                reg[0],
                qubit_indices=(0, 1, 2),
            )
            gemini.logical.terminal_measure(reg)


def test_star_rz_pipeline_produces_physical_local_rz_after_transversal_rewrite():
    theta = math.pi / 16

    @gemini.logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = squin.qalloc(1)
        gemini.logical.star_rz(theta, reg[0])
        gemini.logical.terminal_measure(reg)

    physical_move = compile_squin_to_move(
        kernel,
        transversal_rewrite=True,
        no_raise=False,
        insert_return_moves=False,
    )

    assert not any(
        isinstance(stmt, move.StarRz) for stmt in physical_move.callable_region.walk()
    )
    local_rz_nodes = [
        stmt
        for stmt in physical_move.callable_region.walk()
        if isinstance(stmt, move.LocalRz)
    ]
    star_rz_nodes = [
        stmt
        for stmt in local_rz_nodes
        if tuple(stmt.location_addresses)
        == (
            LocationAddress(0, 4),
            LocationAddress(0, 5),
            LocationAddress(0, 6),
        )
    ]
    assert len(star_rz_nodes) == 1
    angle_owner = star_rz_nodes[0].rotation_angle.owner
    assert isinstance(angle_owner, py.Constant)
    assert cast(Any, angle_owner.value).data == pytest.approx(_star_theta(theta))


def test_star_rz_pipeline_preserves_support_after_prior_single_qubit_gate():
    theta = math.pi / 16

    @gemini.logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = squin.qalloc(1)
        squin.h(reg[0])
        gemini.logical.star_rz(theta, reg[0], qubit_indices=(0, 2, 4))
        gemini.logical.terminal_measure(reg)

    physical_move = compile_squin_to_move(
        kernel,
        transversal_rewrite=True,
        no_raise=False,
        insert_return_moves=False,
    )

    local_rz_nodes = [
        stmt
        for stmt in physical_move.callable_region.walk()
        if isinstance(stmt, move.LocalRz)
        and tuple(stmt.location_addresses)
        == (
            LocationAddress(0, 0),
            LocationAddress(0, 2),
            LocationAddress(0, 4),
        )
    ]
    assert len(local_rz_nodes) == 1


def test_star_rz_without_transversal_rewrite_remains_logical_move_statement():
    @gemini.logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = squin.qalloc(1)
        gemini.logical.star_rz(math.pi / 16, reg[0])
        gemini.logical.terminal_measure(reg)

    logical_move = compile_squin_to_move(
        kernel,
        transversal_rewrite=False,
        no_raise=False,
        insert_return_moves=False,
    )

    star_nodes = [
        stmt
        for stmt in logical_move.callable_region.walk()
        if isinstance(stmt, move.StarRz)
    ]
    assert len(star_nodes) == 1
    assert star_nodes[0].qubit_indices == (4, 5, 6)
    assert star_nodes[0].location_addresses == (LocationAddress(0, 0, 0),)
