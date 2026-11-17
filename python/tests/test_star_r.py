import math

import bloqade.squin as squin
import numpy as np
import pytest
from bloqade.types import QubitType
from kirin import ir, types
from kirin.dialects import func, ilist, py
from kirin.exception import StaticCheckError

from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import move, place, steane713
from bloqade.lanes.dialects.steane713 import extensions
from bloqade.lanes.dialects.steane713.extensions.stmts import StarR
from bloqade.lanes.dialects.steane713.star import VALID_STEANE_STAR_SUPPORTS
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.rewrite.transversal import (
    steane_star_physical_axis,
    steane_star_physical_theta,
    steane_star_r_physical,
    steane_star_theta,
)
from bloqade.lanes.transform import LogicalPipeline
from bloqade.lanes.types import StateType

from .validation._steane_kernel import kernel as steane_kernel

DEFAULT_SUPPORT_ADDRESSES = (
    LocationAddress(0, 4),
    LocationAddress(0, 5),
    LocationAddress(0, 6),
)


def _physical(kernel) -> ir.Method:
    return LogicalPipeline(
        transversal_rewrite=True,
        placement_strategy=LogicalPlacementStrategyNoHome(),
    ).emit(kernel, no_raise=False)


def _support_rotations(physical_move, addresses=DEFAULT_SUPPORT_ADDRESSES):
    return [
        stmt
        for stmt in physical_move.callable_region.walk()
        if isinstance(stmt, move.LocalR) and stmt.location_addresses == addresses
    ]


def _callee(value: ir.SSAValue) -> ir.Method:
    owner = value.owner
    assert isinstance(owner, func.Invoke), owner
    return owner.callee


def _const(value: ir.SSAValue) -> float:
    owner = value.owner
    assert isinstance(owner, py.Constant), owner
    return float(owner.value.unwrap())


# --- construction and support ----------------------------------------------


def test_star_r_dialect_uses_turns_and_default_support():
    axis, theta = 0.25, 1 / 32

    @steane_kernel
    def kernel():
        reg = squin.qalloc(1)
        extensions.star_r(axis, theta, reg)
        steane713.terminal_measure(reg)

    star_nodes = [
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarR)
    ]
    assert len(star_nodes) == 1
    assert star_nodes[0].qubit_indices == (4, 5, 6)
    assert _const(star_nodes[0].axis_angle) == pytest.approx(axis)
    assert _const(star_nodes[0].rotation_angle) == pytest.approx(theta)


def test_star_r_keyword_names_match_native_r():
    """Kirin matches keywords against the statement's fields, so the wrapper's
    parameter names must be the field names -- as on native ``r``."""

    @steane_kernel
    def kernel():
        reg = squin.qalloc(1)
        extensions.star_r(axis_angle=0.5, rotation_angle=0.125, qubits=reg)
        extensions.star_rz(rotation_angle=0.125, qubits=reg)
        steane713.terminal_measure(reg)

    star = next(s for s in kernel.callable_region.walk() if isinstance(s, StarR))
    assert _const(star.axis_angle) == pytest.approx(0.5)


def test_star_r_dialect_applies_to_multiple_logical_qubits():
    @steane_kernel
    def kernel():
        reg = squin.qalloc(2)
        extensions.star_r(0.0, 1 / 32, reg)
        steane713.terminal_measure(reg)

    star_nodes = [
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarR)
    ]
    assert len(star_nodes) == 1
    assert isinstance(star_nodes[0].qubits.owner, ilist.New)
    assert len(star_nodes[0].qubits.owner.values) == 2


@pytest.mark.parametrize("support", sorted(VALID_STEANE_STAR_SUPPORTS))
def test_star_r_accepts_steane_weight_three_support(support):
    @steane_kernel
    def kernel():
        reg = squin.qalloc(1)
        extensions.star_r(0.0, 0.125, reg, qubit_indices=support)
        steane713.terminal_measure(reg)

    star = next(
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarR)
    )
    assert star.qubit_indices == support


def test_star_r_rejects_invalid_support():
    axis = ir.TestValue(type=types.Float)
    angle = ir.TestValue(type=types.Float)
    reg = ir.TestValue(type=ilist.IListType[QubitType, types.Any])
    stmt = StarR(axis, angle, reg, qubit_indices=(0, 1, 2))
    with pytest.raises(StaticCheckError, match="qubit_indices"):
        stmt.check()


# --- physical lowering -------------------------------------------------------


def test_star_r_x_axis_physical_lowering():
    theta = 1 / 32

    @steane_kernel
    def kernel():
        reg = squin.qalloc(1)
        extensions.star_r(0.0, theta, reg)
        steane713.terminal_measure(reg)

    physical_move = _physical(kernel)

    assert not any(
        isinstance(stmt, move.StarR) for stmt in physical_move.callable_region.walk()
    )
    (local_r,) = _support_rotations(physical_move)
    assert _const(local_r.axis_angle) == 0.0
    angle_owner = local_r.rotation_angle.owner
    assert isinstance(angle_owner, func.Invoke)
    assert angle_owner.callee is steane_star_theta
    assert _const(angle_owner.inputs[0]) == pytest.approx(theta)


@pytest.mark.parametrize(
    ("logical_axis", "physical_axis"),
    [(0.0, 0.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25)],
)
def test_star_r_quarter_turn_axis_is_negated(logical_axis, physical_axis):
    """On the STAR support Y_L = -Ybar, so physical axis = -(logical axis)."""

    @steane_kernel
    def kernel():
        reg = squin.qalloc(1)
        extensions.star_r(logical_axis, 0.125, reg)
        steane713.terminal_measure(reg)

    (local_r,) = _support_rotations(_physical(kernel))
    assert _const(local_r.axis_angle) % 1.0 == pytest.approx(physical_axis)
    assert _callee(local_r.rotation_angle) is steane_star_theta


def test_star_r_custom_support_survives_prior_gate_and_lowering():
    @steane_kernel
    def kernel():
        reg = squin.qalloc(1)
        squin.x(reg[0])
        extensions.star_r(0.0, 0.125, reg, qubit_indices=(0, 2, 4))
        steane713.terminal_measure(reg)

    (local_r,) = _support_rotations(
        _physical(kernel),
        (LocationAddress(0, 0), LocationAddress(0, 2), LocationAddress(0, 4)),
    )
    assert _const(local_r.axis_angle) == 0.0


def test_star_r_parameterized_angle_remains_symbolic():
    @steane_kernel
    def kernel(theta: float):
        reg = squin.qalloc(1)
        extensions.star_r(0.0, theta, reg)
        steane713.terminal_measure(reg)

    (local_r,) = _support_rotations(_physical(kernel))
    assert _callee(local_r.rotation_angle) is steane_star_theta


def test_star_r_parameterized_axis_goes_through_physical_kernels():
    @steane_kernel
    def kernel(axis: float):
        reg = squin.qalloc(1)
        extensions.star_r(axis, 0.125, reg)
        steane713.terminal_measure(reg)

    (local_r,) = _support_rotations(_physical(kernel))
    assert _callee(local_r.axis_angle) is steane_star_physical_axis
    assert _callee(local_r.rotation_angle) is steane_star_physical_theta


@pytest.mark.parametrize(
    ("axis", "theta"),
    [(-0.1, 0.05), (0.15, -0.03), (-0.6, 0.2), (0.37, 0.11), (-0.9, -0.07)],
)
def test_symbolic_star_r_kernels_match_compile_time_folding(axis, theta):
    """The kernels a symbolic axis lowers to compute the same physical axis
    and angle as the constant-folding path, so the two paths cannot drift.
    ``axis`` is in EliminateRz's convention (the negated logical axis)."""
    physical_axis = steane_star_physical_axis(axis)
    physical_theta = steane_star_physical_theta(physical_axis, theta)

    folded_axis, folded_theta = steane_star_r_physical(axis, theta)
    # Axes are equal mod one turn: compare the wrapped difference.
    assert (physical_axis - folded_axis + 0.5) % 1.0 - 0.5 == pytest.approx(
        0.0, abs=1e-12
    )
    assert physical_theta == pytest.approx(folded_theta, abs=1e-12)


def test_star_r_without_transversal_rewrite_remains_logical_move_statement():
    @steane_kernel
    def kernel():
        reg = squin.qalloc(1)
        extensions.star_r(0.25, 1 / 32, reg)
        steane713.terminal_measure(reg)

    logical_move = LogicalPipeline(
        transversal_rewrite=False,
        placement_strategy=LogicalPlacementStrategyNoHome(),
    ).emit(kernel, no_raise=False)
    star_nodes = [
        stmt
        for stmt in logical_move.callable_region.walk()
        if isinstance(stmt, move.StarR)
    ]
    assert len(star_nodes) == 1
    assert star_nodes[0].qubit_indices == (4, 5, 6)
    assert star_nodes[0].location_addresses == (LocationAddress(0, 0, 0),)
    # no transversal rewrite: the axis stays logical, un-negated
    assert _const(star_nodes[0].axis_angle) == pytest.approx(0.25)


def _metrics_path(kernel) -> ir.Method:
    from bloqade.lanes.heuristics.logical import layout as logical_layout
    from bloqade.lanes.metrics import _logical_squin_to_move
    from bloqade.lanes.transform import transversal_rewrites

    return transversal_rewrites(
        _logical_squin_to_move(
            kernel,
            layout_heuristic=logical_layout.LogicalLayoutHeuristic(),
            placement_strategy=LogicalPlacementStrategyNoHome(),
            transversal_rewrite=True,
        )
    )


def _benchmark_path(kernel) -> ir.Method:
    from benchmarks.harness.runner import _squin_to_move

    from bloqade.lanes.heuristics.logical import layout as logical_layout
    from bloqade.lanes.transform import transversal_rewrites

    return transversal_rewrites(
        _squin_to_move(
            kernel,
            layout_heuristic=logical_layout.LogicalLayoutHeuristic(),
            placement_strategy=LogicalPlacementStrategyNoHome(),
            logical_initialize=True,
            transversal_rewrite=True,
        )
    )


@pytest.mark.parametrize("compile_path", [_metrics_path, _benchmark_path])
@pytest.mark.parametrize("logical_axis", [0.0, 0.25, 0.5, 0.75])
def test_hand_built_transversal_paths_lower_star_r_like_the_pipeline(
    compile_path, logical_axis
):
    """metrics and the benchmark runner assemble the logical stages by hand
    and then call transversal_rewrites. RewriteStarR assumes the native stage
    applied the Steane axis convention, so they must lower StarR exactly as
    LogicalPipeline(transversal_rewrite=True) does."""

    @steane_kernel
    def kernel():
        reg = squin.qalloc(1)
        extensions.star_r(logical_axis, 0.125, reg)
        steane713.terminal_measure(reg)

    def support_axis(physical_move) -> float:
        (local_r,) = _support_rotations(physical_move)
        return _const(local_r.axis_angle) % 1.0

    assert support_axis(compile_path(kernel)) == pytest.approx(
        support_axis(_physical(kernel))
    )


def test_metrics_compiles_with_the_transversal_conventions(monkeypatch):
    """Metrics' noisy-physical path calls transversal_rewrites, so it must ask
    the native stage for the Steane conventions."""
    from bloqade.lanes import metrics

    seen: dict[str, object] = {}

    class _Stop(Exception):
        pass

    def fake_logical_squin_to_move(mt, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(metrics, "_logical_squin_to_move", fake_logical_squin_to_move)
    m = metrics.Metrics(arch_spec=None)
    with pytest.raises(_Stop):
        m._compile_to_noisy_physical_squin(
            object(),  # type: ignore[arg-type]
            placement_strategy=LogicalPlacementStrategyNoHome(),
        )
    assert seen["transversal_rewrite"] is True


# --- Rz frame absorption -----------------------------------------------------


def _emits_physical_rz(physical_move) -> bool:
    return any(
        isinstance(stmt, (move.LocalRz, move.GlobalRz))
        for stmt in physical_move.callable_region.walk()
    )


def test_star_r_absorbs_pending_s_without_emitting_rz():
    @steane_kernel
    def kernel():
        reg = squin.qalloc(1)
        squin.s(reg[0])
        extensions.star_r(0.0, 0.125, reg)
        steane713.terminal_measure(reg)

    physical_move = _physical(kernel)
    assert not _emits_physical_rz(physical_move)
    (local_r,) = _support_rotations(physical_move)
    # logical S is transversal S^dag = native Rz(-1/4); axis 0 - (-1/4)
    assert _const(local_r.axis_angle) % 1.0 == pytest.approx(0.25)


def test_star_r_on_other_qubit_leaves_frame_for_its_next_gate():
    """The repro from the review: today the StarRx drains q0's frame into a
    LocalRz on q0's atoms even though the gadget is on q1."""

    @steane_kernel
    def kernel():
        reg = squin.qalloc(2)
        squin.x(reg[0])
        squin.x(reg[1])
        squin.s(reg[0])
        extensions.star_r(0.0, 0.125, ilist.IList([reg[1]]))
        squin.x(reg[0])
        steane713.terminal_measure(reg)

    assert not _emits_physical_rz(_physical(kernel))


# --- end-to-end: compiled gadget vs the intended logical gate ----------------
#
# 7-qubit Steane model in lanes' labeling, post-selected on a trivial syndrome
# (projection onto the code space). Xbar = X^7, Zbar = Z^7, Ybar = i Xbar Zbar.

_I2 = np.eye(2)
_X = np.array([[0, 1], [1, 0]], complex)
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1.0, -1.0]).astype(complex)


def _kron(ops):
    out = np.array([[1.0]], complex)
    for op in ops:
        out = np.kron(out, op)
    return out


def _code_basis() -> np.ndarray:
    words = {0}
    for line in VALID_STEANE_STAR_SUPPORTS:
        mask = sum(1 << (6 - j) for j in set(range(7)) - set(line))
        words |= {w ^ mask for w in words}
    zero = np.zeros(2**7, complex)
    zero[list(words)] = 1 / math.sqrt(len(words))
    return np.stack([zero, _kron([_X] * 7) @ zero], axis=1)


def _r(axis, angle):  # turns
    n = math.cos(math.tau * axis) * _X + math.sin(math.tau * axis) * _Y
    return math.cos(math.pi * angle) * _I2 - 1j * math.sin(math.pi * angle) * n


def _rz(angle):  # turns
    return math.cos(math.pi * angle) * _I2 - 1j * math.sin(math.pi * angle) * _Z


def _same_up_to_phase(a, b) -> bool:
    a = a / np.sqrt(np.linalg.det(a))
    b = b / np.sqrt(np.linalg.det(b))
    return 1 - abs(np.trace(a.conj().T @ b)) / 2 < 1e-9


def _star_theta_turns(theta_turns: float) -> float:
    theta = theta_turns * math.tau
    return (
        -math.copysign(2 * math.atan(abs(math.tan(theta / 2)) ** (1 / 3)), theta)
        / math.tau
    )


def _numeric(value: ir.SSAValue) -> float:
    owner = value.owner
    if isinstance(owner, py.Constant):
        return float(owner.value.unwrap())
    assert isinstance(owner, func.Invoke) and owner.callee is steane_star_theta
    return _star_theta_turns(_numeric(owner.inputs[0]))


@pytest.mark.parametrize("axis", [0.0, 0.25, 0.5, 0.75, 0.1, 0.3, 0.6, 0.9])
@pytest.mark.parametrize(
    ("prefix", "native_frame", "logical_prefix"),
    [
        ("none", 0.0, _I2),
        ("s", -0.25, np.diag([1, 1j])),  # logical S = transversal S^dag
        ("z", 0.5, _Z),
    ],
)
def test_compiled_star_r_implements_the_logical_gate(
    axis, prefix, native_frame, logical_prefix
):
    theta = 0.07

    @steane_kernel
    def kernel():
        reg = squin.qalloc(1)
        if prefix == "s":
            squin.s(reg[0])
        elif prefix == "z":
            squin.z(reg[0])
        extensions.star_r(axis, theta, reg)
        steane713.terminal_measure(reg)

    physical_move = _physical(kernel)
    assert not _emits_physical_rz(physical_move)
    (local_r,) = _support_rotations(physical_move)
    phys_axis = _numeric(local_r.axis_angle)
    phys_theta = _numeric(local_r.rotation_angle)

    # EliminateRz moved the prefix's Rz past the gadget and dropped it; put it
    # back to compare against the full logical circuit.
    gadget = _kron(
        [_r(phys_axis, phys_theta) if j in (4, 5, 6) else _I2 for j in range(7)]
    )
    compiled = _kron([_rz(native_frame)] * 7) @ gadget
    basis = _code_basis()
    logical = basis.conj().T @ compiled @ basis
    assert _same_up_to_phase(logical, _r(axis, theta) @ logical_prefix)


# --- lowered IR support checks -----------------------------------------------


@pytest.mark.parametrize("stage", ["place", "move"])
@pytest.mark.parametrize("support", sorted(VALID_STEANE_STAR_SUPPORTS))
def test_star_r_lowered_ir_accepts_valid_support(stage, support):
    state = ir.TestValue(type=StateType)
    axis = ir.TestValue(type=types.Float)
    angle = ir.TestValue(type=types.Float)
    if stage == "place":
        stmt = place.StarR(state, axis, angle, qubits=(0,), qubit_indices=support)
    else:
        stmt = move.StarR(
            state,
            axis,
            angle,
            location_addresses=(LocationAddress(0, 0),),
            qubit_indices=support,
        )
    stmt.check()
    assert stmt.qubit_indices == support


@pytest.mark.parametrize("stage", ["place", "move"])
@pytest.mark.parametrize("support", [(0, 1, 2), (4, 4, 6), (4, 5, 7)])
def test_star_r_lowered_ir_rejects_invalid_support(stage, support):
    state = ir.TestValue(type=StateType)
    axis = ir.TestValue(type=types.Float)
    angle = ir.TestValue(type=types.Float)
    if stage == "place":
        stmt = place.StarR(state, axis, angle, qubits=(0,), qubit_indices=support)
    else:
        stmt = move.StarR(
            state,
            axis,
            angle,
            location_addresses=(LocationAddress(0, 0),),
            qubit_indices=support,
        )
    with pytest.raises(StaticCheckError, match="qubit_indices"):
        stmt.check()
