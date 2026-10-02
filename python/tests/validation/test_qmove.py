from typing import Any, Literal

from bloqade.types import Qubit, QubitType
from kirin import ir, rewrite, types as kirin_types
from kirin.analysis import const
from kirin.dialects import func, ilist, py, scf
from kirin.validation import ValidationSuite
from tests._qmove_helpers import first_of

from bloqade import squin
from bloqade.gemini import physical
from bloqade.gemini.common.dialects import arrange
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.dialects.qmove import Effects, Frame, FrameShape
from bloqade.lanes.rewrite.native2qmove import RewriteNativeToQMove
from bloqade.lanes.rewrite.qmove_state import thread_method
from bloqade.lanes.transform.qmove_frontend import NativeProgram, lower_to_native
from bloqade.lanes.validation.qmove import get_qmove_validation
from bloqade.lanes.validation.spectator import (
    SingleZonePolicy,
    SpectatorPolicy,
    ZonedPolicy,
)

ARCH = get_arch_spec()
A, B = LocationAddress(0, 0), LocationAddress(1, 0)  # an entangling pair
FAR = LocationAddress(5, 0)
Z0 = ZoneAddress(0)


def _build(
    kernel: ir.Method, subroutines: dict[ir.Method, Frame | None] | None = None
) -> NativeProgram:
    subroutines = subroutines or {}
    program = lower_to_native(kernel, subroutines, ARCH)
    clones = frozenset(program.subroutines.values())
    rewrite.Walk(RewriteNativeToQMove(clones)).rewrite(program.entry.code)
    thread_method(program.entry, subroutine=False)
    for original, clone in program.subroutines.items():
        rewrite.Walk(RewriteNativeToQMove(clones)).rewrite(clone.code)
        thread_method(clone, subroutine=True, frame=subroutines[original])
    return program


def _messages(method: ir.Method, policy: SpectatorPolicy | None = None) -> list[str]:
    validation = get_qmove_validation(ARCH, policy or ZonedPolicy())
    result = ValidationSuite([validation]).validate(method)
    return [str(err.args[0]) for errs in result.errors.values() for err in errs]


@squin.kernel
def sub(qs: ilist.IList[Qubit, Literal[2]]):
    squin.cz(qs[0], qs[1])


@squin.kernel
def main():
    qs = squin.qalloc(3)
    if squin.is_one(squin.measure(qs[0])):
        squin.x(qs[1])
    for q in qs:
        squin.h(q)
    sub(ilist.IList([qs[0], qs[1]]))
    return squin.broadcast.measure(qs)


PAIR = Frame(FrameShape(((0, 2),)), (A, B), Effects(cz_zones=frozenset({Z0})))


def test_lowered_program_is_valid():
    program = _build(main, {sub: PAIR})
    assert _messages(program.entry) == []
    assert _messages(program.subroutines[sub], SingleZonePolicy()) == []


def test_hole_subroutine_is_valid():
    program = _build(main, {sub: None})
    assert _messages(program.subroutines[sub]) == []


# --- V1 ---------------------------------------------------------------------


def test_v1_double_consumption():
    program = _build(main)
    measure = first_of(program.entry, qmove.Measure)
    qmove.Measure(measure.current_state, measure.qubits).insert_after(measure)
    assert any("consumed twice on one path" in m for m in _messages(program.entry))


def test_v1_dropped_update():
    program = _build(main)
    first_of(program.entry, move.Store).delete()
    assert any("never used" in m for m in _messages(program.entry))


def test_v1_state_captured_into_a_loop():
    program = _build(main)
    loop = first_of(program.entry, scf.For)
    gate = first_of(loop, qmove.R)
    qmove.Rz(loop.initializers[0], gate.rotation_angle, gate.qubits).insert_before(gate)
    assert any("inside a loop body" in m for m in _messages(program.entry))


def test_v1_one_use_per_if_arm_is_fine():
    program = _build(main)
    branch = first_of(program.entry, scf.IfElse)
    else_yield = branch.else_body.blocks[0].last_stmt
    assert isinstance(else_yield, scf.Yield)
    assert len(else_yield.values[0].uses) >= 2  # then-arm gate and else-arm yield
    assert _messages(program.entry) == []


# --- V2 ---------------------------------------------------------------------


def _insert_load_store_before(anchor: ir.Statement | None) -> None:
    assert anchor is not None
    load = move.Load()
    load.insert_before(anchor)
    move.Store(load.result).insert_after(load)


def test_v2_load_inside_a_region():
    program = _build(main)
    branch = first_of(program.entry, scf.IfElse)
    _insert_load_store_before(branch.then_body.blocks[0].first_stmt)
    messages = _messages(program.entry)
    assert "V2: load must be in the method's top-level block" in messages
    assert "V2: store must be in the method's top-level block" in messages


def test_v2_mixing_load_with_enter():
    program = _build(main, {sub: None})
    clone = program.subroutines[sub]
    _insert_load_store_before(clone.callable_region.blocks[0].first_stmt)
    assert "V2: method uses both load/store and enter/exit" in _messages(clone)


# --- V3 ---------------------------------------------------------------------


def _method(*stmts: ir.Statement) -> ir.Method:
    @squin.kernel
    def stub():
        return None

    out = stub.similar()
    out.code = func.Function(
        sym_name="stub",
        signature=func.Signature(inputs=(), output=kirin_types.NoneType),
        body=ir.Region(ir.Block([*stmts, func.Return()])),
    )
    return out


def _qubits(n: int) -> ir.TestValue:
    return ir.TestValue(ilist.IListType[QubitType, kirin_types.Literal(n)])


def test_v3_cz_lengths_must_match():
    load = move.Load()
    cz = qmove.CZ(load.result, _qubits(1), _qubits(2))
    messages = _messages(_method(load, cz, move.Store(cz.result)))
    assert messages == ["V3: cz has 1 controls but 2 targets"]


@physical.kernel(verify=False)
def bad_permute():
    q = squin.qalloc(2)
    arrange.permute(q, ilist.IList([0, 0]))


def test_v3_perm_must_be_a_permutation():
    assert "V3: perm (0, 0) is not a permutation" in _messages(
        _build(bad_permute).entry
    )


def test_v3_move_to_needs_one_location_per_qubit():
    # From Python, TypeInfer unifies MoveTo's Len and RefineQubitTypes reports
    # the mismatch first; this check covers hand-built IR.
    locations = py.Constant(ilist.IList([FAR]))
    locations.result.hints["const"] = const.Value(ilist.IList([FAR]))
    load = move.Load()
    move_to = qmove.MoveTo(load.result, _qubits(2), locations.result)
    messages = _messages(_method(locations, load, move_to, move.Store(move_to.result)))
    assert messages == ["V3: move_to has 1 locations for 2 qubits"]


# --- call rules -------------------------------------------------------------


def test_func_invoke_of_a_subroutine_is_rejected():
    program = _build(main, {sub: None})
    invoke = first_of(program.entry, qmove.Invoke)
    func.Invoke(tuple(invoke.inputs), callee=invoke.callee).insert_after(invoke)
    messages = _messages(program.entry)
    assert "subroutine sub must be called with qmove.invoke" in messages


def test_qmove_invoke_of_a_non_subroutine_is_rejected():
    load = move.Load()
    call = qmove.Invoke(load.result, (), callee=main)
    messages = _messages(_method(load, call, move.Store(call.result)))
    assert messages == ["invoke target main is not a subroutine"]


# --- frame rules --------------------------------------------------------------


def _frame_messages(
    frame: Frame,
    policy: SpectatorPolicy | None = None,
    kernel: ir.Method = main,
    subroutine: ir.Method = sub,
) -> list[str]:
    program = _build(kernel, {subroutine: frame})
    return _messages(program.subroutines[subroutine], policy)


def test_f1_binding_size_and_distinctness():
    messages = _frame_messages(Frame(FrameShape(((0, 2),)), (A, A, B)))
    assert "F1: binding has 3 locations for 2 slots" in messages
    assert "F1: binding locations are not distinct" in messages


def test_f1_invalid_location_and_zone():
    frame = Frame(
        FrameShape(((0, 2),)),
        (A, LocationAddress(999, 0)),
        Effects(cz_zones=frozenset({ZoneAddress(7)})),
    )
    messages = _frame_messages(frame)
    assert any(m.startswith("F1: invalid location") for m in messages)
    assert "F1: zone 7 does not exist" in messages


def test_f2_shape_must_match_parameters():
    messages = _frame_messages(Frame(FrameShape(((0, 1),)), (A,)))
    assert "F2: frame slots ((0, 1),) do not match parameters ((0, 2),)" in messages


@squin.kernel
def any_length(qs: ilist.IList[Qubit, Any]):
    squin.h(qs[0])


@squin.kernel
def calls_any_length():
    qs = squin.qalloc(2)
    any_length(qs)


def test_f2_parameter_length_must_be_static():
    frame = Frame(FrameShape(((0, 2),)), (A, B))
    messages = _frame_messages(frame, kernel=calls_any_length, subroutine=any_length)
    assert any("needs IList[Qubit, Literal[N]]" in m for m in messages)


@physical.kernel(verify=False)
def mover(qs: ilist.IList[Qubit, Literal[2]]):
    arrange.move_to(ilist.IList([qs[0]]), ilist.IList([FAR]))


# physical.kernel inlines calls by default; inline=False keeps subroutine calls.
@physical.kernel(verify=False, inline=False)
def calls_mover():
    qs = squin.qalloc(2)
    mover(qs)


def test_f3_move_to_must_stay_in_the_frame():
    frame = Frame(FrameShape(((0, 2),)), (A, B))
    messages = _frame_messages(frame, kernel=calls_mover, subroutine=mover)
    assert (
        "F3: move_to target (zone 0, word 5, site 0) is outside the frame" in messages
    )


def test_f3_global_pulse_needs_permission():
    clone = _build(main, {sub: PAIR}).subroutines[sub]
    exit_ = first_of(clone, qmove.Exit)
    pulse = move.GlobalRz(exit_.current_state, ir.TestValue(kirin_types.Float))
    pulse.insert_before(exit_)
    exit_.current_state = pulse.result
    assert "F3: global pulse is not in the frame's effects" in _messages(clone)


@physical.kernel(verify=False)
def relabels(qs: ilist.IList[Qubit, Literal[2]]):
    arrange.permute(qs, ilist.IList([1, 0]))


@physical.kernel(verify=False)
def relabels_back(qs: ilist.IList[Qubit, Literal[2]]):
    arrange.permute(qs, ilist.IList([1, 0]))
    arrange.permute(qs, ilist.IList([1, 0]))


@physical.kernel(verify=False, inline=False)
def calls_relabels():
    qs = squin.qalloc(2)
    relabels(qs)
    relabels_back(qs)


def test_f4_relabels_must_compose_to_identity():
    frame = Frame(FrameShape(((0, 2),)), (A, B))
    program = _build(calls_relabels, {relabels: frame, relabels_back: frame})
    assert any(m.startswith("F4:") for m in _messages(program.subroutines[relabels]))
    assert _messages(program.subroutines[relabels_back]) == []


@physical.kernel(verify=False)
def inner(qs: ilist.IList[Qubit, Literal[2]]):
    squin.cz(qs[0], qs[1])


@physical.kernel(verify=False, inline=False)
def outer(qs: ilist.IList[Qubit, Literal[2]]):
    inner(qs)


@physical.kernel(verify=False, inline=False)
def calls_outer():
    qs = squin.qalloc(2)
    outer(qs)


def test_f5_nested_frame_must_fit_inside():
    outer_frame = Frame(FrameShape(((0, 2),)), (A, B))
    inner_frame = Frame(FrameShape(((0, 2),)), (A, FAR))
    program = _build(calls_outer, {outer: outer_frame, inner: inner_frame})
    messages = _messages(program.subroutines[outer])
    assert "F5: frame of inner is not inside this frame" in messages


def test_policy_problems_are_reported():
    frame = Frame(FrameShape(((0, 2),)), (A, B), Effects(global_pulses=True))
    assert "ZonedPolicy: subroutines may not use global pulses" in _frame_messages(
        frame
    )
