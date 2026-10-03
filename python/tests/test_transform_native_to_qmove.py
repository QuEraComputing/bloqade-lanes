from typing import Literal

import pytest
from bloqade.types import Qubit
from kirin import ir
from kirin.dialects import func, ilist, scf
from kirin.ir.exception import ValidationErrorGroup
from kirin.validation import ValidationSuite
from tests._qmove_helpers import (
    assert_methods_match,
    erase_qmove,
    statements_of,
    top_level,
)

from bloqade import qubit, squin
from bloqade.gemini import physical
from bloqade.gemini.common.dialects import arrange, qubit as gemini_qubit
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.dialects.qmove import Effects, Frame, FrameShape, MachineFrame
from bloqade.lanes.transform import native_to_qmove
from bloqade.lanes.transform.native_to_qmove import (
    RECURSIVE_ALLOCATION_MESSAGE,
    NativeToQMove,
)
from bloqade.lanes.transform.qmove_frontend import lower_to_native
from bloqade.lanes.types import StateType
from bloqade.lanes.validation.qmove import get_qmove_validation
from bloqade.lanes.validation.qmove_input import ALLOCATION_MESSAGE
from bloqade.lanes.validation.spectator import SingleZonePolicy, ZonedPolicy

ARCH = get_arch_spec()
A, B = LocationAddress(0, 0), LocationAddress(1, 0)


@squin.kernel
def sub(qs: ilist.IList[Qubit, Literal[2]]):
    squin.cx(qs[0], qs[1])


@squin.kernel
def nested(qs: ilist.IList[Qubit, Literal[3]]):
    for q in qs:
        if squin.is_one(squin.measure(q)):
            squin.x(q)


@squin.kernel
def main():
    qs = squin.qalloc(3)
    squin.h(qs[0])
    if squin.is_one(squin.measure(qs[0])):
        squin.x(qs[1])
    else:
        squin.z(qs[1])
    for q in qs:
        squin.h(q)
    for i in range(2):
        squin.z(qs[i + 1])
    sub(ilist.IList([qs[0], qs[1]]))
    nested(qs)
    return squin.broadcast.measure(qs)


@physical.kernel(verify=False)
def arranged():
    a = gemini_qubit.new_at(0, 0, 0)
    b = gemini_qubit.new_at(0, 2, 0)
    arrange.move_to(ilist.IList([b]), ilist.IList([B]))
    squin.cz(a, b)
    arrange.permute(ilist.IList([a, b]), ilist.IList([1, 0]))
    if squin.is_one(squin.broadcast.measure(ilist.IList([a, b]))[0]):
        squin.x(b)
    return squin.broadcast.measure(ilist.IList([a, b]))


SUBROUTINES: dict[ir.Method, Frame | None] = {sub: None, nested: None}


def _callees(entry: ir.Method, kind: type[qmove.Invoke] | type[func.Invoke]):
    return {s.callee.sym_name: s.callee for s in statements_of(entry, kind)}


def test_entry_kernel_is_one_chain():
    out = NativeToQMove(ARCH, SUBROUTINES).emit(main)
    top = top_level(out)
    enter = top[0]
    assert isinstance(enter, qmove.Enter) and enter.frame == MachineFrame()
    assert isinstance(top[-2], qmove.Exit)
    assert len(statements_of(out, qmove.Enter)) == 1
    assert not statements_of(out, move.Load) and not statements_of(out, move.Store)


def test_subroutines_are_called_on_the_chain():
    out = NativeToQMove(ARCH, SUBROUTINES).emit(main)
    callees = _callees(out, qmove.Invoke)
    assert set(callees) == {"sub", "nested"}
    for callee in callees.values():
        assert isinstance(top_level(callee)[0], qmove.Enter)


def test_pinned_frame_reaches_enter():
    frame = Frame(
        FrameShape(((0, 2),)), (A, B), Effects(cz_zones=frozenset({ZoneAddress(0)}))
    )
    subroutines: dict[ir.Method, Frame | None] = {sub: frame, nested: None}
    out = NativeToQMove(ARCH, subroutines, SingleZonePolicy()).emit(main)
    enter = top_level(_callees(out, qmove.Invoke)["sub"])[0]
    assert isinstance(enter, qmove.Enter) and enter.frame == frame


def test_loops_and_branches_carry_the_state():
    out = NativeToQMove(ARCH, SUBROUTINES).emit(main)
    for method in (out, _callees(out, qmove.Invoke)["nested"]):
        structured = statements_of(method, scf.For) + statements_of(method, scf.IfElse)
        assert structured
        for stmt in structured:
            assert stmt.results[0].type.is_subseteq(StateType)


@pytest.mark.parametrize("kernel, subroutines", [(main, SUBROUTINES), (arranged, {})])
def test_lowering_adds_only_state_plumbing(kernel, subroutines):
    reference = lower_to_native(kernel, subroutines, ARCH)
    out = NativeToQMove(ARCH, subroutines).emit(kernel)

    erase_qmove(out.callable_region.blocks[0])
    assert_methods_match(out, reference.entry)
    lowered = _callees(out, func.Invoke)
    for original, native_clone in reference.subroutines.items():
        clone = lowered[original.sym_name]
        erase_qmove(clone.callable_region.blocks[0])
        assert_methods_match(clone, native_clone)


def test_unlisted_recursion_is_rejected():
    @squin.kernel
    def rec(qs: ilist.IList[Qubit, Literal[1]], n: int):
        if n > 0:
            squin.h(qs[0])
            rec(qs, n - 1)

    @squin.kernel
    def calls_rec():
        qs = squin.qalloc(1)
        rec(qs, 3)

    with pytest.raises(ValidationErrorGroup):
        NativeToQMove(ARCH).emit(calls_rec)
    NativeToQMove(ARCH, {rec: None}).emit(calls_rec)


def test_unsupported_input_is_rejected():
    @squin.kernel
    def resets():
        qs = squin.qalloc(1)
        squin.reset(qs[0])

    with pytest.raises(ValidationErrorGroup):
        NativeToQMove(ARCH).emit(resets)


def test_mismatched_move_to_is_caught_by_type_refinement():
    @physical.kernel(verify=False)
    def bad():
        q = squin.qalloc(2)
        arrange.move_to(q, ilist.IList([A]))  # type: ignore[arg-type]

    with pytest.raises(ValidationErrorGroup):
        NativeToQMove(ARCH).emit(bad)


def test_original_kernels_are_untouched():
    before = main.print_str()
    NativeToQMove(ARCH, SUBROUTINES).emit(main)
    assert main.print_str() == before


@squin.kernel
def bottom_result():
    qs = squin.qalloc(3)
    # measure takes one Qubit, so its result is Bottom.
    m = squin.measure(qs)  # type: ignore[arg-type]
    x = m[0]  # type: ignore[index]
    squin.h(qs[0])
    return x


def test_bottom_typed_result_is_reported_as_a_type_error():
    with pytest.raises(ValidationErrorGroup) as excinfo:
        NativeToQMove(ARCH).emit(bottom_result)
    messages = [str(err.args[0]) for err in excinfo.value.errors]
    assert any("no valid type" in m and "Bottom" in m for m in messages)
    assert not any(m.startswith("V1") for m in messages)


def test_bottom_typed_value_is_not_threaded_as_state():
    out = NativeToQMove(ARCH).emit(bottom_result, no_raise=True)
    result = ValidationSuite([get_qmove_validation(ARCH, ZonedPolicy())]).validate(out)
    assert [err for errs in result.errors.values() for err in errs] == []


@squin.kernel
def sub_a(qs: ilist.IList[Qubit, Literal[2]]):
    squin.cx(qs[0], qs[1])


@squin.kernel
def sub_b(qs: ilist.IList[Qubit, Literal[2]]):
    squin.cz(qs[0], qs[1])


@squin.kernel
def calls_both():
    qs = squin.qalloc(2)
    sub_a(qs)
    sub_b(qs)


def test_frame_errors_are_reported_across_all_subroutines():
    wrong_shape = Frame(FrameShape(((0, 1),)), (A,))  # the parameter has 2 slots
    subroutines: dict[ir.Method, Frame | None] = {
        sub_a: wrong_shape,
        sub_b: wrong_shape,
    }
    with pytest.raises(ValidationErrorGroup) as excinfo:
        NativeToQMove(ARCH, subroutines).emit(calls_both)
    messages = [str(err.args[0]) for err in excinfo.value.errors]
    assert len([m for m in messages if m.startswith("F2:")]) == 2


def test_no_raise_returns_despite_frame_errors():
    wrong_shape = Frame(FrameShape(((0, 1),)), (A,))
    NativeToQMove(ARCH, {sub_a: wrong_shape}).emit(calls_both, no_raise=True)


@pytest.mark.parametrize(
    "listed", [(sub, nested, sub_a, sub_b), (sub_b, sub_a, nested, sub)]
)
def test_subroutines_are_lowered_in_listed_order(monkeypatch, listed):
    # A frozenset would order them by id hash, which varies from run to run.
    seen: list[tuple[ir.Method, ...]] = []
    real = native_to_qmove.lower_to_native

    def spy(entry, subroutines, *args, **kwargs):
        subroutines = tuple(subroutines)
        seen.append(subroutines)
        return real(entry, subroutines, *args, **kwargs)

    monkeypatch.setattr(native_to_qmove, "lower_to_native", spy)
    NativeToQMove(ARCH, dict.fromkeys(listed)).emit(main)
    assert seen == [listed]


@squin.kernel
def allocating_sub(qs: ilist.IList[Qubit, Literal[1]]):
    extra = squin.qalloc(1)
    squin.cz(qs[0], extra[0])


@squin.kernel
def calls_allocating_sub():
    qs = squin.qalloc(1)
    allocating_sub(qs)


def test_whole_machine_subroutine_may_allocate():
    out = NativeToQMove(ARCH, {allocating_sub: MachineFrame()}).emit(
        calls_allocating_sub
    )
    callee = top_level(_callees(out, qmove.Invoke)["allocating_sub"])[0]
    assert isinstance(callee, qmove.Enter) and callee.frame == MachineFrame()


@pytest.mark.parametrize(
    "frame", [None, Frame(FrameShape(((0, 1),)), (A,))], ids=["unframed", "partial"]
)
def test_allocation_in_a_partial_or_unframed_subroutine_is_rejected(frame):
    with pytest.raises(ValidationErrorGroup) as excinfo:
        NativeToQMove(ARCH, {allocating_sub: frame}).emit(calls_allocating_sub)
    assert ALLOCATION_MESSAGE in [str(e.args[0]) for e in excinfo.value.errors]


@squin.kernel
def recursive_qalloc(qs: ilist.IList[Qubit, Literal[1]], n: int):
    if n > 0:
        extra = squin.qalloc(1)
        squin.cz(qs[0], extra[0])
        recursive_qalloc(qs, n - 1)


@squin.kernel
def calls_recursive_qalloc():
    qs = squin.qalloc(1)
    recursive_qalloc(qs, 3)


@pytest.mark.parametrize(
    "frame",
    [MachineFrame(), None, Frame(FrameShape(((0, 1),)), (A,))],
    ids=["whole-machine", "unframed", "partial"],
)
def test_recursive_allocation_through_qalloc_is_rejected_up_front(frame):
    # kirin's constant propagation used to crash on this in lower_to_native, with
    # NotImplementedError for qubit.New, whatever the frame.
    with pytest.raises(ValidationErrorGroup) as excinfo:
        NativeToQMove(ARCH, {recursive_qalloc: frame}).emit(calls_recursive_qalloc)
    (message,) = [str(e.args[0]) for e in excinfo.value.errors]
    assert message.startswith("recursive_qalloc -> recursive_qalloc")
    assert "reaches qalloc" in message
    assert message.endswith(RECURSIVE_ALLOCATION_MESSAGE)


def test_no_raise_lowers_recursive_allocation_through_qalloc():
    out = NativeToQMove(ARCH, {recursive_qalloc: MachineFrame()}).emit(
        calls_recursive_qalloc, no_raise=True
    )
    assert set(_callees(out, qmove.Invoke)) == {"recursive_qalloc"}


@squin.kernel
def recursive_new(qs: ilist.IList[Qubit, Literal[1]], n: int):
    if n > 0:
        extra = qubit.new()
        squin.cz(qs[0], extra)
        recursive_new(qs, n - 1)


@squin.kernel
def calls_recursive_new():
    qs = squin.qalloc(1)
    recursive_new(qs, 3)


def test_recursive_whole_machine_subroutine_may_allocate_with_qubit_new():
    out = NativeToQMove(ARCH, {recursive_new: MachineFrame()}).emit(calls_recursive_new)
    callee = _callees(out, qmove.Invoke)["recursive_new"]
    enter = top_level(callee)[0]
    assert isinstance(enter, qmove.Enter) and enter.frame == MachineFrame()
    assert statements_of(callee, qubit.stmts.New)
    assert _callees(callee, qmove.Invoke) == {"recursive_new": callee}
