from typing import Literal

import pytest
from bloqade.types import Qubit
from kirin import ir
from kirin.dialects import func, ilist, scf
from kirin.ir.exception import ValidationErrorGroup
from tests._qmove_helpers import (
    assert_methods_match,
    erase_qmove,
    statements_of,
    top_level,
)

from bloqade import squin
from bloqade.gemini import physical
from bloqade.gemini.common.dialects import arrange, qubit as gemini_qubit
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.dialects.qmove import Effects, Frame, FrameShape
from bloqade.lanes.transform.native_to_qmove import NativeToQMove
from bloqade.lanes.transform.qmove_frontend import lower_to_native
from bloqade.lanes.types import StateType
from bloqade.lanes.validation.spectator import SingleZonePolicy

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
    assert isinstance(top[0], move.Load) and isinstance(top[-2], move.Store)
    assert len(statements_of(out, move.Load)) == 1


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
