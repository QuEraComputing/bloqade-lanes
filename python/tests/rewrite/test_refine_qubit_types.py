from typing import Literal

import pytest
from bloqade.types import Qubit, QubitType
from kirin import passes, types
from kirin.dialects import ilist, scf
from kirin.ir.exception import ValidationErrorGroup
from kirin.passes.inline import InlinePass

from bloqade import qubit, squin
from bloqade.lanes.rewrite.refine_qubit_types import RefineQubitTypes


def _inline_and_infer(kernel):
    out = kernel.similar()
    InlinePass(out.dialects).fixpoint(out)
    passes.TypeInfer(out.dialects, no_raise=False)(out)
    return out


def _map_result(mt):
    (stmt,) = [s for s in mt.callable_region.walk() if isinstance(s, ilist.Map)]
    return stmt.result


def test_qalloc_register_gets_its_length():
    @squin.kernel
    def k():
        qs = squin.qalloc(3)
        squin.h(qs[0])

    out = _inline_and_infer(k)
    assert _map_result(out).type == ilist.IListType[QubitType, types.Any]
    assert RefineQubitTypes(out.dialects, no_raise=False)(out).has_done_something
    assert _map_result(out).type == ilist.IListType[QubitType, types.Literal(3)]


def test_loop_variable_keeps_qubit():
    @squin.kernel
    def k():
        qs = squin.qalloc(2)
        for q in qs:
            squin.h(q)

    out = _inline_and_infer(k)
    RefineQubitTypes(out.dialects, no_raise=False)(out)
    (loop,) = [s for s in out.callable_region.walk() if isinstance(s, scf.For)]
    assert loop.body.blocks[0].args[0].type == QubitType


def test_subroutine_parameter_keeps_its_annotation():
    @squin.kernel
    def sub(qs: ilist.IList[Qubit, Literal[2]]):
        squin.h(qs[0])

    out = _inline_and_infer(sub)
    RefineQubitTypes(out.dialects, no_raise=False)(out)
    assert out.arg_types[0] == ilist.IListType[QubitType, types.Literal(2)]


def test_qubit_typed_value_holding_a_register_is_a_contradiction():
    @squin.kernel
    def k():
        qs = squin.qalloc(3)
        squin.h(qs[0])

    out = _inline_and_infer(k)
    _map_result(out).type = QubitType  # disagrees with its AddressReg
    with pytest.raises(ValidationErrorGroup):
        RefineQubitTypes(out.dialects, no_raise=False)(out)


def test_bottom_typed_values_are_left_alone():
    @squin.kernel
    def k():
        qs = squin.qalloc(3)
        return squin.measure(qs)  # type: ignore[arg-type]  # takes one Qubit

    out = _inline_and_infer(k)
    (measure,) = [
        s for s in out.callable_region.walk() if isinstance(s, qubit.stmts.Measure)
    ]
    assert measure.result.type.is_subseteq(types.Bottom)
    RefineQubitTypes(out.dialects, no_raise=False)(out)
    assert measure.result.type.is_subseteq(types.Bottom)


def test_running_twice_changes_nothing():
    @squin.kernel
    def k():
        qs = squin.qalloc(3)
        squin.h(qs[0])

    out = _inline_and_infer(k)
    RefineQubitTypes(out.dialects, no_raise=False)(out)
    assert not RefineQubitTypes(out.dialects, no_raise=False)(out).has_done_something
