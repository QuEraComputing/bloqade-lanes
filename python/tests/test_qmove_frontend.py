from typing import Literal

import pytest
from bloqade.native.dialects.gate import stmts as gate
from bloqade.types import Qubit, QubitType
from kirin import ir, types
from kirin.dialects import func, ilist, scf
from kirin.ir.exception import ValidationErrorGroup

from bloqade import squin
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.transform.qmove_frontend import lower_to_native, unlisted_recursion


@squin.kernel
def sub(qs: ilist.IList[Qubit, Literal[2]]):
    squin.cx(qs[0], qs[1])


@squin.kernel
def main():
    qs = squin.qalloc(3)
    for q in qs:
        squin.h(q)
    sub(ilist.IList([qs[0], qs[1]]))
    return squin.broadcast.measure(qs)


def _calls(mt):
    return [
        s for s in mt.callable_region.walk() if isinstance(s, (func.Invoke, func.Call))
    ]


def test_gates_inside_loops_are_native():
    program = lower_to_native(main, [sub], get_arch_spec())
    (loop,) = [
        s for s in program.entry.callable_region.walk() if isinstance(s, scf.For)
    ]
    body = list(loop.body.walk())
    assert any(isinstance(s, (gate.R, gate.Rz)) for s in body)
    assert not any(isinstance(s, (func.Invoke, func.Call)) for s in body)


def test_only_subroutine_calls_survive_and_target_the_clone():
    program = lower_to_native(main, [sub], get_arch_spec())
    clone = program.subroutines[sub]
    (call,) = _calls(program.entry)
    assert isinstance(call, func.Invoke) and call.callee is clone
    assert clone is not sub and clone.sym_name == "sub"
    assert any(isinstance(s, gate.CZ) for s in clone.callable_region.walk())


def test_originals_are_untouched():
    before = sum(1 for _ in sub.callable_region.walk())
    lower_to_native(main, [sub], get_arch_spec())
    assert sum(1 for _ in sub.callable_region.walk()) == before
    assert not any(isinstance(s, gate.CZ) for s in sub.callable_region.walk())


def test_unlisted_calls_are_inlined():
    program = lower_to_native(main, [], get_arch_spec())
    assert _calls(program.entry) == []


def test_register_types_are_refined():
    program = lower_to_native(main, [sub], get_arch_spec())
    (alloc,) = [
        s for s in program.entry.callable_region.walk() if isinstance(s, ilist.Map)
    ]
    assert alloc.result.type == ilist.IListType[QubitType, types.Literal(3)]


@squin.kernel
def rec(qs: ilist.IList[Qubit, Literal[1]], n: int):
    if n > 0:
        squin.h(qs[0])
        rec(qs, n - 1)


@squin.kernel
def calls_rec():
    qs = squin.qalloc(1)
    rec(qs, 3)


def test_unlisted_recursion_is_reported():
    (cycle,) = unlisted_recursion(calls_rec, frozenset())
    assert "rec -> rec" in cycle
    assert unlisted_recursion(calls_rec, frozenset({rec})) == []


def test_listed_recursion_calls_its_own_clone():
    program = lower_to_native(calls_rec, [rec], get_arch_spec())
    clone = program.subroutines[rec]
    (self_call,) = _calls(clone)
    assert self_call.callee is clone


def _call_chain(depth: int) -> ir.Method:
    """An entry kernel that reaches a gate through ``depth`` nested calls."""

    @squin.kernel
    def leaf(q: Qubit):
        squin.h(q)

    def wrap(callee: ir.Method) -> ir.Method:
        @squin.kernel
        def step(q: Qubit):
            callee(q)

        return step

    inner = leaf
    for _ in range(depth):
        inner = wrap(inner)

    @squin.kernel
    def entry():
        qs = squin.qalloc(1)
        inner(qs[0])

    return entry


def test_moderately_nested_calls_are_fully_inlined():
    program = lower_to_native(_call_chain(10), [], get_arch_spec())
    assert _calls(program.entry) == []


def test_inlining_that_does_not_converge_is_an_error():
    # Each fixpoint round inlines a bounded number of levels; a call chain deeper
    # than that used to leave an un-inlined call behind without any error.
    with pytest.raises(ValidationErrorGroup) as excinfo:
        lower_to_native(_call_chain(100), [], get_arch_spec())
    (error,) = excinfo.value.errors
    message = str(error.args[0])
    assert "inlining did not converge" in message and "entry" in message
