from itertools import pairwise
from typing import Literal

from bloqade.types import Qubit
from kirin import ir, rewrite, types
from kirin.dialects import func, ilist, scf
from tests._qmove_helpers import first_of, statements_of, top_level

from bloqade import squin
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.rewrite.native2qmove import RewriteNativeToQMove
from bloqade.lanes.rewrite.qmove_state import thread_block, thread_method
from bloqade.lanes.transform.qmove_frontend import NativeProgram, lower_to_native
from bloqade.lanes.types import StateType


def _lowered(kernel: ir.Method, subroutines=()) -> NativeProgram:
    program = lower_to_native(kernel, subroutines, get_arch_spec())
    clones = frozenset(program.subroutines.values())
    for method in (program.entry, *clones):
        rewrite.Walk(RewriteNativeToQMove(clones)).rewrite(method.code)
    return program


def _yield(region: ir.Region) -> scf.Yield:
    stmt = region.blocks[0].last_stmt
    assert isinstance(stmt, scf.Yield)
    return stmt


@squin.kernel
def straight():
    qs = squin.qalloc(2)
    squin.h(qs[0])
    squin.cz(qs[0], qs[1])
    return squin.broadcast.measure(qs)


def test_straight_line_becomes_one_chain():
    program = _lowered(straight)
    thread_method(program.entry, subroutine=False)
    stmts = top_level(program.entry)
    (load,) = [s for s in stmts if isinstance(s, move.Load)]
    (store,) = [s for s in stmts if isinstance(s, move.Store)]
    assert stmts[0] is load
    assert stmts[-2] is store and isinstance(stmts[-1], func.Return)
    chain = [s for s in stmts if s.args and s.args[0].type.is_subseteq(StateType)]
    for prev, nxt in pairwise(chain):
        assert nxt.args[0] is prev.results[0]


@squin.kernel
def branchy():
    qs = squin.qalloc(2)
    if squin.is_one(squin.measure(qs[0])):
        squin.x(qs[1])
    squin.h(qs[0])


def test_if_else_captures_and_yields_the_state():
    program = _lowered(branchy)
    thread_method(program.entry, subroutine=False)
    branch = first_of(program.entry, scf.IfElse)
    measure = first_of(program.entry, qmove.Measure)
    assert branch.results[0].type.is_subseteq(StateType)
    assert _yield(branch.else_body).values[0] is measure.results[0]
    assert _yield(branch.then_body).values[0] is not measure.results[0]
    after = [s for s in top_level(program.entry) if isinstance(s, (qmove.R, qmove.Rz))]
    assert after[0].current_state is branch.results[0]


@squin.kernel
def loopy():
    qs = squin.qalloc(3)
    for i in range(2):
        squin.z(qs[i + 1])


def test_for_carries_the_state_ahead_of_existing_iter_args():
    program = _lowered(loopy)
    carried_before = len(first_of(program.entry, scf.For).initializers)
    assert carried_before > 0  # kirin carries `qs` through the loop
    thread_method(program.entry, subroutine=False)
    loop = first_of(program.entry, scf.For)
    assert len(loop.initializers) == carried_before + 1
    assert loop.body.blocks[0].args[1].type.is_subseteq(StateType)
    assert _yield(loop.body).values[0].type.is_subseteq(StateType)
    assert first_of(program.entry, move.Store).current_state is loop.results[0]


def test_hand_built_if_without_else_gets_one():
    cond = ir.TestValue(types.Bool)
    outer = ir.TestValue(StateType)
    then_block = ir.Block()
    then_block.args.append_from(types.Bool)
    load = move.Load()
    gate = qmove.Rz(load.result, ir.TestValue(types.Float), ir.TestValue(types.Any))
    then_block.stmts.extend([load, gate, move.Store(gate.result), scf.Yield()])
    branch = scf.IfElse(cond, then_block)
    assert not branch.else_body.blocks  # the constructor leaves else empty
    block = ir.Block([branch])
    final = thread_block(block, outer)
    (new_branch,) = [s for s in block.stmts if isinstance(s, scf.IfElse)]
    assert final is new_branch.results[0]
    assert _yield(new_branch.else_body).values[0] is outer


@squin.kernel
def sub(qs: ilist.IList[Qubit, Literal[2]]):
    squin.cz(qs[0], qs[1])


@squin.kernel
def calls_sub():
    qs = squin.qalloc(2)
    sub(qs)


def test_subroutine_uses_enter_and_exit():
    program = _lowered(calls_sub, [sub])
    clone = program.subroutines[sub]
    frame = qmove.Frame(
        qmove.FrameShape(((0, 2),)), (LocationAddress(0, 0), LocationAddress(1, 0))
    )
    thread_method(clone, subroutine=True, frame=frame)
    stmts = top_level(clone)
    enter = stmts[0]
    assert isinstance(enter, qmove.Enter) and enter.frame == frame
    assert isinstance(stmts[-2], qmove.Exit)
    assert not statements_of(clone, move.Load) and not statements_of(clone, move.Store)


def test_method_without_quantum_operations_is_untouched():
    @squin.kernel
    def classical(x: int) -> int:
        return x + 1

    out = classical.similar()
    before = len(top_level(out))
    thread_method(out, subroutine=False)
    assert len(top_level(out)) == before
