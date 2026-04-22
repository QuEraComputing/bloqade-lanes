from kirin import ir
from kirin.dialects import py
from kirin.rewrite import Walk

from bloqade.lanes.bytecode import LocationAddress
from bloqade.lanes.dialects import move, stack_move
from bloqade.lanes.rewrite.lower_stack_move import LowerStackMove


def _build_stack_move_block(stmts: list[ir.Statement]) -> ir.Block:
    block = ir.Block()
    for stmt in stmts:
        block.stmts.append(stmt)
    return block


def test_empty_block_emits_load_and_func_return():
    block = _build_stack_move_block([stack_move.Return()])
    result = Walk(LowerStackMove()).rewrite(block)
    assert result.has_done_something
    # Expect a move.Load at block start and a func.Return; the stack_move
    # Return should have been deleted.
    assert any(isinstance(s, move.Load) for s in block.stmts)
    assert not any(isinstance(s, stack_move.Return) for s in block.stmts)


def test_const_float_emits_py_constant_and_tracks_value():
    cf = stack_move.ConstFloat(value=1.5)
    block = _build_stack_move_block([cf, stack_move.Return()])
    rule = LowerStackMove()
    Walk(rule).rewrite(block)
    # py.Constant statement emitted with value 1.5.
    py_const = next(s for s in block.stmts if isinstance(s, py.Constant))
    assert py_const.value.unwrap() == 1.5
    # Its result is tracked in ssa_to_attr for attribute lifting by
    # downstream stateful-op handlers (key is the new SSA, because
    # replace_by rewired all consumer operands to point there).
    assert rule.ssa_to_attr[py_const.result] == 1.5


def test_const_loc_tracks_attribute_value():
    addr = LocationAddress(0, 0, 0)
    cl = stack_move.ConstLoc(value=addr)
    block = _build_stack_move_block([cl, stack_move.Return()])
    rule = LowerStackMove()
    Walk(rule).rewrite(block)
    # The stack_move SSA is mapped to its raw attribute (for lifting into
    # downstream move.* attributes).
    assert rule.ssa_to_attr[cl.result] == addr


def test_pop_is_dropped():
    cf = stack_move.ConstFloat(value=1.0)
    pop = stack_move.Pop(value=cf.result)
    block = _build_stack_move_block([cf, pop, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    # No target statement for Pop, and the original stack_move.Pop is gone.
    assert not any(isinstance(s, stack_move.Pop) for s in block.stmts)


def test_dup_redirects_uses_to_input():
    cf = stack_move.ConstFloat(value=1.0)
    dup = stack_move.Dup(value=cf.result)
    # Downstream consumer that references Dup's result.
    consumer = stack_move.Pop(value=dup.result)
    block = _build_stack_move_block([cf, dup, consumer, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    # Dup is gone; Pop is also lowered away.
    assert not any(isinstance(s, stack_move.Dup) for s in block.stmts)
    assert not any(isinstance(s, stack_move.Pop) for s in block.stmts)


def test_swap_permutes_uses():
    a = stack_move.ConstInt(value=1)
    b = stack_move.ConstInt(value=2)
    sw = stack_move.Swap(in_top=b.result, in_bot=a.result)
    # Consumers that read Swap's outputs; pop them so the test has
    # something observable.
    p_top = stack_move.Pop(value=sw.out_top)
    p_bot = stack_move.Pop(value=sw.out_bot)
    block = _build_stack_move_block([a, b, sw, p_top, p_bot, stack_move.Return()])
    Walk(LowerStackMove()).rewrite(block)
    # Swap is gone.
    assert not any(isinstance(s, stack_move.Swap) for s in block.stmts)
