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
