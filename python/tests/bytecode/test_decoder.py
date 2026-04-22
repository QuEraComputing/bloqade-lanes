from bloqade.lanes.bytecode import (
    Instruction,
    LaneAddress,
    LocationAddress,
    MoveType,
    Program,
    ZoneAddress,
)
from bloqade.lanes.bytecode.lowering import load_program
from bloqade.lanes.dialects import stack_move


def test_empty_program_returns_method_with_empty_body():
    prog = Program(version=(1, 0), instructions=[Instruction.return_()])
    method = load_program(prog)
    assert method.sym_name == "main"
    # Body should have one statement (stack_move.Return), no other statements.
    block = method.callable_region.blocks[0]
    assert len(block.stmts) == 1


def _decode(instructions):
    prog = Program(version=(1, 0), instructions=instructions + [Instruction.return_()])
    return load_program(prog).callable_region.blocks[0]


def test_decode_const_float():
    block = _decode([Instruction.const_float(2.5)])
    assert any(
        isinstance(s, stack_move.ConstFloat) and s.value == 2.5 for s in block.stmts
    )


def test_decode_const_int():
    block = _decode([Instruction.const_int(42)])
    assert any(
        isinstance(s, stack_move.ConstInt) and s.value == 42 for s in block.stmts
    )


def test_decode_const_loc():
    block = _decode([Instruction.const_loc(0, 0, 0)])
    stmt = next(s for s in block.stmts if isinstance(s, stack_move.ConstLoc))
    assert stmt.value == LocationAddress(0, 0, 0)


def test_decode_const_lane():
    block = _decode([Instruction.const_lane(MoveType.SITE, 0, 0, 0, 0)])
    stmt = next(s for s in block.stmts if isinstance(s, stack_move.ConstLane))
    assert stmt.value == LaneAddress(MoveType.SITE, 0, 0, 0, 0)


def test_decode_const_zone():
    block = _decode([Instruction.const_zone(3)])
    stmt = next(s for s in block.stmts if isinstance(s, stack_move.ConstZone))
    assert stmt.value == ZoneAddress(3)


def test_decode_pop_consumes_top():
    block = _decode([Instruction.const_int(1), Instruction.pop()])
    assert any(isinstance(s, stack_move.Pop) for s in block.stmts)


def test_decode_dup_duplicates_top():
    block = _decode([Instruction.const_int(1), Instruction.dup()])
    dup = next(s for s in block.stmts if isinstance(s, stack_move.Dup))
    cint = next(s for s in block.stmts if isinstance(s, stack_move.ConstInt))
    assert dup.value is cint.result


def test_decode_swap_permutes_top_two():
    block = _decode(
        [Instruction.const_int(1), Instruction.const_int(2), Instruction.swap()]
    )
    swap = next(s for s in block.stmts if isinstance(s, stack_move.Swap))
    ints = [s for s in block.stmts if isinstance(s, stack_move.ConstInt)]
    assert swap.in_top is ints[1].result
    assert swap.in_bot is ints[0].result


def test_decode_pop_underflow_raises():
    import pytest

    from bloqade.lanes.bytecode.lowering import LoweringError

    with pytest.raises(LoweringError):
        _decode([Instruction.pop()])


def test_decode_fill_consumes_arity_locations():
    block = _decode(
        [
            Instruction.const_loc(0, 0, 0),
            Instruction.const_loc(0, 0, 1),
            Instruction.fill(2),
        ]
    )
    fill = next(s for s in block.stmts if isinstance(s, stack_move.Fill))
    locs = [s for s in block.stmts if isinstance(s, stack_move.ConstLoc)]
    assert fill.locations == (locs[0].result, locs[1].result)


def test_decode_initial_fill():
    block = _decode([Instruction.const_loc(0, 0, 0), Instruction.initial_fill(1)])
    assert any(isinstance(s, stack_move.InitialFill) for s in block.stmts)


def test_decode_move_consumes_arity_lanes():
    block = _decode(
        [
            Instruction.const_lane(MoveType.SITE, 0, 0, 0, 0),
            Instruction.move_(1),
        ]
    )
    mv = next(s for s in block.stmts if isinstance(s, stack_move.Move))
    lane = next(s for s in block.stmts if isinstance(s, stack_move.ConstLane))
    assert mv.lanes == (lane.result,)


def test_decode_local_r():
    block = _decode(
        [
            Instruction.const_loc(0, 0, 0),  # loc
            Instruction.const_float(0.1),  # theta
            Instruction.const_float(0.2),  # phi
            Instruction.local_r(1),
        ]
    )
    r = next(s for s in block.stmts if isinstance(s, stack_move.LocalR))
    floats = [s for s in block.stmts if isinstance(s, stack_move.ConstFloat)]
    locs = [s for s in block.stmts if isinstance(s, stack_move.ConstLoc)]
    # bytecode pops phi first, then theta, then locations (per .pyi docstring)
    assert r.phi is floats[1].result
    assert r.theta is floats[0].result
    assert r.locations == (locs[0].result,)


def test_decode_global_rz():
    block = _decode([Instruction.const_float(0.5), Instruction.global_rz()])
    rz = next(s for s in block.stmts if isinstance(s, stack_move.GlobalRz))
    cf = next(s for s in block.stmts if isinstance(s, stack_move.ConstFloat))
    assert rz.theta is cf.result


def test_decode_cz():
    block = _decode([Instruction.const_zone(0), Instruction.cz()])
    cz = next(s for s in block.stmts if isinstance(s, stack_move.CZ))
    cz_zone = next(s for s in block.stmts if isinstance(s, stack_move.ConstZone))
    assert cz.zone is cz_zone.result
