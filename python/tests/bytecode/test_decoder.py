from kirin.dialects import func

from bloqade.lanes.bytecode import (
    Instruction,
    MoveType,
    Program,
)
from bloqade.lanes.bytecode.decode import load_program
from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress, ZoneAddress
from bloqade.lanes.dialects import stack_move


def test_empty_program_returns_method_with_empty_body():
    # ``return`` lowers directly to ``func.Return`` (overlap with the
    # kirin.basic dialect group), so we push a dummy const_int(0)
    # before return to satisfy the virtual-stack invariant.
    prog = Program(
        version=(1, 0),
        instructions=[Instruction.const_int(0), Instruction.return_()],
    )
    method = load_program(prog)
    assert method.sym_name == "main"
    # Body should have two statements: the dummy ConstInt and func.Return.
    block = method.callable_region.blocks[0]
    assert len(block.stmts) == 2
    assert any(isinstance(s, func.Return) for s in block.stmts)


def _decode(instructions):
    prog = Program(
        version=(1, 0),
        instructions=[*instructions, Instruction.const_int(0), Instruction.return_()],
    )
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
    assert stmt.value == LaneAddress(MoveType.SITE, 0, 0, 0)


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

    from bloqade.lanes.bytecode.decode import DecodingError

    with pytest.raises(DecodingError):
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
    # bytecode pops phi first, then theta, then locations (per .pyi docstring);
    # these map to axis_angle and rotation_angle on the stack_move statement.
    assert r.axis_angle is floats[1].result
    assert r.rotation_angle is floats[0].result
    assert r.locations == (locs[0].result,)


def test_decode_global_rz():
    block = _decode([Instruction.const_float(0.5), Instruction.global_rz()])
    rz = next(s for s in block.stmts if isinstance(s, stack_move.GlobalRz))
    cf = next(s for s in block.stmts if isinstance(s, stack_move.ConstFloat))
    assert rz.rotation_angle is cf.result


def test_decode_cz():
    block = _decode([Instruction.const_zone(0), Instruction.cz()])
    cz = next(s for s in block.stmts if isinstance(s, stack_move.CZ))
    cz_zone = next(s for s in block.stmts if isinstance(s, stack_move.ConstZone))
    assert cz.zone is cz_zone.result


def test_decode_measure():
    block = _decode([Instruction.const_zone(0), Instruction.measure(1)])
    m = next(s for s in block.stmts if isinstance(s, stack_move.Measure))
    zone_const = next(s for s in block.stmts if isinstance(s, stack_move.ConstZone))
    assert m.zones == (zone_const.result,)


def test_decode_await_measure():
    # measure pushes a future; await_measure consumes it (linear) and
    # pushes an array ref of measurement results — frame.push auto-pushes
    # the AwaitMeasure result onto the virtual stack.
    block = _decode(
        [
            Instruction.const_zone(0),
            Instruction.measure(1),
            Instruction.await_measure(),
        ]
    )
    aw = next(s for s in block.stmts if isinstance(s, stack_move.AwaitMeasure))
    m = next(s for s in block.stmts if isinstance(s, stack_move.Measure))
    # measure(arity=1) produces a single future, which is the only
    # stack_move.Measure result available for await_measure to consume.
    assert aw.future is m.results[0]
    # AwaitMeasure now produces an array-ref result (measurement results).
    assert aw.result is not None


def test_decode_new_array():
    block = _decode(
        [
            Instruction.const_float(0.0),
            Instruction.const_float(1.0),
            Instruction.const_float(2.0),
            Instruction.const_float(3.0),
            Instruction.new_array(type_tag=0, dim0=4),
        ]
    )
    na = next(s for s in block.stmts if isinstance(s, stack_move.NewArray))
    assert na.dim0 == 4
    assert len(na.values) == 4


def test_decode_get_item():
    block = _decode(
        [
            Instruction.const_float(0.0),
            Instruction.const_float(1.0),
            Instruction.const_float(2.0),
            Instruction.const_float(3.0),
            Instruction.new_array(type_tag=0, dim0=4),
            Instruction.const_int(2),
            Instruction.get_item(1),
        ]
    )
    gi = next(s for s in block.stmts if isinstance(s, stack_move.GetItem))
    assert len(gi.indices) == 1


def test_decode_halt():
    prog = Program(version=(1, 0), instructions=[Instruction.halt()])
    method = load_program(prog)
    block = method.callable_region.blocks[0]
    # ``halt`` lowers to ``func.ConstantNone`` + ``func.Return`` directly
    # (overlap with kirin.basic's func dialect).
    assert any(isinstance(s, func.ConstantNone) for s in block.stmts)
    assert any(isinstance(s, func.Return) for s in block.stmts)


def test_load_program_infers_return_type():
    # load_program runs Kirin's TypeInfer pass after decoding, so the
    # method's return_type should be narrowed from the default types.Any.
    from kirin import types

    prog = Program(
        version=(1, 0),
        instructions=[Instruction.const_int(0), Instruction.return_()],
    )
    method = load_program(prog)
    assert (
        method.return_type != types.Any
    ), f"expected narrowed return type, got {method.return_type}"


_TWO_FUNCTIONS = """sst v1

.section(root):
.header(root):
version 1.0
.header(root).
.text(root):
fn @helper() {
  cpu::cpu.halt
}

fn @main() {
  lanes::lanes.const_loc 0x0000000000000000
  lanes::lanes.initial_fill 1
  cpu::cpu.halt
}
.text(root).
.section(root).
"""


def test_multi_function_program_is_rejected():
    # Decoding lowers the whole stream into one kirin block, which silently
    # concatenates the bodies: this program used to produce a kernel whose
    # first statement was `@helper`'s `func.return` and which carried two
    # terminators -- and `verify()` accepted it. The compiler neither emits
    # multi-function bytecode nor lowers it further, so there is no correct
    # lowering to fall back to; refusing is the fix.
    import pytest

    from bloqade.lanes.bytecode.decode import DecodingError

    prog = Program.from_text(_TWO_FUNCTIONS)
    with pytest.raises(DecodingError) as excinfo:
        load_program(prog)

    message = str(excinfo.value)
    assert "2 functions" in message, message
    # It names the second `func_start`, not a value-carrying instruction.
    assert "func_start" in message, message


def test_single_function_program_still_decodes():
    # The guard keys on the marker count, so the single `func_start` /
    # `func_end` pair every program carries must not trip it.
    src = _TWO_FUNCTIONS.replace(
        "fn @helper() {\n  cpu::cpu.halt\n}\n\n",
        "",
    )
    method = load_program(Program.from_text(src))
    block = method.callable_region.blocks[0]

    terminators = [s for s in block.stmts if isinstance(s, func.Return)]
    assert len(terminators) == 1, f"expected one terminator, got {len(terminators)}"
    assert any(isinstance(s, stack_move.InitialFill) for s in block.stmts)
