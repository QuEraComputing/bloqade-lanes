"""Round-trip tests for BytecodeEncoder: decode → encode → compare."""

import pytest

from bloqade.lanes.bytecode import Instruction, MoveType, Program
from bloqade.lanes.bytecode.decode import load_program
from bloqade.lanes.bytecode.encode import BytecodeEncoder, EncodingError, dump_program


def _roundtrip(instructions: list[Instruction]) -> tuple[Program, Program]:
    prog = Program(version=(1, 0), instructions=instructions)
    return prog, dump_program(load_program(prog))


def _assert_roundtrip(instructions: list[Instruction]) -> None:
    orig, encoded = _roundtrip(instructions)
    assert orig.to_text() == encoded.to_text()


def test_roundtrip_halt():
    _assert_roundtrip([Instruction.halt()])


def test_roundtrip_return_int():
    _assert_roundtrip([Instruction.const_int(42), Instruction.return_()])


def test_roundtrip_const_float():
    _assert_roundtrip(
        [
            Instruction.const_float(3.14),
            Instruction.pop(),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_const_loc():
    _assert_roundtrip(
        [
            Instruction.const_loc(0, 1, 2),
            Instruction.pop(),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_const_lane():
    _assert_roundtrip(
        [
            Instruction.const_lane(MoveType.SITE, 0, 1, 2, 3),
            Instruction.pop(),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_const_zone():
    _assert_roundtrip(
        [
            Instruction.const_zone(5),
            Instruction.pop(),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_pop():
    _assert_roundtrip(
        [
            Instruction.const_int(1),
            Instruction.pop(),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_dup():
    _assert_roundtrip(
        [
            Instruction.const_int(7),
            Instruction.dup(),
            Instruction.pop(),
            Instruction.return_(),
        ]
    )


def test_roundtrip_swap():
    _assert_roundtrip(
        [
            Instruction.const_int(1),
            Instruction.const_int(2),
            Instruction.swap(),
            Instruction.pop(),
            Instruction.return_(),
        ]
    )


def test_roundtrip_initial_fill():
    _assert_roundtrip(
        [
            Instruction.const_loc(0, 0, 0),
            Instruction.const_loc(0, 0, 1),
            Instruction.initial_fill(2),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_fill():
    _assert_roundtrip(
        [
            Instruction.const_loc(0, 0, 0),
            Instruction.fill(1),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_move():
    _assert_roundtrip(
        [
            Instruction.const_lane(MoveType.SITE, 0, 0, 0, 0),
            Instruction.move_(1),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_local_r():
    _assert_roundtrip(
        [
            Instruction.const_loc(0, 0, 0),
            Instruction.const_float(0.1),  # theta (rotation_angle)
            Instruction.const_float(0.2),  # phi (axis_angle)
            Instruction.local_r(1),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_local_rz():
    _assert_roundtrip(
        [
            Instruction.const_loc(0, 0, 0),
            Instruction.const_float(0.5),
            Instruction.local_rz(1),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_global_r():
    _assert_roundtrip(
        [
            Instruction.const_float(0.1),  # theta (rotation_angle)
            Instruction.const_float(0.2),  # phi (axis_angle)
            Instruction.global_r(),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_global_rz():
    _assert_roundtrip(
        [
            Instruction.const_float(0.5),
            Instruction.global_rz(),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_cz():
    _assert_roundtrip(
        [
            Instruction.const_zone(0),
            Instruction.cz(),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_measure_await():
    _assert_roundtrip(
        [
            Instruction.const_zone(0),
            Instruction.measure(1),
            Instruction.await_measure(),
            Instruction.pop(),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_new_array_1d():
    _assert_roundtrip(
        [
            Instruction.const_float(0.0),
            Instruction.const_float(1.0),
            Instruction.new_array(type_tag=0, dim0=2),
            Instruction.pop(),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_new_array_2d():
    _assert_roundtrip(
        [
            Instruction.const_float(0.0),
            Instruction.const_float(1.0),
            Instruction.const_float(2.0),
            Instruction.const_float(3.0),
            Instruction.new_array(type_tag=0, dim0=2, dim1=2),
            Instruction.pop(),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_get_item_1d():
    _assert_roundtrip(
        [
            Instruction.const_float(0.0),
            Instruction.const_float(1.0),
            Instruction.new_array(type_tag=0, dim0=2),
            Instruction.const_int(0),
            Instruction.get_item(1),
            Instruction.pop(),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_get_item_2d():
    _assert_roundtrip(
        [
            Instruction.const_float(0.0),
            Instruction.const_float(1.0),
            Instruction.const_float(2.0),
            Instruction.const_float(3.0),
            Instruction.new_array(type_tag=0, dim0=2, dim1=2),
            Instruction.const_int(0),
            Instruction.const_int(1),
            Instruction.get_item(2),
            Instruction.pop(),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_set_detector():
    _assert_roundtrip(
        [
            Instruction.const_zone(0),
            Instruction.measure(1),
            Instruction.await_measure(),
            Instruction.set_detector(),
            Instruction.pop(),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_roundtrip_set_observable():
    _assert_roundtrip(
        [
            Instruction.const_zone(0),
            Instruction.measure(1),
            Instruction.await_measure(),
            Instruction.set_observable(),
            Instruction.pop(),
            Instruction.const_int(0),
            Instruction.return_(),
        ]
    )


def test_encoding_error_for_unknown_statement():
    from kirin import ir, types
    from kirin.decl import info, statement

    unknown_dialect = ir.Dialect(name="test.unknown")

    @statement(dialect=unknown_dialect)
    class UnknownStmt(ir.Statement):
        result: ir.ResultValue = info.result(types.Int)

    encoder = BytecodeEncoder()
    with pytest.raises(EncodingError):
        encoder._visit(UnknownStmt())
