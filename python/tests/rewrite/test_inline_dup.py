"""Tests for the InlineDup canonicalisation."""

from kirin import ir
from kirin.rewrite import Walk

from bloqade.lanes.bytecode import Instruction as I, Program
from bloqade.lanes.bytecode.decode import load_program
from bloqade.lanes.bytecode.encode import dump_program
from bloqade.lanes.dialects import stack_move
from bloqade.lanes.rewrite.inline_dup import InlineDup
from bloqade.lanes.rewrite.stackify import stackify


def _decoded(instructions: list) -> ir.Method:
    return load_program(Program(version=(1, 0), instructions=instructions))


def _stmts(method: ir.Method) -> list[ir.Statement]:
    return list(method.callable_region.blocks[0].stmts)


def _text(instructions: list) -> str:
    return Program(version=(1, 0), instructions=instructions).to_text()


def test_inline_dup_forwards_every_copy_to_the_operand():
    """A copy of a copy included: one walk leaves each ``cz`` reading the
    zone itself, and no ``Dup``."""
    method = _decoded(
        [I.const_zone(0), I.dup(), I.dup(), I.cz(), I.cz(), I.cz(), I.halt()]
    )
    zone = next(s for s in _stmts(method) if isinstance(s, stack_move.ConstZone))

    result = Walk(InlineDup()).rewrite(method.code)

    assert result.has_done_something
    stmts = _stmts(method)
    assert not any(isinstance(s, stack_move.Dup) for s in stmts)
    gates = [s for s in stmts if isinstance(s, stack_move.CZ)]
    assert [g.zone for g in gates] == [zone.result] * 3
    method.verify()


def test_inline_dup_removes_a_dup_of_a_non_constant():
    """Unlike ``ConstantFold`` + DCE, which need a constant to fold, this
    removes a ``Dup`` of anything — here a measurement array."""
    method = _decoded(
        [
            I.const_zone(0),
            I.measure(1),
            I.await_measure(),
            I.dup(),
            I.store("undef", 0),
            I.store("undef", 1),
            I.halt(),
        ]
    )
    array = next(s for s in _stmts(method) if isinstance(s, stack_move.AwaitMeasure))

    Walk(InlineDup()).rewrite(method.code)

    stores = [s for s in _stmts(method) if isinstance(s, stack_move.StoreLocal)]
    assert [s.value for s in stores] == [array.result] * 2
    assert not any(isinstance(s, stack_move.Dup) for s in _stmts(method))


def test_inline_dup_does_nothing_without_a_dup():
    method = _decoded([I.const_zone(0), I.cz(), I.halt()])

    assert not Walk(InlineDup()).rewrite(method.code).has_done_something


def test_stackify_redoes_what_an_inlined_dup_did():
    """The operand now has a consumer per copy: ``stackify`` parks it in a
    local and reloads it for each, and the program validates."""
    method = _decoded(
        [
            I.const_zone(0),
            I.measure(1),
            I.await_measure(),
            I.dup(),
            I.store("undef", 0),
            I.store("undef", 1),
            I.halt(),
        ]
    )

    Walk(InlineDup()).rewrite(method.code)
    stackify(method)

    program = dump_program(method)
    program.validate(stack=True)
    assert program.to_text() == _text(
        [
            I.const_zone(0),
            I.measure(1),
            I.await_measure(),
            I.store("undef", 2),
            I.load("undef", 2),
            I.store("undef", 0),
            I.load("undef", 2),
            I.store("undef", 1),
            I.halt(),
        ]
    )
