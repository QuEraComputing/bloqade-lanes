"""End-to-end: bytecode Program -> decode -> lower -> measure_lower."""

from kirin.dialects import func, ilist
from kirin.rewrite import Walk

from bloqade.lanes._prelude import kernel
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.bytecode import Instruction, Program
from bloqade.lanes.bytecode.decode import load_program
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.rewrite.measure_lower import MeasureLower
from bloqade.lanes.rewrite.stack_move2move import RewriteStackMoveToMove


def test_minimal_program_runs_end_to_end():
    """Bytecode Program -> stack_move IR -> multi-dialect IR with EndMeasure.

    Asserts the full expected block structure after both rewrites:
    move.Load, move.Fill (InitialFill lowers to Fill), move.ConstZone
    (orphaned after measure_lower), move.EndMeasure, ilist.New
    (AwaitMeasure stub), move.Store, func.Return.
    """
    prog = Program(
        version=(1, 0),
        instructions=[
            Instruction.const_loc(0, 0, 0),
            Instruction.initial_fill(1),
            Instruction.const_zone(0),
            Instruction.measure(1),
            Instruction.await_measure(),
            Instruction.return_(),
        ],
    )

    # Stage 1: decode bytecode -> stack_move IR.
    method = load_program(prog)

    # Stage 2: lower stack_move -> multi-dialect IR. Walk can operate
    # directly on the method's top-level Function statement — no need to
    # dig out the block.
    Walk(RewriteStackMoveToMove()).rewrite(method.code)

    # The method's dialect group still points at the decoder's
    # stack_move+func group; AtomInterpreter needs the full move-pipeline
    # dialect group to dispatch against the lowered statements.
    method.dialects = kernel

    # Stage 3: measure_lower.
    rule = MeasureLower.from_method(method, arch_spec=get_arch_spec())
    Walk(rule).rewrite(method.code)

    # Assert the full block structure (types, order, and key attribute
    # values for the stateful-op sites).
    block = method.callable_region.blocks[0]
    stmts = list(block.stmts)

    expected_types = [
        move.Load,
        move.Fill,
        move.ConstZone,
        move.EndMeasure,
        ilist.New,
        move.Store,
        func.Return,
    ]
    assert [
        type(s) for s in stmts
    ] == expected_types, (
        f"block structure mismatch; got {[type(s).__name__ for s in stmts]}"
    )

    fill = next(s for s in stmts if isinstance(s, move.Fill))
    assert fill.location_addresses == (LocationAddress(0, 0, 0),)

    const_zone = next(s for s in stmts if isinstance(s, move.ConstZone))
    assert const_zone.value == ZoneAddress(0)

    end_measure = next(s for s in stmts if isinstance(s, move.EndMeasure))
    assert end_measure.zone_addresses == (ZoneAddress(0),)
