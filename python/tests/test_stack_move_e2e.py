"""End-to-end: bytecode Program -> decode -> lower -> measure_lower."""

from kirin.rewrite import Walk

from bloqade.lanes._prelude import kernel
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.bytecode import Instruction, Program
from bloqade.lanes.bytecode.decode import load_program
from bloqade.lanes.dialects import move
from bloqade.lanes.rewrite.measure_lower import MeasureLower
from bloqade.lanes.rewrite.stack_move2move import RewriteStackMoveToMove


def test_minimal_program_runs_end_to_end():
    """Bytecode Program -> stack_move IR -> multi-dialect IR with EndMeasure."""
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
    block = method.callable_region.blocks[0]

    # Stage 2: lower stack_move -> multi-dialect IR (in place).
    Walk(RewriteStackMoveToMove()).rewrite(block)

    # The method's dialect group still points at the decoder's
    # stack_move+func group; AtomInterpreter needs the full move-pipeline
    # dialect group to dispatch against the lowered statements.
    method.dialects = kernel

    # Stage 3: measure_lower (in place).
    rule = MeasureLower.from_method(method, arch_spec=get_arch_spec())
    Walk(rule).rewrite(block)

    # The method now contains move.EndMeasure, not move.Measure.
    assert any(isinstance(s, move.EndMeasure) for s in block.stmts)
    assert not any(isinstance(s, move.Measure) for s in block.stmts)
