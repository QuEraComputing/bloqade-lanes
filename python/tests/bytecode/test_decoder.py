from bloqade.lanes.bytecode import Instruction, Program
from bloqade.lanes.bytecode.lowering import load_program


def test_empty_program_returns_method_with_empty_body():
    prog = Program(version=(1, 0), instructions=[Instruction.return_()])
    method = load_program(prog)
    assert method.sym_name == "main"
    # Body should have one statement (stack_move.Return), no other statements.
    block = method.callable_region.blocks[0]
    assert len(block.stmts) == 1
