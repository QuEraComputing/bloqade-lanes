from kirin import interp, ir
from kirin.analysis import const
from kirin.dialects import py, ssacfg
from kirin.rewrite import Chain, ConstantFold, DeadCodeElimination, Fixpoint, Walk

from bloqade.lanes.bytecode import Instruction, Program
from bloqade.lanes.bytecode.decode import load_program
from bloqade.lanes.dialects import stack_move


def test_dialect_exists():
    assert stack_move.dialect.name == "lanes.stack_move"


def test_no_typeinfer_table_registered():
    # Return and Halt lowered to kirin.basic's func dialect, so
    # stack_move no longer needs its own typeinfer MethodTable — every
    # remaining stack_move statement's result type is fully determined
    # by its declaration, and func.Return / func.ConstantNone carry
    # their own type-inference methods.
    assert stack_move.dialect.interps.get("typeinfer") is None


def _decoded(instructions) -> ir.Method:
    """A decoded program in a group a kirin interpreter can run: the decoder's
    own lacks ``ssacfg``, which interprets the function body, and ``py`` for
    the constants ``ConstantFold`` inserts."""
    method = load_program(Program(version=(1, 0), instructions=instructions))
    return method.similar(method.dialects.union([ssacfg.dialect, py.constant.dialect]))


def test_concrete_dup_returns_its_operand_twice():
    """``dup`` copies the top: running the program returns the constant."""
    method = _decoded(
        [Instruction.const_int(7), Instruction.dup(), Instruction.return_()]
    )

    _, result = interp.Interpreter(method.dialects).run(method)
    assert result == 7


def test_constants_propagate_through_a_dup_that_does_not_fold():
    """Constant propagation goes through ``Dup``, so its copies are known to be
    the constant. But a ``Dup`` is not ``Pure``: ``ConstantFold`` + DCE leave it
    where it is, for ``InlineDup`` to take out if anything should."""
    method = _decoded(
        [
            Instruction.const_float(0.5),
            Instruction.dup(),
            Instruction.global_rz(),
            Instruction.global_rz(),
            Instruction.halt(),
        ]
    )
    dup = next(
        s for s in method.callable_region.walk() if isinstance(s, stack_move.Dup)
    )

    frame, _ = const.Propagate(method.dialects).run(method)
    assert frame.entries[dup.top] == frame.entries[dup.below] == const.Value(0.5)

    for stmt in method.callable_region.walk():
        for result in stmt.results:
            if result in frame.entries:
                result.hints["const"] = frame.entries[result]
    Fixpoint(Walk(Chain(ConstantFold(), DeadCodeElimination()))).rewrite(method.code)

    stmts = list(method.callable_region.walk())
    assert dup in stmts
    gates = [s for s in stmts if isinstance(s, stack_move.GlobalRz)]
    assert [g.rotation_angle for g in gates] == [dup.top, dup.below]


def test_dup_is_not_pure():
    """Part of the program the bytecode spells: DCE must not drop one."""
    assert not stack_move.Dup.has_trait(ir.Pure)
