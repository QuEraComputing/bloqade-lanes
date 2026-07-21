from dataclasses import dataclass

from kirin import ir, rewrite

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode import Program
from bloqade.lanes.bytecode.encode import dump_program
from bloqade.lanes.dialects import move, stack_move
from bloqade.lanes.rewrite.move2stack_move import RewriteMoveToStackMove
from bloqade.lanes.rewrite.stackify import stackify
from bloqade.lanes.utils import statements_outside_dialect_group


@dataclass
class MoveToStackMove:
    """Lower a ``move``-dialect kernel to a canonicalized ``stack_move`` kernel.

    ``emit`` runs the full move → stack_move lowering pipeline, producing an
    ``ir.Method`` that is stack-consistent and ready for bytecode emission:

    1. ``RewriteMoveToStackMove`` — in-place move → stack_move rewrite
       (strips Load/Store state threading, materialises address attributes
       as ``Const*`` SSA values, rebuilds Measure/AwaitMeasure/GetItem).
    2. DCE + CSE to a fixpoint — the cleanup the encoder pipeline expects
       before stackification (see ``stackify``'s docstring).
    3. ``stackify`` — normalise into stack-consistent form (single block,
       each SSA value used at most once, defining statements in stack order).

    ``RewriteMoveToStackMove`` only lowers the subset of ``move`` statements
    the bytecode path supports. When ``no_raise`` is ``False``, ``emit`` runs
    ``statements_outside_dialect_group`` after dropping ``move`` from the group
    and raises if any statement is left outside it — Kirin's ``verify()`` does
    not check dialect-group membership, so an unlowered statement would
    otherwise slip through and fail lazily inside ``dump_program``.

    ``emit_bytecode`` runs ``emit`` and encodes the result to a bytecode
    ``Program`` via ``dump_program``.
    """

    arch_spec: ArchSpec

    def emit(self, main: ir.Method, no_raise: bool = True) -> ir.Method:
        # Copy into a dialect group that includes stack_move so the rewritten
        # statements are legal members of the method's dialects.
        out = main.similar(main.dialects.union([stack_move.dialect]))

        # move → stack_move (single pass; the rule deletes the move statements).
        rewrite.Walk(RewriteMoveToStackMove(arch_spec=self.arch_spec)).rewrite(out.code)

        # DCE + CSE, matching the cleanup the real pipeline runs before stackify.
        rewrite.Fixpoint(
            rewrite.Walk(
                rewrite.Chain(
                    rewrite.DeadCodeElimination(),
                    rewrite.CommonSubexpressionElimination(),
                )
            )
        ).rewrite(out.code)

        # Drop the now-unused move dialect from the group.
        out = out.similar(out.dialects.discard(move.dialect))

        if not no_raise:
            # RewriteMoveToStackMove passes through any move statement it does
            # not handle, and verify() does not police dialect-group membership,
            # so an unlowered statement would otherwise surface as a confusing
            # EncodingError deep inside dump_program. Fail fast with a precise
            # error naming the offending statement kinds.
            leftover = statements_outside_dialect_group(out)
            if leftover:
                kinds = sorted({type(stmt).__name__ for stmt in leftover})
                raise ValueError(
                    "MoveToStackMove left statements outside the stack_move "
                    f"dialect group: {', '.join(kinds)}; RewriteMoveToStackMove "
                    "does not lower them, so the kernel cannot be emitted as "
                    "stack_move IR"
                )

        # Canonicalize into stack-consistent form, ready for dump_program.
        stackify(out)

        if not no_raise:
            out.verify()

        return out

    def emit_bytecode(
        self,
        main: ir.Method,
        version: tuple[int, int] = (1, 0),
        no_raise: bool = True,
    ) -> Program:
        return dump_program(self.emit(main, no_raise=no_raise), version=version)
