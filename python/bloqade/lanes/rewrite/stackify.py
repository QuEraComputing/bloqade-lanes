"""Stackification for single-block stack_move ir.Method.

``stackify(method)`` is the primary entry point.  It normalises a
stack_move ``ir.Method`` in place so that ``BytecodeEncoder`` can walk
the entry block in statement order and emit correct bytecode.

Restriction: the method must contain exactly one block with no
branches or back-edges.  This is enforced at runtime.  All IR produced
by the current compiler pipeline (Move → StackMove → Bytecode) satisfies
this invariant.

Three sub-passes run in sequence:

Pass 1 — ``CloneConstants`` (``RewriteRule`` via ``Walk``)
    For each consuming statement, clones every ``ConstantLike`` (``Const*``)
    argument and inserts the clone immediately before the consumer in
    stack-depth order: deepest arg first (highest ``stmt.args`` index),
    top-of-stack arg last (index 0).  The arg reference on the consumer
    is updated to the clone in-place; the original becomes dead and is
    removed by Pass 2.

Pass 2 — DCE
    Removes the now-dead original constant definitions left behind by
    Pass 1.

Pass 3 — Dup/Swap insertion
    For each non-Pure SSA value consumed by more than one statement
    (e.g. an ``AwaitMeasure`` result used by N ``GetItem`` statements),
    inserts ``Dup``/``Swap`` instructions so the value remains available
    for later consumers without violating the stack discipline.
    For consumers ``c_0, …, c_{N-1}`` in block order the pattern emitted
    before each non-last consumer ``c_k`` is:

        Dup(live_copy)          ← save a copy below c_k's const args
        <const args of c_k>     ← already placed by Pass 1
        c_k(dup_result, …)      ← consumes the dup
        Swap(c_k.result, live)  ← out_top = live copy (for c_{k+1})

    The last consumer uses ``out_top`` from the preceding Swap directly.
"""

from __future__ import annotations

from kirin import ir
from kirin.rewrite import Walk
from kirin.rewrite.abc import RewriteResult, RewriteRule
from kirin.rewrite.dce import DeadCodeElimination

from bloqade.lanes.dialects import stack_move


class CloneConstants(RewriteRule):
    """Clone ``ConstantLike`` (``Const*``) args to be immediately before their consumer.

    Stack discipline: for a consumer with args ``[a0, ..., aN]`` where index 0
    is the top of the stack and index N is the deepest, the defining statement
    of ``aN`` must be emitted first and ``a0`` last.  This rule iterates args
    from highest index to lowest; each ``insert_before(node)`` call places the
    new clone right before the consumer, so successive insertions build up:
    ``[clone_N, ..., clone_0, consumer]``.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        changed = False
        for i in range(len(node.args) - 1, -1, -1):
            arg = node.args[i]
            if not isinstance(arg, ir.ResultValue):
                continue
            owner = arg.owner
            if not owner.has_trait(ir.ConstantLike):
                continue
            clone = owner.from_stmt(owner)
            node.args[i] = clone.results[0]
            clone.insert_before(node)
            changed = True
        return RewriteResult(has_done_something=changed)


def stackify(method: ir.Method) -> None:
    """Normalise a single-block stack_move ir.Method for bytecode encoding.

    Applies all three stackification sub-passes in sequence (CloneConstants,
    DCE, Dup/Swap insertion) so that ``dump_program`` can walk the block in
    statement order and emit correct bytecode.

    Raises ``ValueError`` if the method contains more than one block.  All IR
    produced by the current compiler pipeline is single-block; call this
    function after ``RewriteMoveToStackMove`` and before ``dump_program``.
    """
    blocks = method.callable_region.blocks
    if len(blocks) != 1:
        raise ValueError(
            f"stackify only supports single-block methods; got {len(blocks)} blocks"
        )

    # Pass 1 + 2: clone constants into correct stack-depth order, then DCE.
    Walk(CloneConstants()).rewrite(method.code)
    Walk(DeadCodeElimination()).rewrite(method.code)

    # Pass 3: insert Dup/Swap for non-ConstantLike multi-use SSA values.
    block = blocks[0]
    stmts: list[ir.Statement] = list(block.stmts)

    # O(n): last position in stmts where each non-ConstantLike arg appears.
    # Single forward scan; later assignments overwrite earlier ones → last wins.
    last_consumer: dict[ir.ResultValue, int] = {}
    for idx, stmt in enumerate(stmts):
        for arg in stmt.args:
            if isinstance(arg, ir.ResultValue) and not arg.owner.has_trait(
                ir.ConstantLike
            ):
                last_consumer[arg] = idx

    # O(n): for each position, the start of the contiguous ConstantLike run
    # that ends immediately before it.  Dup must be inserted there so it sits
    # below all const-clone args on the stack.
    const_run_start: list[int] = list(range(len(stmts)))
    for i in range(1, len(stmts)):
        if stmts[i - 1].has_trait(ir.ConstantLike):
            const_run_start[i] = const_run_start[i - 1]

    # Deferred insertion plan: (new_stmt, insert_before_target).
    insertion_plan: list[tuple[ir.Statement, ir.Statement]] = []
    # current_live[orig] tracks which SSA value currently carries orig's stack slot.
    # Populated lazily: only set when a Dup/Swap chain has been started for orig.
    current_live: dict[ir.ResultValue, ir.SSAValue] = {}

    for idx, stmt in enumerate(stmts):
        for i, arg in enumerate(stmt.args):
            if not isinstance(arg, ir.ResultValue):
                continue
            # ConstantLike args are already cloned by CloneConstants; skip.
            if arg.owner.has_trait(ir.ConstantLike):
                continue
            if len(arg.uses) <= 1:
                continue

            live = current_live.get(arg, arg)
            is_last = last_consumer.get(arg, idx) == idx

            if is_last:
                stmt.args[i] = live
                continue

            dup = stack_move.Dup(value=live)
            insertion_plan.append((dup, stmts[const_run_start[idx]]))
            stmt.args[i] = dup.results[0]

            # Swap goes immediately after the consumer so the live copy
            # bubbles back to the top for the next consumer.
            swap = stack_move.Swap(in_top=stmt.results[0], in_bot=live)
            insertion_plan.append((swap, stmts[idx + 1]))
            current_live[arg] = swap.out_top

    for new_stmt, target in insertion_plan:
        new_stmt.insert_before(target)
