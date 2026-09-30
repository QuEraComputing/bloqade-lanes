"""Stackification for single-block stack_move ir.Method.

``stackify(method)`` is the primary entry point.  It normalises a
stack_move ``ir.Method`` in place so that ``BytecodeEncoder`` can walk
the entry block in statement order and emit correct bytecode.

Restriction: the method must contain exactly one block with no
branches or back-edges.  This is enforced at runtime.  All IR produced
by the current compiler pipeline (Move → StackMove → Bytecode) satisfies
this invariant.

The target is SSA stack ordering: a block in which every statement finds its
arguments on top of the operand stack, in the order it pops them, and pushes
its results there — so each value is popped exactly once, where it was
pushed. Such a block is a stack program (see ``_is_stack_program``), and
``stackify`` leaves one exactly as it is. The decoder builds every statement
by popping a simulated stack, so a decoded block is one, and so is
``stackify``'s own output: decoded bytecode comes back as it was, and a
second run changes nothing.

A block that is not one is repaired by two rules, one per kind of value:

- A constant is always re-created: cloned in front of each of its consumers,
  in stack order (Pass 1), so it never waits on the stack. A ``Dup`` of a
  constant is inlined first — a copy of a constant is that constant — so its
  copies are re-created the same way. Every instruction is one 14-byte word
  and a constant is one instruction, as a ``load`` is, so re-creating one is
  never dearer than keeping it in a local.
- Any other value stays on the stack while it is in stack order, and is
  moved to a local otherwise (Pass 3): stored right after its producer, and
  loaded again in front of each consumer, where that consumer's arguments
  have it.

Pass 1 — ``CloneConstants`` (``RewriteRule`` via ``Walk``)
    For each consuming statement, clones every ``ConstantLike`` (``Const*``)
    argument and inserts the clone immediately before the consumer in
    stack-depth order: deepest arg first, top-of-stack arg last, by the
    layout ``_stack_order`` reads off the decoder.  The arg reference on the
    consumer is updated to the clone in-place; the original becomes dead and
    is removed by Pass 2.

Pass 2 — DCE
    Removes the now-dead original constant definitions left behind by
    Pass 1, to a fixpoint: a dead ``Pure`` consumer takes the clones placed
    in front of it with it. So does a constant nothing consumes: the
    bytecode leaves it on the stack and never reads it, so dropping it
    changes nothing the program computes. A ``Dup`` is not ``Pure``, so one
    whose copies nothing reads stays, as the ``dup`` it was.

Pass 3 — move out-of-order values to locals
    A value is in stack order when it has one consumer and waits on the stack
    for it among the deepest of that consumer's arguments that are exactly
    the top of the stack when it runs. Whatever a consumer does not take from
    the stack is supplied on top — clones and reloads, interleaved deepest
    first — so every argument above the first one supplied is supplied too.
    ``_values_to_spill`` finds the rest greedily, walking the stack; each is
    moved:

        v = AwaitMeasure(...)
        StoreLocal(v, k)          ← off the operand stack
        ...                       ← anything at all; local k is untouched
        <args of c_i below v, taken from the stack or supplied>
        LoadLocal(k)              ← one reload per consumer
        <args of c_i above v, supplied>
        c_i(…, reload, …)

    A value with several consumers is out of order at all but one of them at
    least, and is moved outright (e.g. an ``AwaitMeasure`` result read by N
    ``GetItem`` statements). One used once is moved when it is not where its
    consumer pops it: below a constant it belongs above, or beneath values
    consumed later — two detector arrays built before either is set, so the
    first ``SetDetector`` finds the second array on top.

    Slots are handed out lowest-free-first and returned after a value's last
    reload, so a function reserves one local per moved value live at once.

    This is the work the bytecode's ``dup``/``swap``/``pop`` did before a
    function had locals of its own (#1038).
"""

from __future__ import annotations

import heapq
import itertools
from typing import cast

from kirin import ir
from kirin.rewrite import Fixpoint, Walk
from kirin.rewrite.abc import RewriteResult, RewriteRule
from kirin.rewrite.dce import DeadCodeElimination

from bloqade.lanes.dialects import stack_move
from bloqade.lanes.rewrite.inline_dup import InlineDup


class CloneConstants(RewriteRule):
    """Clone ``ConstantLike`` (``Const*``) args to be immediately before their consumer.

    Stack discipline: the defining statement of the deepest operand must be
    emitted first and the top-of-stack one last. This rule visits args in
    ``_stack_order`` — deepest first — and each ``insert_before(node)`` places
    the new clone right before the consumer, so successive insertions build
    up ``[clone_deepest, ..., clone_top, consumer]``.

    Iterating from the highest ``args`` index down, as this once did, gets
    the scalars of ``LocalR``/``GlobalR`` right but reverses every operand
    group of two or more — ``initial_fill``'s locations, ``new_array``'s
    elements — which the decoder builds bottom-to-top.
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        args = list(node.args)
        changed = False
        for i in _stack_order(node):
            arg = args[i]
            if not isinstance(arg, ir.ResultValue):
                continue
            owner = arg.owner
            if not owner.has_trait(ir.ConstantLike):
                continue
            clone = owner.from_stmt(owner)
            args[i] = clone.results[0]
            clone.insert_before(node)
            changed = True
        # Once, not per argument: kirin rebuilds the whole tuple on each
        # assignment, which made a wide `new_array` quadratic.
        if changed:
            node.args = args
        return RewriteResult(has_done_something=changed)


def stackify(method: ir.Method) -> None:
    """Normalise a single-block stack_move ir.Method for bytecode encoding.

    Applies all three stackification sub-passes in sequence (CloneConstants,
    DCE, spilling to locals) so that ``dump_program`` can walk the block in
    statement order and emit correct bytecode — unless the block already is a
    stack program, which is left unchanged.

    Raises ``ValueError`` if the method contains more than one block, or if
    it keeps more values spilled at once than a frame has locals. All IR
    produced by the current compiler pipeline is single-block; call this
    function after ``RewriteMoveToStackMove`` and before ``dump_program``.
    """
    blocks = method.callable_region.blocks
    if len(blocks) != 1:
        raise ValueError(
            f"stackify only supports single-block methods; got {len(blocks)} blocks"
        )
    if _is_stack_program(list(blocks[0].stmts)):
        return

    # A copy of a constant is that constant: give its consumers the constant,
    # for Pass 1 to re-create in front of each like any other use.
    for stmt in list(blocks[0].stmts):
        if isinstance(stmt, stack_move.Dup) and _is_constant(stmt.value):
            InlineDup().rewrite_Statement(stmt)

    # Pass 1 + 2: re-create every constant in front of each consumer, in stack
    # order, then DCE — to a fixpoint, because a dead `Pure` consumer's clones
    # die with it; they would otherwise be left on the stack for the next
    # operand to pop.
    Walk(CloneConstants()).rewrite(method.code)
    Fixpoint(Walk(DeadCodeElimination())).rewrite(method.code)

    # Pass 3: move every value that is not in stack order to a local.
    _spill_to_locals(blocks[0])


# Highest local index a frame may name: the Rust validator's
# ``MAX_LOCAL_INDEX``, which ``test_stackify`` pins this against.
_MAX_LOCAL_INDEX = 1023


def _is_constant(value: ir.SSAValue) -> bool:
    return isinstance(value, ir.ResultValue) and value.owner.has_trait(ir.ConstantLike)


def _is_stack_program(stmts: list[ir.Statement]) -> bool:
    """Whether ``stmts`` already run as they stand on the operand stack.

    Walks the stack with SSA identities, the layout the decoder builds and
    ``dump_program`` emits: each statement pops its arguments deepest first
    by ``_stack_order`` and pushes its results with the first declared on
    top. True when every statement finds its arguments exactly on top, which
    also means each value is popped once, where it was pushed. A value
    nothing pops may stay: ``halt`` and ``ret`` discard what is left.
    """
    stack: list[ir.SSAValue] = []
    for stmt in stmts:
        order = [stmt.args[i] for i in _stack_order(stmt)]
        if order:
            top = stack[len(stack) - len(order) :] if len(order) <= len(stack) else []
            if len(top) != len(order) or any(a is not b for a, b in zip(top, order)):
                return False
            del stack[len(stack) - len(order) :]
        stack.extend(reversed(stmt.results))
    return True


def _spillable(arg: ir.SSAValue) -> bool:
    return isinstance(arg, ir.ResultValue) and not arg.owner.has_trait(ir.ConstantLike)


def _value_type(value: ir.SSAValue) -> str:
    """The vihaco type ``value`` has at run time, spelled for ``StoreLocal``.

    Decided by what produced the value rather than its kirin type. Every value
    a lanes op returns without simulating it — a measurement future, an array,
    an element of one, a detector — is an ``Undefined`` placeholder on the
    machine, whatever it stands for, until #776 decides what those values
    are. Constants are re-created rather than moved, and so are the copies
    of one, so the other producers of a moved value are a decoded program's
    own ``LoadLocal``, which says, and its ``Dup``, whose copies are whatever
    it copied.
    """
    while isinstance(value, ir.ResultValue) and isinstance(value.owner, stack_move.Dup):
        value = value.owner.value
    if isinstance(value, ir.ResultValue) and isinstance(
        value.owner, stack_move.LoadLocal
    ):
        return value.owner.value_type
    return "undef"


def _stack_order(stmt: ir.Statement) -> list[int]:
    """Indices into ``stmt.args``, deepest stack slot first.

    The order the bytecode pushes a statement's operands in, which is the
    reverse of the order ``BytecodeDecoder`` pops them. That is ``args``
    order — an operand group bottom-to-top, after any fixed operand beneath
    it, like ``GetItem``'s array — except for the rotations, which take their
    angles from above their locations with the axis on top.
    """
    n = len(stmt.args)
    if isinstance(stmt, stack_move.LocalR):  # (axis, rotation, *locations)
        return [*range(2, n), 1, 0]
    if isinstance(stmt, stack_move.LocalRz):  # (rotation, *locations)
        return [*range(1, n), 0]
    if isinstance(stmt, stack_move.GlobalR):  # (axis, rotation)
        return [1, 0]
    return list(range(n))


def _values_to_spill(stmts: list[ir.Statement]) -> set[ir.SSAValue]:
    """Every non-constant value Pass 3 moves to a local: each one not in
    stack order.

    A value is in stack order when it has one consumer and waits on the
    operand stack for it, as one of the deepest of that consumer's arguments
    that are exactly the top of the stack when it runs — every argument above
    the first one that is not has to be supplied on top, so is not either.
    Constants are always re-created in front of their consumers (Pass 1), so
    each is supplied, never waits on the stack, and never stands in anyone's
    way.

    Found greedily, by walking the stack: whatever is not moved stays where
    its producer pushed it. At each statement, the longest run of its
    deepest arguments that is exactly the top stays; every other argument
    that is waiting on the stack is moved. One walk is enough: a value is
    only ever moved at its one consumer, and taking it off the stack beneath
    whatever that consumer pops cannot put anything else out of order,
    because whatever was popped while it waited was popped from above it.
    """
    # A value popped twice is out of order at one of them at least.
    moved = {
        arg
        for stmt in stmts
        for arg in stmt.args
        if _spillable(arg) and len(arg.uses) > 1
    }
    # Insertion-ordered, so the last key is the top — and a value moved for
    # being out of order comes out wherever it is without rebuilding the rest.
    stack: dict[ir.SSAValue, None] = {}
    for stmt in stmts:
        if stmt.has_trait(ir.ConstantLike):
            continue
        order = [stmt.args[i] for i in _stack_order(stmt)]
        kept = 0
        top = next(reversed(stack), None)
        if top is not None and top in order:
            depth = order.index(top) + 1
            run = order[:depth]
            # A constant is never on the stack, so a run holding one is not.
            if run == list(itertools.islice(reversed(stack), depth))[::-1]:
                kept = depth
        for _ in range(kept):
            stack.popitem()
        for arg in order[kept:]:
            if arg in stack:
                del stack[arg]
                moved.add(arg)
        # Results pushed deepest first; the first declared ends up on top.
        stack.update((r, None) for r in reversed(stmt.results) if r not in moved)
    return moved


def _spill_to_locals(block: ir.Block) -> None:
    """Pass 3: park every value that cannot wait on the stack for its
    consumers in a local, and reload it for each (see the module docs)."""
    stmts: list[ir.Statement] = list(block.stmts)

    spilled = _values_to_spill(stmts)
    if not spilled:
        return

    uses_left = {value: len(value.uses) for value in spilled}
    slots: dict[ir.SSAValue, int] = {}
    free: list[int] = []
    # Past every local the block already names: a decoded program has its own,
    # and handing one of those out would overwrite it.
    next_slot = 1 + max(
        (
            stmt.index
            for stmt in stmts
            if isinstance(stmt, (stack_move.StoreLocal, stack_move.LoadLocal))
        ),
        default=-1,
    )

    def take() -> int:
        nonlocal next_slot
        if free:
            return heapq.heappop(free)
        if next_slot > _MAX_LOCAL_INDEX:
            raise ValueError(
                f"stackify needs local {next_slot}, past the {_MAX_LOCAL_INDEX + 1} "
                f"a frame may hold: more values are spilled at once than a "
                f"function can keep"
            )
        next_slot += 1
        return next_slot - 1

    def reload(value: ir.SSAValue, slot: int) -> stack_move.LoadLocal:
        load = stack_move.LoadLocal(index=slot, value_type=_value_type(value))
        load.result.type = value.type
        return load

    # (new statement, the statement it goes before), applied in plan order:
    # statements planned before one anchor land in the order they were
    # planned.
    plan: list[tuple[ir.Statement, ir.Statement]] = []

    for idx, stmt in enumerate(stmts):
        # Reloads, deepest first, interleaved with the constants Pass 1 put
        # right in front of the consumer: each goes before the clone of the
        # next constant above it, or the consumer if none is. A value's slot
        # is free again after its last one.
        args = list(stmt.args)
        reloaded = False
        below_next_clone: list[stack_move.LoadLocal] = []
        for i in _stack_order(stmt):
            arg = args[i]
            if _is_constant(arg):
                clone = cast(ir.ResultValue, arg).owner
                plan.extend((load, clone) for load in below_next_clone)
                below_next_clone.clear()
                continue
            if arg not in spilled:
                continue
            load = reload(arg, slots[arg])
            below_next_clone.append(load)
            args[i] = load.result
            reloaded = True
            uses_left[arg] -= 1
            if uses_left[arg] == 0:
                heapq.heappush(free, slots.pop(arg))
        plan.extend((load, stmt) for load in below_next_clone)
        # Once per consumer: kirin rebuilds the argument tuple on every
        # assignment, which made this quadratic in a consumer's width.
        if reloaded:
            stmt.args = args

        # Spills, right after the producer. Its results are on top, the first
        # declared highest, so they come off in declaration order down to the
        # deepest spilled one; any above it that is not itself spilled is
        # parked on the way and put back.
        deepest = max(
            (i for i, result in enumerate(stmt.results) if result in spilled),
            default=None,
        )
        if deepest is None:
            continue
        # A spilled result has a consumer, so something follows its producer.
        after = stmts[idx + 1]
        parked: list[tuple[ir.ResultValue, int, stack_move.StoreLocal]] = []
        for result in stmt.results[: deepest + 1]:
            slot = take()
            store = stack_move.StoreLocal(
                value=result, index=slot, value_type=_value_type(result)
            )
            plan.append((store, after))
            if result in spilled:
                slots[result] = slot
            else:
                parked.append((result, slot, store))
        for result, slot, store in reversed(parked):
            load = reload(result, slot)
            plan.append((load, after))
            for use in list(result.uses):
                if use.stmt is not store:
                    use.stmt.args[use.index] = load.result
            heapq.heappush(free, slot)

    for new_stmt, target in plan:
        new_stmt.insert_before(target)
