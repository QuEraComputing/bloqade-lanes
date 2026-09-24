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
    stack-depth order: deepest arg first, top-of-stack arg last, by the
    layout ``_stack_order`` reads off the decoder.  The arg reference on the
    consumer is updated to the clone in-place; the original becomes dead and
    is removed by Pass 2.

Pass 2 — DCE
    Removes the now-dead original constant definitions left behind by
    Pass 1.

Pass 3 — spill to locals
    An operand is consumed by the op that pops it, so a non-constant value
    with more than one consumer (e.g. an ``AwaitMeasure`` result read by N
    ``GetItem`` statements) cannot stay where it was pushed. It is spilled
    instead: a ``StoreLocal`` right after its producer parks it in a local,
    out of every operand op's reach, and a ``LoadLocal`` before each consumer
    brings a copy back where the consumer expects it — below the constants
    Pass 1 placed:

        v = AwaitMeasure(...)
        StoreLocal(v, k)          ← off the operand stack
        ...                       ← anything at all; local k is untouched
        LoadLocal(k)              ← one reload per consumer
        <const args of c_i>       ← placed by Pass 1
        c_i(reload, …)

    A consumer that takes a spilled value takes its other non-constant
    arguments from locals too, reloaded in stack order. A reload lands on
    top of whatever is already on the stack, so on its own it would sit
    above an argument that belongs above it.

    So does a consumer whose arguments are not on top of the stack in the
    order it pops them. Values each used once are still consumed out of
    order — two detector arrays built before either is set, so the first
    ``SetDetector`` finds the second array on top — and a walk of the
    operand stack finds those, and takes them from locals, where the order
    they were pushed in no longer matters.

    Slots are handed out lowest-free-first and returned after a value's last
    reload, so a function reserves one local per spilled value live at once.

    This is the work the bytecode's ``dup``/``swap``/``pop`` did before a
    function had locals of its own (#1038).
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable

from kirin import ir
from kirin.rewrite import Walk
from kirin.rewrite.abc import RewriteResult, RewriteRule
from kirin.rewrite.dce import DeadCodeElimination

from bloqade.lanes.dialects import stack_move


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
    statement order and emit correct bytecode.

    Raises ``ValueError`` if the method contains more than one block, if it
    has one of the two shapes below, or if it keeps more values spilled at
    once than a frame has locals. All IR produced by the current compiler
    pipeline is single-block and has neither shape; call this function after
    ``RewriteMoveToStackMove`` and before ``dump_program``.
    """
    blocks = method.callable_region.blocks
    if len(blocks) != 1:
        raise ValueError(
            f"stackify only supports single-block methods; got {len(blocks)} blocks"
        )
    # Before Pass 1, which hoists constants and so erases the second shape.
    _reject_unsupported(list(blocks[0].stmts))

    # Pass 1 + 2: clone constants into correct stack-depth order, then DCE.
    Walk(CloneConstants()).rewrite(method.code)
    Walk(DeadCodeElimination()).rewrite(method.code)

    # Pass 3: spill every value that more than one statement consumes.
    _spill_to_locals(blocks[0])


# Highest local index a frame may name: the Rust validator's
# ``MAX_LOCAL_INDEX``, which ``test_stackify`` pins this against.
_MAX_LOCAL_INDEX = 1023


def _is_constant(value: ir.SSAValue) -> bool:
    return isinstance(value, ir.ResultValue) and value.owner.has_trait(ir.ConstantLike)


def _reject_unsupported(stmts: list[ir.Statement]) -> None:
    """Refuse the two shapes Pass 3 would silently get wrong.

    Only decoded bytecode has either; supporting them is #1050.

    - A ``Dup``. Pass 3 models every statement as popping its operands, but
      ``dup`` only copies the top, so its operand would be spilled and the
      reload left behind on the stack.
    - A constant operand below a non-constant one. Pass 1 hoists every
      constant above the consumer's other operands, and the reloads go
      beneath those constants, so the operands would come out reordered.
    """
    dup = next((stmt for stmt in stmts if isinstance(stmt, stack_move.Dup)), None)
    if dup is not None:
        raise ValueError(
            f"stackify does not support a decoded dup ({dup}): it copies the "
            f"top without popping it, which Pass 3 cannot express yet (#1050)"
        )
    for stmt in stmts:
        order = [stmt.args[i] for i in _stack_order(stmt)]
        first_constant = next(
            (depth for depth, arg in enumerate(order) if _is_constant(arg)), None
        )
        if first_constant is not None and any(
            not _is_constant(arg) for arg in order[first_constant:]
        ):
            raise ValueError(
                f"stackify does not support a constant operand below a "
                f"non-constant one ({stmt}): constants are placed above every "
                f"other operand, which would reorder them (#1050)"
            )


def _value_type(value: ir.SSAValue) -> str:
    """The vihaco type ``value`` has at run time, spelled for ``StoreLocal``.

    Decided by what produced the value rather than its kirin type. Every value
    a lanes op returns without simulating it — a measurement future, an array,
    an element of one, a detector — is an ``Undefined`` placeholder on the
    machine, whatever it stands for, until #776 decides what those values
    are. Constants are cloned rather than spilled, so the one other producer
    of a spilled value is a decoded program's own ``LoadLocal``, which says.
    """
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


def _values_to_spill(
    stmts: list[ir.Statement], spillable: Callable[[ir.SSAValue], bool]
) -> set[ir.SSAValue]:
    """Every value Pass 3 parks in a local.

    Those with more than one consumer, every other non-constant argument of a
    statement that takes one, and the arguments of any statement that would
    not find them on top of the operand stack in the order it pops them. The
    last is found by walking the stack: whatever is not spilled stays where
    its producer pushed it, so each statement's remaining arguments have to be
    exactly the top of it.

    One walk is enough. A value spilled for being out of place has one
    consumer, the statement that found it so, and parking it only takes it
    out of the stack — which cannot put anything else out of place, because
    whatever was consumed while it waited was consumed from above it.
    """
    shared = {
        arg
        for stmt in stmts
        for arg in stmt.args
        if spillable(arg) and len(arg.uses) > 1
    }
    spilled = set(shared)
    for stmt in stmts:
        if any(arg in shared for arg in stmt.args):
            spilled.update(arg for arg in stmt.args if spillable(arg))

    # Insertion-ordered, so the last key is the top — and a value parked for
    # being out of place comes out wherever it is without rebuilding the rest.
    stack: dict[ir.SSAValue, None] = {}
    for stmt in stmts:
        # A constant is cloned in front of its one consumer, above the rest
        # of its arguments; it never waits on the stack.
        if stmt.has_trait(ir.ConstantLike):
            continue
        order = (stmt.args[i] for i in _stack_order(stmt))
        need = [arg for arg in order if spillable(arg) and arg not in spilled]
        top = list(itertools.islice(reversed(stack), len(need)))[::-1]
        if need and top == need:
            for _ in need:
                stack.popitem()
        elif need:
            spilled.update(need)
            for arg in need:
                stack.pop(arg, None)
        # Results pushed deepest first; the first declared ends up on top.
        stack.update((r, None) for r in reversed(stmt.results) if r not in spilled)
    return spilled


def _spill_to_locals(block: ir.Block) -> None:
    """Pass 3: park every value that cannot wait on the stack for its
    consumers in a local, and reload it for each (see the module docs)."""
    stmts: list[ir.Statement] = list(block.stmts)

    def spillable(arg: ir.SSAValue) -> bool:
        return isinstance(arg, ir.ResultValue) and not arg.owner.has_trait(
            ir.ConstantLike
        )

    spilled = _values_to_spill(stmts, spillable)
    if not spilled:
        return

    # O(n): for each position, the start of the contiguous ConstantLike run
    # that ends immediately before it. A reload goes there, so it sits below
    # the const clones Pass 1 placed.
    const_run_start: list[int] = list(range(len(stmts)))
    for i in range(1, len(stmts)):
        if stmts[i - 1].has_trait(ir.ConstantLike):
            const_run_start[i] = const_run_start[i - 1]

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
        # Reloads, deepest first, below the consumer's constants. A value's
        # slot is free again after its last one.
        anchor = stmts[const_run_start[idx]]
        args = list(stmt.args)
        reloaded = False
        for i in _stack_order(stmt):
            arg = args[i]
            if arg not in spilled:
                continue
            load = reload(arg, slots[arg])
            plan.append((load, anchor))
            args[i] = load.result
            reloaded = True
            uses_left[arg] -= 1
            if uses_left[arg] == 0:
                heapq.heappush(free, slots.pop(arg))
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
