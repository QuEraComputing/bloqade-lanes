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
from collections.abc import Callable

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
    DCE, spilling to locals) so that ``dump_program`` can walk the block in
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

    # Pass 3: spill every value that more than one statement consumes.
    _spill_to_locals(blocks[0])


# What each constant pushes, spelled as the text format spells vihaco's types.
_CONSTANT_TYPES: dict[type[ir.Statement], str] = {
    stack_move.ConstFloat: "f64",
    stack_move.ConstInt: "i64",
    stack_move.ConstLoc: "u64",
    stack_move.ConstLane: "u64",
    stack_move.ConstZone: "u32",
}


def _value_type(value: ir.SSAValue) -> str:
    """The vihaco type ``value`` has at run time, spelled for ``StoreLocal``.

    Decided by what produced the value rather than its kirin type. Every value
    a lanes op returns without simulating it — a measurement future, an array,
    an element of one, a detector — is an ``Undefined`` placeholder on the
    machine, whatever it stands for, until #776 decides what those values
    are. Only the constants push real values, and ``Dup`` and ``LoadLocal``
    copy whatever they were given.
    """
    while isinstance(value, ir.ResultValue):
        owner = value.owner
        if isinstance(owner, stack_move.Dup):
            value = owner.value
            continue
        if isinstance(owner, stack_move.LoadLocal):
            return owner.value_type
        return _CONSTANT_TYPES.get(type(owner), "undef")
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

    stack: list[ir.SSAValue] = []
    for stmt in stmts:
        # A constant is cloned in front of its one consumer, above the rest
        # of its arguments; it never waits on the stack.
        if stmt.has_trait(ir.ConstantLike):
            continue
        order = (stmt.args[i] for i in _stack_order(stmt))
        need = [arg for arg in order if spillable(arg) and arg not in spilled]
        if need and stack[-len(need) :] == need:
            del stack[-len(need) :]
        elif need:
            spilled.update(need)
            stack = [value for value in stack if value not in spilled]
        # Results pushed deepest first; the first declared ends up on top.
        stack.extend(r for r in reversed(stmt.results) if r not in spilled)
    return spilled


def _spill_to_locals(block: ir.Block) -> None:
    """Pass 3: park every multi-consumer value in a local and reload it for
    each consumer (see the module docs)."""
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
    next_slot = 0

    def take() -> int:
        nonlocal next_slot
        if free:
            return heapq.heappop(free)
        next_slot += 1
        return next_slot - 1

    def reload(value: ir.SSAValue, slot: int) -> stack_move.LoadLocal:
        load = stack_move.LoadLocal(index=slot, value_type=_value_type(value))
        load.result.type = value.type
        return load

    # Deferred, because inserting while walking would shift ``stmts``:
    # (new statement, the statement it goes before), applied in order, so two
    # planned before one anchor keep the order they were planned in.
    plan: list[tuple[ir.Statement, ir.Statement]] = []

    for idx, stmt in enumerate(stmts):
        # Reloads, deepest first, below the consumer's constants. A value's
        # slot is free again after its last one.
        anchor = stmts[const_run_start[idx]]
        for i in _stack_order(stmt):
            arg = stmt.args[i]
            if arg not in spilled:
                continue
            load = reload(arg, slots[arg])
            plan.append((load, anchor))
            stmt.args[i] = load.result
            uses_left[arg] -= 1
            if uses_left[arg] == 0:
                heapq.heappush(free, slots.pop(arg))

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
        if idx + 1 == len(stmts):
            raise ValueError(f"{stmt.name} is last, but its results are consumed")
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
