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

    The one constant left where it is is one a ``Dup`` copies: ``dup`` only
    copies the top, so its operand has to stay beneath the copy for its own
    consumer (see "Dup" below).

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

    So, too, does a consumer's non-constant argument above one of its
    constants. The constant is cloned right in front of the consumer, so
    whatever belongs above it can only get there as a reload. A consumer's
    operands are therefore materialised in one deepest-first sequence —
    each constant cloned and each non-constant above it reloaded, interleaved
    in ``_stack_order`` — on top of whatever it takes from the stack. The
    compiler never puts a constant below a non-constant operand; decoded
    bytecode can (``new_array`` of a constant and a measurement, ``local_r``
    with a computed rotation).

    Slots are handed out lowest-free-first and returned after a value's last
    reload, so a function reserves one local per spilled value live at once.

    This is the work the bytecode's ``dup``/``swap``/``pop`` did before a
    function had locals of its own (#1038).

Dup
    Only decoded bytecode has a ``Dup``, and it is a peek: it pushes a copy
    of the top and leaves its operand where it was, for that operand's own
    consumer. So a ``Dup`` counts as neither a consumer of its operand nor a
    pop of it, and a program in which every ``dup`` still finds its operand on
    top comes back out unchanged.

    One whose operand has to be spilled anyway cannot: a reload under it
    would be copied and then left behind. That ``Dup`` is lowered before
    Pass 1 — its result replaced by its operand, which then takes one more
    reload like any other shared value — until every ``Dup`` left finds its
    operand on the stack.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import cast

from kirin import ir
from kirin.rewrite import Walk
from kirin.rewrite.abc import RewriteResult, RewriteRule
from kirin.rewrite.dce import DeadCodeElimination

from bloqade.lanes.dialects import stack_move


@dataclass
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

    ``pinned`` are the constants a ``Dup`` copies. They stay where they were
    pushed, for the ``Dup`` and their consumer alike, and are not cloned.
    """

    pinned: frozenset[ir.SSAValue] = frozenset()

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        args = list(node.args)
        changed = False
        for i in _stack_order(node):
            arg = args[i]
            if not _is_constant(arg) or arg in self.pinned:
                continue
            owner = cast(ir.ResultValue, arg).owner
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
    statement order and emit correct bytecode, after lowering any decoded
    ``Dup`` that cannot stay a peek.

    Raises ``ValueError`` if the method contains more than one block, or if
    it keeps more values spilled at once than a frame has locals. Call this
    function after ``RewriteMoveToStackMove`` and before ``dump_program``.
    """
    blocks = method.callable_region.blocks
    if len(blocks) != 1:
        raise ValueError(
            f"stackify only supports single-block methods; got {len(blocks)} blocks"
        )
    if any(isinstance(stmt, stack_move.Dup) for stmt in blocks[0].stmts):
        # DCE first, so no `Dup` is left unused to vanish in the one after
        # Pass 1 — the constant it pinned would be neither cloned nor copied.
        Walk(DeadCodeElimination()).rewrite(method.code)
        _lower_dups_off_the_stack(blocks[0])

    # Pass 1 + 2: clone constants into correct stack-depth order, then DCE.
    pinned = _pinned(list(blocks[0].stmts))
    Walk(CloneConstants(pinned)).rewrite(method.code)
    Walk(DeadCodeElimination()).rewrite(method.code)

    # Pass 3: spill every value that more than one statement consumes.
    _spill_to_locals(blocks[0])


# Highest local index a frame may name: the Rust validator's
# ``MAX_LOCAL_INDEX``, which ``test_stackify`` pins this against.
_MAX_LOCAL_INDEX = 1023

# The vihaco type each constant pushes, spelled for ``StoreLocal``: what the
# Rust validator's stack simulation gives each ``const``.
_CONSTANT_TYPE: dict[type[ir.Statement], str] = {
    stack_move.ConstFloat: "f64",
    stack_move.ConstInt: "i64",
    stack_move.ConstLoc: "u64",
    stack_move.ConstLane: "u64",
    stack_move.ConstZone: "u32",
}


def _is_constant(value: ir.SSAValue) -> bool:
    return isinstance(value, ir.ResultValue) and value.owner.has_trait(ir.ConstantLike)


def _pinned(stmts: list[ir.Statement]) -> frozenset[ir.SSAValue]:
    """The constants a ``Dup`` copies, which stay on the stack uncloned."""
    return frozenset(
        stmt.value
        for stmt in stmts
        if isinstance(stmt, stack_move.Dup) and _is_constant(stmt.value)
    )


def _consumers(value: ir.SSAValue) -> int:
    """How many statements pop ``value``. A ``Dup`` only copies it."""
    return sum(not isinstance(use.stmt, stack_move.Dup) for use in value.uses)


def _lower_dups_off_the_stack(block: ir.Block) -> None:
    """Lower every ``Dup`` whose operand Pass 3 would spill (see the module
    docs), leaving the ones that find their operand on top of the stack.

    Lowering one can spill more — its operand gains the ``Dup``'s consumers,
    and with them their other arguments — so this repeats until none is
    left to lower. Each round lowers at least one, so it ends.
    """
    while True:
        stmts = list(block.stmts)
        spilled = _values_to_spill(stmts, _pinned(stmts))
        doomed = [
            stmt
            for stmt in stmts
            if isinstance(stmt, stack_move.Dup) and stmt.value in spilled
        ]
        if not doomed:
            return
        for dup in doomed:
            dup.result.replace_by(dup.value)
            dup.delete()


def _value_type(value: ir.SSAValue) -> str:
    """The vihaco type ``value`` has at run time, spelled for ``StoreLocal``.

    Decided by what produced the value rather than its kirin type. Every value
    a lanes op returns without simulating it — a measurement future, an array,
    an element of one, a detector — is an ``Undefined`` placeholder on the
    machine, whatever it stands for, until #776 decides what those values
    are. Constants are cloned rather than spilled, so the other producers of
    a spilled value are a decoded program's own ``LoadLocal``, which says,
    and its ``Dup`` statements, which copy whatever they are handed — a constant
    included.
    """
    while isinstance(value, ir.ResultValue) and isinstance(value.owner, stack_move.Dup):
        value = value.owner.value
    if isinstance(value, ir.ResultValue):
        if isinstance(value.owner, stack_move.LoadLocal):
            return value.owner.value_type
        if type(value.owner) in _CONSTANT_TYPE:
            return _CONSTANT_TYPE[type(value.owner)]
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
    stmts: list[ir.Statement], pinned: frozenset[ir.SSAValue]
) -> set[ir.SSAValue]:
    """Every value Pass 3 parks in a local.

    Those with more than one consumer, every other non-constant argument of a
    statement that takes one, every non-constant argument above one of its
    statement's constants, and the arguments of any statement that would not
    find them on top of the operand stack in the order it pops them. The last
    is found by walking the stack: whatever is not spilled stays where its
    producer pushed it, so each statement's remaining arguments have to be
    exactly the top of it. A ``Dup`` is the one statement that finds its
    argument there and leaves it.

    ``pinned`` constants are on the stack like any other value; every other
    constant is cloned in front of its consumer, so where it was defined does
    not matter — this gives the same answer before Pass 1 as after it.

    One walk is enough. A value spilled for being out of place has one
    consumer, the statement that found it so, and parking it only takes it
    out of the stack — which cannot put anything else out of place, because
    whatever was consumed while it waited was consumed from above it.
    """

    def cloned(arg: ir.SSAValue) -> bool:
        return _is_constant(arg) and arg not in pinned

    def spillable(arg: ir.SSAValue) -> bool:
        return isinstance(arg, ir.ResultValue) and not cloned(arg)

    # Counted once per value: a wide array has as many uses as readers.
    args = {arg for stmt in stmts for arg in stmt.args if spillable(arg)}
    shared = {arg for arg in args if _consumers(arg) > 1}
    spilled = set(shared)
    for stmt in stmts:
        order = [stmt.args[i] for i in _stack_order(stmt)]
        if any(arg in shared for arg in order):
            spilled.update(arg for arg in order if spillable(arg))
            continue
        # Above a cloned constant, only a reload can put an argument.
        first_clone = next(
            (depth for depth, arg in enumerate(order) if cloned(arg)), None
        )
        if first_clone is not None:
            spilled.update(arg for arg in order[first_clone:] if spillable(arg))

    # Insertion-ordered, so the last key is the top — and a value parked for
    # being out of place comes out wherever it is without rebuilding the rest.
    stack: dict[ir.SSAValue, None] = {}
    for stmt in stmts:
        # A constant is cloned in front of its one consumer, above the rest
        # of its arguments; it never waits on the stack.
        if stmt.has_trait(ir.ConstantLike) and stmt.results[0] not in pinned:
            continue
        order = (stmt.args[i] for i in _stack_order(stmt))
        need = [arg for arg in order if spillable(arg) and arg not in spilled]
        top = list(itertools.islice(reversed(stack), len(need)))[::-1]
        if need and top == need:
            if not isinstance(stmt, stack_move.Dup):
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
    pinned = _pinned(stmts)
    spilled = _values_to_spill(stmts, pinned)
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
        # is free again after its last one. A `Dup` never takes one: its
        # operand is on the stack, or the `Dup` was lowered.
        args = list(stmt.args)
        reloaded = False
        below_next_clone: list[stack_move.LoadLocal] = []
        for i in _stack_order(stmt):
            arg = args[i]
            if _is_constant(arg) and arg not in pinned:
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
