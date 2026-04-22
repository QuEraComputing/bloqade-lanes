# Lowering Stack-Based Bytecode into SSA IR

**Date**: 2026-04-21
**Audience**: Team members who want to understand how we turn a bytecode program into Kirin SSA IR.
**Companion spec**: [2026-04-21-stack-move-dialect-design.md](./2026-04-21-stack-move-dialect-design.md) — the full `stack_move` dialect design this technique is used for.

This document walks through the general problem of lowering a **stack-based bytecode** into an **SSA intermediate representation**, using the Bloqade Lanes bytecode decoder (the `stack_move` dialect) as a concrete use case. The technique generalizes to any stack-machine input; the `stack_move` specifics are just the example.

> **Scope — straight-line programs only.** The virtual-stack technique described here is sufficient for bytecode programs with **no branching control flow**. That matches the Bloqade Lanes bytecode today (a single linear instruction stream terminating in `return` / `halt`) and is what the `stack_move` decoder targets.
>
> Once branches are introduced, stack state has to be reconciled across control-flow joins. That reconciliation requires stack snapshots at branch boundaries and phi nodes / block arguments in the SSA output — a meaningfully more complex decoder. The "Things that are genuinely subtle" section at the end revisits this. **Plan for it before adding branching opcodes to the bytecode.**

## The two models

### Stack-based bytecode

Each instruction implicitly consumes some values from the top of an operand stack and pushes its results back. Order of operands is encoded in the **execution order** of the instruction stream; there are no names.

Example (Bloqade Lanes bytecode):

```text
CONST_LOC (0, 0, 0)     # push LocationAddress(0,0,0)
CONST_LOC (0, 0, 1)     # push LocationAddress(0,0,1)
FILL       arity=2      # pop 2 locations, fill those sites
```

### SSA intermediate representation

Every value is named exactly once (single static assignment). Each operation takes named operands and produces a named result. Order of operands is encoded in the **argument list** of each statement.

The same program in SSA form:

```text
%loc0 = stack_move.ConstLoc [value=LocationAddress(0,0,0)]
%loc1 = stack_move.ConstLoc [value=LocationAddress(0,0,1)]
        stack_move.Fill %loc0, %loc1
```

## Why bridge the two

Stack bytecode is a great *transport* and *execution* format — dense, linear, trivially interpreted. SSA IR is a great *analysis and rewrite* format — every value has a name, every use points back to its definition, and dataflow passes become structural.

If we want to take bytecode as an input to a compiler that was designed for SSA (as is the case for the Bloqade Lanes pipeline, built on the Kirin framework), we need a decoder that bridges these two models.

## The core technique: a decoder-internal SSA stack

The decoder maintains a **virtual stack of SSA values**, one entry per live operand:

```python
stack: list[ir.SSAValue] = []
```

This stack exists *only during lowering*. It does not appear in the output IR. It mirrors what the bytecode interpreter's runtime operand stack would look like at the same program point, but instead of holding *runtime values*, it holds *references to the SSA values that will be produced at runtime*.

Rules:

1. **Each bytecode instruction is a visitor** that reads `stack`, optionally emits IR, and optionally mutates `stack`.
2. **Constant and value-producing instructions emit IR and push** the new SSA value onto `stack`.
3. **Consuming instructions pop** the operands they need (in reverse push order), emit IR using those operands, and push any result.
4. **Stack-only instructions (`dup`, `swap`, `pop`) emit no IR** — they just rearrange `stack` references.

That's the whole trick.

## Walked-through example

Bytecode:

```text
CONST_LOC (0, 0, 0)
CONST_LOC (0, 0, 1)
DUP
FILL arity=2
POP
```

### After `CONST_LOC (0, 0, 0)`

- Emit: `%loc0 = stack_move.ConstLoc [value=LocationAddress(0,0,0)]`
- Stack: `[%loc0]`

### After `CONST_LOC (0, 0, 1)`

- Emit: `%loc1 = stack_move.ConstLoc [value=LocationAddress(0,0,1)]`
- Stack: `[%loc0, %loc1]`

### After `DUP`

- Emit: nothing
- Stack: `[%loc0, %loc1, %loc1]`  ← two references to the same SSA value

### After `FILL arity=2`

- Pop 2: operands are `%loc1` (top) and `%loc1` (below) in *reverse* pop order → `fill(%loc1, %loc1)`.
  The decoder rule is: stack pop is LIFO, but the IR argument list reads top-to-bottom of the consumed window in a documented order. A common choice is "top-of-stack becomes last argument"; another is "top-of-stack becomes first." Pick one and stick with it (`stack_move`'s choice is documented in the dialect spec). For this walkthrough assume **top = last argument**, so the operand list is `[%loc1, %loc1]` reversed back into `[%loc1, %loc1]` — the deeper-stack value first.
- Emit: `stack_move.Fill %loc1, %loc1`
- Stack: `[%loc0]`

### After `POP`

- Emit: nothing
- Stack: `[]`

The dangling `%loc0` was produced but never consumed. It's a live SSA value with no uses. This is fine — it'll be pruned by dead-code elimination later, or the decoder can raise if strict "stack must be empty at return" checking is desired.

## Integrating with a lowering framework

If the host compiler provides a lowering framework (Kirin's `LoweringABC[T]`, LLVM's IRBuilder, MLIR's Builder, etc.) you usually get three things for free:

1. **Frame / State object** — tracks the current basic block, handles IR emission, and is where you park your `stack` list.
2. **Visitor dispatch** — you implement one visitor per instruction kind; the framework iterates your source and calls the right one.
3. **Error plumbing** — a `BuildError`/equivalent that carries location info and gets raised with clean diagnostics.

For Kirin specifically:

```python
class StackMoveLowering(lowering.LoweringABC[Program]):
    def run(self, program: Program, state: lowering.State) -> ir.Method:
        for instruction in program.instructions:
            self.visit(state, instruction)
        return state.current_frame.finalize()

    def visit(self, state: lowering.State, instr: Instruction) -> lowering.Result:
        # Dispatch by opcode to visit_const_loc, visit_fill, visit_dup, ...
        ...

    def visit_const_loc(self, state, instr):
        stmt = stack_move.ConstLoc(value=instr.location_address())
        state.current_frame.push(stmt)
        self._push(state, stmt.result)  # append to virtual SSA stack

    def visit_fill(self, state, instr):
        locs = [self._pop(state) for _ in range(instr.arity)]
        locs.reverse()  # or not, depending on your chosen ordering convention
        self._check_type(locs, stack_move.LocationAddressType)
        state.current_frame.push(stack_move.Fill(locations=tuple(locs)))

    def visit_dup(self, state, instr):
        top = self._peek(state)
        self._push(state, top)

    def visit_pop(self, state, instr):
        self._pop(state)

    def visit_swap(self, state, instr):
        a = self._pop(state); b = self._pop(state)
        self._push(state, a); self._push(state, b)
```

The virtual `stack` lives on either the `State` object itself (as an added attribute) or on a custom `Frame` subclass. Both are fine; the choice depends on whether the framework lets you subclass its frame cleanly.

## Error conditions

A decent decoder surfaces these as `BuildError`:

| Condition                                     | Example                                                                               |
|-----------------------------------------------|---------------------------------------------------------------------------------------|
| **Stack underflow**                           | `FILL arity=3` when stack has 2 values.                                               |
| **Operand type mismatch**                     | `FILL` consumed something that isn't a `LocationAddressType` SSA value.               |
| **Unsupported opcode**                        | Opcodes outside the target dialect's scope (e.g. reserved / experimental ones).        |
| **Dangling stack at end-of-program**          | `return` / `halt` reached with non-empty stack. Optional; depends on source semantics. |
| **Semantic precondition violated**            | `INITIAL_FILL` not first, etc.                                                        |

The first two are essentially a type-checker run against the bytecode's implicit type rules. Doing this in the decoder catches malformed programs at the earliest possible point.

## Why this pattern scales

Once you have the virtual-stack visitor machinery:

- **Adding a new opcode** is a one-visitor, localized change.
- **Adding a new address type** is a new `Const*` visitor plus whatever consumers use it.
- **Changing the output dialect** (e.g., if we later target a different SSA IR) is localized to the visitor bodies — the stack discipline doesn't change.
- **Writing a fuzz tester** is straightforward: generate random well-typed bytecode (easy because you can simulate the stack and only emit opcodes whose argument types are on top), decode, round-trip through your rewrite pipeline, and check for consistency.

## Things that are genuinely subtle

Not everything is mechanical. A few points that deserve care:

- **Argument ordering convention.** When you pop N operands for an N-ary op, the IR argument list can read either "top-of-stack first" or "top-of-stack last." Both are valid; neither is obviously right. Pick one, document it, and stick to it project-wide. Bugs here are silent and painful.
- **Shared SSA values from `dup`.** A single SSA value may appear in multiple argument positions of the same operation. This is perfectly valid SSA (one definition, many uses) — but some downstream passes or testing tools have bad intuitions about it. Test this case explicitly.
- **Control flow — the big one.** The example above, and the entire virtual-stack technique as described, is a *single-basic-block* algorithm. It breaks as soon as the bytecode gains branches, because stack shape at a branch join has to be reconciled across predecessors. The fix is well understood — snapshot the virtual stack at each branch boundary, merge snapshots at joins, and materialize merged values as SSA block arguments (or phi nodes) in the output IR — but it turns the decoder from "straight visitor with a list" into something closer to a proper abstract interpreter. Java's JVM verifier is an existence proof that this is tractable, but it's not a small addition. The current Bloqade Lanes bytecode has no branching, so this is deferred; **when branching opcodes are added, the decoder will need a dedicated redesign**, not a small patch.
- **Stack-typed disagreement at joins.** The specific failure mode at joins: different predecessors may have stacks of different depth or different element types. The decoder has to either reject the program or reconcile via widening. Either answer is hard to retrofit.
- **Implicit arity instructions.** Some opcodes have arity encoded as an operand; others have a fixed arity. The decoder's per-opcode visitor handles this locally, but the bytecode format should be unambiguous on this point.

## Summary

Turning a stack-based bytecode into SSA IR is a small idea: *simulate the operand stack with SSA value references at lowering time.* The scaffolding around it — visitor dispatch, frame management, error plumbing — is whatever the host lowering framework provides. With a virtual stack on the side and one visitor per opcode, the decoder is mechanical, testable opcode-by-opcode, and easy to extend. The `stack_move` dialect decoder in Bloqade Lanes is an instance of this pattern.
