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
4. **Stack-only instructions (`dup`, `swap`, `pop`)** have two valid treatments — see §"Linear-IR style vs vanishing stack ops" below. The textbook choice is to emit no IR for them and just rearrange `stack` references; Bloqade Lanes makes a different choice.

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

## Linear-IR style vs vanishing stack ops

The walked-through example above uses the **textbook mainstream treatment** of `dup`/`swap`/`pop`: they mutate the decoder's virtual stack and emit no IR. The rationale is that `Dup` is a semantic identity in SSA (`%y = Dup %x` means `%y ≡ %x`, a tautology), `Swap` is a permutation that doesn't affect SSA def-use edges, and `Pop` is a discard with no runtime effect. In an IR where dataflow is fully explicit, preserving these operations adds no information. It's the choice Soot, Cranelift, V8 TurboFan, and the JVM verifier all make (with minor variations — Soot lifts them into local-variable copies that are then optimized away).

The **alternative — the "linear IR" style — keeps them as explicit SSA statements**:

```text
%loc0 = stack_move.ConstLoc [value=LocationAddress(0,0,0)]
%loc1 = stack_move.ConstLoc [value=LocationAddress(0,0,1)]
%loc2 = stack_move.Dup %loc1                # identity — %loc2 ≡ %loc1
%a, %b = stack_move.Swap %loc0, %loc1       # permutation
        stack_move.Pop %loc2                # explicit discard
        stack_move.Fill %a, %b
```

This draws on categorical / substructural IRs (Joyal-Street string diagrams, Selinger's categorical quantum mechanics, linear-type compiler IRs) where `dup`, `swap`, and `discard` are first-class morphisms because the category doesn't assume cartesian (freely-duplicating) structure. Keeping them explicit gives downstream passes a hook to attach invariants.

**Bloqade Lanes uses the linear-IR style in `stack_move`**, for two reasons:

1. **Atom non-cloning as a first-class invariant.** Atoms physically cannot be duplicated. An explicit `Dup` statement in the IR is exactly the right place for a later linear-type pass to check: any `Dup` whose operand has an atom-bearing type is a physical error. The mainstream vanishing-stack-ops treatment would have to reconstruct this information from alias analysis or equivalent.
2. **Structural round-trippability to bytecode.** Every bytecode instruction has exactly one corresponding `stack_move` statement. Compiling Kirin IR back down to bytecode is a structural inverse of decoding — no reconstruction of missing stack ops.

The cost is that stack ops have to be explicitly collapsed when lowering `stack_move` into the downstream target dialects (`move`, `ilist`, etc.): `Dup` and `Swap` map their result SSA values back to their inputs, `Pop` produces nothing. This is a trivial per-statement transformation in `lower_stack_move` — but it's a step that the mainstream treatment avoids by never introducing the statements in the first place.

## Where the technique lives: decoder vs rewrite pass

The virtual-stack-of-SSA-values technique is a self-contained algorithm. It doesn't care whether it runs inside a **decoder** (`source bytecode → SSA IR in one step`) or inside a **rewrite pass** over a pre-existing intermediate dialect. Both are valid splits.

The Bloqade Lanes implementation puts the stack-to-SSA conversion **in the decoder**. Each bytecode stack slot becomes a named SSA value at decode time; the decoder maintains a virtual stack of SSA references and consumes/produces them as it walks the bytecode. The output is already-SSA `stack_move` IR where every statement has explicit operands and results.

**Stack ops are preserved as explicit SSA statements** rather than being eliminated invisibly during decoding. `Dup` is an SSA identity statement (`%y = Dup %x`, semantically `%y ≡ %x`); `Swap` is an SSA permutation statement (`%a, %b = Swap %x, %y`, with `%a ≡ %y` and `%b ≡ %x`); `Pop` is an SSA discard statement (`Pop %x`, no result). This is the "linear IR" style described earlier — it preserves the bytecode's structural operations at the SSA level so that downstream analyses (notably a future atom-non-cloning type check) can attach invariants to them, and so that lowering from Kirin *back* to bytecode is a structural inverse of decoding.

The follow-up rewrite `lower_stack_move` becomes a **mechanical per-statement translation** from `stack_move` into target dialects (`move`, `ilist`, `py.constant`, `py.indexing`, `annotate`, `func`). `Dup` and `Swap` collapse during that rewrite: their result SSA values are mapped to the same target-dialect SSA values as their inputs. `Pop` produces no target statement. The linear-IR shape lives only in `stack_move`; the target IR is conventional SSA.

Conceptual shape of the decoder:

```python
class BytecodeDecoder:
    stack: list[ir.SSAValue]
    block: ir.Block

    def decode(self, program: Program, kernel_name: str) -> ir.Method:
        for idx, instruction in enumerate(program.instructions):
            try:
                self._visit(instruction)
            except _StackError as e:
                raise DecodeError.from_context(idx, instruction, self.stack, e)
        return self._finalize(kernel_name)

    def _visit(self, instr):
        # dispatch by opcode to _visit_const_loc, _visit_fill, _visit_dup, …
        ...

    def _visit_const_loc(self, instr):
        stmt = stack_move.ConstLoc(value=instr.location_address())
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_fill(self, instr):
        locs = [self.stack.pop() for _ in range(instr.arity)]
        locs.reverse()  # or not, depending on your chosen ordering convention
        self._check_type(locs, LocationAddressType)
        self.block.stmts.append(stack_move.Fill(locations=tuple(locs)))

    def _visit_dup(self, instr):
        top = self.stack[-1]
        stmt = stack_move.Dup(value=top)
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)  # new SSA name, semantically ≡ top

    def _visit_swap(self, instr):
        in_top = self.stack.pop()
        in_bot = self.stack.pop()
        stmt = stack_move.Swap(in_top=in_top, in_bot=in_bot)
        self.block.stmts.append(stmt)
        self.stack.append(stmt.out_bot)  # new SSA name ≡ in_top
        self.stack.append(stmt.out_top)  # new SSA name ≡ in_bot

    def _visit_pop(self, instr):
        top = self.stack.pop()
        self.block.stmts.append(stack_move.Pop(value=top))
```

The virtual stack is a `list[ir.SSAValue]` on the decoder instance. `DecodeError` carries the offending instruction index, opcode, and a snapshot of the virtual stack at failure.

**Compatibility with future control flow.** This decoder design composes cleanly with Approach 3 (block-argument SSA, see §"Unifying with the host dataflow framework" below) when branching is added: the virtual stack at a block exit becomes the argument list for the successor block, and the explicit `stack_move.Dup`/`Swap`/`Pop` statements operate uniformly on both block-argument SSA values and locally produced SSA values — they don't care which kind of SSA value they're manipulating.

**Toward a bytecode decoder framework (planned follow-up).** The v1 implementation is deliberately monolithic — one `stack_move` dialect, one `BytecodeDecoder`, one `lower_stack_move` rewrite — so the end-to-end mechanics can be validated on a small, reviewable surface. Once those boundaries are stable, the natural next step is to decompose the work into a **bytecode decoder framework**: instruction families become sub-dialects (stack ops, constants, atom ops, gates, measurement, arrays, annotations, control flow), each sub-dialect registers a **decoding handler** that the top-level decoder dispatches to for its opcodes, and each sub-dialect also owns a **rewrite chunk** that consumes its own statements and emits into a target dialect. The virtual stack threads through the decoder's chain of handlers as shared state. The payoff is independently testable, replaceable, reusable pieces — and a framework that can handle new stack-based bytecode variants by swapping sub-dialect handlers rather than rewriting the whole decoder. This is post-v1 work but it's the direction this prototype is paving toward.

## Error conditions

Whichever pass owns the virtual stack surfaces these as a dedicated exception (the Bloqade Lanes decoder raises `DecodeError`; a host framework's decoder would raise `BuildError`/equivalent):

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

## Related work and future direction

Stack-machine-to-SSA is a well-studied problem. The one-line framing the literature uses: stack machines don't lose dataflow, they **encode it implicitly in execution order**. Recovering the dataflow for SSA output is a **dataflow analysis in itself** — abstract interpretation over an abstract state that includes the operand stack in addition to local variables.

### The general technique (for branching bytecode)

Given a bytecode program with control flow, the standard approach is:

1. **Define an abstract state** `(stack: list[AbstractValue], locals: map[int, AbstractValue])`.
2. **Define transfer functions** per opcode: each instruction's effect on the abstract state.
3. **Run a dataflow fixpoint** over the CFG. At each CFG join point, require **equal stack heights** across predecessors and compute the **pointwise ⊔ (join)** of abstract stack slots.
4. **Materialize SSA** from the stable fixpoint: each stack slot at each join becomes a phi node (or block argument, depending on your SSA flavor).

For **straight-line bytecode** (the current Bloqade Lanes case), step 3 collapses to a single forward pass with no joins, and step 4 is implicit — the visitor technique described in this doc is the degenerate case.

### Key references

**Directly on stack-machine → SSA via abstract interpretation:**

- Xavier Leroy, **"Java bytecode verification: algorithms and formalizations"**, *Journal of Automated Reasoning* 30(3–4):235–269, 2003. The canonical exposition: the JVM verifier is literally abstract interpretation over (stack, locals), and §3 formalizes the lattice and join rules. [[PDF]](https://xavierleroy.org/publi/bytecode-verification-JAR.pdf) [[DOI]](https://doi.org/10.1023/A:1025055424017)
- Xavier Leroy, **"Bytecode verification on Java smart cards"**, *Software: Practice and Experience* 32(4):319–340, 2002. Same core ideas in a resource-constrained setting — particularly relevant for restricted target hardware. [[PDF]](https://xavierleroy.org/publi/oncard-verifier-spe.pdf) [[DOI]](https://doi.org/10.1002/spe.438)
- Raja Vallée-Rai, Phong Co, Etienne Gagnon, Laurie Hendren, Patrick Lam, Vijay Sundaresan, **"Soot — a Java bytecode optimization framework"**, *CASCON* 1999. Production-scale framework for lowering JVM bytecode through multiple IRs (Baf → Jimple → Shimple/SSA). Good case study in making the abstract interpretation engineerable. [[PDF]](https://patricklam.ca/papers/99.cascon.soot.pdf) [[ACM]](https://dl.acm.org/doi/10.5555/781995.782008)
- Etienne Gagnon, Laurie Hendren, Guillaume Marceau, **"Efficient inference of static types for Java bytecode"**, *SAS* 2000. The type-inference side of the Soot approach — the specific dataflow analysis that lifts stack/local slots to static types. [[Springer]](https://link.springer.com/chapter/10.1007/978-3-540-45099-3_11)

**SSA construction (what you do once you have the abstract state):**

- Ron Cytron, Jeanne Ferrante, Barry K. Rosen, Mark N. Wegman, F. Kenneth Zadeck, **"Efficiently computing static single assignment form and the control dependence graph"**, *ACM TOPLAS* 13(4):451–490, 1991. The classic dominance-frontier construction. [[ACM]](https://dl.acm.org/doi/10.1145/115372.115320)
- Preston Briggs, Keith D. Cooper, Timothy J. Harvey, L. Taylor Simpson, **"Practical improvements to the construction and destruction of static single assignment form"**, *Software: Practice and Experience* 28(8):859–881, 1998. Often what modern compilers actually implement — simpler than Cytron et al. and often faster. [[PDF]](https://web.eecs.umich.edu/~mahlke/courses/583w23/reading/briggs_spe_98.pdf) [[Wiley]](https://onlinelibrary.wiley.com/doi/abs/10.1002/\(SICI\)1097-024X\(19980710\)28:8%3C859::AID-SPE188%3E3.0.CO;2-8)

**Foundations:**

- Patrick Cousot, Radhia Cousot, **"Abstract interpretation: a unified lattice model for static analysis of programs by construction or approximation of fixpoints"**, *POPL* 1977. Foundational. Worth re-reading specifically for the "abstract domain = concrete semantic state" framing — a stack machine's state naturally includes the stack. [[PDF]](https://www.di.ens.fr/~cousot/publications.www/CousotCousot-POPL-77-ACM-p238--252-1977.pdf) [[ACM]](https://dl.acm.org/doi/10.1145/512950.512973)
- Flemming Nielson, Hanne Riis Nielson, Chris Hankin, **_Principles of Program Analysis_**, Springer, 1999. Textbook; the dataflow-analysis chapters walk through abstract interpretation over stack machines as a worked example. Easier entry than the papers cold. [[Springer]](https://link.springer.com/book/10.1007/978-3-662-03811-6)
- Thomas Reps, Susan Horwitz, Mooly Sagiv, **"Precise interprocedural dataflow analysis via graph reachability"**, *POPL* 1995. IFDS framework. Not immediately needed; relevant if interprocedural analysis over bytecode is ever required. [[PDF]](https://pages.cs.wisc.edu/~fischer/cs701.f14/popl95.pdf) [[ACM]](https://dl.acm.org/doi/10.1145/199448.199462)

**Modern, production reference:**

- Andreas Haas et al., **"Bringing the Web up to Speed with WebAssembly"**, *PLDI* 2017. Wasm design document; Section 2.1 discusses why they mandated structured control flow, which dodges the unstructured-CFG complexity in the bytecode-to-SSA pipeline. [[PDF]](https://people.mpi-sws.org/~rossberg/papers/Haas,%20Rossberg,%20Schuff,%20Titzer,%20Gohman,%20Wagner,%20Zakai,%20Bastien,%20Holman%20-%20Bringing%20the%20Web%20up%20to%20Speed%20with%20WebAssembly.pdf) [[ACM]](https://dl.acm.org/doi/10.1145/3062341.3062363)
- **Cranelift** — production Rust compiler backend that lowers Wasm (stack machine) to CLIF (SSA IR with block arguments instead of phi nodes). Source is readable and idiomatic; a useful real-world reference for a Rust-adjacent codebase. [[cranelift.dev]](https://cranelift.dev/) [[source]](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift)

### Unifying with the host dataflow framework

A natural follow-up question: *can the stack-shape-at-joins problem be handled by the host compiler's existing dataflow/SSA machinery, rather than bolting on a bespoke abstract interpreter?* Three approaches show up in the literature, and they differ exactly on this point.

**Approach 1 — Verifier-style, stack stays meta-level.** The JVM verifier (Leroy 2003) enforces that predecessors into a join have equal stack depth and type-compatible slots. The lattice can be trivially flat ("give up to `Top` on any inconsistency and error"), but the machinery is still a **separate pass** with its own traversal, worklist, and fixpoint — distinct from the normal IR dataflow.

**Approach 2 — Lift stack slots into named locals.** Soot's Baf → Jimple conversion normalizes `PUSH` / `POP` into reads and writes of numbered locals, then runs ordinary SSA construction on the result. The stack disappears as a concept; what remains is standard SSA over named variables.

**Approach 3 — Block arguments carry the stack.** Cranelift (CLIF), MLIR, and Kirin all use block-argument SSA (no phi nodes — block entry parameters play the phi role). Each basic block declares typed parameters; branches pass values as arguments to the target block. At a join, the block-argument unification *is* the merge. If the current virtual stack has N values at a branch, the successor block is declared with N parameters and those N SSA values are passed as arguments. Multiple predecessors with consistent stack shapes feed the same block cleanly; inconsistent shapes are a type error at block-argument binding time, surfaced by the same machinery that catches any other argument-type mismatch.

**Approaches 2 and 3 unify stack handling with the existing dataflow framework**: stack state is lifted from "meta level" (separate abstract interpreter) to "object level" (ordinary IR values). This is the same move that state-threading makes for world state. No additional analysis pass; no dedicated lattice; no bespoke fixpoint. The compiler framework's existing block-argument / phi-insertion / dataflow machinery is what reconciles state at joins.

**Recommendation for Bloqade Lanes: Approach 3.** Kirin is block-argument-based, so this is the path of least resistance. Concretely, once branching is introduced:

- **Block entry**: parameters = the stack values live at entry.
- **Within a block**: the existing forward visitor with an ephemeral virtual stack (what the current straight-line decoder already does).
- **Block exit / branch**: pass the current virtual stack as block arguments to the successor(s).
- **Join**: Kirin's existing block-argument machinery reconciles — no new code.

The enabling precondition is **structured control flow in the bytecode** (e.g. Wasm-style `block` / `loop` / `if` with declared stack types). With structured control flow, block boundaries and stack deltas are declared in the bytecode format itself, so Approach 3 needs zero CFG discovery. With unstructured `goto`, you first need a pass to discover block boundaries before Approach 3 applies — which reintroduces exactly the bespoke pre-pass Approach 3 was supposed to avoid.

This is a second reason to pick Wasm-style structured branching when Bloqade Lanes bytecode gains control flow: it lets the decoder stay inside Kirin's dataflow framework with no custom analysis infrastructure.

### Decisions to make when branching is added

The two prior subsections together point at a concrete plan: **Wasm-style structured control flow + Approach 3 (block arguments)**. Open design decisions at that point:

1. **Structured vs unstructured branching in the bytecode format.** Take the Wasm position — it's the precondition for Approach 3 to apply without a CFG-discovery pre-pass.
2. **Block-argument type discipline** — how strict is "compatible stack shape" at a join? Exact type match, or widening via a lattice join?
3. **Error taxonomy** — unreachable-code handling, unbalanced-stack at join, type-widened joins, how these surface to the user.
4. **Unstructured fallback** — if some bytecode source *does* need arbitrary `goto`, the fallback is a CFG-discovery pre-pass that runs Approach 1 (the meta-level verifier) just to identify block boundaries and stack shapes, then the main decoder runs Approach 3. Worth having a story for, even if the format itself is structured.
