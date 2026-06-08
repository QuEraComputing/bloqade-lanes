# `lanes.slot` Dialect Design

**Date:** 2026-05-13
**Status:** Draft

## Overview

The `lanes.slot` dialect sits between the circuit level (squin / `lanes.place`) and the
physical transport level (`lanes.move`). It has three purposes:

1. **Replace `StaticPlacement`** — qubits carry explicit slot assignments rather than
   having locations resolved implicitly by a placement analysis pass.
2. **User-level atom micro-management** — users can write explicit `Move` and
   `ResetLocation` operations to control atom positions directly.
3. **Per-kernel layout decomposition** — each kernel function is an independent layout
   sub-problem; slot allocation is scoped to the kernel, enabling the routing and initial
   layout problems to be solved per kernel rather than globally.

In the linear form the dialect additionally serves as a **multi-pass optimization target**.
The SSA def-use chain through moves and gates is simultaneously the dataflow graph for
three independent optimization passes, each of which operates on the same IR without
requiring a separate dialect or lowering step:

- **Gate scheduling** — reorder and parallelize gate operations using the def-use
  dependency graph between `QubitRefType` values.
- **Slot assignment** — resolve abstract `LocationAddressType` values to concrete trap
  sites using forward traversal of the def-use chain as look-ahead.
- **Move scheduling** — merge independent scalar `Move` statements into parallel variadic
  batches using ASAP/ALAP on the move dependency graph.

These three passes are decoupled from each other and from the lowering to `lanes.move`.

## Motivation

`StaticPlacement` in `lanes.place` resolves qubit locations through an implicit
`PlacementAnalysis` pass: the qubit's `location_address` attribute is `None` until the
analysis fills it in. This makes it difficult to reason about physical qubit positions at
the IR level, prevents users from expressing explicit movement strategies, and forces
layout to be solved globally across the entire circuit rather than independently per
kernel.

`lanes.slot` replaces this with an explicit, SSA-level representation of physical qubit
locations. Qubits are bound to typed slot values (`LocationAddressType`) that are either
abstract (compiler-assigned) or concrete (user-specified). The dependency structure of
atom movement is explicit in the IR, enabling per-kernel layout decomposition and
hardware-parallel move scheduling.

## Dialect Position in the Pipeline

```
squin / lanes.place    (circuit level — QubitType, gate statements)
         ↓
    lanes.slot         (this dialect — QubitRefType, slot statements, explicit moves)
         ↓
    lanes.move         (physical level — LocationAddress, LaneAddress, StateType)
         ↓
    bytecode
```

## Types

### `QubitType`

Abstract qubit identity. Borrowed from the upstream `qubit` dialect; not redefined here.
Carries no physical location. Not a linear type — ownership is enforced by a validation
pass rather than by the type system.

### `LocationAddressType`

A physical slot (trap site). Whether the slot is abstract or concrete depends entirely on
which statement produced the value:

- Produced by `AllocSlot()` → abstract; address assigned by the assignment pass.
- Produced by `ConstLocation(...)` → concrete; address known at IR construction time.

All consumers of `LocationAddressType` are unchanged by the assignment pass — only the
producer statement is rewritten.

### `CZPairType`

An opaque handle to a hardware CZ pair: two adjacent trap sites whose simultaneous
occupation allows a CZ gate to be executed. The anchor and partner sites are derived via
statements rather than stored as fields, because the pairing relationship is owned by the
`ArchSpec`.

### `QubitRefType`

An active physical reference to a qubit — an ownership token binding a `QubitType`
identity to its current physical slot. Exists in two forms that are interconverted by a
rewrite pass:

- **Non-linear form**: `Move` and `ResetLocation` are void. The qubit's current location
  is tracked by `SlotAnalysis`, not encoded in the SSA value. Simpler to write; intended
  for the user-facing frontend.
- **Linear form**: `Move` and `ResetLocation` return a new `QubitRefType`. The dep chain
  of moves is explicit in the SSA def-use graph, enabling ASAP/ALAP scheduling. Intended
  for the scheduling and lowering passes.

At most one live `QubitRefType` may exist for any given `QubitType` at any program point.
This invariant is enforced by the ownership validation pass (see §Validation).

## Statements

### Slot Producers

```
AllocSlot() → LocationAddressType
```

Allocates an abstract slot. Address is `None` until the assignment pass runs. Every
`AllocSlot()` in the program corresponds to exactly one physical trap site in the final
compiled output. **`AllocSlot()` must only appear in the root kernel function** — never
inside a helper that could be called multiple times without inlining — to guarantee that
each call site gets a distinct slot (see §Call Convention).

```
ConstLocation(word: int, site: int, zone: int) → LocationAddressType
```

Produces a concrete slot with a known address. Used in user micro-management mode or
after the assignment pass has run.

### CZ Pair

```
AllocCZPair(anchor: LocationAddressType) → CZPairType
```

Allocates a CZ pair rooted at `anchor`. The partner site is determined by
`ArchSpec.get_partner(anchor)`.

```
anchor(pair: CZPairType)  → LocationAddressType
partner(pair: CZPairType) → LocationAddressType
```

Extract the anchor and partner `LocationAddressType` from a pair handle. `partner` is
useful in user micro-management mode when the user needs to know where to `Move` the
second qubit before a `CZ`.

### Qubit Lifecycle

```
NewQubit(θ: float, φ: float, λ: float) → QubitType
```

Creates a new logical qubit initialised to `U3(θ, φ, λ)|0⟩`. Returns an abstract
identity with no physical slot.

```
PlaceQubit(qubit: QubitType, slot: LocationAddressType) → QubitRefType
```

Binds a `QubitType` to a physical slot, producing an active `QubitRefType`. This is the
point at which a qubit enters the physical domain. Inside a callee, the first
`PlaceQubit` on each function argument is the **IO slot declaration** for that argument
(see §Call Convention).

```
MeasureQubit(qref: QubitRefType) → ClassicalResult
```

Measures the qubit and ends its physical lifetime. The `QubitRefType` must not be used
after this statement.

### Physical Reference

```
GetRef(qubit: QubitType) → QubitRefType
```

Takes physical ownership of a qubit that was previously placed (via `PlaceQubit`) but
whose `QubitRefType` live range has ended. Used to re-acquire a reference after it was
implicitly released (e.g. after a function call). At most one live `QubitRefType` per
`QubitType` at any point.

```
GetSlot(qref: QubitRefType) → LocationAddressType
```

Returns the current physical location of the qubit. In the non-linear form this is
resolved by `SlotAnalysis` rather than being a pure SSA computation. In the linear form
it is derivable from the SSA chain.

### Movement — Non-Linear Form (User-Facing)

```
Move(qref: QubitRefType, location: LocationAddressType)   [void]
```

Moves the atom to `location`. The `QubitRefType` token is unchanged; the current location
is updated in `SlotAnalysis`.

```
ResetLocation(qref: QubitRefType)   [void]
```

Moves the atom back to the qubit's **home slot** — the `LocationAddressType` from the
`PlaceQubit` that introduced this qubit into the current scope. Two uses:

1. **Caller pre-call preparation**: moves live qubits to their home slots before a
   function call, without requiring geometry reasoning at the call site.
2. **Join-point convergence** (required by validation): all live `QubitRefType` values
   must have their last move be `ResetLocation` before any branching terminator whose
   successor has multiple predecessors.

### Movement — Linear Form (Scheduling-Facing)

```
Move(refs: IListType[QubitRefType, N],
     locs: IListType[LocationAddressType, N]) → IListType[QubitRefType, N]

ResetLocation(qref: QubitRefType) → QubitRefType
```

The scalar case (N=1) and variadic case (N>1) are the same statement. The variadic form
represents a single hardware atom transport step in which all N atoms move simultaneously.
The ASAP/ALAP scheduling pass produces the variadic form by merging independent scalar
moves within a basic block.

### Gates — Non-Linear Form (User-Facing)

```
CZ(pairs:    IListType[CZPairType,   N],
   controls: IListType[QubitRefType, N],
   targets:  IListType[QubitRefType, N])   [void]

R(qrefs: IListType[QubitRefType, N], axis_angle: float, rotation_angle: float)   [void]

Rz(qrefs: IListType[QubitRefType, N], rotation_angle: float)   [void]
```

Gates consume `QubitRefType` values but do not return them. The qubit's physical lifetime
continues implicitly after the gate; the `QubitRefType` token remains valid.

### Gates — Linear Form (Scheduling-Facing)

```
CZ(pairs:    IListType[CZPairType,   N],
   controls: IListType[QubitRefType, N],
   targets:  IListType[QubitRefType, N])
   → (IListType[QubitRefType, N],   # post-CZ controls refs
      IListType[QubitRefType, N])   # post-CZ targets refs

R(qrefs: IListType[QubitRefType, N], axis_angle: float, rotation_angle: float)
   → IListType[QubitRefType, N]

Rz(qrefs: IListType[QubitRefType, N], rotation_angle: float)
   → IListType[QubitRefType, N]
```

Gates return new `QubitRefType` values, continuing the SSA def-use chain through the
gate. Combined with the linear form of `Move` and `ResetLocation`, this means the
def-use chain runs continuously from `PlaceQubit` through every move and every gate all
the way to `MeasureQubit`. The assignment pass uses this complete chain for look-ahead
placement (see §Assignment Pass).

`MeasureQubit` remains terminal — it does not return a `QubitRefType`.

CZ preconditions (checked by validation pass):
- `controls[i]` is at `anchor(pairs[i])`.
- `targets[i]` is at `partner(pairs[i])`.
- Every other pair in the same zone is in sole-occupancy configuration (at most one site
  occupied), preventing accidental entanglement.

## Validation Rules

The following invariants are checked by dedicated validation passes.

### 1. Single ownership of `QubitRefType`

At each `GetRef(q)` or `PlaceQubit(q, _)`, no other `QubitRefType` derived from `q` may
be live at that program point.

### 2. Clean call sites

At each call `f(q, ...)`, no `QubitRefType` for `q` may be live. All active references
must have their live range end before the call. `ResetLocation` followed by the natural
end-of-live-range is the standard pattern.

### 3. Join-point convergence

At each branching terminator whose target has multiple predecessors, every live
`QubitRefType` must have its last move operation be a `ResetLocation`. This ensures all
qubits are at their home slots at every join-point entry, making the block-entry state
consistent and the placement analysis local to each block.

### 4. CZ preconditions

At each `CZ(pairs, controls, targets)`:
- `controls[i]` is at `anchor(pairs[i])`.
- `targets[i]` is at `partner(pairs[i])`.
- Every other pair in the same zone is in sole-occupancy configuration.

## Call Convention

### Function boundaries use `QubitType`

The physical slot layout is an internal concern of each kernel. Callers pass `QubitType`
values; callees receive `QubitType` values. The compiler inserts `Move` operations at
call sites to place the caller's atoms at the callee's IO slot addresses.

### IO slots

Inside a callee, the first `PlaceQubit(q, slot)` on each function argument declares the
IO slot for that argument. The compiler reads these slots after the assignment pass has
resolved their concrete addresses, then inserts the necessary `Move` operations at each
call site in the caller.

```python
@kernel
def bell_pair(q0: QubitType, q1: QubitType):
    io_0   = AllocSlot()
    io_1   = AllocSlot()
    q0_ref = PlaceQubit(q0, io_0)    # IO slot declaration for q0
    q1_ref = PlaceQubit(q1, io_1)    # IO slot declaration for q1
    ...
```

### Slot allocation lives at the call site

`AllocSlot()` must appear in the root kernel, not inside a helper that could be called
multiple times. Two calls to a helper that internally allocates a slot would resolve to
the same concrete address without inlining. The correct pattern is caller-allocated
storage:

```python
# Correct: caller provides distinct slots
def make_qubit(θ, φ, λ, slot: LocationAddressType) -> QubitRefType:
    return PlaceQubit(NewQubit(θ, φ, λ), slot)

@kernel
def main():
    slot_0 = AllocSlot()
    slot_1 = AllocSlot()                    # distinct SSA value → distinct address
    q0_ref = make_qubit(θ, φ, λ, slot_0)
    q1_ref = make_qubit(θ, φ, λ, slot_1)
```

This is analogous to C++ placement new: the caller allocates the storage, the callee
constructs at that location.

### Callee-saved invariant

The IO slot a qubit arrives at in the callee is its home slot for the duration of the
call. Before the function returns, all argument qubits must be back at their IO slots,
satisfied by calling `ResetLocation` on each argument `QubitRefType` before the return.

### CZ pair ownership at call boundaries

A `CZPair` allocated for a qubit argument transfers to the callee for the duration of the
call and reverts when the callee returns the qubit to its IO slot. Non-argument qubits
that remain live across the call must be in sole-occupancy of their CZ pair (partner site
empty) or parked in a storage pair before the call.

## Control Flow

The dialect uses basic blocks with goto-style branching terminators. Join-point
convergence (validation rule 3) requires `ResetLocation` on all live `QubitRefType`
values before any branch to a join-point successor.

```
bb1:
    ...
    ResetLocation(q0_ref)       # required: last move before branch to join
    ResetLocation(q1_ref)
    goto bb3(q0_ref, q1_ref)

bb2:
    ...
    ResetLocation(q0_ref)
    ResetLocation(q1_ref)
    goto bb3(q0_ref, q1_ref)

bb3(q0_ref, q1_ref):            # block arguments unify the two chains
    ...                         # all qubits at home slots — consistent entry state
```

The linearization pass auto-inserts `ResetLocation` before branches to join-point
successors if not already present, making this transparent to users writing in the
non-linear frontend.

With all qubits at home slots at every block entry, the `SlotAnalysis` and ASAP/ALAP
scheduler operate per basic block with a known, fixed entry state. No inter-block
location reasoning is required.

## Compilation Pipeline

`lanes.slot` has two entry points:

1. **Compiler-generated** — The existing squin / `lanes.place` lowering pipeline produces
   `lanes.slot` IR automatically. Users write standard squin circuits; the compiler lowers
   them to `lanes.slot` with abstract slots, then runs the assignment pass to resolve
   concrete addresses. This is the default path and requires no user awareness of slots.

2. **User-authored** — A user who wants explicit atom micro-management writes `lanes.slot`
   IR directly using `AllocSlot`, `PlaceQubit`, `Move`, `ResetLocation`, etc. The same
   passes apply, but the user's explicit `ConstLocation` values constrain or replace what
   the assignment pass would otherwise choose.

The same pass sequence (Passes 1–5 below) applies regardless of entry point. `lanes.slot`
is therefore the canonical mid-level IR for the physical placement stage, not an opt-in
extension.

### Pass 1 — Estimation

**Input**: circuit-level IR (squin / `lanes.place`)
**Output**: `lanes.slot` IR with abstract slots

Walk the kernel function bodies. For each qubit emit `AllocSlot()` and
`AllocCZPair(AllocSlot())` for unknown locations. Count peak simultaneously live CZ pairs
(gate pairs and storage pairs) per kernel.

### Pass 2 — Assignment

**Input**: abstract `lanes.slot` IR
**Output**: concrete `lanes.slot` IR

Replace each `AllocSlot()` with `ConstLocation(word, site, zone)`. Solve the per-kernel
layout problem using the `ArchSpec`, respecting:
- Blockade radius constraints between CZ pairs.
- Call convention constraints (callee IO slots must not conflict with caller's retained
  qubits).
- CZ pair availability from the `ArchSpec`.

Only producers (`AllocSlot()` statements) are rewritten. All consumers are unchanged.

**Look-ahead placement strategy**: this pass runs on the linear IR (after linearization),
so the full def-use chain of each `QubitRefType` — from `PlaceQubit` through every move
and gate to `MeasureQubit` — is visible as plain SSA graph edges. The solver can examine
future operations by simply traversing the use-def chain forward from each `AllocSlot()`
producer. No separate dataflow analysis is required; the SSA structure is the dataflow.
In particular, qubits that will share a `CZPair` later in their chains can be assigned to
physically compatible slots upfront, avoiding unnecessary transport.

**Decoupling from lane execution order**: this pass assigns physical trap sites only. The
question of which transport lanes are used to move atoms between sites is left entirely to
the lowering pass (Pass 5). The two concerns are independent and do not need to be solved
together.

### Pass 3 — Linearization

**Input**: non-linear `lanes.slot` IR (void `Move`, `ResetLocation`, and gate statements)
**Output**: linear scalar `lanes.slot` IR (chaining `Move`, `ResetLocation`, and gate statements)

For each non-linear `QubitRefType` token, thread a fresh SSA value through every `Move`,
`ResetLocation`, `CZ`, `R`, and `Rz` operation on that token in program order within each
basic block. Insert `QubitRefType` values as block arguments at join points. Auto-insert
`ResetLocation` before branches to join-point successors where not already present.

After linearization the def-use chain of each `QubitRefType` is a complete timeline of
all moves and gates on that qubit, from `PlaceQubit` to `MeasureQubit`.

### Pass 4 — ASAP/ALAP Scheduling

**Input**: linear scalar `lanes.slot` IR
**Output**: linear variadic `lanes.slot` IR

Within each basic block, inspect the SSA def-use graph of `QubitRefType` values. Group
scalar `Move` statements with no shared dependencies into the same variadic `Move` batch.
Each variadic `Move` represents one hardware atom transport step in which all listed atoms
move simultaneously.

ASAP assigns each scalar move to the earliest possible batch; ALAP to the latest. The
choice of policy affects parallelism vs. latency.

### Pass 5 — Lowering

**Input**: linear variadic `lanes.slot` IR
**Output**: `lanes.move` dialect IR

Map each variadic `Move`'s concrete `LocationAddressType` pairs to `LaneAddress` sets
derived from the `ArchSpec`. Emit `move.Move`, `move.Fill`, `move.CZ`, etc.

## Analysis: `SlotAnalysis`

Analogous to `PlacementAnalysis` in `lanes.place`. Tracks the current
`LocationAddressType` of each live `QubitRefType` at each program point by following
`PlaceQubit`, `Move`, `ResetLocation`, `CZ`, `R`, and `Rz` operations in order.

`GetSlot(qref)` is resolved by this analysis in the non-linear form. In the linear form
the location is derivable directly from the SSA chain without a separate analysis pass —
the complete def-use chain through moves and gates encodes the full position history of
each qubit.

The analysis is per-basic-block: because join-point convergence guarantees all qubits are
at home slots at block entry, each block's analysis starts from a clean, known state.

The analysis is used by validation passes to resolve qubit positions at each program
point and verify CZ preconditions, single-ownership rules, and join-point convergence.

The assignment pass performs its own forward traversal of the SSA def-use chains
independently — it does not go through `SlotAnalysis`. Keeping the two separate avoids
coupling correctness checking to placement strategy and allows each to evolve
independently.

## Future Work

### Join-point normalization

Instead of requiring `ResetLocation` before every branch to a join, the compiler could
automatically insert normalizing `Move` operations at join points after the join. This
would allow qubits to be at arbitrary locations when branching. Deferred because it
requires more sophisticated analysis and is not needed for the initial use cases.

### Batch statements

Once Kirin gains function specialization, batch versions of the scalar statements
(`AllocSlots(n)`, `GetRefs(qubits)`, `BatchMove(qrefs, locs)`) can be added as proper
first-class statements. Currently the equivalent is expressed using scalar statements
composed with `ilist.New`, relying on the inlining pass to propagate concrete
`IListType` sizes for type inference. The scalar-only dialect is the correct design for
the current Kirin infrastructure.
