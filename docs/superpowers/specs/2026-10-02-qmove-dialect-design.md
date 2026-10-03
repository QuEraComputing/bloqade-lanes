# `qmove`: a qubit-addressed, state-threaded dialect for structured control flow

Add a new dialect, `lanes.qmove`, that expresses quantum operations on **qubit
references** while threading an explicit machine `State` through them. Every
qmove method opens its state chain with `qmove.enter(frame)` and closes it with
`qmove.exit`. It keeps `scf` control flow and non-inlined calls intact, and
gives every method an explicit placement contract (a *frame*). The entry
kernel's frame is the whole machine. This spec covers the dialect, the lowering from native IR into it,
and its verifier. It does not cover placement or move synthesis.

**Target: the physical pipeline.** Input kernels are physical-qubit kernels, as
accepted by `PhysicalPipeline` today. Logical-pipeline constructs
(`gemini.logical.*`, `gemini.extensions.*`) are out of scope and rejected.

The new path coexists with the current pipeline. No existing module, pipeline,
test or benchmark baseline changes.

## Motivation

### `StaticPlacement` exists to give integer ids a frame of reference

Every `place` statement stores `qubits: tuple[int, ...]` as an attribute. The
ints are positions into the enclosing `StaticPlacement.qubits` tuple, not
global addresses (`dialects/place.py:106-249`). `RewritePlaceOperations` builds
one `StaticPlacement` per gate with `qubits=range(len(inputs))`, and
`MergeStaticPlacement` spends ~90 lines remapping those positions through
`new_input_map` when it merges neighbours (`rewrite/circuit2place.py:704-798`).

If statements take qubit references instead, the index space, the remapping
and the region all disappear. Merging becomes deleting a redundant
`Store`/`Load` pair.

### The region shape forces straight-line code

- **Every `StaticPlacement` starts from the home layout.** `get_inintial_state`
  rebuilds its state from `initial_layout` (`analysis/placement/analysis.py:49-66`).
  Only `move_count` carries across regions, so there is an implicit convention:
  every atom is home at every region boundary.
- **CZ lookahead stops at the region boundary.** The buffer is keyed by the
  region's body block (`place.py:392`), and `StaticPlacement.check` requires
  that body to be a single block.
- **Control flow is unrolled or lowered away.** `AggressiveUnroll` unrolls loops
  (and inlines every call), then `ScfToCfRule` lowers what is left
  (`transform/native_to_place.py:157-160`).
  - The logical pipeline rejects `scf.IfElse` and `scf.For` with a non-constant
    range (`gemini/logical/validation/clifford/impls.py:16-46`).
  - The physical pipeline lets leftover `cf` blocks through. Each block gets its
    own `StaticPlacement` that starts from home, and nothing checks that atoms are
    home at block exits. That is a latent hazard in the current pipeline, outside
    this spec.

### What the new structure must be able to represent

1. **Runtime branches:** `scf.IfElse` on a runtime value, such as a mid-circuit
   measurement.
2. **Index-dependent qubit references:** a loop body addressing `q[i]`, or
   iterating an `IList[Qubit]` directly, kept rolled.
3. **Non-inlined calls:** subroutine kernels kept as calls, with a placement
   contract at the boundary.

Representing these is in scope. Compiling them to atom moves is not.

### Decomposition

| # | Sub-project | Core problem | Status |
|---|---|---|---|
| 1 | **This spec:** `qmove` dialect, lowering, verifier | An IR that can hold all three kinds of control flow | — |
| 2 | Upstream qubit-identity analysis | Symbolic qubit identities (parameter slots, loop indices, per-callee summaries) | [bloqade-circuit#332](https://github.com/QuEraComputing/bloqade-circuit/issues/332), reopened |
| 3 | Straight-line synthesis on `qmove` | Match the current pipeline on control-flow-free kernels | future spec |
| 4 | Runtime branches | Agreeing layouts where branch arms rejoin | future spec |
| 5 | Subroutines | Choosing frames, emitting `prepare`, `check_call` | future spec |
| 6 | Index-dependent references | Lane tables indexed at runtime, or per-iteration schedules | future spec |
| 7 | Logical pipeline on `qmove` | Logical measurement dataflow, `StarRz`, a replacement for initialization with angles | future spec |

Sub-projects 3–7 depend on 1, and 3–6 also depend on 2.

## Design overview

- **Allocation** stays in the existing statements: `qubit.New` and
  `gemini.common.NewAt`. They create references and do not touch the state.
- **Quantum operations** become `qmove` statements. Each one consumes a `State`,
  produces a new `State`, and takes its qubits as `IList[Qubit]` SSA values,
  mirroring the native `gate` dialect.
- **The state cell** is read and written only by `qmove.enter` / `qmove.exit`
  (one pair per method). `move.Load` / `move.Store` appear only transiently
  inside the lowering, and they remain in address-level `move` IR. `qmove` is a
  sibling dialect, not part of `move`, so the current pipeline's dialect groups
  (and its `raise_if_statements_outside_dialect_group` check) are unaffected.
- **Structured control flow** threads the `State` explicitly through `scf.IfElse`
  and `scf.For`.
- **Calls** to subroutines go through `qmove.invoke`. Every method declares a
  *frame* on its `enter`. For the entry kernel it is `MachineFrame()`; for a
  subroutine it says where its arguments must be, which scratch locations it may
  use, and which zone-wide effects it may have.

## The `qmove` dialect

**Location:** package `python/bloqade/lanes/dialects/qmove/` (dialect
`lanes.qmove`), laid out like `dialects/arch/`: `_dialect.py`, `stmts.py`,
`frame.py`.

### Reused from `move`

`StateType`, `move.Load`, `move.Store`, the `ConsumesState`/`EmitsState`
traits, and the `move.StatefulStatement` base class. Every gate statement, and
`prepare`/`invoke`, subclasses `StatefulStatement`, so it has a
`current_state: State` argument and a `result: State` first result. `enter` and
`exit` mirror `Load` and `Store` instead. Nothing in `move` changes.

### Gate statements

Operand types mirror the source statement in `gate` / `arrange` / `qubit`.
`MoveTo.locations` and `Permute.perm` stay SSA values (as in `arrange`), not
attributes (as in `place`), so they can depend on runtime values. Synthesis
will need them to be constant.

| Statement | Operands beyond `current_state` | Results beyond `State` | Source |
|---|---|---|---|
| `CZ` | `controls: IList[Qubit, N]`, `targets: IList[Qubit, N]` | — | `gate.CZ` |
| `R` | `axis_angle, rotation_angle: Float`, `qubits: IList[Qubit, Any]` | — | `gate.R` |
| `Rz` | `rotation_angle: Float`, `qubits: IList[Qubit, Any]` | — | `gate.Rz` |
| `MoveTo` | `qubits: IList[Qubit, Len]`, `locations: IList[LocationAddress, Len]`; attribute `multi_move_warning` | — | `arrange.MoveTo` |
| `Permute` | `qubits: IList[Qubit, Len]`, `perm: IList[Int, Len]`; attribute `insert_moves` | — | `arrange.Permute` |
| `Measure` | `qubits: IList[Qubit, Len]` | `measurements: IList[MeasurementResult, Len]` | `qubit.Measure` |

`Measure` is **not terminal**. The state continues after it, so measurement
results can drive later branches. Whether a measured qubit may be used again is
for architecture validation to decide, not the dialect. The current physical
pipeline's terminal-only rule (`PhysicalTerminalMeasurementValidation`) is not
run on this path.

The call and frame statements (`enter`, `exit`, `prepare`, `invoke`) are
defined under [Calls and frames](#calls-and-frames).

### What `State` means

A `State` value is the whole machine configuration:

- which atoms exist;
- where each atom is;
- which qubit reference each atom carries (the **binding**).

Qubit references are stable identities for the life of the kernel. `qmove`
statements never name locations. Each one acts on whichever atoms hold the
referenced qubits at that point in the chain. (Below, "the quantum information"
means the state a reference carries, as opposed to where its atom sits; all
qubits here are physical qubits.)

- **`Permute`** is a permutation gate on quantum information: afterwards
  `qubits[i]` holds the quantum information `qubits[perm[i]]` held. Both
  `insert_moves` settings have this same effect on quantum information.
  `insert_moves` is a binding **constraint** on how synthesis realizes it:
  - `False`: realized by relabeling the binding, with no atom moves;
  - `True`: realized with moves that restore the previous binding.

  Because the binding lives in `State`, qubit identity is unaffected by
  `Permute`. The upstream identity analysis (#332) never needs to model it; only
  placement analysis does.
- **`MoveTo`** leaves quantum information unchanged and adds a layout
  constraint: afterwards, the atoms carrying those qubits are at those locations.
- **Path-dependent bindings are legal IR.** A relabel `Permute` in one
  `scf.IfElse` arm makes the binding after the join depend on which arm ran. A
  relabel in an `scf.For` body makes the binding after `k` iterations `perm^k`,
  which repeats with the order of `perm`. Synthesis must either support this
  (lane tables indexed at runtime, unrolling by the order, or restoring moves)
  or report it as unsupported.

### Mixing levels on one chain

`qmove` statements and address-level `move` statements (`Move`, `LocalR`,
`CZ(zone)`, …) may share one chain. Address-level statements act on locations;
`qmove` statements act on wherever their qubits currently are. This is defined
now so that later synthesis is a local rewrite: one `qmove` statement becomes
`move.Move…; move.LocalR…` on the same chain.

## State threading

### Normal form

- **Every qmove method**, entry kernel and subroutine alike, keeps its signature.
  Its top-level block opens one chain with `qmove.enter(frame)` and closes it
  with exactly one `qmove.exit`. The entry kernel's frame is `MachineFrame()`
  (see [Calls and frames](#calls-and-frames)).
- No `move.Load` / `move.Store` remains in qmove IR. The chain ends exactly once,
  so there are no extra writes to the cell mid-chain. (An earlier draft allowed
  extra `Store`s in an entry kernel; unifying the entry with subroutines removed
  them.)
- State is threaded only through structure whose body **touches** the state. A
  body touches the state if it contains a stateful statement, directly or in a
  nested region. A call to a subroutine is a stateful statement
  (`qmove.invoke`), so no call-graph scan is needed.

### `scf.IfElse`

`IfElse` arms take only the condition as a block argument, so the incoming
state is **captured** from the enclosing scope by both arms. Each arm's
`scf.Yield` carries the arm's final state as its **first** value, and the
`IfElse`'s first result is the state after the join. An arm with no stateful
statements yields the captured state unchanged. A missing `else` body is
created so it can do that.

### `scf.For`

The state is the **first** initializer and the **first** loop-carried value:
the body's block arguments become `(loop_var, state, *carried)`, the body's
`scf.Yield` yields the state first, and the `For`'s first result is the final
state. A zero-trip loop returns the initializer.

### Example

```
%s0 = qmove.enter() {frame = MachineFrame()}
%s1 = qmove.r(%s0, %qs, ...)
%s2, %m = qmove.measure(%s1, %a)
%c  = py.indexing.getitem(qubit.is_one(%m), 0)
%s3 = scf.if %c {                       # both arms capture %s2
        %t = qmove.cz(%s2, %a, %b)
        scf.yield %t
      } else {
        scf.yield %s2                   # untouched arm passes the state through
      }
%s4 = scf.for %q in %qs iter(%st = %s3) {
        %u = qmove.r(%st, ilist.new(%q), ...)
        scf.yield %u
      }
%s5, %r = qmove.invoke(%s4, sub, %qs)
qmove.exit(%s5)
```

## Calls and frames

Subroutine calls are not threaded through by cloning the callee with a `State`
parameter. Instead, each subroutine has a **frame**: a contract on where its
argument atoms must be on entry, which scratch locations it may use, and which
zone-wide operations it may perform. The frame rule from separation logic
applies: the callee owns its footprint, and everything outside the footprint is
guaranteed unchanged across the call. That guarantee lets a caller reason about
a call without looking inside it.

### Statements

| Statement | Signature | Meaning |
|---|---|---|
| `enter` | `enter(frame: Frame \| MachineFrame \| None) -> State` | Opens a method's chain: reads the machine state, plus the frame's precondition. Exactly one per qmove method. Trait `EmitsState(originates=True)`. |
| `exit` | `exit(state)` | Closes a method's chain: writes the machine state, plus the frame's postcondition. Exactly one per qmove method. Trait `ConsumesState(terminates=False)`. |
| `prepare` | `prepare(state, *args; callee) -> State` | Establishes `callee`'s precondition for `args`. Logically a no-op; physically, afterwards the arguments are at their entry slots, the scratch slots are empty, and no other atom is inside the footprint. Names the callee rather than repeating the frame, so the two cannot disagree. For a whole-machine callee the precondition is trivial: `prepare` is a no-op, and an `invoke` guarantees nothing about other atoms, because there are none outside its footprint. |
| `invoke` | `invoke(state, *args; callee) -> (State, T)` | Calls `callee` on the caller's chain. The callee's signature is unchanged. The second result, `value`, always exists and has the callee's return type (`NoneType` for a callee that returns `None`), because a kirin statement's result count is fixed by its declaration. |

`frame`, `callee` are attributes (`callee` is a `Method`, as in `func.Invoke`).
A frame is one of three kinds:

| Kind | Meaning |
|---|---|
| `Frame(...)` | Partial: entry slots, scratch slots and effects (below) |
| `MachineFrame()` | The whole machine: no footprint limit, every effect allowed |
| `None` | A **hole**: an unframed subroutine whose frame later synthesis fills with a partial `Frame` |

A `prepare` of an unframed subroutine is also a hole.

A `prepare` is **not** required to come right before an `invoke`. A state can
already satisfy the precondition, for example after an earlier call to the same
subroutine in a loop, because exit = entry (below). Whether the precondition
holds at a given `invoke` is a check on synthesized IR. The lowering in this
spec never emits `prepare`; it exists for synthesis and for hand-written IR,
and lets a `prepare` be hoisted out of a loop of calls.

### Kernel roles

Roles follow from the frame kind, not from separate statements:

- **Every qmove method** has exactly one `enter` and one `exit` in its top-level
  block, and is only called through `qmove.invoke`. A plain `func.invoke` of a
  qmove method is an error.
- The **entry kernel** gets `MachineFrame()`. A listed subroutine may also be
  given `MachineFrame()`; it then behaves like a nested entry kernel.
- **Who may call whom** follows from nesting (F5). A whole-machine method can
  only be called from a whole-machine method, so the entry kernel cannot be
  called from a partially framed subroutine. A partial frame inside a
  whole-machine caller is always fine.
- **Allocation** (`qubit.New`, `NewAt`) is only allowed under `MachineFrame()`.
  Ancillas are passed to partially framed subroutines as arguments. A
  recursive whole-machine subroutine may allocate with `qubit.new()` but not
  with `squin.qalloc`, until a kirin bug is fixed (see the input check).

### The frame

`MachineFrame` is a frozen marker with no fields. The rest of this section is
about partial frames.

A partial `Frame` is a **shape** plus a **binding**, so that a relocatable
variant can be added later without changing the IR. For now the binding is
always concrete.

```python
@dataclass(frozen=True)
class FrameShape:
    param_slots: tuple[tuple[int, int], ...]  # (qubit-typed parameter index excl. self, slot count)
    scratch_slots: int

@dataclass(frozen=True)
class Effects:
    cz_zones: frozenset[ZoneAddress]
    measure_zones: frozenset[ZoneAddress]
    global_pulses: bool

@dataclass(frozen=True)
class Frame:
    shape: FrameShape
    binding: tuple[LocationAddress, ...]  # parameter slots in parameter order, then scratch
    effects: Effects
```

- **Slots:** a `Qubit` parameter has 1 slot; an `IList[Qubit, Literal[N]]`
  parameter has `N`, in element order. A framed subroutine cannot take an
  `IList[Qubit, Any]` parameter.
- **Exit = entry.** On return, every argument atom is back in its own entry slot,
  and every scratch slot is empty again. There is no exit field. So a loop of
  calls needs only one `prepare` before the loop. A relabel `Permute` inside a
  subroutine is allowed only if its net effect on the binding is the identity at
  `exit`. An `insert_moves=True` `Permute` is always fine.
- **Effects are permissions.** Before synthesis, `qmove.CZ` and `qmove.Measure`
  name no zone; the frame bounds which zones synthesis may choose for them, and
  whether it may use global pulses.
- **Nesting.** A method that calls a partially framed subroutine must pass it a
  footprint inside its own footprint, with effects that are a subset of its own,
  like stack frames. `MachineFrame()` contains every partial frame and is
  contained only in another `MachineFrame()`.

Users pin frames through the lowering's `subroutines` mapping (see
[Lowering](#lowering-native-ir--qmove)).

### Spectator policies

A footprint alone does not isolate a callee from the caller's other atoms
(spectators):

- `move.CZ(zone)` entangles every complete pair in the zone, including a pair
  formed by a callee atom and a spectator at the partner site.
- Measurement reads out every atom in the measured zones.
- `GlobalR` / `GlobalRz` hit every atom.

How strict to be depends on the machine. A single-zone machine needs a more
permissive rule than a zoned one. So spectator safety is a **policy**: a
pluggable, checkable set of rules, passed to validation in the same way as the
`ArchSpec`, one per compilation. The frame records facts (its `Effects`); the
policy decides whether those facts are acceptable and what they require of a
call site.

```python
class SpectatorPolicy(ABC):
    @abstractmethod
    def check_frame(self, frame: Frame, arch: ArchSpec) -> list[str]:
        """Static: are these effects acceptable for this footprint on this arch?"""

    def check_call(self, frame: Frame, atoms: AtomStateData, arch: ArchSpec) -> list[str]:
        """At an invoke, once atom positions exist: are the spectators safe?"""
        raise NotImplementedError
```

Both return human-readable problems; the validator turns each into an
`ir.ValidationError` anchored at the subroutine. `check_frame` runs as part of
this spec's validation. `check_call` needs real atom positions, so it runs on
synthesized IR. In this spec it is declared only.
The rules in the table below are its specification for the synthesis spec.

Two policies ship with this spec:

| | `ZonedPolicy` (strict) | `SingleZonePolicy` (permissive) |
|---|---|---|
| `check_frame` | `global_pulses` must be false | `global_pulses` false; footprint is pair-closed in `cz_zones` (every entangling-pair partner of a footprint location in those zones is also in the footprint, from the `ArchSpec`); `measure_zones` empty |
| `check_call` (specified, not implemented) | no spectator anywhere in `cz_zones ∪ measure_zones` | spectators may share `cz_zones`, but no two spectators may form a complete pair |

`SingleZonePolicy` forbids measurement in subroutines because reading out the
only zone reads every atom. A different policy could allow it with conditions.

## Lowering: native IR → `qmove`

### Entry point

```python
@dataclass
class NativeToQMove:
    arch_spec: ArchSpec
    subroutines: Mapping[ir.Method, Frame | MachineFrame | None]
    policy: SpectatorPolicy

    def emit(self, mt: ir.Method, no_raise: bool = False) -> ir.Method: ...
```

Location: `python/bloqade/lanes/transform/native_to_qmove.py`.

- **Input:** squin- or native-level IR. "Native level" means `gate.*`,
  `qubit.New`/`Measure`/`IsZero`/`IsOne`/`IsLost`, `gemini.common.NewAt` and
  `arrange.MoveTo`/`Permute`, plus `scf`, `func`, `py` and `ilist`.
- **No unrolling.** There is no `AggressiveUnroll` and no `ScfToCf`.
- **Originals are never mutated.** Every method the transform changes is a
  `similar()` clone, including subroutine bodies; `qmove.invoke` statements
  target the clones. Subroutines are identified by the user's original `Method`
  objects, which is what the IR references before any cloning.
- **`no_raise` defaults to `False`.** The transforms in the existing pipeline
  default to `True`, but `no_raise=True` can hide analysis errors behind
  degenerate output, which is a bad default for a new path.

### Subroutines are opt-in

The keys of `subroutines` stay as calls; each value is a pinned partial frame,
`MachineFrame()`, or `None` (a hole). The entry kernel always gets
`MachineFrame()`. **Every other call is inlined.** This matters because squin's gate
functions are kernels: `squin.h`, `squin.cx` and `squin.broadcast.h` are all
`Method`s. Today `AggressiveUnroll` is what inlines them. With subroutines
opt-in, stdlib gate kernels always disappear, and existing kernels lower as they
do today, minus the unrolling. A recursive kernel that is not listed is an
error, since it cannot be inlined.

A subroutine call has to survive until this transform sees it.
`bloqade.gemini.physical.kernel` inlines calls when the kernel is decorated
(`inline=True` by default), so a physical kernel that calls a subroutine must be
decorated with `inline=False`. `squin.kernel` does not inline.

### Passes

1. **Clone and inline, per method.** For the entry method and for each
   `subroutines` key, make a `similar()` clone over a dialect group that is the
   union of every reachable method's dialects plus the native `kernel` group. In
   each clone, run `Fixpoint(Walk(Chain(ConstCallToInvoke(), Inline(keep))))`,
   where `keep(code)` is true unless `code` is a subroutine's `code` object.
   (`rewrite.Inline` passes its heuristic the callee's `func.Function`
   statement, not the call site.)
   - **Why not `CallGraphPass` or `SquinToNative.emit`:** both clone every
     callee and retarget the calls, so after them a subroutine call no longer
     references the user's `Method`. In a trial run, the subroutine was then
     silently inlined. Per-method cloning keeps calls pointing at the originals
     until step 2 retargets them.
   - **`ConstCallToInvoke`** (new, small) turns `func.Call` of a
     `py.Constant(Method)` into `func.Invoke`. `rewrite.Inline` only inlines
     `func.Call` of a lambda, and kirin's `Call2Invoke` relies on const-prop
     hints, which are not set inside `scf.For` bodies. In a trial run, gate calls
     inside loops survived for that reason.
2. **Squin → native, per clone.** Apply `Walk(DecomposeCliffordToNative())`
   and then `Walk(GateRule())` (`SquinToNative`'s own rule) directly, both
   unchanged. `GateRule` turns each squin gate into a call of a native stdlib
   kernel, so repeat step 1's inline fixpoint to expose the `gate.*` statements.
   Then retarget every `func.invoke` of a subroutine original to its clone
   (bloqade's `ReplaceMethods`).
3. **Types:** `TypeInfer`, then `RefineQubitTypes` (see
   [Qubit type refinement](#qubit-type-refinement)).
4. **Input check** (a `ValidationPass`, reporting every problem at once):
   - `qubit.Reset`;
   - any `gemini.logical.*` or `gemini.extensions.*` statement (this spec targets
     the physical pipeline; `Initialize` in particular has no counterpart, since
     its initialize-with-angles semantics is being retired);
   - multi-block (`cf`) regions;
   - a function value that applies gates or measurements: a `func.Lambda` body,
     or the constant `Method` passed as `fn` to `ilist.map` / `for_each` /
     `foldl` / `foldr` / `scan`, containing a quantum statement other than
     allocation. Allocation-only function values are fine: `qalloc`'s
     `ilist.map(_new, range(n))`, and the physical-kernel idiom
     `ilist.map(lambda addr: qubit.new_at(...), addrs)`;
   - an early return: a `func.Return` inside an `scf` body (the state would
     leave the chain without reaching `exit`);
   - a `func.Call` left after inlining, i.e. a call through a function value.
     The lowering only threads the state through `qmove.invoke`, so any gates
     behind such a call would never join the chain;
   - allocation in a method whose frame is not `MachineFrame()`;
   - any remaining statement from a quantum dialect (`squin`, `gate`, `qubit`,
     `arrange`, `gemini.*`) that the lowering table does not cover;
   - a `func.Invoke` whose callee is not a subroutine clone. Everything else
     should have been inlined, and an invoke the lowering does not lower stays
     off the chain. (`lower_to_native` also raises if its inline fixpoint stops
     at the iteration limit without converging.);
   - a quantum statement or `func.Invoke` whose result type inference typed
     `Bottom`, for example `squin.measure(qs)` on a register (`squin.measure`
     takes one `Qubit`). `TypeInfer` does not raise on this, so the input check
     is where the user's real type error is reported. Later passes also treat a
     `Bottom`-typed value as never being a `State`;
   - a recursive kernel that is not a `subroutines` key. This one is checked
     **before** step 1, with the call graph from
     `bloqade.gemini.common.validation.recursion`, because inlining a recursive
     kernel never reaches a fixpoint;
   - a call-graph cycle that reaches allocation through a function value, such
     as `squin.qalloc`'s `ilist.map(_new, range(n))`. Also checked before
     step 1, from the entry and from every subroutine (each is lowered, called
     or not), and skipped under `no_raise`. This is a kirin bug, not a qmove
     rule. kirin's constant propagation follows recursion down to its depth
     limit and takes the call it cuts off there for pure. One level above that,
     the allocating `ilist.map` therefore looks pure, with constant operands,
     and is evaluated concretely. `TypeInfer` in step 3 then raises
     `NotImplementedError` for `qubit.New`, whatever the frame. A bare
     `qubit.new()` is not affected, so a recursive whole-machine subroutine
     can allocate that way. Drop this check once kirin marks a call it cuts
     off as impure.
5. **Local lowering** (`python/bloqade/lanes/rewrite/native2qmove.py`). Every
   source statement becomes `Load; qmove.X(state, …); Store`:
   - `gate.*` and `arrange.*` map one-to-one;
   - `qubit.Measure` becomes `qmove.Measure`, whose `measurements` replaces the
     original result;
   - a `func.invoke` of a subroutine becomes `qmove.invoke` targeting the clone.

   Each rule is local, and its output is valid C-style IR.
6. **Threading** (`python/bloqade/lanes/rewrite/qmove_state.py`). One recursive
   procedure, `thread_block(block, state) -> state`, is applied to each method's
   top-level block:
   - replace each `Load` with the current state and delete it; delete each
     `Store`;
   - each stateful statement's state result becomes the new current state;
   - at an `scf` statement whose body touches the state, recurse into the body,
     starting an `IfElse` arm from the captured state and a `For` body from a new
     block argument, then **rebuild** the statement with the state as its first
     yield and result (kirin fixes a statement's result count when it is built),
     and remap the old results' uses to the shifted results. `scf.For` often
     already has loop-carried values: kirin's Python lowering carries a register
     read in a loop body as `iter_args`, even when it is only yielded back
     unchanged (`for i in range(2): squin.z(qs[i + 1])` carries `qs`). The state
     goes in front of them;
   - at the top level, open the chain with `qmove.enter(frame)`
     (`MachineFrame()` for the entry kernel, the `subroutines` value for a
     subroutine) and close it with `qmove.exit`. Every method gets the pair,
     even one with no quantum statements, so "is a qmove method" means "has an
     `enter`" and a pinned frame is never dropped.

   This is a direct recursive traversal, not `kirin.rewrite.Walk`. `Walk` visits
   a region's blocks in reverse and reaches a nested statement's regions before
   the statement that owns them (`kirin/rewrite/walk.py`,
   `populate_worklist_Statement` / `populate_worklist_Region`, kirin 0.22.16).
   Threading must see statements in execution order. Both behaviours are pinned
   by `python/tests/rewrite/test_walk_order.py`.

   The existing `rewrite/state.py:RewriteLoadStore` cannot be reused here. It
   recognizes stateful statements only by their `ConsumesState`/`EmitsState`
   traits, and a threaded `scf.IfElse` has a `State` result but no trait (kirin
   owns `scf`). `RewriteLoadStore` would treat the branch as stateless, replace
   the following `Load` with the pre-branch state, and drop the branch's effect.
7. **Cleanup:** DCE, `TypeInfer`, then `RefineQubitTypes` again.
8. **Validation:** V1–V3, F1–F5, then `policy.check_frame` for every method
   under a partial frame. Errors from every method are collected and raised together.
   Under `no_raise=True` validation is skipped.

### Qubit type refinement

`RefineQubitTypes` (`python/bloqade/lanes/rewrite/refine_qubit_types.py`) is a
separate, reusable pass that narrows the types of qubit-valued SSA values using
`AddressAnalysis`. The new path needs it because type inference cannot recover
qubit types once calls are inlined without unrolling:

- **`func.invoke` is typed from the callee's signature.** Type inference does
  not specialize a call to its arguments, because that would mutate the callee's
  signature, which every call site shares. So `qalloc(3)` is typed
  `IList[Qubit, Any]`.
- **Constants are not `Literal`s.** `py.Constant` is typed
  `PyClass(type(value))`, e.g. `!py.int`. Values live in the const-prop
  lattice; `types.Literal` appears only where an impl builds it, as
  `ilist.range` does from const hints.
- **Re-inference cannot narrow an inlined type.**
  `TypeInference.eval_fallback` solves the type variables correctly (for
  `ilist.map(_new, range(3))` it finds `ListLen := Literal(3)`), but substitutes
  them into the result's *current* SSA type rather than the statement's declared
  type. An inlined statement arrives already typed from the callee's generic
  body (`IList[Qubit, Any]`), with no type variables left to substitute.

`AddressAnalysis` re-interprets each callee with the real arguments, so it does
know `qalloc(3)` is a 3-qubit register (`AddressReg((0, 1, 2))`).

**Rule.** For each SSA value (statement results and block arguments) in the
analysis frame, derive a type from its address:

| Address | Derived type |
|---|---|
| `AddressQubit`, `UnknownQubit` | `Qubit` |
| `AddressReg(data)` | `IList[Qubit, Literal(len(data))]` |
| `UnknownReg` | `IList[Qubit, Any]` |
| `PartialIList(elems)` whose elements all derive a type | `IList[join of element types, Literal(len(elems))]` |
| anything else, including `Unknown` | no refinement |

Set the value's type to the **meet** of its current type and the derived type,
so the pass only ever narrows. A loop variable whose address joins to
`Unknown` keeps its inferred `Qubit`. If the meet is bottom, the value's type
contradicts what it holds (a value typed `Qubit` that holds a register), and
the pass reports a validation error instead of setting the type.

Values that type inference already typed `Bottom` are skipped. Their address
says nothing: `AddressAnalysis` derives `UnknownReg` from a `Bottom` type,
because `Bottom` is a subtype of every `IList[Qubit]`. Meeting with it would
turn the value into a spurious contradiction. An example is the `Bottom`-typed
result of `squin.measure(qs)` on a register. Type inference does *not* report
that mistake; the input check (step 4) does.

**Placement.** `TypeInfer` overwrites every type it infers (`ApplyType`), so
`RefineQubitTypes` runs immediately after each `TypeInfer` in this transform
(passes 3 and 7). It runs on each method separately, entry and subroutine
clones, because `AddressAnalysis` discards callee frames. Subroutine parameters
are `UnknownReg` / `UnknownQubit` (or `Unknown`), so they keep their annotated
types. It does not change the existing pipeline.

## Validation

All checks are structural, so they are direct walks written as
`kirin.validation.ValidationPass`es, following the `FlatBlockValidation` and
`gemini.no_recursion` precedent, not `Forward` analyses. (A `Forward` pass only
visits reachable code.) Each pass collects every violation and reports them
together, naming the rule and the statement, for example
`V1: %s3 consumed twice on one path, by qmove.r and qmove.cz`. Errors inside
inlined code point at the callee's source, not the call site; that is a known
kirin limitation.

Location: `python/bloqade/lanes/validation/qmove.py`, and
`python/bloqade/lanes/validation/spectator.py` for the policies.

### State rules (any IR containing stateful statements)

These apply to every stateful statement, including address-level `move`
statements on a mixed chain.

- **V1 — Use-def, ignoring `Store`.** (A `Store` can only appear in mixed or
  hand-built IR, and V2 rejects it there; the carve-out keeps it from being
  reported twice.)
  - Every `State` value has at least one use. A state that is neither exited,
    yielded, passed on nor consumed is a dropped update.
  - No execution path consumes a state twice, not counting `Store`. Concretely,
    its non-`Store` uses must each lie in a different arm of a common enclosing
    `scf.IfElse`, and none may lie inside an `scf.For` body that does not also
    contain the definition.
- **V2 — Chain ends.**
  - A qmove method has exactly one `enter` and one `exit`, both in its top-level
    block.
  - `Load`/`Store` do not appear in qmove IR.

  V1 and V2 together guarantee that a chain inside a region cannot re-read the
  cell. It must take its state from outside and yield it back, and any update
  that goes nowhere fails V1.
- **V3 — Per-statement checks, where operands are constant.**
  - `CZ` controls and targets have equal length.
  - `Permute.perm` is a permutation of `range(len(qubits))`.
  - `MoveTo` has as many locations as qubits.

  These are the checks `circuit2place` performs today by raising `ValueError`.

### Call rules

- `qmove.invoke` and `qmove.prepare` target qmove methods (methods with
  `enter`).
- A plain `func.invoke` of a qmove method is an error.
- A whole-machine callee needs a whole-machine caller. This is F5's
  whole-machine half, checked here because it holds whatever the caller's frame
  is (partial, hole or whole machine).

### Frame rules (methods under a partial frame only)

Holes, unframed `prepare`s and `MachineFrame()` methods skip these. A whole
machine has no footprint or effect limit to check.

- **F1** — Binding locations are distinct, there is one per slot, and each is
  valid in the `ArchSpec`. Effect zones are valid in the `ArchSpec`.
- **F2** — The shape matches the subroutine's qubit parameters: every qubit-typed
  parameter has an entry, of the right size, and no other parameter does.
- **F3** — The body stays within its footprint and effects:
  - a `qmove.MoveTo` with constant locations stays inside the binding;
  - address-level `move` statements in the body respect `effects` (`move.CZ`
    zones in `cz_zones`, measurement zones in `measure_zones`, no
    `GlobalR`/`GlobalRz` unless `global_pulses`);
  - `move.Move` lanes start and end inside the footprint.
- **F4** — Exit = entry: constant-`perm` relabel `Permute`s on parameter slots
  compose to the identity. F4 judges only the case it can see completely.
  - The relabels it composes are top-level, constant-perm, valid relabels
    acting directly on a parameter.
  - If the method has any other relabel (nested in a region, non-constant,
    malformed, or on a non-parameter operand that may alias), F4 abstains for
    that method.
  - The general case needs qubit identity and is an obligation for the
    #332-based analysis.
- **F5** — For an `invoke` of a partially framed subroutine from a partially
  framed method: the inner footprint is inside the outer footprint, and the
  inner effects are a subset of the outer effects. (The whole-machine half is
  under Call rules.)
- **Policy** — `policy.check_frame(frame, arch)`.

### Not checked here

Anything that depends on qubit identity: distinct qubits within one statement
(`q[i]`, `q[j]` with `i == j` at runtime), the general case of F4, and whether a
`prepare`/`invoke` precondition actually holds. These need #332 and synthesized
layouts.

## Testing

Tests live under `python/tests/`, mirroring the package layout:
`dialects/test_qmove.py`, `rewrite/test_native2qmove.py`,
`rewrite/test_qmove_state.py`, `rewrite/test_refine_qubit_types.py`,
`validation/test_qmove.py`,
`validation/test_spectator.py`, and `test_transform_native_to_qmove.py`.

- **Dialect:** build, print and type-check each statement.
- **Verifier:** hand-built IR with one passing case and at least one failing case
  per rule (V1–V3, the call rules, F1–F5), plus one IR with several violations to
  confirm they are all reported in one run.
- **Policies:** `check_frame` for `ZonedPolicy` and `SingleZonePolicy` on
  synthetic architectures built with `ArchBuilder` (the `python/tests/conftest.py`
  fixtures), not only the shipped Gemini specs.
- **Type refinement** (`rewrite/test_refine_qubit_types.py`):
  - `qalloc(3)` is narrowed from `IList[Qubit, Any]` to
    `IList[Qubit, Literal(3)]`;
  - a loop variable over a register keeps `Qubit` (an `Unknown` address never
    widens a type);
  - a subroutine's parameter keeps its annotated type;
  - a value typed `Qubit` that holds a register is reported as a contradiction;
  - a `Bottom`-typed value (from `squin.measure(qs)` on a register) is left
    alone;
  - running it twice changes nothing.
- **Lowering:** kernels covering
  - straight-line code;
  - a physical kernel using `new_at`, `arrange.move_to` and `arrange.permute`;
  - `scf.IfElse` on a mid-circuit measurement, including an arm with no gates and
    a missing `else`;
  - `scf.For` over an `IList[Qubit]`, and over a `range` using `q[i]`;
  - a loop and a branch nested inside each other;
  - a subroutine with a pinned frame, and one with a hole;
  - nested subroutines;
  - a recursive subroutine, and the error for a recursive kernel that is not
    listed;
  - a recursive whole-machine subroutine allocating with `qubit.new()`, and the
    up-front error, under any frame, for one that allocates through
    `squin.qalloc`, directly or through a callee;
  - calls that are not listed, including squin stdlib, being inlined;
  - pinned allocation (`NewAt`) alongside `qubit.New`, and `IsZero`/`IsOne`/
    `IsLost` on `Measure` results feeding a branch;
  - the input-check errors for logical-pipeline statements
    (`gemini.logical.Initialize`, `TerminalLogicalMeasurement`, `StarRz`).

  Every output must pass validation. Every method's top-level block must open
  with `qmove.enter` (`MachineFrame()` for the entry kernel) and end with
  `qmove.exit`, with no `Load`/`Store` anywhere. Also tested: a whole-machine
  subroutine may allocate, a partial or unframed one may not, and a
  whole-machine callee from a partially framed caller fails F5.
- **Oracle: the lowering adds only state plumbing.** A test-only `EraseQMove`
  pass (in `python/tests/`) removes everything state-related:
  - state operands and results;
  - `Load`/`Store`/`enter`/`exit`;
  - the state's yields, initializers and block arguments;

  and turns `qmove` statements back into the native statements they came from
  (`qmove.invoke` back into `func.invoke`). For the test kernels, the result
  must equal the IR after passes 1–3 and DCE, compared **block by block** with a
  test-only comparator.
  - This does not hold for every kernel, but only result types can differ.
  - Pass 3 refines once. A loop-carried copy of a register that
    `AddressAnalysis` never visits (for example in a loop with an unknown range)
    can keep `IList[Qubit, Any]` in the reference.
  - Pass 7's re-inference then propagates the refined type.
  - Iterating `TypeInfer`/`RefineQubitTypes` to a fixpoint in pass 3 would
    close this; it is left for later.
  kirin's `is_structurally_equal` cannot be used for this: `Region`'s version
  records every block pair in its context before comparing, so `Block`'s
  version returns early and nested region contents are never compared (two
  function bodies `x+1` and `x*2` compare equal), and result types are never
  compared. The comparator recurses into every nested region, maps SSA values,
  compares result types, and compares callee `Method`s by `sym_name`. Its own
  tests include a difference that exists only inside an `scf` body.
- **No regressions:** only new files are added, apart from registering the new
  dialect module in `python/tests/test_import_cycles.py`. No existing pipeline
  changes, so the existing suite and the benchmark baseline CSVs stay the
  same.

## Out of scope

- **All synthesis.** No `qmove → move` lowering, no layout or placement analysis
  on `qmove`, `prepare` never emitted, holes never filled. `check_call` is
  declared only.
- **Checks that depend on qubit identity**, which wait on
  [bloqade-circuit#332](https://github.com/QuEraComputing/bloqade-circuit/issues/332).
- `cf` control flow, closures that apply gates, `qubit.Reset`, and allocation
  outside a `MachineFrame()`.
- **The logical pipeline.** `gemini.logical.*` and `gemini.extensions.*` are
  rejected by the input check. That covers the logical measurement dataflow
  (`TerminalLogicalMeasurement` and `ConvertToPhysicalMeasurements`), `StarRz`,
  and initialization with angles (`gemini.logical.Initialize`, and the
  `place`-level `Initialize` / `NewLogicalQubit` angles). Initialization with
  angles is being retired rather than ported; what replaces it is a separate
  decision.
- Relocatable frames, and a kernel-body intrinsic for pinning a frame (frames
  are pinned through `subroutines` for now).
- Any change to `PhysicalPipeline`, the logical kernel decorator or other
  user-facing entry points.
- Retiring `place` / `StaticPlacement`. The new path runs alongside the old one
  until synthesis on `qmove` matches the current pipeline's output.
- **Deferred until after this refactor:** restructuring how qubit-addressed and
  location-addressed statements relate.

## Dependencies

- [bloqade-circuit#332](https://github.com/QuEraComputing/bloqade-circuit/issues/332)
  (reopened 2026-10-02): `SlotQubit` / `SlotRegister` for parameters,
  `AddressQubit(parent, index)`, `LoopInteger(loop, offset)`. Only its
  `PartialLambda` part landed. Today's `AddressAnalysis` loses qubit identity in
  exactly the cases this IR is built for:
  - `func.Invoke` re-interprets the callee and discards its frame, so callee SSA
    values have no recorded addresses (also the source of the recursion blowup,
    bloqade-circuit#852);
  - an `IList[Qubit]` / `Qubit` parameter becomes `UnknownReg` / `UnknownQubit`,
    with no "element `i` of parameter 0" (bloqade-circuit#584 covers the entry
    arguments);
  - `scf.For` joins each body value across iterations, so the loop variable
    becomes `UnknownQubit`;
  - `GetItem` with a non-constant index returns `Unknown`.

  Not needed by this spec; needed by every synthesis spec that follows.

## Related

- `python/bloqade/lanes/dialects/place.py`, `dialects/move.py` — the dialects
  this design sits beside.
- `python/bloqade/lanes/rewrite/state.py` — `RewriteLoadStore`,
  `InsertBlockArgs`, `RewriteBranches`: the existing cell-to-SSA conversion for
  `cf`.
- `python/bloqade/lanes/rewrite/circuit2place.py` — `RewritePlaceOperations`,
  `MergeStaticPlacement`.
- `python/bloqade/lanes/dialects/stack_move.py` — bytecode-level `Move` /
  `LocalR` take lane and location addresses as SSA values, and `NewArray` /
  `GetItem` take runtime indices, so index-dependent schedules can be expressed
  at the bytecode level.
