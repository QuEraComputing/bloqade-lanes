# Rz Elimination (Virtual Z) — Design

**Status:** Draft
**Date:** 2026-09-14
**Author:** brainstormed with Naoki Kanazawa
**Branch:** `nk/add-rz-removal`

## Goal

Eliminate `Rz` from compiled Gemini logical programs. The backend runs
`R(axis_angle, rotation_angle)` and `CZ`; a subset-addressed `Rz`
(`move.local_rz`) is not a primitive it can execute, yet the logical pipeline
emits three of them for an `h`/`s`/`cx` teleportation kernel today.

Two exact identities make the elimination total:

```
R(φ, θ) · Rz(α) = Rz(α) · R(φ − α, θ)      commutation, updates the axis
CZ · Rz(α)      = Rz(α) · CZ               both diagonal
```

Sweeping a circuit in order, every `Rz` accumulates into a per-qubit *frame*
that ends up at the wire end, where the terminal Z-basis measurement absorbs it.
Gate count never increases: `Rz` statements are deleted, everything else is
rewritten in place with a new `axis_angle`.

## Non-goals

- **Collapsing the leading single-qubit prefix into `logical_initialize`.** A
  larger optimization; it would stop the emitted circuit resembling the user's
  source. Virtual Z preserves structure, so no opt-out flag is needed here.
- **Gate fusion or reordering.** `FuseAdjacentGates` and the ASAP/ALAP policies
  compose with this; the pass never moves a statement.
- **Making STAR rotations disappear.** `operations.StarRz` is a user-requested
  payload, not a compiler artifact.
- **Absorbing the residual into `Initialize`** instead of discarding it — needed
  only where no Z-basis readout is guaranteed, i.e. a future `PhysicalPipeline`
  use. Follow-up.

## Layer choice: native

The pass runs inside `NativeToPlaceBase.emit`, after
`AggressiveUnroll(...).fixpoint()` and before `RewritePlaceOperations`.

`Rz` elimination is a circuit identity: it reads no architecture, cares about no
layout, and would hold on a machine with different geometry. The place layer
exists to turn qubits into placement indices and feed layout and routing —
putting gate algebra there would mix a layout-shaped stage with an optimization
unrelated to layout. At native the IR has no placement, no layout and no location
addresses, so the rewrite is architecture-agnostic by construction.

Rejected, recorded for future context:

- **squin layer.** Not flat yet — `AggressiveUnroll` has not run, so the frame
  would need interprocedural threading through `func.Invoke` and `scf.For`.
- **place layer.** Viable and marginally easier (qubits are `tuple[int, ...]`
  attributes; wiring would be additive). Rejected on the layering argument above,
  and because its indices are assigned per-placement by `circuit2place` then
  remapped by `MergeStaticPlacement` — a dependency native does not have.

## The rewrite

`EliminateRzPass` is a single pass: sweep the block in order, absorb every `Rz` into
the frame, rewrite each `R`, and **discard the residual frame** when the sweep
ends. Nothing is ever materialized.

The pure sweep returns `(rewritten, residual_frame)`, with the invariant

```
U_before  =  Rz(residual)  ·  U_after
```

so tests can apply the residual themselves and assert exact unitary equality.

*Rejected:* splitting this into "commute to the wire end" plus "drop the trailing
`Rz`", to make the first half exactly unitary-preserving. Callers may run
`RemovePostProcessing(delete_terminal_measure=True)`, leaving no measurement
statement to insert before, so the first pass would be *choosing* an insertion
point rather than being told one. The invariant above gives the same test
strength without materializing anything.

Worked example (teleportation kernel; angles in turns, `%r1` holds logical qubit
1), verified numerically against that invariant:

```
before                                  after
  rz(-¼)                qubits=%r1        r (axis=¼, rot=-¼)   qubits=%r1
  r (axis=0,  rot=-¼)   qubits=%r1        r (axis=0, rot=-¼)   qubits=%r1
  rz(-¼)                qubits=%r1        cz(controls=%r0, targets=%r1)
  rz(-¼)                qubits=%r1        r (axis=0, rot=¼)    qubits=%r1
  r (axis=¼,  rot=-¼)   qubits=%r1        terminal_logical_measurement
  cz(controls=%r0, targets=%r1)
  r (axis=¼,  rot=¼)    qubits=%r1        residual frame: {q1: -¾}  (discarded)
  terminal_logical_measurement
```

### Why discarding is sound

The residual is diagonal, so it commutes with a Z-basis readout and cannot shift
any outcome probability. This holds whether or not the measurement statement is
still in the IR: `operations.TerminalLogicalMeasurement` is present in the native
window when the caller keeps it (verified), and absent after
`RemovePostProcessing` — but the device's final readout is in the Z basis either
way, and `GeminiTerminalMeasurementValidation` has already established that the
source program's one measurement is terminal.

**The assumption this rests on, stated explicitly:** the program's observable
output is a Z-basis readout of the final state, not the final state itself. A
future flow that consumed the output *state* — feeding it to another program
phase-sensitively — would invalidate the pass. Nothing today does this, and the
terminal-measure validation rules it out for logical kernels.

## Implementation

### Module

`python/bloqade/lanes/rewrite/eliminate_rz.py`, exporting `EliminateRzPass` and
the pure sweep it wraps. Neither imports `place`, `arch`, or anything layout-shaped.

### Frame: continuous float, keyed on qubit SSA values

```
frame[v] += rz.rotation_angle            # Rz absorbed, statement deleted
new_axis  = (r.axis_angle - frame[v]) % 1.0
```

A `float` in turns, not an integer `k ∈ ℤ₄`: angles are already continuous floats
in native IR, so quantizing would add a rounding concern the pass does not
otherwise have, and would block reuse by `PhysicalPipeline` where arbitrary
angles are legitimate. Accumulate the frame as a running sum and derive each new
axis from the *original* axis in one subtraction, so rounding is one ulp per gate
rather than compounding.

### Qubit identity

A native gate's `qubits` is one SSA value of `IList` type. Post-unroll the
register is always materialized, so the qubit values are one destructuring away:

```
%qubit  = qubit.new()
%qubits = py.ilist.new(values=(%qubit)){elem_type=!py.Qubit}
          native.gate.rz(rotation_angle=%angle, qubits=%qubits)
```

Destructure `stmt.qubits.owner` as `ilist.New`, read `.values`, key the frame on
those SSA values. No address analysis, no value resolution —
`circuit2place.rewrite_R` already relies on this shape. `ir.SSAValue.__hash__`
returns `id(self)`, so the dict is identity-keyed (verified).

In the logical pipeline one such value is one logical qubit = one Steane block.
Physical sites do not exist at this layer; a word address is expanded into 7 site
addresses only later, at the move layer.

Entries are created lazily (`frame.setdefault(v, 0.0)`) — no pre-scan for
`qubit.new`. A qubit never touched has no entry, which reads as frame 0.

### Traversal

One forward iteration over the block's statements in order, carrying
`dict[ir.SSAValue, float]`. No DAG and no analysis pass — the block is already in
execution order.

| statement | action |
|---|---|
| `native.gate.Rz` | `frame[v] += angle` per qubit; **delete** |
| `native.gate.R` | `axis = (axis − frame[v]) % 1`; split if frames differ |
| `native.gate.CZ` | untouched; frames are per-qubit and need not agree |
| `operations.StarRz` | untouched — diagonal, frame commutes past it exactly |
| `operations.Initialize` | assert frame is 0 for its qubits |
| `operations.TerminalLogicalMeasurement` | discard frames for its qubits; assert no later gate touches them |
| `qubit.New`, `ilist.New`, `py.Constant`, `func.Return` | pass through |
| anything else touching a qubit | raise |

Whatever frame remains when the block ends is discarded — that is the
`RemovePostProcessing` path, where no measurement statement exists to discard it
earlier.

`Initialize` is always at the head of a wire (validation forces non-Clifford
gates first), so the frame is zero. Asserting rather than absorbing makes a
future mid-wire `Initialize` fail loudly instead of silently dropping a phase.

### Splitting

Across logical qubits, **never within a code block** — a Steane block is one
qubit value carrying one frame, so its phase is uniform across all 7 atoms by
construction and transversality is preserved automatically.

What splits is a broadcast gate such as `squin.sqrt_x(reg)`, which lowers to one
`native.gate.R` over an `ilist.New` of several qubits with differing frames. It
becomes one statement per distinct frame value, each with a **new `ilist.New`**
holding that group, preserving original relative ordering.

Assert a true partition at each split: groups pairwise disjoint, union equal to
the original value list as a multiset. A bug here silently drops a gate on one
qubit — it survives a 2-qubit unitary test and only fails on a wider one.

### Angle constants must be shared

`FuseAdjacentGates` runs later, at place, and matches parameters by **SSA
identity** (`stmt.axis_angle is head.axis_angle`). Angle values are carried
through unchanged by `circuit2place.rewrite_R`, so whatever sharing this pass
leaves is what fusion sees. Sharing is real: in the native dump one `%angle =
0.25` serves an `rz` and two `r` statements.

Emit rewritten angles through a constant cache — one `py.constant` per distinct
value, reusing the existing SSA value when unchanged. Minting a fresh constant
per statement breaks fusion between numerically equal angles. Hard requirement,
not an optimization.

### Preconditions, checked at entry

| precondition | on violation |
|---|---|
| single block, no `cf`/`scf`, no unresolved calls | raise |
| every gate's `qubits.owner` is an `ilist.New` | raise |
| every `Rz` angle is a multiple of ¼ turn, when `require_clifford_angles` | raise, quoting the angle |
| no duplicate qubit value within one statement's register | raise, naming value + statement |
| only the statement kinds above | raise |

Raising rather than skipping matters: `circuit2place` handles a shape mismatch by
silently returning, but a skipped statement here leaves an `Rz` behind and breaks
the guarantee.

`require_clifford_angles` defaults `True`. It keeps axis angles on the ¼-turn
lattice, so the gate set stays `{X, Y, √X, √Y}`+adjoints — all Steane-transversal
Cliffords. This matters because a transversal gate at a non-Clifford angle is not
a logical gate at all (Eastin–Knill; see References). Set `False` only for a
future `PhysicalPipeline` use, where there is no code and no such constraint.

The tolerance is a backstop, not load-bearing: values are exact today
(`clifford2native` emits literal `0.25`/`-0.25`/`0.5`, exact dyadic floats).

A related precondition, inherited rather than checked: several rewrites hold only
up to global phase, which is unobservable because `GeminiLogicalValidation`
rejects `scf.IfElse` — there is no gate controlled on a measurement for a global
phase to become relative.

### Wiring — a fifth hook

The native window lives entirely inside `NativeToPlaceBase.emit`, so the pass
cannot be wired additively from `LogicalPipeline.emit`. The template documents
four hooks, none in the right place (`_post_unroll_validation` is adjacent but is
a validation hook and must not be abused for a rewrite). Add a fifth:

```python
AggressiveUnroll(out.dialects, no_raise=no_raise).fixpoint(out)
self._post_unroll_rewrites(out, no_raise)      # new hook, default no-op
self._post_unroll_validation(out, no_raise)
```

`LogicalNativeToPlace` overrides it; `PhysicalNativeToPlace` and the generic
`NativeToPlace` inherit the no-op, so no physical compile changes. Update the
base class docstring's hook list to five.

On by default and not flag-guarded: a flag that turns it off produces IR the
backend cannot run.

## Scope of the guarantee

The logical pipeline gains a hard invariant: **no `Rz` reaches the backend.** Any
`Rz` the pass cannot eliminate is a compile error, not a silent pass-through.

`StarRz` is exempt and is not an `Rz` as far as the pass is concerned. The
pipeline never synthesizes it — the only producer is the user-facing
`gemini.logical.star_rz`, and a mid-circuit non-Clifford `squin.rz` is a
validation error rather than a fallback to the gadget. So on the default path
every `local_rz` is Clifford-derived and removable; when a user calls `star_rz`,
the surviving `local_rz` is the payload they asked for.

## Testing

- **Core sweep, no IR.** The sweep is a pure function over
  `(qubit_keys, kind, angle)` records returning `(rewritten, residual_frame)`, so
  frame accumulation, axis update and split-by-frame partitioning are testable
  without building IR.
- **Unitary equality via the residual.** On hand-built native IR, assert
  `U_before == Rz(residual) · U_after` up to global phase — exact, not
  distributional. Covers a broadcast register spanning differing frames (forcing
  a split), `CZ` with unequal frames on its pair, and `StarRz` mid-wire.
- **Distribution equality**, for both shapes (terminal measure present, and
  deleted by `RemovePostProcessing`): outcome distributions match before and
  after, with a nonzero residual in play.
- **Gate-set closure.** Under `require_clifford_angles`, assert every rewritten
  axis angle stays on the ¼-turn lattice.
- **Constant sharing.** Two gates whose frames coincide: assert they share one
  `axis_angle` SSA value (`is`, not `==`) and that `FuseAdjacentGates` still
  fuses them after lowering to place. Must span the native→place boundary to be
  meaningful.
- **Splitting.** A broadcast `sqrt_x` over two logical qubits with different
  frames: assert exactly two statements whose registers partition the original,
  neither finer than one logical qubit.
- **End-to-end.** Teleportation kernel through `LogicalPipeline`: zero
  `move.LocalRz`, and `move.LocalR` count unchanged from the pre-pass compile.
- **Physical pipeline unchanged.** A physical compile must be identical before
  and after this change.
- **Preconditions.** One test per row, asserting the message names the offending
  construct.
- **STAR regression.** Its `local_rz` survives with the `steane_star_theta` angle
  intact and the frame commuted past it correctly.

## Risks & follow-ups

- **Loss of pulse parallelism — the primary open risk, not settled by argument.**
  When two logical qubits' histories diverge their frames diverge, so an `R` that
  previously fused across them cannot fuse. Measured on place IR:

  ```
  frames DIVERGE (S on q0 only)          frames AGREE (S on both)
    Rz(-0.25)    q=(0,)                    Rz(-0.25)    q=(0,1)
    R(axis=0)    q=(1,0)  ← fused          R(axis=0)    q=(0,1)
    ---------------------------            ---------------------------
    R(axis=0)    q=(1,)                    R(axis=0.25) q=(0,1)
    R(axis=0.25) q=(0,)
    2 -> 2  (wash)                         2 -> 1  (win)
  ```

  The wash is the *optimistic* end: the benefit is fixed at the `Rz` count, while
  the cost is every subsequent fused statement spanning divergent frames, and a
  frame persists to the end of the wire. **Do not assume the win** — measure the
  logical benchmark suite and record the result here.
- **Benchmark baselines.** On by default, so the committed logical CSVs move and
  must be regenerated per `AGENT.md`.
- **The `ilist.New` shape invariant is unenforced** — it holds because
  `AggressiveUnroll` ran, and nothing checks it. The pass raises rather than
  assuming, but the native gate statements live upstream in `bloqade-circuit`, so
  a shared check is not available here.
- **Absorbing the residual into `Initialize`** unimplemented; it additionally
  needs the transversal-`S` = logical-`S†` sign convention.

## References

- `python/bloqade/lanes/transform/native_to_place.py` — template gaining the fifth hook
- `bloqade/native/dialects/gate/stmts.py` (upstream) — `R`/`Rz`/`CZ` shapes
- `python/bloqade/lanes/rewrite/circuit2place.py` — precedent for `ilist.New` destructuring
- `python/bloqade/lanes/rewrite/clifford2native.py` — defines the gate set this is closed over
- `python/bloqade/lanes/rewrite/fuse_gates.py` — the downstream fusion this must not break
- `python/bloqade/gemini/logical/validation/clifford/impls.py` — Clifford-only / first-gate-only validation
- Eastin & Knill, PRL 102, 110502 (2009), [arXiv:0811.4262](https://arxiv.org/abs/0811.4262) — why the Clifford-angle precondition is load-bearing
- McKay et al., PRA 96, 022330 (2017), [arXiv:1612.00858](https://arxiv.org/abs/1612.00858) — the virtual-Z technique
- QuEraComputing/bloqade-lanes#748 — earlier, closed-unimplemented approach
- QuEraComputing/bloqade-circuit#690 — open upstream question on where such rewrites belong
