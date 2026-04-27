# Explicit Qubit Allocation — Design

**Status:** Draft
**Date:** 2026-04-27
**Author:** brainstormed with Phillip Weinberg
**Branch:** `worktree-feat-explicit-qubit-allocation`

## Goal

Let users (and compiler passes) bypass the auto-placement heuristic for individual logical qubits by specifying a concrete `(zone_id, word_id, site_id)` address. Mix-and-match with auto-placed qubits is supported: the heuristic places the rest while respecting pinned addresses.

## Secondary goal — internal cleanup

Make `place.NewLogicalQubit.location_address` the canonical source of truth for "where this qubit lives." Downstream passes stop consulting `LayoutAnalysis` frames; they read the attribute. After the resolve rewrite runs, every `place.NewLogicalQubit` carries a concrete address, which simplifies `Place→Move` consumers (`InsertInitialize`, `InsertFill`).

## Non-goals

- **Partial-address allocations** (zone-only, word-only) — follow-up. This proposal establishes the basic semantic; partial forms can build on top.
- **`arch_query` dialect** — follow-up. A small dialect exposing arch dimensions (`arch.sites_per_word(zone_id=0)` etc.) so users can write loops like *"allocate a stripe across all sites in word 3"* without hard-coded magic numbers. Out of scope here, but flagged so the design doesn't paint itself into a corner.
- **"Address forces later-illegal op" semantic check** — stays in the existing post-compile validator (`validation/address.py`); not duplicated here.
- **Placement strategy / atom-movement scheduling** — untouched. This feature only affects *initial* placement.

## Package split

- **`bloqade.gemini`** — owns the user-facing surface: the new statement, the lowering, eager validation. Depends on `bloqade.lanes`.
- **`bloqade.lanes`** — owns the IR shape (`location_address` attribute on `place.NewLogicalQubit`), the `LayoutHeuristic` API change, the resolve rewrite, and the `Place→Move` consumer refactor.

Both live in this repo.

## Architecture overview

```
user kernel:        gemini.operations.new_at(zone, word, site)   ← runtime SSA ints
                    OR squin.qubit.new(...)                      ← unchanged plain alloc

  ┌─ eager gemini-level validation ──────────────────────┐
  │  • per-stmt method-table impl (lanes validator key):  │
  │      - const-foldability of args                      │
  │      - in-range check vs ArchSpec                     │
  │  • gemini validation pass:                            │
  │      - duplicate address detection                    │
  └───────────────────────────────────────────────────────┘
       ↓
RewriteInitializeToLogicalInitialize  (existing, unchanged)
       ↓
RewriteLogicalInitializeToNewLogical  (augmented; reads const values via Kirin's
                                       AbstractInterpreter.maybe_const / expect_const,
                                       stamps location_address onto NewLogicalQubit)
       ↓
CleanUpLogicalInitialize  (existing, unchanged)
       ↓
InitializeNewQubits  (augmented; same pattern for bare new_at)
       ↓
... rest of circuit→place lowering ...
       ↓
LayoutAnalysis (extracts pinned addresses from NewLogicalQubit.location_address;
                forwards them as `pinned` to compute_layout; hard-fails if
                heuristic returns no legal layout)
       ↓
ResolvePinnedAddresses (new, lanes-side, ALWAYS runs):
    every NewLogicalQubit ends with a concrete location_address attribute.
       ↓
InsertInitialize / InsertFill (refactored: read attribute directly,
                               no longer take address_entries / initial_layout)
       ↓
existing post-compile lanes validator (unchanged) catches semantic illegality
```

## §1 — IR statements & attributes

### `bloqade.gemini.operations.NewAt` (new statement)

User-facing allocation primitive. Lives in `bloqade.gemini.logical.dialects.operations` (sibling to `Initialize`).

```python
@statement(dialect=operations)
class NewAt(ir.Statement):
    name = "new_at"
    traits = frozenset({...}) #include python lowering trait
    zone_id: ir.SSAValue = info.argument(types.Int)
    word_id: ir.SSAValue = info.argument(types.Int)
    site_id: ir.SSAValue = info.argument(types.Int)
    qubit:   ir.ResultValue = info.result(QubitType) # import from bloqad.types

# stub for lowering

@wraps(NewAt)
def new_at(zone_id: int, word_id: int, site_id: int) -> Qubit # import from bloqad.types
    ...

```

- Three runtime SSA int args.
- Emits a `Qubit` value, identical type to `qubit.stmts.New`.
- Args **must be const-foldable** at compile time. Enforced by validation (§3), not the type system.
- Naming follows `qubit.new` / `qubit.new_at` convention.

### `place.NewLogicalQubit.location_address` (new optional attribute)

The existing op gains:

```python
location_address: LocationAddress | None = None
```

- `None` = un-pinned, layout heuristic chooses.
- `LocationAddress(zone, word, site)` = pinned.
- After `ResolvePinnedAddresses` (§2.4) runs, every `place.NewLogicalQubit` in the IR has a non-`None` attribute. This invariant is load-bearing for `InsertInitialize` and `InsertFill` (§4).

### `LocationAddress` (reused, unchanged)

Existing `(zone_id, word_id, site_id)` tuple from `bloqade.lanes.layout.encoding`. Already a Kirin-friendly attribute (immutable + hashable).

### `gemini.logical.Initialize` — no changes

Initialize is purely a transient carrier of `(theta, phi, lam)` initialization data. It does **not** carry the address. The existing rewrite chain (`RewriteLogicalInitializeToNewLogical`) walks back through SSA references from `Initialize.qubits` to the qubit-producing statements (`qubit.stmts.New` today; `gemini.operations.new_at` after this change) and replaces those allocation statements in place — that's where the `location_address` attribute gets stamped.

## §2 — Rewrites

### 2.1 — `RewriteLogicalInitializeToNewLogical` (augmented)

Current behavior (`circuit2place.py:36–53`): walks `place.LogicalInitialize`, filters its qubits to those owned by `qubit.stmts.New`, replaces each with `place.NewLogicalQubit(theta, phi, lam)`.

**Augmentation:** extend the type guard to also accept `gemini.operations.NewAt`. When found, read const values for `(zone_id, word_id, site_id)` via Kirin's `AbstractInterpreter.maybe_const` / `expect_const` (idiomatic; avoids reaching into raw `hints`), build a `LocationAddress`, and replace with `place.NewLogicalQubit(theta, phi, lam, location_address=LocationAddress(...))`.

### 2.2 — `InitializeNewQubits` (augmented)

Current behavior (`circuit2place.py:71–86`): finds bare `qubit.stmts.New` (not referenced by any `Initialize`) and replaces with `place.NewLogicalQubit(0, 0, 0)`.

**Augmentation:** also handle bare `gemini.operations.NewAt`, producing `place.NewLogicalQubit(0, 0, 0, location_address=LocationAddress(...))`.

### 2.3 — Reading const-prop values

Both 2.1 and 2.2 use Kirin's `AbstractInterpreter.maybe_const` / `expect_const` to read constant values for the SSA args. By the time these rewrites run, eager validation (§3) has already guaranteed the values are present and in range, so failures here can be `assert`s rather than user-facing diagnostics.

### 2.4 — `ResolvePinnedAddresses` (new, lanes-side, always runs)

Runs after `LayoutAnalysis`. Walks every `place.NewLogicalQubit`:

- `location_address is None` → fetch heuristic's pick from the analysis output, stamp it onto the attribute.
- `location_address is not None` → already pinned; leave it alone.

**Post-condition:** every `place.NewLogicalQubit` has a concrete `location_address` after this pass.

Always-on. Even when no `new_at` is used in a kernel, this pass runs and stamps heuristic picks into attributes — that's what enables the `Place→Move` consumers (§4) to read attributes uniformly.

## §3 — Validation

Three failure modes surface eagerly at the gemini level; the fourth stays in the existing post-compile validator. All eager errors are Kirin diagnostics pointing at the user's `operations.new_at(...)` source line.

| # | Failure | Where | When |
|---|---|---|---|
| 1 | SSA arg isn't const-foldable | gemini per-stmt method-table impl (registered against lanes validator key) | eager (gemini IR) |
| 2 | `(z, w, s)` out of ArchSpec range | gemini per-stmt method-table impl | eager (gemini IR) |
| 3 | Duplicate address across qubits | gemini cross-stmt validation pass | eager (gemini IR) |
| 4 | Address forces later-illegal op | existing `validation/address.py` post-compile validator | unchanged |

### 3.1 — Per-statement method-table impl

A method-table impl for `gemini.operations.NewAt` registered against the existing lanes validation interpreter key (using Kirin's idiomatic registration so the existing infrastructure picks it up).

**Checks:**

1. **Const-foldability.** For each of `zone_id`, `word_id`, `site_id`: `expect_const` (or equivalent). Failure → diagnostic *"address argument is not a compile-time constant; explicit allocation requires constant zone/word/site"*.
2. **Range check.** With three constants in hand, build a candidate `LocationAddress` and call `arch_spec.check_location_group([candidate])`. Failure → diagnostic *"address (z, w, s) is not valid for this architecture: <reason>"*.

### 3.2 — Cross-statement duplicate-address validation pass

A pass in `bloqade.gemini` walking all `gemini.operations.NewAt`, building a `dict[LocationAddress, NewAt]` from already-validated constants. Collision → diagnostic on the second occurrence referencing the first: *"address (z, w, s) is pinned by two `operations.new_at` calls"*.

Requires 3.1 to have run first (so const hints are guaranteed populated).

### 3.3 — Eager invocation

Both 3.1 and 3.2 run on the gemini IR before any lowering to the place dialect.

### 3.4 — Failure mode #4 — unchanged

"Pinned to wrong zone for needed gate" / semantic illegality stays in the existing post-compile lanes validator. It's a whole-program property dependent on later analyses; not duplicated here.

## §4 — Place→Move consumer refactor

`InsertInitialize` and `InsertFill` (in `python/bloqade/lanes/rewrite/place2move.py`) currently consume the analysis frame. After §2.4 establishes the post-resolve invariant, they read attributes directly.

Note for implementer: for all rewrite rules. If the rewrite can't happen because of whatever reason
The rewrite rule should simply return with `RewriteResult(has_done_something=False)` intead of throwing
An exception. For examples look at the existing rewrite rules for how to do this in an idomatic way.

### 4.1 — `InsertInitialize` (revised)

- **Drops:** `address_entries: dict[ir.SSAValue, address.Address]`, `initial_layout: tuple[LocationAddress, ...]`.
- **New:** walks the block's `place.NewLogicalQubit` statements in order, reads `stmt.location_address` directly (asserted non-`None` by post-resolve invariant), accumulates into `location_addresses` alongside `theta`/`phi`/`lam`.
- **Output:** the same `move.LogicalInitialize` it emits today.

### 4.2 — `InsertFill` (revised)

- **Drops:** `initial_layout: tuple[LocationAddress, ...]`.
- **New:** walks the function body collecting `place.NewLogicalQubit.location_address` in allocation order.
- **Output:** the same `move.Fill` it emits today.

### 4.3 — `LayoutAnalysis.initial_layout` output

No remaining production consumer in `place2move.py` once §4.1/§4.2 ship. **Drop it from the analysis output.** If it turns out to be useful for debugging or visualization later, it can be resurrected behind a debug flag.

## §5 — `LayoutHeuristicABC` API change

`LayoutHeuristicABC.compute_layout(...)` (in `analysis/layout/analysis.py:12–37`) gains an optional `pinned` parameter so it can respect user-supplied addresses.

### 5.1 — Signature

```python
class LayoutHeuristicABC(abc.ABC):
    @abc.abstractmethod
    def compute_layout(
        self,
        logical_qubit_ids,           # existing
        cz_stages,                   # existing
        pinned: dict[int, LocationAddress] | None = None,   # NEW
    ) -> tuple[LocationAddress, ...]:
        ...
```

- `pinned` maps logical qubit ID → already-pinned `LocationAddress`.
- Returned tuple has every entry concrete: pinned IDs return their pinned address; un-pinned IDs return the heuristic's choice. Internal exclusion set is `set(pinned.values())`.
- Default `None`/empty preserves existing behavior bit-for-bit.

### 5.2 — `LayoutAnalysis` driver

`LayoutAnalysis` gains one new responsibility: scan `place.NewLogicalQubit` ops, collect their `location_address` attributes, build the `pinned` map, and pass it to `compute_layout`.

### 5.3 — Existing heuristic implementations

For each existing `LayoutHeuristicABC` impl:

1. Accept the new `pinned` parameter.
2. Pre-stamp pinned IDs into the output tuple.
3. Run the existing search/heuristic logic restricted to addresses not in `pinned.values()`.
4. Stitch results.

**Behavioral guarantee:** when `pinned` is empty/None, output is byte-identical to today.

### 5.4 — Hard-failure case

If pinning leaves no legal layout for un-pinned qubits, **compilation must fail** with a clear diagnostic from `LayoutAnalysis`. No silent degradation, no "best-effort" fallback. Diagnostic surfaces at the kernel level (not source-line) and names the conflict (e.g. *"layout heuristic cannot place 4 un-pinned qubits given 12 pinned addresses; no legal positions remain in zone 0"*).

The "degraded but legal" case is fine — the user asked for these pins; suboptimal layouts are on them.

## §6 — Testing

### 6.1 — Eager validation (§3)

For each failure mode, write a kernel that exhibits it and assert the diagnostic surfaces at the right source line:

- `new_at(non_const_var, 3, 5)` → const-foldability error.
- `new_at(99, 0, 0)` (out-of-range zone) → range error.
- Two `new_at` calls with identical addresses → duplicate-address error, points at the second occurrence and references the first.
- Two `new_at` calls with constants that evaluate to the same address through different code paths (e.g. `new_at(0, 1+0, 2)` and `new_at(0, 1, 2)`) — should also be caught after const-prop normalizes them.
- Negative cases: valid addresses, distinct addresses → no diagnostics.

### 6.2 — Rewrites (§2)

- **`RewriteLogicalInitializeToNewLogical` (augmented):**
  - Mixed kernel: `Initialize` references both `qubit.stmts.New` and `gemini.operations.new_at`; both become `place.NewLogicalQubit`, only the latter has `location_address` set.
  - All-`new_at` kernel: every qubit becomes a pinned `place.NewLogicalQubit`.
  - Regression: pure `qubit.stmts.New` kernel produces byte-identical output.
- **`InitializeNewQubits` (augmented):**
  - Bare `new_at` becomes `place.NewLogicalQubit(0, 0, 0, location_address=...)`.
  - Regression: bare `qubit.stmts.New` still becomes `place.NewLogicalQubit(0, 0, 0)` with no attribute.
- **`ResolvePinnedAddresses` (new):**
  - Mix of `None` and pinned attributes after `LayoutAnalysis` → every op has a non-`None` attribute; pinned ones unchanged; un-pinned filled from the analysis.
  - All-pinned kernel → pass is a no-op.
  - Zero-pinned kernel → every attribute filled.

### 6.3 — Place→Move refactor (§4)

- **`InsertInitialize` (revised):**
  - Mixed pinned + un-pinned kernel through `ResolvePinnedAddresses` → emitted `move.LogicalInitialize` has correct `location_addresses` and `theta`/`phi`/`lam` ordering.
  - Construction asserts no `address_entries` or `initial_layout` parameters.
  - Assertion-on-invariant: a `None` attribute when this rewrite runs raises (post-resolve invariant is load-bearing).
- **`InsertFill` (revised):**
  - Same setup; emitted `move.Fill` has correct `location_addresses`.
  - Construction asserts no `initial_layout` parameter.
- **Regression:** unchanged python+rust corpus produces byte-identical move IR for kernels with zero `new_at` usage.

### 6.4 — Heuristic API (§5)

- **Backward compat:** for each `LayoutHeuristicABC` impl, `compute_layout(..., pinned={})` (or `pinned=None`) produces byte-identical output to pre-change.
- **Pinning honored:** mixed pinned/un-pinned call asserts pinned indices match exactly; un-pinned indices never collide with pinned addresses; all entries valid for the arch.
- **Hard-failure:** small `ArchSpec` + over-constraining pins → heuristic raises with a diagnostic naming the conflict.
- **`LayoutAnalysis` driver:** mixed-pin kernel → `pinned` dict correctly extracted from attributes and forwarded; over-constraining pins → analysis-level diagnostic.

### 6.5 — Integration

- **Mixed-pinning happy path:** squin kernel mixing `new_at` and `qubit.new`, runs gates and measurements over all of them, end-to-end through `compile_squin_to_move`. Pinned qubits land at requested addresses (visible in `move.Fill` and `move.LogicalInitialize`); resulting IR passes the existing post-compile validator.
- **No-regressions gate (un-pinned only):** existing demo kernels under `demo/` (zero `new_at`) compile to byte-identical move IR before vs after this change. Most important integration test — proves Option B's refactor preserved semantics.
- **End-to-end failure modes:** const-prop failure, overconstraining pins, semantic illegality — each surfaces at the expected stage.
- **New demo:** a script under `demo/` showing real-ish use of explicit allocation (e.g. pinning a small register to known-good addresses for a benchmark circuit). Doubles as documentation.

### 6.6 — Existing test corpus

Full python + rust test suites must pass unchanged. `Place→Move` refactor is the highest-risk piece for accidental drift; the unmodified corpus is the canary.

## Open follow-ups (out of scope, tracked here)

- Partial-address allocations (zone-only, word-only) — needs Kirin support that doesn't exist yet.
- `arch_query` dialect — small dialect exposing arch dimensions for ergonomic loops over sites/words.

## Risk register

- **Highest risk:** `Place→Move` refactor (§4) introducing semantic drift in existing pipelines. Mitigation: byte-identical regression test on the existing demo corpus is the gate.
- **Medium risk:** existing `LayoutHeuristicABC` impls don't cleanly factor a "pre-stamp pinned, restrict heuristic to remaining" structure. Mitigation: each impl is updated independently with its own regression test.
- **Low risk:** const-prop hints not populated when the new rewrites run. Mitigation: validation (§3) gates this; the rewrites can `assert` rather than handle.
