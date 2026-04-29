# Place-Stage Gate Fusion — Design

**Status:** Draft
**Date:** 2026-04-28
**Author:** brainstormed with Phillip Weinberg
**Branch:** `weinbe58/plan/greedy-circuit-opt`

## Goal

Within the body of a `place.StaticPlacement` statement, fuse runs of textually-adjacent quantum statements that have the same opcode, identical non-qubit SSA arguments, and pairwise-disjoint qubit sets into a single statement covering the union of their qubits. The pass is a place-dialect → place-dialect rewrite.

Targeted opcodes: `place.R`, `place.Rz`, `place.CZ`.

The immediate win is fewer pulses on the hardware (each fused statement collapses to one global pulse instead of N). The downstream win is that wider statements give the placement strategy more freedom to co-locate atoms across what were previously distinct gate layers, reducing rearrangement moves.

## Non-goals

- **Cross-opcode reordering / commutation-aware fusion.** A separate "canonicalizing" pass that pulls commuting same-op statements adjacent is a follow-up. This pass does strict-adjacency only; the canonicalizer's job is to set it up.
- **Fusion of `Initialize` or `EndMeasure`.** Excluded — `Initialize` is typically already a single statement, and `EndMeasure` has per-qubit result-rewiring complexity that warrants its own design.
- **Pattern B (CZ pair bundling across heterogeneous CZ statements with shared qubits).** Out of scope; the disjoint-qubits invariant rules it out by construction.
- **Constant-folding / structural equality** for the parameter-equivalence check. We rely on SSA-identity; users get more fusion opportunities by running CSE / DCE upstream.
- **Pipeline integration.** The pass ships as a standalone module not wired into any existing rewrite pipeline. Tests invoke it directly. Wiring it into `compile_squin_to_*` orchestration is a follow-up.

## Architecture overview

```
StaticPlacement body block
    ┌─────────────────────────────────────────────────┐
    │  Initialize(...)                                 │
    │  R(state₀, axis=%a, angle=%φ, qubits=[0])        │ ─┐
    │  R(state₁, axis=%a, angle=%φ, qubits=[2,3])      │  ├─ fusable run (3-way)
    │  R(state₂, axis=%a, angle=%φ, qubits=[5])        │ ─┘
    │  Rz(state₃, angle=%θ, qubits=[1])                │
    │  CZ(state₄, qubits=[0,1])                        │ ─┐
    │  CZ(state₅, qubits=[2,3])                        │  ├─ fusable run (2-way)
    │  R(state₆, axis=%a, angle=%φ, qubits=[1])        │ ─── singleton (axis matches but blocked by CZ)
    │  EndMeasure(state₇, qubits=[0,1,2,3,4,5])        │
    │  Yield(state₈, ...)                              │
    └─────────────────────────────────────────────────┘

After FuseAdjacentGates:
    ┌─────────────────────────────────────────────────┐
    │  Initialize(...)                                 │
    │  R(state₀, axis=%a, angle=%φ, qubits=[0,2,3,5])  │   ← fused
    │  Rz(state₃', angle=%θ, qubits=[1])               │
    │  CZ(state₄', qubits=[0,2,1,3])                   │   ← fused, controls-then-targets
    │  R(state₆', axis=%a, angle=%φ, qubits=[1])       │
    │  EndMeasure(state₇', qubits=[...])               │
    │  Yield(state₈', ...)                             │
    └─────────────────────────────────────────────────┘
```

## Components

### 1. Module location

New file: `python/bloqade/lanes/rewrite/fuse_gates.py`. Sibling to `transversal.py`, `resolve_pinned.py`, etc. Not re-exported from `python/bloqade/lanes/rewrite/__init__.py` for now — direct import only.

### 2. Rule shape

A single `kirin.rewrite.abc.RewriteRule` matching on `place.StaticPlacement`:

```python
@dataclass
class FuseAdjacentGates(rewrite_abc.RewriteRule):
    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not isinstance(node, place.StaticPlacement):
            return rewrite_abc.RewriteResult()
        body_block = node.body.blocks[0]   # StaticPlacement.check() guarantees single block
        changed = self._fuse_block(body_block)
        return rewrite_abc.RewriteResult(has_done_something=changed)
```

The body-block scan inside `_fuse_block` is a single linear left-to-right pass that handles N-way fusion in one go. Callers wrap the rule with `Walk(Fixpoint(FuseAdjacentGates()))` (per the Kirin idiom used elsewhere in the repo); fixpoint converges on the second iteration when no group changes.

### 3. Fusion predicate

A statement `S` is fusable into a current group `G = [S₀, …, Sₖ]` iff **all four** hold:

1. **Same opcode**: `type(S) is type(S₀)` and `type(S₀) ∈ {place.R, place.Rz, place.CZ}`.
2. **Identical non-qubit SSA arguments** (SSA-identity, no constant folding):
   - `R`: `S.axis_angle is S₀.axis_angle` and `S.rotation_angle is S₀.rotation_angle`
   - `Rz`: `S.rotation_angle is S₀.rotation_angle`
   - `CZ`: trivially satisfied (no non-qubit args)
3. **State-chain adjacency**: `S.state_before is Sₖ.state_after`. Stronger than textual adjacency — guarantees no other quantum statement sits between `Sₖ` and `S` in the data-flow chain.
4. **Disjoint qubits**: `set(S.qubits)` is disjoint from `⋃ᵢ set(Sᵢ.qubits)`.

If any check fails, the current group is *flushed* (fused if `|G| ≥ 2`, else left alone), then a new group is started with `S` if `S` is itself an `R`/`Rz`/`CZ`. Otherwise no group is started and we move on.

**Statements that interrupt grouping:** `Initialize`, `EndMeasure`, `Yield`, or any other place-dialect statement. They flush the current group and do not start a new one.

**End-of-block flush:** at the end of the block, flush whatever group is open.

### 4. Merge construction & SSA rewiring

For a group `G = [S₀, …, Sₖ]` with `k ≥ 1`:

**Build the merged statement.** Same opcode as `S₀`; non-qubit SSA args copied from `S₀`; `state_before = S₀.state_before`. The `qubits` attribute differs by opcode:

- **`R` / `Rz`**: `merged.qubits = S₀.qubits + S₁.qubits + … + Sₖ.qubits` (flat concat in iteration order).
- **`CZ`**: re-interleave to preserve the controls-then-targets convention enforced by `place.CZ.controls` / `.targets` (which split `qubits` in half):

  ```
  merged.qubits = (S₀.controls + S₁.controls + … + Sₖ.controls)
                + (S₀.targets + S₁.targets + … + Sₖ.targets)
  ```

  This guarantees `merged.controls == ⋃ Sᵢ.controls` and `merged.targets == ⋃ Sᵢ.targets`.

**SSA rewiring.**

1. Insert `merged` into the block before `S₀` (or after `Sₖ` — order doesn't matter as long as it's in the block before deletions).
2. Replace all uses of `Sₖ.state_after` with `merged.state_after`. The only consumer of any *intermediate* `Sᵢ.state_after` (`0 ≤ i < k`) is `Sᵢ₊₁.state_before` — those consumers vanish when we delete the inner statements.
3. Delete `S₀, S₁, …, Sₖ` from the block.

In practice, calling `Sₖ.replace_by(merged)` (per the existing pattern at `python/bloqade/lanes/rewrite/transversal.py:27`) handles steps 1–2 in one shot; we then call `.delete()` on `S₀ … Sₖ₋₁`.

**Singleton groups (`|G| == 1`)** are no-ops — leave the original statement, do not increment `has_done_something`.

## Testing strategy

**Location:** `python/tests/rewrite/test_fuse_gates.py` (mirrors source layout). Run with `uv run pytest python/tests/rewrite/test_fuse_gates.py`.

**Construction approach:** synthesize `StaticPlacement` IR directly using `kirin.ir` primitives (Block + Statement constructors) and a small `_make_static_placement(body_stmts)` helper that threads the `state_before`/`state_after` SSA chain. Avoids coupling tests to upstream lowering bugs in `circuit2place`.

**Test matrix (our logic only — Kirin framework plumbing skipped per project policy):**

| Case | Asserts |
|---|---|
| Two adjacent `R` with disjoint qubits, same SSA params | merged into one `R` with concatenated qubits; `state_before` from first, `state_after` rewired to all post-group consumers |
| Two adjacent `Rz` with disjoint qubits, same `rotation_angle` | merged |
| Two adjacent `CZ` with disjoint qubit sets | merged with `controls` re-interleaved (`controls == c₀+c₁`, `targets == t₀+t₁`); explicit assertion on `.controls`, `.targets`, raw `.qubits` |
| Three+ adjacent fusable statements | collapse to one in a single rewrite pass |
| Overlapping qubits | left untouched (e.g. `R([0,1])` followed by `R([1,2])`) |
| Different SSA value for same constant | left untouched (two separate `constant.float` statements with equal payload produce different SSA values) |
| Different non-qubit SSA arg | left untouched (two `R` with different `rotation_angle` SSA values) |
| Different opcode adjacent | strict-adjacency invariant — `R; Rz; R` does not fuse the two `R`s even though qubits are disjoint |
| Boundary statements | `R; R` between `Initialize` and `EndMeasure` still fuses; `Initialize` and `EndMeasure` flush the group and do not start a new one |
| Empty body / single statement / no fusable groups | `has_done_something == False`, IR unchanged |
| Idempotence | applying the rule a second time returns `has_done_something == False` and produces no further changes |

**Explicitly out of scope for tests:**
- That `Walk + Fixpoint` invokes the rule (Kirin framework concern).
- That the place-dialect statement constructors work (dialect concern).
- End-to-end pipeline tests through `compile_squin_to_*` — pass is not yet wired in.

## Risks & follow-ups

- **Canonicalization pass** (out of scope here): a separate pass that reorders commuting same-op statements to be textually adjacent so this pass can fuse them. Until that lands, the win is bounded by what the lowering already produces in adjacent positions.
- **`Initialize` / `EndMeasure` fusion**: deferred. `Initialize` rarely has fusion opportunities; `EndMeasure` needs a separate design for the per-qubit result-rewiring.
- **Pipeline integration**: deferred. Once exercised in tests, a follow-up issue should wire the pass into the appropriate position post-`circuit2place` and gate it on whatever option flag is conventional.
- **CSE / DCE dependency**: with SSA-identity comparison, the pass catches nothing if every statement carries a fresh `constant.float` for the same numeric parameter. Document this in the eventual user-facing release notes; recommend CSE upstream.
- **Move-stage interaction**: a wider `CZ` statement still passes through `place2move`'s placement strategy unchanged. If the strategy chokes on larger control/target tuples (e.g. exceeds a per-zone capacity), that's a strategy-layer issue, not a fusion-layer issue. Flag for verification when the pass is wired in.

## References

- `python/bloqade/lanes/dialects/place.py` — `R`, `Rz`, `CZ`, `StaticPlacement`, `Initialize`, `EndMeasure`, `Yield`.
- `python/bloqade/lanes/rewrite/transversal.py` — reference for `RewriteRule` idiom, `replace_by` usage.
- `python/bloqade/lanes/analysis/placement/strategy.py` — downstream consumer; sees fused statements in `cz_placements` / `sq_placements`.
