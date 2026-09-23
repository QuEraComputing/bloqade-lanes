# Search-crate refactor — epic breakdown

**Date:** 2026-08-18. **Revised 2026-09-23 (binding-first).**
**Status:** DRAFT / not started. Planning artifact.
**Pairs with:**
- [`specs/2026-08-18-search-trait-redesign-design.md`](../specs/2026-08-18-search-trait-redesign-design.md):
  the target trait design. Now **partially superseded**; see its §0.
- [`specs/2026-08-20-search-redesign-critique.md`](../specs/2026-08-20-search-redesign-critique.md):
  the independent critique, plus the 2026-09-23 re-verification against `main`, with
  current file:line locations.
- [`specs/2026-08-18-search-trait-inventory.md`](../specs/2026-08-18-search-trait-inventory.md):
  current-state evidence as of `b823c308`.

## What changed in this revision

The August plan was organised around the trait redesign: a new `SearchCore` substrate,
a push-time `best_reached` fold, a resumable `TargetSolver` trait with `RouteOutcome`,
a capability-trait split and the placement lift. Three things have since changed it:

1. **The critique deflated the new machinery.** `best_reached` becomes a lazy
   failure-path scan over the `SearchResult.graph` the engines already return;
   `MeasurableGoal` and the dyn tier are dropped; and two of the sketched engine
   signatures do not fit the real drivers.
2. **Binding-first.** Once the new machinery is gone, most of "the API problem" is on the
   Python side: string status labels, dict-shaped stats and struct-literal construction.
   A rebuilt, typed PyO3 layer fixes that. The behavioural payoffs need only targeted
   Rust changes, not the trait overhaul.
3. **The direction moved to placement.** The branch-and-bound driver was abandoned on
   2026-09-07 (PR #1004 closed); the guidance is that better results will come from a
   better placement algorithm. The candidate-ranking measurement (`aae13ab9` on
   `phil/class-completion-bound`) found that the choice of target dwarfs every search
   change measured, and that ranking candidates by a cheap Push-and-Rotate plan captures
   52–94% of the available win. That becomes **Epic 4**.

The placement lift (§7 of the design) leaves the critical path; it only concerns the
loose-goal path, which the candidate-ranking work does not need.

---

## The boundary rule (every epic enforces it)

> **Core types never grow fields, string labels or shapes for Python's benefit.** The
> PyO3 layer owns its own DTOs and all conversion. Core exposes domain facts through
> constructors and accessors; the bindings never build core structs with struct
> literals.

This is what the August trait redesign was trying to enforce structurally. Binding-first
enforces it by convention and review, plus the thin adapter from Epic 3. Recent code
shows the pattern it is meant to stop:
- `BoundStats.bound_enabled` exists only to drive Python's empty-dict behaviour.
- `bound_terminates`, an A/B measurement knob, is threaded through core, PyO3 and the
  Python dataclass.
- PyO3 builds `SearchContext` with a struct literal, so adding `capacity` broke the
  bindings.

## Structuring principle: Phase A, checkpoint, Phase B

Each epic does its **structural** work first. That work is zero-drift, is verified
against the benchmarks and the Epic-1 golden, and lands as its own commit/PR. Only after
that checkpoint does its **behavioural** payoff land, as a separately gated Phase B:
new tests first, then land, then regenerate baselines under review. Phase A and Phase B
are always distinct commits/PRs, with zero drift verified between them. Otherwise the
"any drift = bug" signal is lost.

---

## Epic 0 — Close the benchmark gate gap (quick win)

**Goal.** Put Push-and-Rotate and cascade under the zero-diff CI gate now, before any
code moves. Today they appear in neither baseline and are guarded only by unit tests
(design §9).

**Scope.**
- Add rows to the registry (`python/benchmarks/harness/matrix.py`). At minimum:
  `rust_push_rotate`, `rust_cascade`, and one fallback row (`astar` with
  `fallback_push_rotate=True`).
- Regenerate both baselines. **Only new rows may appear; every existing row must be
  byte-identical** in the deterministic columns.

**Acceptance.** New rows are present and deterministic across two runs; existing rows
show zero diff; `success` is recorded for the new rows. Some large-case failures are
expected and pinned.

**Dependencies:** none. It strengthens Epic 1 but doesn't replace it: the CSV can't see
plan identity, resume semantics or proof provenance.

---

## Epic 1 — Behaviour test net with a decoupled interface layer

Unchanged in substance from the August plan.

**Goal.** A Rust suite that detects *any* behaviour change in `bloqade-lanes-search`,
with a **single point of failure** when the API changes.

**Architecture (three layers, strictly separated):**
1. **Cases: pure data, no crate types.**
   - Each case is `{ name, input: ProblemSpec, expected: Outcome, tags }`, using
     test-domain types.
   - `Outcome` carries `status`, `move_layer_count`, `cost`, `nodes_expanded`,
     `deadlocks`, `final_placement`, a `plan_digest` (a stable hash of the move sequence),
     the proof/termination verdict, and `bound_stats` for bounded runs.
2. **Interface layer: the only importer of the crate API.**
   - `fn run(spec: &ProblemSpec) -> Outcome`. The module doc states: *"API changed? Fix
     THIS module, not the cases. A compile/map failure means re-map here; an assertion
     failure means a behaviour regression."*
   - It binds through the **outer solver surface** (`TargetSolver`, the `CzPlacement`
     implementations), so Epic 2's demotions can't break it.
3. **Runner.** Semantic cases use hand-verified expectations; characterization cases
   compare against goldens.

**Coverage (must-haves):**
- **Strategies:** every one, including push-rotate and cascade.
- **Placement paths:**
  - fixed-target;
  - loose-goal, including the two-leg accidental-CZ cleanup path;
  - `NoHome`;
  - RecedingHorizon;
  - single-heuristic with multiple candidates.
- **Fallback and mirroring:**
  - the fallback, including proof promotion (`fallback.proven`) and the lost search
    counters;
  - mirror success and mirror failure (`backwards_search`).
- **Bounds and edge cases:**
  - bounded vs unbounded, including the root-certificate stop (`bound_terminates`);
  - unsolvable, already-at-goal, blocked destination, malformed target;
  - partial targets: the known push-rotate panic corner (inventory audit F1).
- **Anticipatory goldens**, recording current behaviour, for each later Phase B:
  - the §5 bounded-cascade memory;
  - P&R resume vs restart;
  - **candidate-order dependence of `solve_single_heuristic`**, which returns on the
    first solved candidate, so Epic 4 will change it.

**Where it lives.** `crates/bloqade-lanes-search/tests/`, run by `just test-rust`.

**Acceptance.** The interface layer is a single documented module; cases are data-only;
the coverage above is met; the suite is deterministic in CI; the golden baseline is
captured.

**Dependencies:** none; it can run in parallel with Epic 0. It blocks Epics 2–4.

---

## Epic 2 — Targeted Rust changes (shrunk)

**Goal.** Remove dead and misplaced surface, and land the two behavioural payoffs, with
targeted edits rather than a trait overhaul. The Python-facing PyO3 surface stays
unchanged throughout this epic.

### Phase 2A — structural, zero-drift

- **Dead-code hygiene.** Current caller counts are in critique §5.
  - Delete `MaxHopHeuristic`, `SumHopHeuristic` and `EntropyScorer`, all with zero
    callers.
  - Collapse the `entropy_search` → `_with_objective` → `_with_bound` delegation chain:
    the head is bench-only and each link has one caller.
  - Relocate `tests/public_bound_api.rs` to in-crate access, so `MaxBound`,
    `WeightedDuration`, `as_heuristic` and the chain can be demoted to `pub(crate)` or
    deleted.
  - Decide on the dangling `SearchEngine::exhaustive_preconditions()` and
    `ConfigError::UnsupportedArchitecture`: wire them up or delete them.
- **`proven` becomes a method** derived from `termination`, not a stored copy. The PyO3
  getter keeps its output.
- **Constructors for what PyO3 builds today with struct literals**, starting with
  `SearchContext`. This is groundwork for the boundary rule; PyO3 output is unchanged.
- **Move Python-facing fields out of core, output identical.**
  - `BoundStats.bound_enabled` moves into the adapter.
  - The `entropy_trace` tuple shape is produced by the adapter from domain values.
- **Lazy best-partial helper, additive and unused.**
  - `best_partial(&SearchResult, target) -> Option<NodeId>` returns the node with the
    fewest unresolved atoms over `result.graph`, tie-broken by `(unresolved, g, NodeId)`.
  - It is *not* keyed on `WeightedDistanceBound`, which can be 0 when the goal isn't met.
  - It is only defined for point goals.

**Acceptance (2A).** Benchmarks show zero diff; the Epic-1 golden shows zero drift; the
PyO3 Python-facing surface diff is empty; clippy is clean.

### Phase 2B — behavioural payoffs, each gated separately

1. **P&R resumes from the best partial** inside `solve_with_engine`'s fallback branch
   (`target_solver.rs:328`).
   - Pass `best_partial`'s config as P&R's `initial`, concatenate the layers and replay
     the *whole* chain.
   - Keep the search's counters on the returned result.
   - **The mirror path never resumes.** A failed mirror yields a suffix, not a prefix
     (critique F2), so it keeps today's restart behaviour.
   - The proof policy is an open decision (below).
   - Tests: resume vs restart, and chained-plan replay, *before* landing.
2. **Frontier bound gate, scoped per critique F7.** This is a push-time `g + h ≥ C` prune
   in the cascade refinement, which is the fixed-target memory fix, plus the `h = ∞`
   infeasibility cut.
   - Loose-goal solves are unaffected: set-valued goals are never bounded.
   - Expect `nodes_explored` to shift on cascade, and possibly on astar/ids through the
     infeasibility cut. Regenerate and inspect both baselines; `success` must be
     unchanged.

**Dependencies:** Epics 0 and 1 (2B needs the anticipatory goldens).

---

## Epic 3 — Typed PyO3 adapter + Python migration (the main epic)

**Goal.** Rebuild `crates/bloqade-lanes-bytecode-python/src/search_python.rs` as a thin
typed adapter under the boundary rule, and migrate `python/bloqade/lanes/heuristics/physical/*`.

### Phase 3A — structural, behaviour-preserving

- **A typed status enum replaces the string ABI.**
  - The four Python comparison sites to migrate: `movement.py:454`,
    `move_synthesis.py:47`, `_no_return_base.py:290`, `policy_movement.py:75`.
  - The `as_label` sites in `search_python.rs` go away.
- **Distinct proof outcomes.** "Plan proven optimal" and "proven that no plan exists"
  become separate typed outcomes.
  - This fixes `rust_proven_total` counting both, and the `"exhausted_proof"` docstring
    that calls it "optimal".
  - `MultiSolveResult` exposes proof and termination, which it lacks today.
- **Typed `BoundStats` and `attempts`** instead of `PyDict` / lists of dicts.
- **Options through constructors only.** Decide deliberately whether Python should see:
  - `aod_capacity`, which is core-only today and hard-coded to `None` in PyO3;
  - `bound_terminates`, which is documented as an A/B knob.
- **Typed exceptions** per `ConfigError` variant, instead of `ValueError(to_string())`.
- **Expose the "pick a target, then route" pieces Epic 4 needs:**
  - multi-candidate solve with per-candidate attempts;
  - a Push-and-Rotate solve per candidate;
  - the P&R plan cost.

**Regression tracking.** The Epic-1 suite, with its interface layer re-pointed and
cases/goldens unchanged. Also the Python integration tests, the only automated check on
the string/dict-to-typed change. Benchmarks must show zero diff.

There is no Phase 3B: the placement lift moved to Epic 5.

**Dependencies:** Epics 1 and 2A. It can run alongside 2B.

---

## Epic 4 — Candidate-ranking exploration (placement is the lever)

**Goal.** Find out whether, and how, choosing among candidate target placements with a
cheap upper-bound router improves real compilations. If it does, build it into the
placement layer.

**Starting evidence.** `examples/candidate_ranking.rs` at `aae13ab9` on
`phil/class-completion-bound` (unmerged). It used 8 candidates per start, from
equal-length random walks.
- The mean candidate is 30–45% worse than the best.
- At logical k=16 the mean is more than 2× the best.
- Picking the P&R-cheapest candidate captures 52–94% of the available win.
- P&R costs about 0.6 ms per solve, against about 57 ms for entropy.

**Phases.**
1. **Re-measure with real candidate sets (go/no-go).**
   - The random-walk candidates may overstate the spread. Rerun against the Python
     generators (`CongestionAwareTargetGenerator`, `AODClusterTargetGenerator`,
     `LookaheadCongestionAwareTargetGenerator` in
     `python/bloqade/lanes/heuristics/physical/target_generator.py`).
   - Decide on regret and captured win, not rank correlation.
   - Needs no other epic; it can start now.
2. **Remove the two blockers.**
   - `RustPlacementTraversal.target_generator` defaults to `None`, which makes
     `DefaultTargetGenerator` emit exactly **one** candidate, so the default pipeline has
     nothing to rank.
   - `solve_single_heuristic` returns on the **first** solved candidate
     (`placement/single_heuristic.rs:177`), so candidate order alone decides the output.
3. **A ranking policy in the placement layer.** Route each candidate with P&R (an upper
   bound), pick the cheapest, then route the winner with search. Optional extensions:
   - **Seed incumbent:** use the P&R plan as the search's starting incumbent.
   - **Lower-bound pruning over the candidate list:** discard any candidate whose class
     lower bound (also on `phil/class-completion-bound`) exceeds the best upper bound
     found.

**Behavioural by design.** This moves the `pipeline_default` benchmark row, which is
that row's purpose. Update the candidate-order goldens from Epic 1 under review.

**Dependencies.** Phase 1: none. Phases 2–3: Epic 1, and Epic 3A if the policy is
driven from Python. It follows Epic 2A and runs alongside Epic 3.

---

## Epic 5 — Placement lift for the loose-goal path (optional, deferred)

The design's §7 is softened per critique F6. Pair coordination (`cz_pairs`,
`CzCoordination`, loose-goal target assignment) is lifted out of `HeuristicGenerator`.
Spectator handling (accidental-CZ detection, escape moves) stays unless separately
justified. RecedingHorizon keeps reaching into the search engines directly (critique F5).

**Trigger:** the loose-goal path next needs real work. This epic moves the logical
baseline and carries the highest behaviour risk; budget several regenerate-and-inspect
cycles.

---

## Parked

- **Stable Rust extension surface for a private downstream crate** (discussed
  2026-09-23). A private crate that implements this crate's traits would be the first
  real Rust-level consumer. That would bring back the trait renames and the capability
  split (design §3) as stability work rather than cleanup. Revisit after Epic 2A, once
  the surface has been pruned. Direction discussed so far:
  - a curated stable surface, with everything else behind an `unstable` feature;
  - `#[non_exhaustive]` plus constructors;
  - `cargo-semver-checks`;
  - PyO3 built against the stable surface only.
- **Dropped:** the dyn-dispatch tier (`DynBound`/`ErasedBound`), `MeasurableGoal`, the
  push-time `best_reached` fold, a public `RouteOutcome` / `TargetSolver` trait, and the
  `SearchCore` extraction.

## Ordering

```
Epic 0 (gate rows) ─┐
Epic 1 (test net) ──┴─▶ Epic 2A ─▶ ✔ ─▶ Epic 2B (P&R resume; cascade bound gate)
                            │
                            ├─▶ Epic 3A (typed adapter + Python migration)
                            │
                            └─▶ Epic 4 phases 2–3 (candidate ranking)
Epic 4 phase 1 (re-measure) — any time
Epic 5 (placement lift) — deferred, on demand
```

## Open decisions

1. **P&R-resume proof policy.** A resumed P&R `Unsolvable` is a proof about the partial
   config, not the caller's `initial`. Either follow RecedingHorizon's
   `merge_fallback`, which downgrades the proof after a committed prefix, or argue that it
   carries over (invertible prefix plus an identical blocked set) and assert that
   precondition.
2. **Where candidate ranking lives:** a Rust placement type (a new `CzPlacement`
   implementation, or an option on `SingleHeuristicCzPlacement`) or Python orchestration
   over the Epic-3A surface.
3. **Python exposure of `aod_capacity` and `bound_terminates`** (Epic 3A).
4. **Golden format for Epic 1:** inline expected `Outcome`s or a committed golden file.
5. **Epic 1 capture level:** `SolveResult`, `TargetSolver`, or both.
