# Search-crate refactor — epic breakdown

**Date:** 2026-08-18. **Revised 2026-09-23 (binding-first).**
**Status:** in progress. Epic 0 is done; the rest has not started.
**Branch model:** the refactor lives on `claude/search-crate-refactor`, a long-lived review
branch that is **not merged into `main`**.
- Each epic phase lands as its own PR into that branch, with Phase A and Phase B as
  separate PRs. Each phase therefore gets its own review and its own CI run: `ci.yml` and
  `lint.yml` run on every pull request, whatever its base, including the benchmark gate.
- Because each PR is a single phase, squash-merging it into the branch keeps the
  zero-drift checkpoint between phases.
- Changes on `main` come in by merging `main` into the branch. When `main` moves the
  benchmark baselines, regenerate them on the branch.
**Pairs with:**
- [`specs/2026-08-18-search-trait-redesign-design.md`](../specs/2026-08-18-search-trait-redesign-design.md):
  the target trait design. Now **partially superseded**; see its §0.
- [`specs/2026-08-20-search-redesign-critique.md`](../specs/2026-08-20-search-redesign-critique.md):
  the independent critique, plus the 2026-09-23 re-verification against `main`, with
  current file:line locations.
- [`specs/2026-08-18-search-trait-inventory.md`](../specs/2026-08-18-search-trait-inventory.md):
  current-state evidence as of `b823c308`.

Unless a path is given in full, `file:line` citations are relative to
`crates/bloqade-lanes-search/src/` and refer to `main` at `27db773e`.

## What changed in this revision

The August plan was organised around the trait redesign: a new `SearchCore` substrate,
a push-time `best_reached` fold, a resumable `TargetSolver` trait with `RouteOutcome`,
a capability-trait split and the placement lift. Three things have since changed it:

1. **The critique deflated the new machinery.** The best partial becomes a lazy
   failure-path scan over the search graph; `MeasurableGoal` and the dyn tier are
   dropped; and two of the sketched engine signatures do not fit the real drivers.
2. **Binding-first.** Once the new machinery is gone, most of "the API problem" is on the
   binding side: string status labels, dict-shaped stats and struct-literal construction.
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

This revision was checked by two independent read-only reviews on 2026-09-23. One
fact-checked the claims against the code (about 120 claims); the other reviewed logic and
consistency. Their findings are folded in.

---

## Two rules every epic enforces

### Rule 1 — the Python boundary

> **Core types never grow fields, string labels or shapes for Python's benefit.** The
> PyO3 layer owns its own DTOs and all conversion. Core exposes domain facts through
> constructors and accessors; the bindings never build core structs with struct
> literals.

This is what the August trait redesign was trying to enforce structurally. Binding-first
enforces it by convention and review, plus the thin adapter from Epic 3. Current code
shows the pattern it is meant to stop:
- `EntropyTraceStep` (`drivers/entropy.rs:73-91`) carries the Python visualizer's format
  in core:
  - string `event` and `reason` fields;
  - movesets as `(u8, u8, u32, u32, u32, u32)` tuples;
  - configurations as `(u32, u32, u32, u32)` tuples.
- PyO3 builds `SearchContext` with a struct literal
  (`bloqade-lanes-bytecode-python/src/search_python.rs:723`), so adding `capacity` broke
  the bindings.
- Knobs leak outward too. `bound_terminates`, documented as an A/B measurement knob, is
  threaded through core, PyO3 and the Python dataclass. A knob should reach Python only
  by decision; this one becomes Rust-only in Epic 3A (decision 3).

### Rule 2 — the architecture boundary

> **The search crate reads the architecture only through `LaneIndex`.** `ArchSpec` (from
> `bloqade-lanes-bytecode-core`) appears only where a `LaneIndex` / `SearchEngine` is
> built, and in the PyO3 adapter, which needs it because the Python interface works with
> the `ArchSpec`. New interfaces take `&LaneIndex`, never `&ArchSpec`.

**Why.** The infrastructure should support different compilation stacks. Depending on
`bytecode-core` is acceptable; what matters is that the dependency passes through one
clear interface at the Rust boundary. The address vocabulary (`LocationAddr`,
`LaneAddr`) stays a shared dependency.

**Today.**
- About 22 files mention `ArchSpec`.
- `LaneIndex::arch_spec()` (`primitives/lane_index.rs:287`) hands out the raw spec, and
  about 15 modules reach through it.
- What they actually use is small:
  - move legality: `validate`, `check_lanes`, `check_lane`, `check_lane_group_geometry`,
    mostly in `search/verify.rs`;
  - CZ topology: `get_cz_partner` (11 calls), `left_cz_word_ids`;
  - structure and geometry: `sites_per_word`, `location_position`, `lane_endpoints`,
    `word_zone_map`, `is_home_position`.

Epic 2A closes the leak. The Starlark DSL sidecar (`dsl/`) uses `dsl-core`'s own
`ArchSpec` methods and is out of scope.

## Structuring principle: Phase A, checkpoint, Phase B

Each epic does its **structural** work first. That work is zero-drift, is verified
against the benchmarks and the Epic-1 golden, and lands as its own commit/PR. Only after
that checkpoint does its **behavioural** payoff land, as a separately gated Phase B:
new tests first, then land, then regenerate baselines under review. Phase A and Phase B
are always distinct commits/PRs, with zero drift verified between them. Otherwise the
"any drift = bug" signal is lost.

---

## Epic 0 — Close the benchmark gate gap (quick win) — DONE 2026-09-23

**Landed.**
- **Rows:** `RustPlacementTraversal.fallback_push_rotate`, plus five rows —
  `rust_push_rotate`, `rust_cascade_ids`, `rust_astar_fallback`, `rust_loose_goal` and
  `rust_receding_horizon`. `rust_cascade_ids` names the variant explicitly rather than
  using the `"cascade"` alias; `rust_cascade_entropy` was not added.
- **Baselines:** only rows were added (physical 90 → 135, logical 33 → 48). Every existing
  row is identical in the deterministic columns, and the new rows are identical across two
  independent runs. The rows add about 2.3 min and at most 454 MB to the physical suite,
  so they sit in the default matrix rather than behind a flag.
- **Five new physical failures are pinned**, all "place.CZ statements remain":
  - cascade-ids on `steane_physical_35`;
  - loose-goal and receding-horizon, each on `adder_64` and `trotter_rand_35`.
- **After #1049** (the receding-horizon budget fix), receding-horizon now solves `adder_4`
  (19 events / 24 lanes) and `steane_physical_35`. Its other rows keep identical plans
  and report more `nodes_explored`, because dropped rollouts are now counted. No other
  row moved.
- **The A* fallback row solves the three cases plain A* fails.**

**Goal.** Put Push-and-Rotate, cascade and the loose-goal paths under the zero-diff CI
gate before any search code moves. Today the committed baselines contain only two kinds
of row:
- `pipeline_default`, which is NoHome under palindrome;
- the `rust_*` rows, which are `Palindrome(PhysicalPlacementStrategy)`.

Push-and-Rotate, cascade, loose-goal and RecedingHorizon are in neither baseline and are
guarded only by unit tests.

**Scope.**
- **Registry-only rows** (`python/benchmarks/harness/matrix.py`):
  - `rust_push_rotate` (strategy `"push-rotate"`);
  - `rust_cascade_ids` (`"cascade-ids"`);
  - optionally `rust_cascade_entropy`.
- **A fallback row needs one Python change first.** `RustPlacementTraversal` has no
  `fallback_push_rotate` field, and `_move_search_from_traversal`
  (`python/bloqade/lanes/heuristics/physical/movement.py:119-145`) never passes it.
  Add the field with default `False`, so existing rows are unchanged, then add
  `rust_astar_fallback`.
- **Loose-goal rows.** Add one row each for the existing loose-goal and RecedingHorizon
  Python strategies (`python/bloqade/lanes/heuristics/physical/no_return.py`,
  `receding_horizon.py`). They are Epic 5's only gate.
- **Regenerate both baselines.** Only new rows may appear; every existing row must be
  byte-identical in the deterministic columns.

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
   - Per decision 5, placement cases also record the **attempt log** wherever one exists:
     for each candidate tried, its index, status and expansions, in order, plus which one
     won. Today only `SingleHeuristic` has one (`MultiSolveResult`). Epic 3A.0 gives
     every placement a log through `PlacementResult`, and widens the capture when it
     re-points the interface layer.
2. **Interface layer: the only importer of the crate API.**
   - `fn run(spec: &ProblemSpec) -> Outcome`. The module doc states: *"API changed? Fix
     THIS module, not the cases. A compile/map failure means re-map here; an assertion
     failure means a behaviour regression."*
   - It binds through the **outer solver surface** (`TargetSolver`, the `CzPlacement`
     implementations), so Epic 2's demotions can't break it. Epic 3A.0 re-points it once.
3. **Runner.** Semantic cases use hand-verified expectations; characterization cases
   compare against goldens. Format per decision 4:
   - **Semantic cases:** expectations inline, next to their input.
   - **Characterization goldens:** a committed file under `tests/fixtures/`, following
     the `tests/primer_golden.rs` precedent. It needs a regeneration switch (e.g. an
     environment variable), stable case ordering, and LF pinned in `.gitattributes`,
     since #947 hit a CRLF mismatch on exactly this kind of file.

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
  - **partial targets under push-rotate.** These panic. P&R builds `goal_config` from the
    target list alone (`push_rotate/solver.rs:135-139`), and `validate_target_assignment`
    doesn't reject partial targets. The bug was found by the 2026-08-12 input-API audit,
    which was never committed.
- **Anticipatory goldens**, recording current behaviour, for each later Phase B:
  - **Bounded-cascade memory.** Capture `nodes_expanded` now. Neither `SolveResult` nor
    the CSV counts *generated* nodes; Epic 2A adds `nodes_generated`, and this golden
    extends to it before 2B.
  - **P&R resume vs restart.**
  - **Candidate-order dependence**, in both loops that stop at the first solved
    candidate:
    - the Rust `solve_single_heuristic` (`placement/single_heuristic.rs:177`);
    - the Python `PhysicalPlacementStrategy` loop
      (`python/bloqade/lanes/heuristics/physical/movement.py:430-465`). This is the one
      production uses, and it needs a Python test.
- **Note for Epic 3A.0.** Cases with mismatched `controls`/`targets` lengths become
  unrepresentable after the `CzPlacement` reshape, and are retired there under review.

**Where it lives.** `crates/bloqade-lanes-search/tests/`, run by `just test-rust`. The
Python candidate-order test goes under `python/tests/`.

**Acceptance.** The interface layer is a single documented module; cases are data-only;
the coverage above is met; the suite is deterministic in CI; the golden baseline is
captured.

**Dependencies:** none; it can run in parallel with Epic 0. It blocks Epics 2 and 3 and
Epic 4 phases 2–3.

---

## Epic 2 — Targeted Rust changes (shrunk)

**Goal.** Remove dead and misplaced surface, enforce the architecture boundary, and land
the two behavioural payoffs, with targeted edits rather than a trait overhaul. The
Python-facing PyO3 surface stays unchanged throughout this epic.

### Phase 2A — structural, zero-drift

- **Dead-code hygiene.** Counts are of *production* callers and are in critique §5.
  Unit tests in each item's own module go with it.
  - Delete `MaxHopHeuristic`, `SumHopHeuristic` and `EntropyScorer`, which have no
    production callers.
  - Collapse the `entropy_search` → `_with_objective` → `_with_bound` delegation chain.
    Production skips it and calls `entropy_search_with_tables` directly
    (`search/restarts.rs:307, 320`). The links have test callers, and the head is used by
    `benches/entropy.rs`, which must be updated.
  - Relocate `tests/public_bound_api.rs` to in-crate access, so `MaxBound`,
    `WeightedDuration` and the chain can be demoted to `pub(crate)` or deleted.
    `as_heuristic` is a default method on `CompletionBound`, so it can only be kept or
    removed from the trait; it has no production caller.
  - Decide on the dangling `SearchEngine::exhaustive_preconditions()` and
    `ConfigError::UnsupportedArchitecture`, which have no production caller or producer:
    wire them up or delete them.
- **The architecture boundary (Rule 2).** `LaneIndex` becomes the only architecture
  interface.
  - Replace every `.arch_spec().x()` call with a `LaneIndex` method that delegates, e.g.
    `cz_partner`, `left_cz_word_ids`, `sites_per_word`, `location_position`,
    `lane_endpoints`, `word_zone_map` and `is_home_position`.
  - Replace the legality checks used by `search/verify.rs` with one
    `LaneIndex::check_move_set` that delegates to `bytecode-core`'s execution model.
  - Switch functions that take `&ArchSpec` directly to `&LaneIndex`: `ops/entangling.rs`,
    `LooseTargetGenerator`, RecedingHorizon and `feasibility`.
  - Remove `LaneIndex::arch_spec()`.
  - `ArchSpec` then appears only in `LaneIndex` / `SearchEngine` construction and in the
    PyO3 adapter.
  - **The AOD capacity moves into the architecture model (decision 3).**
    - It becomes an optional `ArchSpec` field, where `None` means unlimited. That is an
      additive `bytecode-core` schema change, and the Python `ArchSpec` wrapper exposes
      the field.
    - `LaneIndex` reads it and exposes an accessor, and the shot assemblers read that
      accessor instead of `SearchContext.capacity`.
    - `SearchContext.capacity` and `SolveOptions.aod_capacity` are removed. Neither is
      reachable from Python today (PyO3 hard-codes both to `None`), so the removal isn't
      Python-visible.
    - Decide whether the exhaustive generator keeps its own cap argument, combined via
      `tighten`, as a per-generator tightening.
    - The default of `None` preserves today's uncapped behaviour, so this is zero-drift.
      Push-and-Rotate still ignores the cap, as documented.
  - Every replacement is a delegating method, so this is caught by the compiler and moves
    no behaviour.
- **`proven` becomes a method** derived from `termination`, not a stored copy. The PyO3
  getter keeps its output.
- **Constructors for what PyO3 builds today with struct literals**, starting with
  `SearchContext` (`search_python.rs:723`). PyO3 output is unchanged.
- **Move Python-shaped trace data out of core (Rule 1), output identical.**
  `EntropyTraceStep`'s fields become domain types:
  - the string `event` and `reason` fields become enums;
  - moveset tuples become `MoveSet`;
  - configuration tuples become `Config`.

  The adapter produces today's format for the visualizer. `BoundStats.bound_enabled`
  **stays**: it records whether a real bound was active, which `optimality_gap()` and the
  cascade stats merge (`search/restarts.rs:406`) read, and the adapter can't reconstruct
  it from the options.
- **Best partial, computed where the graph still exists.** The search graph does not
  survive to the fallback site: `extract` (`search/restarts.rs:37`) consumes each
  restart's and each cascade leg's `SearchResult`, and `SolveResult` has no graph. So:
  - **Where:** inside `extract`, on the non-`Solved` path, and only for point goals
    (`goal.exact_targets()` is `Some`).
  - **What:** pick the node with the fewest unresolved atoms, tie-broken by
    `(unresolved, g, NodeId)`. It is *not* keyed on `WeightedDistanceBound`, which can be
    0 when the goal isn't met.
  - **Carried as:** a new domain field on `SolveResult` holding the reached config plus
    its prefix layers from the root.
  - **Across restarts and cascade legs:** the partial travels with the result that
    `pick_best` keeps, and with the result the cascade merge returns, so the choice is
    deterministic.
  - **Drift:** nothing reads the field until 2B, so this is zero-drift. The scan costs
    time only on failed solves.
- **A generated-node count.** `SolveResult` gains `nodes_generated` (the graph size), so
  the §5 memory effect can be observed through the outer surface. This is additive and
  zero-drift.

**Acceptance (2A).** Benchmarks show zero diff; the Epic-1 golden shows zero drift; the
PyO3 Python-facing surface diff is empty; clippy is clean. Outside test code and
fixtures, `ArchSpec` is named only in `LaneIndex` / `SearchEngine` construction and the
`dsl/` sidecar.

### Phase 2B — behavioural payoffs, each gated separately

1. **P&R resumes from the best partial** in `solve_with_engine`'s fallback branch
   (`search/target_solver.rs:328`), reading 2A's best-partial field.
   - Run P&R from the partial config, then replay the *whole* chain: the search prefix
     followed by P&R's layers.
   - Keep the search's counters on the returned result.
   - **The mirror path never resumes.** A failed mirror yields a suffix, not a prefix
     (critique F2), so it keeps today's restart behaviour.
   - **Proof policy (decision 1):** if the resumed P&R reports `Unsolvable`, rerun P&R
     once from the caller's original `initial`, and report only that run's proof.
   - Tests: resume vs restart, and chained-plan replay, *before* landing.
   - **Its reach is low today.** The default pipeline mirrors (`backwards_search=True`
     under palindrome), and no shipped strategy turns on `fallback_push_rotate`. Only
     opted-in users and the Epic-0 fallback row see the change. Lowest priority in 2B;
     deferrable.
2. **Frontier bound gate, scoped per critique F7.** This is a push-time `g + h ≥ C` prune
   in the cascade refinement, which is the fixed-target memory fix, plus the `h = ∞`
   infeasibility cut.
   - **How it's enabled.** Today a completion bound is built only when the strategy runs
     entropy (plain, or as cascade's inner leg) and `EntropyOptions.completion_bound` is
     set (`search/restarts.rs:231-263`). `WeightedDistanceBound` itself does not need the
     entropy tables. This item widens bound construction to the cascade refinement for
     point goals.
   - **Rollout (decision 7): opt-in, then flip.**
     1. Land it opt-in, with the switch in `SolveOptions` rather than
        `EntropyOptions.completion_bound`, which is an odd home for a knob that affects a
        cascade-ids refinement. This moves no existing row; add a bounded-cascade row
        to measure it.
     2. After measuring memory (`nodes_generated`) and node counts on the Epic-0 cascade
        rows, make it the default in a **separate reviewed change**. That change moves
        the cascade rows, and the baseline diff has to be explained.
   - **What can change besides cost.** The prune never discards a strictly cheaper plan,
     but pruned children never get node IDs. Later IDs shift, so tie-breaks can return a
     different *equal-cost* plan.
   - Measure the memory effect with `nodes_generated` from 2A. Loose-goal solves are
     unaffected, because set-valued goals are never bounded.
   - `success` must be unchanged.

**Dependencies:** Epics 0 and 1 for 2A. 2B needs the anticipatory goldens and 2A.

---

## Epic 3 — `CzPlacement` reshape, typed PyO3 adapter, Python migration

**Goal.** Give placement one coherent trait, rebuild
`crates/bloqade-lanes-bytecode-python/src/search_python.rs` as a thin typed adapter
under both rules, and migrate the Python layer.

### Phase 3A.0 — reshape `CzPlacement` (its own PR, first)

Agreed 2026-09-23. It lands alone, so any drift can be traced to it rather than to the
adapter rebuild.

- **Today:** `solve(initial, controls, targets, blocked, max_expansions) ->
  Result<SolveResult, ConfigError>` is a lowest common denominator.
  - No production code dispatches through `dyn CzPlacement`; the only two `dyn` uses are
    tests (`placement/loose_goal.rs:360`, `placement/single_heuristic.rs:250`).
  - PyO3 calls the trait `solve` statically in all four bindings
    (`search_python.rs:1766, 1894, 1999, 2093`).
  - Python production code calls only the inherent `solve_pairs`
    (`python/bloqade/lanes/heuristics/physical/nohome.py:108`, `receding_horizon.py:192`,
    `no_return.py:101`), and the compiler can't see those calls.
  - Every implementation has a richer entry point of its own. `LooseGoal`, `NoHome` and
    `RecedingHorizon` have `solve_pairs(..., cz_pairs, ..., future_cz_layers)`;
    `SingleHeuristic` has `solve_with_attempts(...) -> MultiSolveResult`.
- **Target shape:**

  ```rust
  pub struct CzStage<'a> {
      pub initial: &'a [(u32, LocationAddr)],
      pub pairs: &'a [(u32, u32)],
      pub blocked: &'a [LocationAddr],
      pub future_layers: &'a [Vec<(u32, u32)>],
  }

  #[non_exhaustive]
  pub struct PlacementBudget {
      pub max_expansions: Option<u32>,
  }

  pub trait CzPlacement {
      fn place(&self, stage: &CzStage<'_>, budget: &PlacementBudget)
          -> Result<PlacementResult, ConfigError>;
  }
  ```

  `PlacementBudget` (decision 6) has a constructor and holds only `max_expansions` for
  now. Each implementation keeps today's scope for it, documented on the implementation:
  - `SingleHeuristic`: shared across candidates;
  - `NoHome`: each solve;
  - `LooseGoal`: each leg;
  - RecedingHorizon: a per-stage cap alongside its own `max_expansions_per_rollout`;
  - inside any solve: per restart.

  Because the struct is `#[non_exhaustive]`, fields such as an evaluation cap or a
  per-call total can be added later without changing the trait.

  `PlacementResult` generalizes `MultiSolveResult`:
  - `result: SolveResult`. Its `goal_config` is the chosen placement on success and the
    root on failure.
  - `chosen: Option<usize>`, `None` for placements that don't enumerate candidates.
  - `attempts`: index, status and expansions, plus an **optional evaluator score**. Epic 4
    fills that slot, so the shape doesn't change twice. Placements that don't enumerate
    candidates leave the list empty.
  - `total_expansions`, summed across every leg, including loose-goal's cleanup leg.
- **P&R evaluations are not budgeted.** Epic 4 records them in the attempt log (the
  evaluator-score slot). Unifying the budget scope, e.g. as a total per `place()` call,
  would change behaviour, so it would be a separately gated change, not part of 3A.0.
- **Steps:**
  1. Reshape the core trait.
  2. In PyO3, keep the existing Python method names (`solve_pairs`,
     `solve_with_attempts`) mapped onto `place()`, so the Python layer is untouched.
  3. Drop the four trait-`solve` bindings, which is a Python-visible removal.
  4. Re-point Epic 1's interface layer.
- **The trait stays coarse**, one call per CZ stage. Generate → evaluate → select stays an
  internal pattern. RecedingHorizon's commit-and-replan loop, the loose-goal set-valued
  goal and NoHome's single Hungarian assignment don't fit one pipeline. If ranking and
  RecedingHorizon both need a shared "score a candidate placement" helper, extract it
  then.
- **`TargetGenerator` keeps its parallel slices for now.** `TargetContext` and
  `validate_candidate` take parallel `controls`/`targets`
  (`placement/target_generator.rs:15-24, 136`). `SingleHeuristic` unzips the pairs to
  feed them, which is zero-drift. Whether to reshape `TargetGenerator` too is open
  decision 8.
- **Drift: none, apart from inputs that can no longer be expressed.** Mismatched
  `controls`/`targets` lengths behave differently today:
  - `LooseGoal` panics via `assert_eq!`;
  - `NoHome` and RecedingHorizon only `debug_assert`, then `zip` truncates silently;
  - `SingleHeuristic` doesn't check.

  Retire those Epic-1 cases as a reviewed golden change.
- **Rule 2:** `CzStage` uses only the address vocabulary and implementations take
  `&LaneIndex`.
- **It is also the natural seam for a placement algorithm from a private downstream
  crate** (see "Parked"). If that seam is stabilized, `CzStage` and `PlacementResult`
  also need `#[non_exhaustive]` plus constructors, as `PlacementBudget` already has.

### Phase 3A — typed adapter + Python migration (structural, behaviour-preserving)

- **A typed status enum replaces the string ABI.**
  - Python comparison sites (paths relative to `python/bloqade/lanes/`):
    - `heuristics/physical/movement.py:454`;
    - `heuristics/physical/_no_return_base.py:290`;
    - `heuristics/move_synthesis.py:47`.
  - `heuristics/physical/policy_movement.py:75` compares PolicyRunner's
    `policy_status`, which comes from a separate label source
    (`bloqade-lanes-bytecode-python/src/policy_runner_python.rs:63`). Decide whether it
    joins this change.
  - The `as_label` sites in `search_python.rs` go away.
- **Distinct proof outcomes.** "Plan proven optimal" and "proven that no plan exists"
  become separate typed outcomes.
  - This fixes `rust_proven_total` counting both, and the `"exhausted_proof"` docstring
    that calls it "optimal".
  - The `PlacementResult` binding exposes proof and termination. Core already carries
    them through `.result`; the PyO3 `MultiSolveResult` class just doesn't expose them
    today.
- **Typed `BoundStats` and attempts** instead of `PyDict` / lists of dicts. This includes
  `python/benchmarks/harness/runner.py`, which reads the `bound_stats` dict keys behind
  the gated CSV columns, and reads `rust_proven_total`.
- **Options through constructors only.** Per decision 3:
  - **`bound_terminates` becomes Rust-only.** Remove it from `RustPlacementTraversal`
    (`python/bloqade/lanes/heuristics/physical/movement.py:80`, passed through at `:141`)
    and from the Python `EntropyOptions`, and update the two tests that set it
    (`python/tests/heuristics/test_physical_placement.py`,
    `python/tests/bytecode/test_zone_bus_search.py`). A/B measurement moves to Rust
    benches and examples. This is a Python-visible removal, so the PR needs a
    breaking-change marker.
  - **`aod_capacity` is not a Python solve option.** Python sees it through the
    `ArchSpec` field that Epic 2A adds.
- **Typed exceptions** per `ConfigError` variant, instead of `ValueError(to_string())`.
- **Accept caller-supplied candidates** at the Rust placement boundary.
  - What already works: `solve_with_attempts` is exposed, and P&R per candidate with its
    cost is reachable through `TargetSolver` + `SearchStrategy.PUSH_ROTATE` (`.cost` is
    the layer count).
  - The gap: `solve_with_attempts` is hard-wired to the Rust `DefaultTargetGenerator`
    (`search_python.rs:1753-1760`), so Python-generated candidates can't be passed in.
- **Retire the old Python method names** kept by 3A.0 once the Python layer calls the new
  surface.

**Regression tracking.** The Epic-1 suite, with its interface layer re-pointed and
cases/goldens unchanged beyond 3A.0's retired cases. Also the Python integration tests,
the only automated check on the string/dict-to-typed change. Benchmarks must show zero
diff.

There is no Phase 3B: the placement lift moved to Epic 5.

**Dependencies:** Epics 1 and 2A. 3A.0 comes before the rest of 3A. It can run alongside
2B.

---

## Epic 4 — Candidate-ranking exploration (placement is the lever)

**Goal.** Find out whether, and how, choosing among candidate target placements with a
cheap upper-bound router improves real compilations. If it does, build it into the
placement layer.

**Starting evidence.** `examples/candidate_ranking.rs` at `aae13ab9` on
`phil/class-completion-bound` (unmerged). It used 8 candidates per start, from
equal-length random walks, and measures regret in operations, where regret is the extra
cost of the cheap router's pick over the best candidate in the group.
- The mean candidate is 30–45% worse than the best.
- At logical k=16 the mean is more than 2× the best.
- Picking the P&R-cheapest candidate captures 52–94% of the available win.
- At k=16, P&R costs about 0.6 ms per solve, against about 57 ms for entropy.

**Where candidates exist today** (paths relative to
`python/bloqade/lanes/heuristics/physical/`):
- **The default pipeline** (`pipeline_default`) is `Palindrome(NoHomePlacementStrategy)`,
  which calls `NoHomeCzPlacement.solve_pairs` (`movement.py:528-594`, `nohome.py:107-114`).
  Under palindrome, NoHome skips its return phase and routes once, to CZ targets chosen by
  a fixed per-pair rule, `resolve_cz_targets` (`placement/nohome.rs:500-530`). The rule
  picks which atom of each pair moves: the control if both are in the same word,
  otherwise the target if it sits on a home site, otherwise the control. So there is **no
  candidate list**, but there is a **natural candidate space**, two choices per pair, that
  the rule collapses to one. NoHome also already ranks candidates off palindrome: its
  return phase routes `1 + top_bus_signatures` candidate layouts and keeps the one with
  the fewest layers.
- **The `rust_*` rows** run `Palindrome(PhysicalPlacementStrategy)`. Its
  `target_generator` defaults to `None`, which yields one candidate, and its Python loop
  stops at the first solve (`movement.py:430-465`).
- **The Rust `SingleHeuristicCzPlacement`** has the same first-solve rule
  (`placement/single_heuristic.rs:177`), but no Python path uses it.

**Phases.**
1. **Re-measure with real candidate sets (go/no-go).** The random-walk candidates may
   overstate the spread. Per decision 2, measure **both** candidate spaces:
   - **NoHome's mover choice:** flips of `resolve_cz_targets`' per-pair rule, the space
     that reaches the default pipeline. This can be measured in Rust, extending
     `examples/candidate_ranking.rs`. With two choices per pair the space is 2^pairs, so
     take a bounded subset (e.g. single-pair flips plus a fixed number of random
     assignments), sized from the instance rather than from the shipped specs.
   - **The Python generators** (`CongestionAwareTargetGenerator`,
     `AODClusterTargetGenerator`, `LookaheadCongestionAwareTargetGenerator` in
     `target_generator.py`). This harness must be Python-driven, because the Rust example
     can't consume Python generators.

   In both, route each candidate through `TargetSolver` twice, with entropy and with
   push-rotate, and compute regret and captured win.
   - Fix the go/no-go threshold before running. Decide on regret and captured win, not
     rank correlation.
   - Needs no other epic; it can start now.
2. **Remove the blockers.**
   - Make `PhysicalPlacementStrategy` produce more than one candidate by default, and stop
     its loop from taking the first solve. `SingleHeuristic` gets the same change.
   - Decide how ranking reaches the default pipeline (decision 2, from phase 1's data):
     give NoHome several CZ-stage candidates, or switch the default placement family.
3. **A ranking policy in the placement layer.** Route each candidate with P&R (an upper
   bound), pick the cheapest, then route the winner with search. Where it lives is
   decision 2, settled from phase 1's data. There are four options:
   - inside `NoHomeCzPlacement`, over the per-pair mover choice, which reaches the default
     pipeline directly;
   - the Python `PhysicalPlacementStrategy` loop, which needs no Rust change;
   - a Rust `CzPlacement` that fills 3A.0's evaluator-score slot and is fed through 3A's
     caller-supplied candidates;
   - generators ported to Rust.

   Optional extensions, each of which needs machinery not on `main`:
   - **Seed incumbent:** use the P&R plan as the search's starting incumbent. There is no
     API today for seeding a search with an outside incumbent; the B&B driver's was
     dropped with PR #1004.
   - **Lower-bound pruning over the candidate list:** discard any candidate whose class
     lower bound exceeds the best upper bound found. The class bound has to be merged from
     `phil/class-completion-bound` first.

**Behavioural by design.** It moves the rows of whichever family gains ranking: the
`rust_*` rows for `PhysicalPlacementStrategy`, and `pipeline_default` only if NoHome or
the default family changes. Update the candidate-order goldens from Epic 1 under review.

**Dependencies.**
- Phase 1: none.
- Phase 2: Epic 1.
- Phase 3: Epic 1, plus Epic 3A.0 for a Rust implementation, plus 3A's caller-supplied
  candidates if Python generators feed a Rust implementation.

---

## Epic 5 — Placement lift for the loose-goal path (optional, deferred)

The design's §7 is softened per critique F6. Pair coordination (`cz_pairs`,
`CzCoordination`, loose-goal target assignment) is lifted out of `HeuristicGenerator`.
Spectator handling (accidental-CZ detection, escape moves) stays unless separately
justified. RecedingHorizon keeps reaching into the search engines directly (critique F5).

**Trigger:** the loose-goal path next needs real work.

**Dependencies:** Epic 0's loose-goal rows (no committed baseline covers this path today)
and Epic 1's loose-goal cases.

**Acceptance.** Loose-goal cases are green. Only the anticipated goldens change, and they
are reviewed. The Epic-0 loose-goal rows are regenerated with `success` unchanged. This
carries the highest behaviour risk; budget several regenerate-and-inspect cycles.

---

## Parked

- **Stable Rust extension surface for a private downstream crate** (discussed
  2026-09-23). A private crate that implements this crate's traits would be the first
  real Rust-level consumer. That would bring back the trait renames and the capability
  split (design §3) as stability work rather than cleanup. Revisit after Epic 2A, once
  the surface has been pruned; `CzPlacement` (3A.0) is the most likely seam. Direction
  discussed so far:
  - a curated stable surface, with everything else behind an `unstable` feature;
  - `#[non_exhaustive]` plus constructors;
  - `cargo-semver-checks`;
  - PyO3 built against the stable surface only.

  The "no production callers" counts behind 2A's deletions were measured inside this
  workspace only. If private downstream code already exists, check it before 2A deletes
  or demotes anything.
- **Dropped:** the dyn-dispatch tier (`DynBound`/`ErasedBound`), `MeasurableGoal`, the
  push-time `best_reached` fold, a public `RouteOutcome` / `TargetSolver` trait, and the
  `SearchCore` extraction.

## Ordering

```
Epic 0 (gate rows) ─┐
Epic 1 (test net) ──┴─▶ Epic 2A ─┬─▶ Epic 2B (cascade bound gate; P&R resume)
                                 │
                                 └─▶ Epic 3A.0 (CzPlacement reshape) ─▶ Epic 3A (typed adapter)
                                            │                               │
                                            └───────────────┬───────────────┘
                                                            ▼
Epic 1 ─▶ Epic 4 phase 2 (blockers) ─────────────▶ Epic 4 phase 3 (ranking)
Epic 4 phase 1 (re-measure) — any time
Epic 5 (placement lift) — deferred; needs Epic 0's loose-goal rows
```

Epic 4 phase 3 needs 3A.0 only for a Rust implementation, and 3A only if Python generators
feed it. A Python-only ranking loop needs just phase 2.

## Decisions

Numbers are stable, because the epics refer to them.

1. **DECIDED (2026-09-23): P&R-resume proof policy.** A resumed P&R `Unsolvable` is a
   proof about the partial config, not the caller's `initial`.
   - **Chosen:** on a resumed `Unsolvable`, rerun P&R once from the caller's original
     `initial` and report only that run's proof. The reported proof is then always about
     the caller's instance, with no argument about whether moves can be inverted (which
     isn't guaranteed for an arbitrary architecture spec).
   - **Cost:** one extra P&R run, only on the path where both the search and the resumed
     P&R have failed.
   - **Rejected:** downgrading the proof, as RecedingHorizon's `merge_fallback`
     (`placement/receding_horizon.rs:1092`) does, which loses the signal; and carrying it
     over on an invertibility argument.
2. **PARTLY DECIDED (2026-09-23): candidate ranking — where it lives and how it reaches
   the default pipeline.**
   - **Decided:** Epic 4 phase 1 measures both candidate spaces:
     - NoHome's per-pair mover choice, ranked in Rust inside `NoHomeCzPlacement`, which
       reaches the default directly;
     - the Python generators, ranked in `PhysicalPlacementStrategy`'s Python loop, where
       reaching the default means switching the placement family.
   - **Still open, decided from phase 1's data:** the home (the Python loop, a Rust
     `CzPlacement` fed with caller-supplied candidates, ranking inside NoHome, or
     generators ported to Rust) and the route to the default pipeline.
3. **DECIDED (2026-09-23): Python exposure of `aod_capacity` and `bound_terminates`.**
   - **`aod_capacity` moves into the architecture model** (Epic 2A). It is a hardware
     property, and under Rule 2 architecture facts reach the search through `LaneIndex`.
     Python sees it through the `ArchSpec`, so no solve knob is added and none needs
     deprecating later.
   - **`bound_terminates` becomes Rust-only** (Epic 3A). It is an A/B measurement knob,
     not a user setting.
4. **DECIDED (2026-09-23): golden format for Epic 1: both, split by kind.**
   - **Semantic cases** keep hand-verified expectations inline, where a reviewer reads
     them next to the input.
   - **Characterization goldens** go in a committed file under `tests/fixtures/`, with a
     regeneration switch, stable ordering and LF pinned. Epics 2B, 3A.0 and 4 each
     deliberately regenerate a subset.
5. **DECIDED (2026-09-23): Epic 1 capture level: `SolveResult` facts plus the attempt
   log.** Every case records `SolveResult`-level facts; placement cases also record the
   attempt log wherever one exists (`SingleHeuristic` now, every placement after 3A.0).
   Epic 4 exists to change which candidate wins, and only the attempt log shows why.
   (The August question about a resumable `TargetSolver` level is moot now that it's
   dropped.)
6. **DECIDED (2026-09-23): the placement budget is a small `#[non_exhaustive]`
   `PlacementBudget` struct** (Epic 3A.0).
   - It holds only `max_expansions: Option<u32>` for now.
   - Each implementation keeps today's scope, which differs: shared across candidates,
     per solve, per leg, or per stage, and per restart inside any solve. That keeps 3A.0
     zero-drift.
   - P&R evaluations are recorded in the attempt log but not budgeted.
   - New budget dimensions become new fields, not a trait change.
   - **Rejected:** a plain `Option<u32>`, which can't evolve without a signature change;
     and unifying the scope now, which is a behaviour change and belongs in a gated
     phase.
7. **DECIDED (2026-09-23): the frontier bound gate is opt-in first, then flipped to
   default.** It lands opt-in, switched from `SolveOptions`, and moves no existing row.
   After measurement it becomes the default in a separate reviewed change that moves the
   cascade rows. The prune is sound but can change which equal-cost plan wins a tie.
8. **OPEN (deferred 2026-09-23): `TargetGenerator`'s parallel slices.** Keep them, with
   `SingleHeuristic` unzipping pairs (zero-drift), or reshape `TargetGenerator` /
   `TargetContext` to pairs to match `CzStage`. Facts for whoever decides:
   - the only Rust implementation is `DefaultTargetGenerator`
     (`placement/target_generator.rs:48`);
   - the Starlark target-generator DSL exposes `ctx.controls` / `ctx.targets` as separate
     lists (`dsl/target_generator_dsl/ctx_handle.rs:74-75`), and that surface is used by
     `policies/reference/default_target.star`, `policies/autotune/candidate.star` and
     `policies/primer.md`;
   - the Python `TargetGeneratorABC` is a separate protocol.

   So the Rust side can be reshaped without breaking any `.star` policy, as long as the
   DSL handle keeps unzipping into `controls`/`targets`; `ctx.pairs` could be added
   alongside. Natural time to decide: when Epic 4 touches `TargetGenerator`.
