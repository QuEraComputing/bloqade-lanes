# Search-crate refactor — epic breakdown

**Date:** 2026-08-18
**Status:** DRAFT / not started. Planning artifact.
**Pairs with:**
[`specs/2026-08-18-search-trait-redesign-design.md`](../specs/2026-08-18-search-trait-redesign-design.md)
(target design + sequencing §10 + safety-net gaps §9) and
[`specs/2026-08-18-search-trait-inventory.md`](../specs/2026-08-18-search-trait-inventory.md)
(current-state evidence). Read those for the *what*; this is the *how, in what order*.

Overall size (independent review): **L–XL, ~8–16 person-weeks, bimodal** — a cheap,
compiler-guarded core and an expensive, weakly-guarded behavioural tail. These
epics sequence that so the safe work lands first behind a test net, and the risky
work lands last with the net in place.

**Structuring principle — two phases per epic, not a separate behavioural epic.**
Each epic does its **structural** work first (interface reshape, *zero behaviour
drift*, verified against benchmarks + the Epic-1 golden and landed as its own
commit/PR), then — only after that checkpoint is green — the **behavioural payoff**
the new structure enables, as a *separately-gated* sub-phase (new tests first, then
land, then regenerate baselines under review). This folds the behavioural work into
the epic that provides its enabling interface, while preserving the one thing that
justified isolating it: the **zero-drift diagnostic** survives as a checkpoint
*inside* each epic rather than a wall between epics. The discipline that keeps it
honest: **Phase A and Phase B are distinct commits/PRs with the zero-drift
verification between them** — never blur them, or "any drift = bug" is lost.

---

## Epic 1 — Behaviour test net with a decoupled interface layer

**Goal.** Before any code moves, build a Rust suite that detects *any* behaviour
change in `bloqade-lanes-search`, structured so it survives the API churn of
Epics 2–3 with a **single point of failure** for diagnosis.

**Why first.** It is the safety net Epics 2–3 lean on, and it fills the gap the
review flagged: the benchmark zero-diff gate covers only the frontier + entropy
strategies — **push-and-rotate and cascade are in neither baseline** (design §9),
so today they ride on ~15 unit tests. Epic 1 closes that.

**Architecture (three layers, strictly separated):**

1. **Cases — pure data, no crate types.** Each case = `{ name, input: ProblemSpec,
   expected: Outcome, tags }`, where `ProblemSpec` / `Outcome` are **test-domain**
   types defined in the test crate, not `bloqade_lanes_search` types.
   - `ProblemSpec`: arch fixture, initial placement, target / CZ-pairs, blocked,
     strategy + option knobs, budget, seed.
   - `Outcome` (the behaviour signal): `status`, `move_layer_count`, `cost`,
     `nodes_expanded`, `deadlocks`, `final_placement`, a **`plan_digest`** (stable
     hash of the move sequence, so plan changes are caught, not just metrics), and
     `bound_stats` for bounded runs.
2. **Interface layer — the single point of failure.** One module, the *only* place
   that imports the crate API: `fn run(spec: &ProblemSpec) -> Outcome`. It builds
   the crate types (`SearchEngine`, `MoveSearch`, `SolveOptions`, goals, …), calls
   the API, and maps the result into `Outcome`. **Module doc states the contract:**
   *"This module is the sole binding between the behaviour cases and the
   bloqade-lanes-search API. If a crate-level interface changes, fix THIS module —
   do not edit case data. A failing compile/map = an API change to re-map here; a
   failing assertion = a behaviour regression to investigate in the crate change."*
3. **Runner.** Iterates cases, calls the interface, asserts `Outcome == expected`
   (semantic cases) or golden-compares (characterization cases).

**Two kinds of case:**
- **Semantic** — hand-verified expected outcomes (e.g. "1 atom site 0→5 ⇒ solved,
  1 move, cost 1.0"). Robust to internal change; assert correctness.
- **Characterization / golden** — record current outputs (esp. `plan_digest`,
  `nodes_expanded`) to trip on *any* drift. The fine-grained refactor tripwire.

**Coverage (must-haves, driven by the §9 gaps):**
- Every strategy incl. **push-rotate and cascade** (the un-gated gap), plus
  astar / bfs / dfs / ids / greedy / entropy{1,5,10,20} / entropy-bounded.
- Fixed-target (`TargetSolver`), loose-goal (`LooseGoal` / `NoHome` /
  `RecedingHorizon`), single-heuristic multi-candidate.
- Fallback / resume path (`fallback_push_rotate`) and mirroring
  (`backwards_search`).
- Bounded vs unbounded; edge cases (unsolvable, already-at-goal, blocked dest,
  malformed target).
- Deterministic (seed-fixed) so drift = real change.
- **Anticipate the Phase-B behavioural changes.** Author the cases that will
  characterize the later behavioural payoffs — bounded-frontier pruning (§5),
  P&R-resume vs restart (§6), the loose-goal placement path (§7) — up front,
  capturing *current* behaviour as their golden. Each behavioural sub-phase then
  updates *only those specific goldens* under review, so the change is visible as a
  small, intentional golden diff rather than hiding among refactor noise.

**Where it lives.** `crates/bloqade-lanes-search/tests/` (Rust integration), run by
`just test-rust`. Complements the Python benchmark harness: Rust-level, per-case,
covers *all* strategies (incl. un-gated), and captures the actual plan — a finer,
faster signal than the 6-column CSV.

**Deliverables:** the three-layer harness; the case corpus above; recorded golden
baseline for characterization cases; the interface-layer contract doc.

**Acceptance:** interface layer is a single documented module; cases are data-only;
coverage includes P&R + cascade + loose-goal + fallback + mirror + bounded; runs
deterministically in CI; golden baseline captured for Epics 2–3 to diff against.

**Dependencies:** none. Blocks Epics 2 and 3.

---

## Epic 2 — Internal refactor, Python-boundary API preserved

**Goal.** Reshape the crate internals (and its *Rust* public surface) to the target
design **without changing the Python-facing PyO3 surface or any behaviour.** The
PyO3 adapter absorbs Rust-side renames so Python sees nothing.

**Scope (design §10 steps 1–5, + 9):** all *behaviour-preserving* work.
- **Step 1 — dead-code + re-export hygiene.** Delete `MaxHopHeuristic` /
  `SumHopHeuristic`; relocate `tests/public_bound_api.rs` to in-crate access so
  `run_search` / `entropy_search_*` / `MaxBound` / `WeightedDuration` can be
  demoted. (Rust-API change only; not Python-facing.)
- **Step 2 — trait renames + capability split**, keeping the **runtime**
  `match goal.exact_targets()` bound gate (defer the compile-time `PointGoal`
  enforcement — see design §3 caveat). Add `MeasurableGoal` impls (unused yet).
- **Step 3 — `SearchCore` extraction + scope `Frontier`** as pure code-motion.
- **Step 4 — `best_reached` as an additive, opt-in `SearchResult` field.**
- **Step 5 — `RouteOutcome` / resumable `TargetSolver` as an *internal* type**:
  `route()` returns best-partial, but `extract()` still maps back to today's
  `SolveResult` and the PyO3 surface is unchanged (isolates the DTO reshape from
  the ABI — the reshape *ships* in Epic 3).
- **Step 9 (optional) — two-tier `DynBound`.** Consumer-less; may be dropped from
  scope.

The above is **Phase 2A — structural, zero-drift.** Regression tracking: benchmarks
(**zero diff**) **and** the Epic-1 suite (**zero golden drift**). Because every step
is a pure refactor, *any* drift in either is a refactor bug, not an expected change.
The Epic-1 interface layer stays pointed at the current API shape (its `Outcome`
mapping barely changes since `SolveResult` is preserved). **Land 2A as its own
commit/PR and verify the checkpoint before starting 2B.**

**Phase 2B — behavioural payoff (the parts that move outputs), gated separately:**
- **§5 — wire `CompletionBound` into the frontier** (push-time `g + h` prune) — the
  memory/perf win the `SearchCore` + bound plumbing from 2A enables. **Moves gated
  baselines** (`nodes_explored`, possibly which optimal-cost plan is returned on
  astar/ids/cascade) → new bound-behaviour tests, then regenerate + inspect both
  baselines, confirm `success` unchanged.
- **§6 — P&R-resume behaviour** (fallback resumes from `best_reached` instead of
  restarting; later top-k / race). Enabled by 2A's `best_reached` + internal
  `RouteOutcome`. **Un-gated by benchmarks** (P&R/cascade absent) → relies entirely
  on the Epic-1 P&R + resume cases; write/settle those first.

**Acceptance.** *2A:* `cargo build` green; PyO3 Python-facing surface diff = none;
benchmarks zero-diff; Epic-1 golden zero-drift. *2B:* new behavioural tests green;
only the anticipated §5/§6 goldens change, reviewed; baselines regenerated with
`success` unchanged; PyO3 surface still unchanged.

**Dependencies:** Epic 1 (2B additionally needs the §5/§6 cases authored in Epic 1).

---

## Epic 3 — Public API refactor + Python-binding migration

**Goal.** Change the **Python-facing** interfaces to the target shape and migrate
the bindings + Python layer.

**Scope:**
- Ship the `SolveResult` → `RouteOutcome` reshape *at the boundary*; replace the
  transport-only coupling (the `status.as_label()` string ABI, the `bound_stats` /
  `attempts` `PyDict` shapes) with the new surface, or preserve equivalents
  deliberately.
- Expose the new `TargetSolver` (resumable) / placement (`StagePlacement`) surface
  to Python; migrate `crates/bloqade-lanes-bytecode-python/src/*` and
  `python/bloqade/lanes/heuristics/physical/*`.
- Any remaining public-API changes that actually reach Python.

**This is where the Epic-1 decoupling pays off.** When the crate API changes, you
update **only the Epic-1 interface layer** to the new API; the case data and
expected `Outcome`s stay fixed. If the golden outcomes still match, behaviour was
preserved *across* the API change — a single, well-lit migration point, exactly as
designed.

The scope above is **Phase 3A — structural, behaviour-preserving** (API/ABI reshape
+ Python migration). Regression tracking: the Epic-1 suite (interface layer
re-pointed, cases and goldens unchanged) + Python integration tests (the ABI is
**not** Rust-gated, so this is the only automated check on the string/dict → typed
reshape) + benchmarks (zero-diff). Land and verify it before 3B.

**Phase 3B — behavioural payoff, gated separately:**
- **§7 — placement lift** (lift `cz_pairs` / `CzCoordination` and loose-goal target
  assignment *up* out of the routing generators, so the generator is goal-agnostic
  and `SearchContext.cz_pairs` disappears). This reshapes the generator/placement
  interface **and** changes candidate scoring/selection on the loose-goal path, so
  it **moves the logical baseline** — the highest behaviour risk in the whole
  refactor. Do it last, on top of the stable 3A interfaces; new/updated loose-goal
  cases first; budget several baseline regen/inspect cycles.

**Acceptance.** *3A:* Python layer migrated + green; Epic-1 interface layer updated
to the new API with cases/goldens unchanged (or intentionally-updated under review);
benchmarks zero-diff. *3B:* loose-goal behavioural cases green; only the anticipated
§7 goldens change, reviewed; logical baseline regenerated with `success` unchanged.

**Dependencies:** Epics 1 and 2.

---

## Behavioural payoffs — folded into each epic as Phase B (not a separate epic)

The design's baseline-moving work is **not** a parallel "Epic 4"; each payoff lives
as the **Phase B** of the epic that provides its enabling interface, so it lands on
stable, already-verified structure:

- **§5 — frontier bound-wiring** → **Phase 2B** (needs 2A's `SearchCore` + bound
  plumbing). Moves gated baselines.
- **§6 — P&R-resume behaviour** → **Phase 2B** (needs 2A's `best_reached` + internal
  `RouteOutcome`). Un-gated → relies on Epic-1's P&R/resume cases.
- **§7 — placement lift** → **Phase 3B** (needs 3A's stable interfaces). Moves the
  logical baseline; highest risk; last.

The isolation that a separate epic gave is preserved by the **Phase A → checkpoint →
Phase B** discipline (each phase a distinct commit/PR; zero-drift verified between):
the refactor still gets its "any drift = bug" signal, and each behavioural change
shows up as a small, intentional golden/baseline diff rather than hiding in refactor
noise.

---

## Ordering & dependencies

```
Epic 1 (test net)
   │  (must include the un-gated P&R + cascade + loose-goal + resume cases)
   ▼
Epic 2 : 2A internal refactor (API frozen, zero-drift) ─▶ ✔checkpoint─▶ 2B §5 bound-wiring + §6 P&R-resume
   ▼
Epic 3 : 3A public API + Python migration (zero-drift) ─▶ ✔checkpoint─▶ 3B §7 placement lift
```

- **Epic 1 blocks everything** — the net must exist first, and must include the
  currently-un-gated P&R + cascade + loose-goal + resume cases *and* the anticipatory
  §5/§6/§7 goldens.
- **Phase A of each epic is the low-risk bulk** — guarded twice (benchmarks +
  Epic-1 golden), zero intended drift, landed and verified before Phase B.
- **Phase B is the only intentionally-behaviour-changing work** — separate commit,
  new tests first, baselines regenerated under review.
- **Epic 3's Phase A exercises the interface-layer seam** — the one place that
  changes when the crate API changes.

## Open decisions

1. **Are the Phase-B payoffs in scope now, or deferred?** Phase A of each epic (the
   structural refactor) is the committed backbone; the Phase-B payoffs (§5/§6/§7) are
   separable and can be deferred until the A-phases land, unless the memory win (§5)
   or P&R-resume is a near-term priority. (Deferring = stop each epic at its
   checkpoint.)
2. **Golden format for Epic 1** — inline expected `Outcome`s in the case files, or
   a committed golden file (like the benchmark CSVs)? Inline is more legible;
   a file is easier to regenerate wholesale.
3. **Does the Epic-1 suite capture at the `SolveResult` level, the `TargetSolver`
   level, or both?** Capturing at the level Epic 2 preserves (`SolveResult`) gives
   the cleanest zero-drift signal; capturing lower catches more but churns more.
