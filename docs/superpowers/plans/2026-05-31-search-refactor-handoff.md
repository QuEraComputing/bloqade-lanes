# bloqade-lanes-search refactor — session handoff (2026-05-31)

## TL;DR for tomorrow's session

The Rust-side §6 type split is **complete and committed**. Branch is
`phil/refactor-search-infra` at `8329763`. All 13 commits build, 296
unit tests + the 60-case `temp_regression` integration suite pass
bit-for-bit on every commit, full Python suite passes
(1423 / 9 skipped / 0 failed). The legacy `MoveSolver` API is
unchanged — every `solve_*` method now delegates to a shared
implementation that the new `CzPlacement` peers also use, with
in-file parity tests proving byte-identity.

**Next milestone:** migrate `bloqade-lanes-bytecode-python` (the
PyO3 layer) to consume the new `CzPlacement`-shaped surface instead
of `MoveSolver.solve_*`. Once Python is on the new types and stable,
delete the legacy `MoveSolver` API.

## What landed on `phil/refactor-search-infra`

Reverse-chronological. Each commit has its own detailed message; this
table is the scanning index.

| Commit | Slice | What it did |
|--------|-------|-------------|
| `8329763` | §6 slice 5 | `RecedingHorizonCzPlacement` + `NoHomeCzPlacement` peers; `MoveSolver::solve_entangling_rh` / `solve_nohome` collapsed to delegations |
| `5152eca` | §6 slice 4 | `LooseGoalCzPlacement`; `MoveSolver::solve_entangling` collapsed |
| `d3b8e7c` | §6 slice 3 | `CzPlacement` trait + `SingleHeuristicCzPlacement`; `MoveSolver::solve_with_generator` collapsed |
| `519406b` | §6 slice 2b | `MoveSearch` + `TargetSolver` composition; `solve_with_engine` shared impl |
| `67f403f` | §6 slice 2a | `SearchEngine` extracted from `MoveSolver`; `MoveSolver { engine: SearchEngine }` |
| `e6292a6` | §6 slice 1 | `search/{result,options,restarts}.rs` extracted from `solve.rs` (god module 2025 → 1430 lines) |
| `4e23a65` | structural | `find_path_occupied` extracted from `drivers/entropy.rs` to `primitives/path.rs` |
| `177d274` | structural | Reorganize module tree into `primitives/`, `ops/`, `drivers/`, `search/`, `placement/` |
| `b659896` | pre-split | Six "ready-to-abstract" cleanups: `SolveResult` ctors, `clipped_future_layers`, `upgraded_for_entangling`, `HeuristicGenerator::configured`, `Status::as_label()`, `LaneIndex::from_arch_spec`, `MoveProblem::*_locations()` |
| `cbdd159` | observer | Unify entropy trace plumbing under `SearchObserver` (Option A from §6.4) |
| `b1f103f` | docs | Post-PR-600 crate review doc |
| `2cfb7cb` | docs | Pre-PR-600 crate review doc |
| `9958619` | test infra | `temp_regression` suite (60 random-seeded fixtures, bit-identity replay) |

## The new public API

Additive. The legacy `MoveSolver` API is unchanged.

```rust
use bloqade_lanes_search::{
    SearchEngine, MoveSearch, TargetSolver, CzPlacement,
    SingleHeuristicCzPlacement, LooseGoalCzPlacement,
    RecedingHorizonCzPlacement, NoHomeCzPlacement,
    DefaultTargetGenerator,
    SolveOptions, EntropyOptions, EntanglingOptions,
    RecedingHorizonOptions, NoHomeOptions,
};
use std::sync::Arc;

// 1. Pick a search algorithm — orthogonal to the CZ strategy.
let engine = Arc::new(SearchEngine::from_arch_spec(&arch));
let search = MoveSearch::entropy();        // or ::astar(1.0), ::ids(), ::cascade(InnerStrategy::Ids)

// 2a. Single-target fixed solve.
let solver = TargetSolver::new(engine.clone(), search.clone());
let result = solver.solve(initial, target, blocked, Some(max_expansions))?;

// 2b. Single-heuristic CZ placement (TargetSolver + plug-in target generator).
let placement = SingleHeuristicCzPlacement::new(
    TargetSolver::new(engine.clone(), search.clone()),
    Box::new(DefaultTargetGenerator),
);
let result = placement.solve(&initial, &controls, &targets, &blocked, Some(max_expansions))?;
// or for per-candidate detail:
let multi = placement.solve_with_attempts(initial, &controls, &targets, blocked, Some(max_expansions))?;

// 2c. Loose-goal CZ placement (drives MoveSearch directly with EntanglingConstraintGoal).
let placement = LooseGoalCzPlacement::new(
    engine.clone(),
    search.clone(),
    EntanglingOptions::default(),
);
let result = placement.solve(&initial, &controls, &targets, &blocked, Some(max_expansions))?;
// or with future-layer lookahead:
let result = placement.solve_pairs(initial, &cz_pairs, blocked, Some(max_expansions), &future_cz_layers)?;

// 2d. Receding-horizon MPC.
let placement = RecedingHorizonCzPlacement::new(
    engine.clone(),
    search.clone(),
    EntanglingOptions::default(),
    RecedingHorizonOptions::default(),
);
let result = placement.solve_pairs(initial, &cz_pairs, blocked, Some(max_expansions), &future_cz_layers)?;

// 2e. No-home two-phase placement.
let placement = NoHomeCzPlacement::new(engine, search, NoHomeOptions::default());
let result = placement.solve_pairs(initial, &cz_pairs, blocked, Some(max_expansions), &future_cz_layers)?;
```

**Trait shape** (in `placement/cz_placement.rs`):

```rust
pub trait CzPlacement {
    fn solve(
        &self,
        initial: &[(u32, LocationAddr)],
        controls: &[u32],
        targets: &[u32],
        blocked: &[LocationAddr],
        max_expansions: Option<u32>,
    ) -> Result<SolveResult, ConfigError>;
}
```

Borrowed-slice arguments (not `impl IntoIterator`) so the trait stays
dyn-compatible. `Box<dyn CzPlacement>` works for runtime polymorphism.

## The shared-impl pattern

This is the load-bearing pattern of the refactor. **Every** legacy
`MoveSolver::solve_*` method is now a one-line delegation to a shared
`pub(crate)` free function under `placement/*`:

| Legacy method (in `search/solve.rs`) | Shared impl (in `placement/*`) |
|--------------------------------------|--------------------------------|
| `MoveSolver::solve` | `search::target_solver::solve_with_engine` |
| `MoveSolver::solve_with_generator` | `placement::single_heuristic::solve_single_heuristic` |
| `MoveSolver::solve_entangling` | `placement::loose_goal::solve_loose_goal` |
| `MoveSolver::solve_entangling_rh` | `placement::receding_horizon::solve_receding_horizon` |
| `MoveSolver::solve_nohome` | `placement::nohome::solve_nohome` |

The new `CzPlacement` peers (`SingleHeuristicCzPlacement` etc.) **also
call these same `pub(crate)` free functions**, so the legacy facade
and the new typed surface are guaranteed to produce identical output
on identical inputs.

**Parity tests** in each `placement/*.rs` file pin byte-identity:

- `placement/single_heuristic.rs`: `single_heuristic_matches_solve_with_generator`, `cz_placement_trait_returns_inner_result`
- `placement/loose_goal.rs`: `loose_goal_parity_simple_pair`, `_multiple_pairs`, `_with_spectators`, `_with_ids_strategy`, `_with_cascade_strategy`, `cz_placement_trait_matches_solve_pairs`
- `placement/receding_horizon.rs`: `rh_placement_matches_solve_entangling_rh`
- `placement/nohome.rs`: `nohome_placement_matches_solve_nohome`
- `search/target_solver.rs`: `target_solver_solves_simple_move`, `target_solver_matches_move_solver_solve`

These prove byte-identity at every CI run, not just for random
fixtures — strongest possible refactor-correctness signal.

## Verification gates

Run after every change:

```bash
# Rust
cargo test -p bloqade-lanes-search                          # 296 + temp_regression
cargo test --workspace                                       # all crates
cargo clippy --workspace --all-targets -- -D warnings        # zero warnings

# Python (slow — only when PyO3 surface or strategy-layer Python changes)
just develop-python
uv run pytest python/tests                                   # 1423 pass, 9 skip
```

`temp_regression` is the regression net for the unchanged
`MoveSolver::solve` path. Per-placement parity tests are the
regression net for the new typed surface. Don't delete `temp_regression`
until the Python migration is complete and `MoveSolver` is gone.

## Next milestone: Python migration

`bloqade-lanes-bytecode-python` currently wraps `MoveSolver` as
`PyMoveSolver` with methods like `solve_with_generator`,
`solve_entangling`, etc. These need to migrate to the new typed
surface. **Strategy: add new PyO3 classes alongside `PyMoveSolver`,
migrate the Python layer to use them, then delete the legacy class.**

### Step 1 — Add new PyO3 wrapper classes

For each new Rust type, add a `Py*` wrapper in
`crates/bloqade-lanes-bytecode-python/src/search_python.rs`:

- `PySearchEngine` wrapping `Arc<SearchEngine>` (construct via
  `PyArchSpec`, no JSON round-trip — `SearchEngine::from_arch_spec`
  already exists, see slice 1's "(E)" cleanup).
- `PyMoveSearch` wrapping `MoveSearch` with `astar(weight)`,
  `entropy()`, `ids()`, `cascade(inner)` class methods + options
  setters.
- `PyTargetSolver` wrapping `TargetSolver` with `solve(initial,
  target, blocked, max_expansions)`.
- `PySingleHeuristicCzPlacement` wrapping the corresponding Rust
  type. Takes `PyTargetSolver` + a `PyTargetGenerator` (the latter
  already exists).
- `PyLooseGoalCzPlacement`, `PyRecedingHorizonCzPlacement`,
  `PyNoHomeCzPlacement` for the other three peers.

Each new class is **additive** — the existing `PyMoveSolver` stays
working. PyO3 module-init registers the new types alongside.

The PyO3 wrappers should release the GIL during the actual solve via
`py.allow_threads(|| ...)` (the existing `PyMoveSolver.solve` already
does this — copy the pattern).

### Step 2 — Migrate Python placement strategies

The Python code that currently constructs `MoveSolver` lives mainly in
`python/bloqade/lanes/heuristics/physical/`:

- `physical_placement.py` — uses `MoveSolver.solve_entangling`
- `no_return_placement.py` — uses `MoveSolver.solve_entangling`
- `nohome_placement.py` — uses `MoveSolver.solve_nohome`
- `receding_horizon_placement.py` — uses `MoveSolver.solve_entangling_rh`

Each should be rewritten to:
1. Build a `SearchEngine` once per arch.
2. Construct the appropriate `*CzPlacement` object.
3. Call `placement.solve(...)` (or `solve_pairs` for the cz_pairs API).

Behavior-preserving check: the existing `python/tests/heuristics/test_*_placement.py`
tests must pass after each migration. They're already end-to-end so
any drift is caught immediately.

### Step 3 — Delete the legacy `MoveSolver` surface

Once Python is fully on the new types and stable:

1. Delete the legacy methods on `MoveSolver`:
   `solve_with_generator`, `solve_entangling`, `solve_entangling_rh`,
   `solve_nohome`, `generate_candidates`. Keep `solve` and the
   constructors (or migrate those too).
2. Delete `MoveSolver` entirely (replace with `TargetSolver` +
   `CzPlacement` peers). At this point the parity tests in
   `placement/*.rs` lose their reference and should be deleted too
   (they compared `LooseGoalCzPlacement` against the now-deleted
   `MoveSolver::solve_entangling`).
3. Delete `crates/bloqade-lanes-search/tests/temp_regression/` —
   its job is done.
4. Delete the legacy `PyMoveSolver` from `search_python.rs`.

## Deferred work (not blocking, flagged in the §6 review doc)

These were called out in `docs/superpowers/reviews/2026-05-31-bloqade-lanes-search-post-pr600-review.md`
but deliberately not done as part of this refactor branch:

- **`astar.rs` ⇄ `frontier.rs` cycle.** `astar::Expander` is
  `pub(crate) + #[allow(dead_code)]` and `astar()` exists only for
  legacy in-crate tests. Delete `drivers/astar.rs`, move `SearchResult`
  into `drivers/frontier.rs` or `search/result.rs`.
- **Per-call observer on `TargetSolver` / `CzPlacement`.** The §6.4
  observer unification landed for entropy, but the new `TargetSolver`
  and `CzPlacement` methods don't yet take a `&mut dyn SearchObserver`
  parameter. Frontier-strategy events are still discarded via
  `&mut NoOpObserver` inside `solve_with_engine`. Adding observer
  threading is a follow-up.
- **`Strategy::Cascade { inner }` reshape.** `InnerStrategy` is a
  strict subset of `Strategy`; could fold into
  `SolveOptions { strategy, refine_with_astar: bool }`. Not yet done.
- **DSL surface (PR 600).** `dsl::move_policy_dsl::PolicyStatus` /
  `PolicyResult` / `solve_with_policy` are a fourth parallel solve
  path that doesn't yet implement `CzPlacement`. Could become
  `PolicyCzPlacement` in a future slice. The DSL stays sidecar.

## Key file locations

```
crates/bloqade-lanes-search/src/
├── lib.rs                                          (crate-root re-exports)
├── primitives/                                     (types reused across drivers)
│   ├── config.rs, context.rs, graph.rs, lane_index.rs,
│   ├── distance.rs                                 (DistanceTable + heuristics)
│   ├── ordering.rs, path.rs
│   └── mod.rs
├── ops/                                            (stateless math)
│   ├── aod_grid.rs, entangling.rs, mod.rs
├── drivers/                                        (search-loop engines)
│   ├── frontier.rs, entropy.rs, astar.rs, mod.rs
├── search/                                         (facade layer)
│   ├── engine.rs              (SearchEngine + EntanglingCache + NoHomeCache)
│   ├── move_search.rs         (MoveSearch value bundle)
│   ├── options.rs             (SolveOptions, EntropyOptions, EntanglingOptions, Strategy)
│   ├── result.rs              (SolveResult, SolveStatus + ctors)
│   ├── restarts.rs            (run_with_components, pick_best, extract)
│   ├── target_solver.rs       (TargetSolver + solve_with_engine)
│   ├── solve.rs               (legacy MoveSolver, all methods 1-line delegations)
│   └── mod.rs
├── placement/                                      (CZ-stage strategies)
│   ├── cz_placement.rs        (the CzPlacement trait)
│   ├── single_heuristic.rs    (+ solve_single_heuristic)
│   ├── loose_goal.rs          (+ solve_loose_goal)
│   ├── receding_horizon.rs    (RecedingHorizonCzPlacement + solve_receding_horizon + solve_entangling_rh_single + supporting helpers)
│   ├── nohome.rs              (NoHomeCzPlacement + solve_nohome + home-site helpers)
│   ├── target_generator.rs    (TargetGenerator trait + DefaultTargetGenerator)
│   └── mod.rs
├── generators/, goals/, scorers/                  (trait impls, unchanged shape)
├── cost.rs, heuristics.rs, observer.rs, traits.rs  (top-level small modules)
├── dsl/                                            (Starlark sidecar — PR 600)
└── tests/temp_regression/                          (60-fixture bit-identity replay)

docs/superpowers/reviews/
├── 2026-05-31-bloqade-lanes-search-review.md           (pre-PR-600 review)
└── 2026-05-31-bloqade-lanes-search-post-pr600-review.md (post-PR-600 review + §6 plan)

docs/superpowers/plans/
└── 2026-05-31-search-refactor-handoff.md               (this document)
```

## Resuming tomorrow — first commands

```bash
cd /path/to/bloqade-lanes              # or clone fresh
git fetch origin
git checkout phil/refactor-search-infra
git pull --ff-only

# Sanity-check the branch state
git log --oneline phil/refactor-search-infra ^origin/main | head -15
cargo test -p bloqade-lanes-search 2>&1 | grep "test result" | head -3

# If pyright fails (pre-existing main issue), the fix is already in 50b7ca4.

# Read this doc + the §6 review for full context
$EDITOR docs/superpowers/plans/2026-05-31-search-refactor-handoff.md
$EDITOR docs/superpowers/reviews/2026-05-31-bloqade-lanes-search-post-pr600-review.md
```

If starting in a fresh Claude session, paste this doc's TL;DR into the
opening prompt and point at the branch — that's the minimum
self-contained context needed to continue.
