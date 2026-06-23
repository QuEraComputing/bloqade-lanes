# bloqade-lanes-search Review Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the eight findings from the 2026-06-16 crate review of `bloqade-lanes-search` — dead seams, an upward layering dependency, an implicit two-mode generator contract, re-export drift, and three localized code-quality fixes — without changing solver behavior (except where a behavior-neutral refactor is explicitly intended).

**Architecture:** Each task is independent and individually committable. They are ordered low-risk → high-risk: three mechanical fixes first (dedup, `Option` return, helper move), then three medium refactors (call-site helper, escape-move dedup, re-export hygiene), then two large structural changes (explicit two-mode trait, legacy A* removal). All work is inside `crates/bloqade-lanes-search/` except Task 6, which touches one consumer file in `crates/bloqade-lanes-bytecode-python/`.

**Tech Stack:** Rust (stable), `cargo test`/`cargo clippy`, the crate's existing trait set (`MoveGenerator`, `CandidateScorer`, `CostFn`, `Goal`, `Heuristic`, `Frontier`). Build/test commands run from the repo root.

**Conventions for every task:**
- Build check: `cargo build -p bloqade-lanes-search`
- Test: `cargo test -p bloqade-lanes-search`
- Lint (must be clean before commit): `cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings`
- Format: `cargo fmt -p bloqade-lanes-search`
- Commit messages follow Conventional Commits (see `AGENT.md`); end the body with the `Co-Authored-By` trailer.

---

## Task 1: Replace O(n²) candidate dedup with a `HashSet` (Rust-1)

**Why:** `generators/heuristic.rs:623` deduplicates candidate movesets with a linear `candidates.iter().any(...)` scan inside the per-bus-group loop — O(n²) in moveset count. `MoveSet` derives `Hash + Eq` (`bloqade-lanes-dsl-core/src/primitives/move_set.rs:20`), and the `candidates` vec is fully re-sorted by `cmp_candidates` at `:642`, so insertion order is irrelevant — a side-index `HashSet` is behavior-neutral.

**Files:**
- Modify: `crates/bloqade-lanes-search/src/generators/heuristic.rs` (the `generate` method, around lines 562–628)

- [ ] **Step 1: Confirm the current dedup site**

Run: `grep -n "candidates.iter().any" crates/bloqade-lanes-search/src/generators/heuristic.rs`
Expected: one hit near line 623.

- [ ] **Step 2: Add the seen-set alongside the candidates vec**

Find the candidates declaration (around line 563):

```rust
        // Step 5: per group, build AOD-compatible rectangular grids.
        let mut candidates: Vec<(i32, MoveSet, Config)> = Vec::new();
```

Change it to also declare a seen-set:

```rust
        // Step 5: per group, build AOD-compatible rectangular grids.
        let mut candidates: Vec<(i32, MoveSet, Config)> = Vec::new();
        // Side-index of movesets already emitted, for O(1) dedup. Order of
        // `candidates` is irrelevant — Step 6 re-sorts by score.
        let mut seen_movesets: HashSet<MoveSet> = HashSet::new();
```

- [ ] **Step 3: Replace the linear scan with the set check**

Replace the dedup block (around lines 622–627):

```rust
                // Deduplicate: skip if we already have this exact moveset.
                if candidates.iter().any(|(_, ms, _)| *ms == move_set) {
                    continue;
                }

                candidates.push((total_score, move_set, new_config));
```

with:

```rust
                // Deduplicate: skip if we already have this exact moveset.
                if !seen_movesets.insert(move_set.clone()) {
                    continue;
                }

                candidates.push((total_score, move_set, new_config));
```

- [ ] **Step 4: Ensure `HashSet` is in scope**

Run: `grep -n "use std::collections::HashSet" crates/bloqade-lanes-search/src/generators/heuristic.rs`
Expected: a hit at the top of the file (the module already uses `HashSet` for `occupied`/`target_locs`). If absent, add `use std::collections::HashSet;` to the import block.

- [ ] **Step 5: Build, test, lint**

Run: `cargo test -p bloqade-lanes-search` — Expected: PASS (existing heuristic-generator tests unchanged).
Run: `cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings` — Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/bloqade-lanes-search/src/generators/heuristic.rs
git commit -m "perf(search): O(1) candidate dedup in HeuristicGenerator

Replace the O(n^2) linear moveset scan with a HashSet side-index.
Behavior-neutral: candidates are re-sorted by score afterward, so
insertion order is irrelevant.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Make `pick_best` return `Option<SolveResult>` (Rust-2)

**Why:** `search/restarts.rs:56` is `pub(crate) fn pick_best(results: Vec<SolveResult>) -> SolveResult` and ends in `.expect("non-empty results")`. The emptiness precondition is real but invisible. Return `Option<SolveResult>` so it's type-visible, and update the three call sites (`:200`, `:235`, `:274`).

**Files:**
- Modify: `crates/bloqade-lanes-search/src/search/restarts.rs`

- [ ] **Step 1: Change the signature and drop the panic**

Replace (lines 55–65):

```rust
/// Pick the best result from multiple restarts (prefer solved, then lowest cost).
pub(crate) fn pick_best(results: Vec<SolveResult>) -> SolveResult {
    results
        .into_iter()
        .min_by(|a, b| {
            let a_solved = a.status == SolveStatus::Solved;
            let b_solved = b.status == SolveStatus::Solved;
            b_solved.cmp(&a_solved).then(a.cost.total_cmp(&b.cost))
        })
        .expect("non-empty results")
}
```

with:

```rust
/// Pick the best result from multiple restarts (prefer solved, then lowest
/// cost). Returns `None` only when `results` is empty.
pub(crate) fn pick_best(results: Vec<SolveResult>) -> Option<SolveResult> {
    results.into_iter().min_by(|a, b| {
        let a_solved = a.status == SolveStatus::Solved;
        let b_solved = b.status == SolveStatus::Solved;
        b_solved.cmp(&a_solved).then(a.cost.total_cmp(&b.cost))
    })
}
```

- [ ] **Step 2: Fix call site at `:200` (parallel restarts)**

The `run_inner_with_restarts` closure builds `results` from `(0..restarts)` where `restarts > 1` (the `else` branch), so it is always non-empty. Replace:

```rust
            pick_best(results)
```

with (use `expect` locally where the non-empty invariant is provable and visible):

```rust
            pick_best(results).expect("restarts > 1 yields a non-empty result set")
```

- [ ] **Step 3: Fix call site at `:235` (cascade two-way)**

Replace:

```rust
            return pick_best(vec![inner_result, astar_solve]);
```

with:

```rust
            return pick_best(vec![inner_result, astar_solve])
                .expect("two-element vec is non-empty");
```

- [ ] **Step 4: Fix call site at `:274`**

Run: `sed -n '270,278p' crates/bloqade-lanes-search/src/search/restarts.rs` to see the surrounding context and how `results` is built. It is the non-cascade restart path, also built from `(0..restarts)` with `restarts > 1`. Replace:

```rust
        pick_best(results)
```

with:

```rust
        pick_best(results).expect("restarts > 1 yields a non-empty result set")
```

- [ ] **Step 5: Build, test, lint**

Run: `cargo test -p bloqade-lanes-search` — Expected: PASS.
Run: `cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings` — Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/bloqade-lanes-search/src/search/restarts.rs
git commit -m "refactor(search): pick_best returns Option<SolveResult>

Make the non-empty precondition type-visible instead of a runtime
expect inside the helper. Each caller asserts its own provable
non-empty invariant at the call site.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Move `home_sites` into `ops::entangling` (Arch-1)

**Why:** `search/engine.rs:123` (the data layer, documented as *below* the placement peers) calls `placement::nohome::home_sites` — an upward dependency. `home_sites` (`placement/nohome.rs:69`) is pure `ArchSpec` math (the no-home analogue of `ops::entangling::all_entangling_locations`) and belongs in `ops::entangling`.

**Files:**
- Modify: `crates/bloqade-lanes-search/src/ops/entangling.rs` (add `home_sites`)
- Modify: `crates/bloqade-lanes-search/src/placement/nohome.rs` (remove definition, import from ops)
- Modify: `crates/bloqade-lanes-search/src/search/engine.rs` (import from ops, drop `use crate::placement::nohome`)

- [ ] **Step 1: Confirm callers of `home_sites`**

Run: `grep -rn "home_sites" crates/bloqade-lanes-search/src/`
Expected: definition at `nohome.rs:69`, call at `engine.rs:123`, plus any in-crate tests.

- [ ] **Step 2: Add `home_sites` to `ops::entangling`**

Cut this function (and its doc comment) from `placement/nohome.rs:65-88`:

```rust
/// Enumerate all home-site locations (encoded) from the architecture.
///
/// A home site is any `(zone_id, word_id, site_id)` where `word_id` is
/// in [`ArchSpec::left_cz_word_ids`].
pub fn home_sites(arch: &ArchSpec) -> Vec<u64> {
    let home_words = arch.left_cz_word_ids();
    let word_zone = arch.word_zone_map();
    let sites_per_word = arch.sites_per_word() as u32;
    let mut result = Vec::new();
    for &word_id in &home_words {
        let zone_id = *word_zone.get(&word_id).unwrap_or(&0);
        for site_id in 0..sites_per_word {
            result.push(
                LocationAddr {
                    zone_id,
                    word_id,
                    site_id,
                }
                .encode(),
            );
        }
    }
    result
}
```

Paste it into `crates/bloqade-lanes-search/src/ops/entangling.rs` near the other location-enumeration helpers (e.g. just after `all_entangling_locations`). Change `pub fn` to `pub(crate) fn` (it has no external consumer; confirmed in the review). Ensure `ArchSpec` and `LocationAddr` are imported in `entangling.rs` — run `grep -n "use bloqade_lanes_bytecode_core\|LocationAddr\|ArchSpec" crates/bloqade-lanes-search/src/ops/entangling.rs`; add the imports if missing (mirror the import lines that were in `nohome.rs`).

- [ ] **Step 3: Update `nohome.rs` to use the moved function**

In `placement/nohome.rs`, wherever `home_sites(...)` was called internally, qualify it as `entangling::home_sites(...)` (or add `use crate::ops::entangling::home_sites;`). Confirm `nohome.rs` already imports `ops::entangling` — run `grep -n "ops::entangling" crates/bloqade-lanes-search/src/placement/nohome.rs`; add `use crate::ops::entangling;` if absent. If `LocationAddr` is now unused in `nohome.rs`, remove its import to avoid a clippy warning.

- [ ] **Step 4: Update `engine.rs` to use the moved function and drop the upward import**

In `crates/bloqade-lanes-search/src/search/engine.rs`:
- Remove line 17: `use crate::placement::nohome;`
- Change line 123 from `let home_locs = nohome::home_sites(arch);` to `let home_locs = entangling::home_sites(arch);` (the module already imports `use crate::ops::entangling::{self, WordPairDistances};` at line 16, so `entangling::home_sites` resolves).

- [ ] **Step 5: Build, test, lint**

Run: `grep -rn "placement::nohome" crates/bloqade-lanes-search/src/search/engine.rs` — Expected: no hits.
Run: `cargo test -p bloqade-lanes-search` — Expected: PASS.
Run: `cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings` — Expected: clean (watch for newly-unused imports).

- [ ] **Step 6: Commit**

```bash
git add crates/bloqade-lanes-search/src/ops/entangling.rs \
        crates/bloqade-lanes-search/src/placement/nohome.rs \
        crates/bloqade-lanes-search/src/search/engine.rs
git commit -m "refactor(search): move home_sites into ops::entangling

home_sites is pure ArchSpec math; hosting it in the nohome placement
strategy forced SearchEngine (the data layer) to depend upward on a
placement peer. Relocating it to ops::entangling removes the inversion.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Collapse the 11-arg `run_search` call sites in `restarts.rs` (Emerging-1)

**Why:** `frontier::run_search` takes 11 arguments and is invoked 6× in `search/restarts.rs` with an identical constant block — `&scorer, &cost_fn, goal, ctx, &mut SearchState::default(), &mut NoOpObserver` — varying only the generator, the frontier, the budget, and `max_depth`. A generic helper fixes the constants and collapses each call to 4 meaningful arguments.

**Files:**
- Modify: `crates/bloqade-lanes-search/src/search/restarts.rs`

- [ ] **Step 1: Add a private generic helper inside `restarts.rs`**

Place this free function in the module (top-level, near `extract`/`pick_best`, not inside `run_with_components`). It captures the always-constant scorer (`DistanceScorer`), cost (`UniformCost`), fresh state, and no-op observer:

```rust
/// Run the trait-based frontier search with the scorer/cost/state/observer
/// that every `restarts` call site uses identically. Collapses the 11-arg
/// `frontier::run_search` signature to the four arguments that actually vary.
fn run_frontier<Gen, Go, F>(
    root: &Config,
    generator: &Gen,
    goal: &Go,
    ctx: &SearchContext,
    frontier: &mut F,
    max_expansions: Option<u32>,
    max_depth: Option<u32>,
) -> SearchResult
where
    Gen: MoveGenerator,
    Go: Goal,
    F: Frontier,
{
    frontier::run_search(
        root.clone(),
        generator,
        &DistanceScorer,
        &UniformCost,
        goal,
        frontier,
        ctx,
        &mut SearchState::default(),
        &mut NoOpObserver,
        max_expansions,
        max_depth,
    )
}
```

Confirm `Frontier` and `Goal` are imported in `restarts.rs` — run `grep -n "use crate::traits\|use crate::drivers::frontier\|Frontier" crates/bloqade-lanes-search/src/search/restarts.rs`; add `use crate::drivers::frontier::Frontier;` and ensure `Goal` is in the `traits` import if the helper needs them.

- [ ] **Step 2: Replace the Ids call site (`:113`)**

```rust
                let result = run_frontier(&root, &move_gen, goal, ctx, &mut f, budget, None);
```

(Removes the local `scorer`/`cost_fn` references at this site. Keep the surrounding `let move_gen = ...; let mut f = IdsFrontier::new(h_sum);` lines.)

- [ ] **Step 3: Replace the Dfs call site (`:131`)**

```rust
                let result = run_frontier(&root, &move_gen, goal, ctx, &mut f, budget, None);
```

- [ ] **Step 4: Replace the cascade A* call site (`:215`)**

```rust
        let astar_result = run_frontier(
            &root,
            &astar_move_gen,
            goal,
            ctx,
            &mut astar_f,
            max_expansions,
            max_depth,
        );
```

- [ ] **Step 5: Replace the remaining three `run_strategy_v2` call sites (`:300`, `:316`, `:332`)**

Run: `grep -n "frontier::run_search" crates/bloqade-lanes-search/src/search/restarts.rs` to list any still using the long form. For each, apply the same transform: `run_frontier(&root, <generator>, goal, ctx, <&mut frontier>, <budget>, <max_depth>)`, matching the original generator/frontier/budget/max_depth at that site.

- [ ] **Step 6: Remove now-dead locals**

The module-level `let scorer = DistanceScorer;` and `let cost_fn = UniformCost;` inside `run_with_components` (lines 104–105) are now unused (the helper supplies them). Run `grep -n "scorer\|cost_fn" crates/bloqade-lanes-search/src/search/restarts.rs` to confirm no remaining references in `run_with_components`, then delete those two lines. (The `entropy_search` branch does not use them.)

- [ ] **Step 7: Build, test, lint**

Run: `grep -c "frontier::run_search" crates/bloqade-lanes-search/src/search/restarts.rs` — Expected: 0 (production call sites all routed through `run_frontier`; note `run_frontier` itself calls `frontier::run_search` once, so expect exactly 1 hit — adjust expectation to 1).
Run: `cargo test -p bloqade-lanes-search` — Expected: PASS (behavior-neutral; same defaults).
Run: `cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings` — Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add crates/bloqade-lanes-search/src/search/restarts.rs
git commit -m "refactor(search): collapse run_search call sites via run_frontier helper

Six call sites passed an identical scorer/cost/state/observer block to
the 11-arg frontier::run_search. A private generic helper fixes those
constants, leaving the four arguments that actually vary. Behavior-neutral.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Unify the three escape-move generators (Emerging-2)

**Why:** `generators/heuristic.rs` has `generate_blocker_escape` (`:209`) and `generate_all_escape` (`:240`) sharing the same `outgoing_lanes → skip-occupied → MoveCandidate` inner loop, plus an inline spectator-escape fallback (`:496`) that open-codes a similar loop but emits a `ScoredTriple` into `selected` instead of a `MoveCandidate`. Unify the shared traversal behind one helper that yields `(lane, dst)` pairs, so all three consume it.

**Files:**
- Modify: `crates/bloqade-lanes-search/src/generators/heuristic.rs`

- [ ] **Step 1: Add a shared escape-traversal helper**

Add this associated function (or free fn in the module) — it yields valid escape `(lane, dst)` pairs for one occupied source location, the piece all three sites share:

```rust
/// Yield `(lane, destination)` pairs for every outgoing lane from `loc`
/// whose destination is currently unoccupied. Shared traversal for all
/// escape-move generation paths.
fn escape_targets<'a>(
    loc: LocationAddr,
    occupied: &'a HashSet<u64>,
    index: &'a LaneIndex,
) -> impl Iterator<Item = (LaneAddr, LocationAddr)> + 'a {
    index
        .outgoing_lanes(loc)
        .iter()
        .filter_map(move |&lane| {
            let (_, dst) = index.endpoints(&lane)?;
            if occupied.contains(&dst.encode()) {
                None
            } else {
                Some((lane, dst))
            }
        })
}
```

Confirm `LaneAddr` and `LocationAddr` are imported (they are used elsewhere in the file). Note: `outgoing_lanes` returns `&[LaneAddr]` and `endpoints` returns `Option<(LocationAddr, LocationAddr)>` per the existing call sites — verify the exact types via `grep -n "fn outgoing_lanes\|fn endpoints" crates/bloqade-lanes-search/src/primitives/lane_index.rs` and adjust the helper's iterator item types to match before relying on them.

- [ ] **Step 2: Rewrite `generate_blocker_escape` to use the helper**

Replace the inner `for &lane in index.outgoing_lanes(loc) { ... }` block (`:222-235`) with:

```rust
            for (lane, dst) in escape_targets(loc, occupied, index) {
                let ms = MoveSet::from_encoded(vec![lane.encode_u64()]);
                let new_config = config.with_moves(&[(qid, dst)]);
                out.push(MoveCandidate {
                    move_set: ms,
                    new_config,
                });
            }
```

- [ ] **Step 3: Rewrite `generate_all_escape` to use the helper**

Replace its inner loop (`:248-261`) with the identical block from Step 2 (it has no `target_locs` filter; only the outer `for (qid, loc) in config.iter()` differs, which stays).

- [ ] **Step 4: Rewrite the inline spectator fallback to use the helper**

At `:494-516`, replace the open-coded `for &lane in ctx.index.outgoing_lanes(loc) { ... break; }` with a `next()` on the helper iterator (it only needs the first valid escape):

```rust
                let (qid, loc_enc) = accidental_cz_qubits[0];
                let loc = LocationAddr::decode(loc_enc);
                if let Some((lane, dst)) = escape_targets(loc, &occupied, ctx.index).next() {
                    let triplet_key = (lane.move_type as u8, lane.bus_id, lane.direction as u8);
                    selected.push((
                        triplet_key,
                        ScoredTriple {
                            qubit_id: qid,
                            score: 1,
                            lane_encoded: lane.encode_u64(),
                            dst_encoded: dst.encode(),
                        },
                    ));
                }
```

(Note: the original computed `dst_enc = dst.encode()` once and reused it for both the occupancy check and `dst_encoded`. The helper already filters on occupancy, so here we only need `dst.encode()` for `dst_encoded`.)

- [ ] **Step 5: Build, test, lint**

Run: `cargo test -p bloqade-lanes-search` — Expected: PASS. The escape generators are exercised by deadlock-policy tests; if any test asserts an exact candidate ordering, confirm the helper preserves `outgoing_lanes` order (it does — `filter_map` is order-preserving).
Run: `cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings` — Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/bloqade-lanes-search/src/generators/heuristic.rs
git commit -m "refactor(search): unify escape-move traversal in HeuristicGenerator

Factor the shared outgoing-lane / skip-occupied traversal behind an
escape_targets iterator consumed by generate_blocker_escape,
generate_all_escape, and the inline spectator fallback. Behavior-neutral.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Re-export hygiene — delete the `solve` shim, slim the `lib.rs` facade (Contract-2 + Drift-1)

**Why:** `search/solve.rs` is a re-export shim: only `CandidateAttempt`/`MultiSolveResult` are defined there; everything else re-exports `search::options`/`search::result`. The `lib.rs` facade re-exports ~50 names of which only `LaneIndex` and `DeadlockPolicy` are consumed via the short path, and `lib.rs:67` points the facade at the `solve` shim (drift — those types live in `options`/`result`). One consumer (`search_python.rs:31`) imports through the shim.

**Files:**
- Modify: `crates/bloqade-lanes-search/src/search/result.rs` (gain `CandidateAttempt`/`MultiSolveResult`)
- Modify: `crates/bloqade-lanes-search/src/search/mod.rs` (drop `pub mod solve;`)
- Delete: `crates/bloqade-lanes-search/src/search/solve.rs`
- Modify: `crates/bloqade-lanes-search/src/lib.rs` (slim facade, fix paths)
- Modify: `crates/bloqade-lanes-bytecode-python/src/search_python.rs` (canonical import paths)
- Modify: any in-crate `use crate::search::solve::...` sites

- [ ] **Step 1: Inventory every `search::solve` reference**

Run: `grep -rn "search::solve\|crate::search::solve" crates/`
Record every hit. Expect: `lib.rs:67`, `search/mod.rs` (the `pub mod solve;`), in-crate imports across `placement::*`/`search::target_solver`, the `search_python.rs:31` consumer, and the `solve.rs` test module's `use super::*`.

- [ ] **Step 2: Relocate `CandidateAttempt` and `MultiSolveResult` into `search::result`**

Open `crates/bloqade-lanes-search/src/search/solve.rs`. Move the `CandidateAttempt` struct (with its `#[derive]`/doc) and the `MultiSolveResult` struct (with its `#[derive]`/doc), plus the `#[cfg(test)] mod tests` block, into `crates/bloqade-lanes-search/src/search/result.rs` (append to the end). Update any `use` lines those moved items need (they reference `SolveResult`/`SolveStatus`, which already live in `result.rs`, and `SolveOptions`/`Strategy` for tests — import from `crate::search::options`).

- [ ] **Step 3: Delete the shim and its module declaration**

```bash
git rm crates/bloqade-lanes-search/src/search/solve.rs
```

In `crates/bloqade-lanes-search/src/search/mod.rs`, remove the `pub mod solve;` line.

- [ ] **Step 4: Repoint in-crate imports to canonical paths**

For every in-crate `use crate::search::solve::{...}` found in Step 1, rewrite to canonical sources:
- `SolveOptions`, `Strategy`, `InnerStrategy`, `EntropyOptions`, `EntanglingOptions` → `crate::search::options::{...}`
- `SolveResult`, `SolveStatus` → `crate::search::result::{...}`
- `CandidateAttempt`, `MultiSolveResult` → `crate::search::result::{...}` (their new home)

Work file-by-file (likely `placement/single_heuristic.rs`, `placement/loose_goal.rs`, `placement/receding_horizon.rs`, `placement/nohome.rs`, `search/target_solver.rs`). After each, run `cargo build -p bloqade-lanes-search` to catch unresolved imports immediately.

- [ ] **Step 5: Fix the `lib.rs` facade**

In `crates/bloqade-lanes-search/src/lib.rs`, replace the drifted block (lines 64–69):

```rust
pub use search::solve::{
    CandidateAttempt, InnerStrategy, MultiSolveResult, SolveOptions, Strategy,
};
```

with canonical-path re-exports:

```rust
pub use search::options::{InnerStrategy, SolveOptions, Strategy};
pub use search::result::{CandidateAttempt, MultiSolveResult};
```

(Decision: keep the full curated re-export set rather than deleting it — the goal of this task is removing *drift* and the *shim*, not migrating the 26 deep-path consumer imports. The wide facade stays but now every re-export points at a canonical module.)

- [ ] **Step 6: Update the consumer import (`search_python.rs:31`)**

Replace:

```rust
use bloqade_lanes_search::search::solve::{
    EntanglingOptions, EntropyOptions, InnerStrategy, MultiSolveResult, SolveOptions, SolveResult,
    Strategy,
};
```

with:

```rust
use bloqade_lanes_search::search::options::{
    EntanglingOptions, EntropyOptions, InnerStrategy, SolveOptions, Strategy,
};
use bloqade_lanes_search::search::result::{MultiSolveResult, SolveResult};
```

- [ ] **Step 7: Verify no `solve` references remain and the whole workspace builds**

Run: `grep -rn "search::solve" crates/` — Expected: no hits.
Run: `cargo build -p bloqade-lanes-search -p bloqade-lanes-bytecode-python -p bloqade-lanes-bytecode-cli` — Expected: clean.
Run: `cargo test -p bloqade-lanes-search` — Expected: PASS (relocated `solve.rs` tests now run under `result.rs`).
Run: `cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings` — Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(search): delete search::solve shim, fix lib.rs facade drift

The §6 type split left search::solve as a re-export shim whose only real
contents were CandidateAttempt/MultiSolveResult. Relocate those into
search::result, delete the shim, repoint internal imports and the one
python consumer at canonical search::options/search::result paths, and
fix the lib.rs facade which still re-exported via the stale solve::* path.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Make `HeuristicGenerator`'s two modes explicit via a trait (Contract-1)

**Why:** `HeuristicGenerator::generate` silently switches behavior on `ctx.cz_pairs.is_some()` in three places — the contested-destination penalty (~`:376`), the no-positive-scores fallback width (3 vs 1, ~`:469`), and pair-coordinated boosting (Step 3c, `:523`). This is "documenting complexity instead of removing it." Introduce a trait that names the contract extension, with a default (no-op / fixed-target) implementation and an entangling implementation, so the mode becomes an explicit type rather than a scattered runtime branch.

> **This is the highest-risk task — it touches the hot scoring path. Implement it as its own PR. Pin behavior with characterization tests BEFORE refactoring (Step 1), and confirm they still pass after (Step 6).**

**Files:**
- Create: `crates/bloqade-lanes-search/src/generators/cz_coordination.rs` (the trait + two impls)
- Modify: `crates/bloqade-lanes-search/src/generators/mod.rs` (declare the module)
- Modify: `crates/bloqade-lanes-search/src/generators/heuristic.rs` (consume the trait at the three gated sites)

- [ ] **Step 1: Characterize current behavior with golden tests**

Before changing anything, add tests in `heuristic.rs`'s `#[cfg(test)] mod tests` that lock in the current output for both modes. Read the existing test helpers (`make_index`, `make_table`, `loc`) at `heuristic.rs:672+` and follow their style. Write one test that runs `generate` with `ctx.cz_pairs = None` and asserts the produced `MoveCandidate` set, and one with `ctx.cz_pairs = Some(&[(qa, qb)])` that exercises the contested-penalty / pair-boost path. Capture the exact candidate movesets/configs the current code produces.

Run: `cargo test -p bloqade-lanes-search heuristic` — Expected: PASS (these document current behavior).

- [ ] **Step 2: Define the trait and two implementations**

Create `crates/bloqade-lanes-search/src/generators/cz_coordination.rs`. The exact method set must mirror the three current branch points — read `heuristic.rs:360-554` first and align signatures to the real local types (`ScoredTriple`, `TripletKey`, the score vectors). A representative shape:

```rust
//! Explicit CZ-coordination policy for `HeuristicGenerator`.
//!
//! Replaces the implicit `ctx.cz_pairs.is_some()` branch with two named
//! policies: [`FixedTargetCoordination`] (the legacy single-target default)
//! and [`EntanglingCoordination`] (loose-goal / no-home / receding-horizon).

use std::collections::{HashMap, HashSet};

use crate::primitives::ordering::TripletKey;

/// One selected, scored lane move (mirrors the local ScoredTriple usage).
pub(crate) use super::heuristic::ScoredTriple;

/// Pluggable scoring behavior that differs between fixed-target and
/// entangling search modes.
pub(crate) trait CzCoordination {
    /// Extra penalty applied to a candidate destination that is contested
    /// by another qubit's target. Default: none (fixed-target mode).
    fn contested_penalty(&self) -> i32 {
        0
    }

    /// How many fallback escape moves to emit when no candidate scores
    /// positive. Default: 1 (fixed-target mode); entangling widens this.
    fn fallback_width(&self) -> usize {
        1
    }

    /// Boost coordinated CZ-pair entries that share a bus triplet so they
    /// land in the same AOD grid. Default: no-op (fixed-target mode).
    fn boost_coordinated_pairs(
        &self,
        _selected: &mut Vec<(TripletKey, ScoredTriple)>,
    ) {
    }
}

/// Legacy single fixed-target behavior — all defaults.
pub(crate) struct FixedTargetCoordination;
impl CzCoordination for FixedTargetCoordination {}

/// Entangling-mode behavior driven by the active CZ pairs.
pub(crate) struct EntanglingCoordination<'a> {
    pub pairs: &'a [(u32, u32)],
}

impl CzCoordination for EntanglingCoordination<'_> {
    // Confirmed from heuristic.rs:381 — the entangling penalty is a flat `-1`
    // applied at the contested-destination guard.
    fn contested_penalty(&self) -> i32 {
        1
    }

    // Confirmed from heuristic.rs:469 — entangling keeps the top 3 entries.
    fn fallback_width(&self) -> usize {
        3
    }

    fn boost_coordinated_pairs(&self, selected: &mut Vec<(TripletKey, ScoredTriple)>) {
        // Move the Step 3c body (heuristic.rs:523-554) here verbatim,
        // using self.pairs in place of `pairs`.
        let mut keys_by_qubit: HashMap<u32, HashSet<TripletKey>> = HashMap::new();
        for entry in selected.iter() {
            keys_by_qubit
                .entry(entry.1.qubit_id)
                .or_default()
                .insert(entry.0);
        }
        let mut boost_set: HashSet<(TripletKey, u32)> = HashSet::new();
        for &(qa, qb) in self.pairs {
            if let (Some(keys_a), Some(keys_b)) =
                (keys_by_qubit.get(&qa), keys_by_qubit.get(&qb))
            {
                for key in keys_a.intersection(keys_b) {
                    boost_set.insert((*key, qa));
                    boost_set.insert((*key, qb));
                }
            }
        }
        if !boost_set.is_empty() {
            for entry in selected.iter_mut() {
                if boost_set.contains(&(entry.0, entry.1.qubit_id)) {
                    entry.1.score += 1;
                }
            }
        }
    }
}
```

**Implementer note:** make `ScoredTriple` and any needed local types `pub(crate)` in `heuristic.rs` so the new module can reference them, or move `ScoredTriple` into `cz_coordination.rs` if that reads cleaner. The `1`/`3` values are the actual constants from `heuristic.rs:381`/`:469` — verify they're unchanged when you implement, but they are not placeholders.

- [ ] **Step 3: Declare the module**

In `crates/bloqade-lanes-search/src/generators/mod.rs` add `mod cz_coordination;` (or `pub(crate) mod` if other generators will reuse it).

- [ ] **Step 4: Select the policy once at the top of `generate`**

Near the start of `HeuristicGenerator::generate`, derive the policy from `ctx.cz_pairs`:

```rust
        let coordination: Box<dyn CzCoordination> = match ctx.cz_pairs {
            Some(pairs) => Box::new(EntanglingCoordination { pairs }),
            None => Box::new(FixedTargetCoordination),
        };
```

(Import the trait/impls: `use crate::generators::cz_coordination::{CzCoordination, EntanglingCoordination, FixedTargetCoordination};`.)

- [ ] **Step 5: Replace the three inline branches with policy calls**

- At the contested-penalty site (`:376-382`): keep the `score > 0 && dst_enc != target_enc && contested.contains(&dst_enc)` guard but drop the `&& ctx.cz_pairs.is_some()` clause, and subtract the policy value instead of the literal `1`:

```rust
                if score > 0 && dst_enc != target_enc && contested.contains(&dst_enc) {
                    score -= coordination.contested_penalty();
                }
```

  (For `FixedTargetCoordination` the penalty is `0`, so the guard still runs but subtracts nothing — preserving legacy behavior.)

- At the fallback-width site (`:469`): replace the literal selection

```rust
            let keep = if ctx.cz_pairs.is_some() { 3 } else { 1 };
            selected.truncate(keep);
```

  with:

```rust
            selected.truncate(coordination.fallback_width());
```

- At Step 3c (`:523-554`): replace the entire `if let Some(pairs) = ctx.cz_pairs { ... }` block with `coordination.boost_coordinated_pairs(&mut selected);`.

After each replacement, run `cargo build -p bloqade-lanes-search` to keep the diff compiling incrementally.

- [ ] **Step 6: Verify the characterization tests still pass**

Run: `cargo test -p bloqade-lanes-search heuristic` — Expected: PASS, identical to Step 1. If any golden assertion changes, the refactor altered behavior — reconcile the penalty/width constants until the tests match.
Run: `cargo test -p bloqade-lanes-search` — Expected: PASS (full suite, including placement strategy tests that drive the entangling mode).
Run: `cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings` — Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add crates/bloqade-lanes-search/src/generators/cz_coordination.rs \
        crates/bloqade-lanes-search/src/generators/mod.rs \
        crates/bloqade-lanes-search/src/generators/heuristic.rs
git commit -m "refactor(search): make HeuristicGenerator CZ-coordination mode explicit

Replace the implicit ctx.cz_pairs.is_some() branch (contested penalty,
fallback width, pair boosting) with a CzCoordination trait: a default
FixedTargetCoordination and an EntanglingCoordination impl. Behavior is
pinned by characterization tests added before the refactor.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Retire the legacy A* path and extract `SearchResult` (Arch-2)

**Why:** `drivers/astar.rs` holds the live, public `SearchResult` type entangled with a dead legacy scaffold: the `pub(crate)` `Expander` trait, the `astar()` shim, and `frontier::run_search_legacy` — all `#[allow(dead_code)]` and exercised only by in-crate tests. Extract the live type to its own module, port the worthwhile invariant tests to the v2 `run_search` API, then delete the legacy trio.

> **Two phases. Phase A (extract `SearchResult`) is behavior-neutral and independently shippable. Phase B (port tests + delete legacy) must land atomically — deleting `run_search_legacy` breaks its tests.**

**Files:**
- Create: `crates/bloqade-lanes-search/src/drivers/result.rs`
- Modify: `crates/bloqade-lanes-search/src/drivers/mod.rs`
- Modify: `crates/bloqade-lanes-search/src/drivers/frontier.rs`, `crates/bloqade-lanes-search/src/drivers/entropy.rs`, `crates/bloqade-lanes-search/src/search/restarts.rs`, `crates/bloqade-lanes-search/src/lib.rs`, `crates/bloqade-lanes-search/src/generators/mod.rs`
- Delete: `crates/bloqade-lanes-search/src/drivers/astar.rs`

### Phase A — extract `SearchResult`

- [ ] **Step 1: Create `drivers/result.rs` with the live type**

Move the `SearchResult` struct (`astar.rs:29-38`) and its `impl SearchResult { fn solution_path(...) }` (`astar.rs:40-55`) into a new file `crates/bloqade-lanes-search/src/drivers/result.rs`. Carry over the needed imports (`Config`, `MoveSet`, `NodeId`, `SearchGraph` from `crate::primitives::graph`/`config` — copy the exact `use` lines from `astar.rs`).

- [ ] **Step 2: Declare the module and update `astar.rs` to re-export temporarily**

In `drivers/mod.rs` add `pub mod result;` (keep `pub mod astar;` for now). In `astar.rs`, replace the moved definitions with `pub(crate) use crate::drivers::result::SearchResult;` so existing `drivers::astar::SearchResult` paths keep resolving during Phase A.

- [ ] **Step 3: Repoint the four `SearchResult` import sites**

Update to `crate::drivers::result::SearchResult`:
- `lib.rs:41` → `pub use drivers::result::SearchResult;`
- `drivers/entropy.rs:18`
- `drivers/frontier.rs:13` (split the import: `SearchResult` from `result`, keep `Expander` from `astar` for now)
- `search/restarts.rs:16`

- [ ] **Step 4: Build, test, commit Phase A**

Run: `cargo test -p bloqade-lanes-search` — Expected: PASS.
Run: `cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings` — Expected: clean.

```bash
git add -A
git commit -m "refactor(search): extract SearchResult into drivers::result

Move the live SearchResult type out of drivers::astar (which otherwise
holds only the dead legacy A* scaffold) into its own module. Behavior-
neutral; astar.rs temporarily re-exports it for the legacy test path.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Phase B — port tests and delete legacy

- [ ] **Step 5: Inventory legacy tests and classify port-vs-drop**

Run: `grep -n "#\[test\]" crates/bloqade-lanes-search/src/drivers/astar.rs crates/bloqade-lanes-search/src/drivers/frontier.rs`
The legacy `Expander`/`run_search_legacy` tests live in `astar.rs:69-292` and `frontier.rs:~812-1104`. Classify each:
- **Must-port** (unique invariant coverage): `nonuniform_cost_finds_cheapest`, `closed_set_prevents_reexpansion`, `transposition_and_closed_set_interaction`, `multi_step_optimal`, `max_expansions_respected`, `root_is_goal`, `no_path_disconnected`, and the BFS/DFS/IDS equivalence tests.
- **Drop-as-redundant**: any whose invariant is already covered by a v2 `run_search` integration test in `generators/heuristic.rs`, `generators/exhaustive.rs`, or `primitives/distance.rs`. **Log each dropped test and its rationale in the commit body** — no silent deletion.

- [ ] **Step 6: Build a v2 test fixture to replace the `Expander` test doubles**

The v2 `run_search` (`frontier.rs:610`) takes `MoveGenerator` + `CandidateScorer` + `CostFn` + `Goal` + `Heuristic` instead of an `Expander`. In a new `#[cfg(test)] mod` (in `frontier.rs` or a `tests/` integration file), build minimal trait doubles replacing `LineExpander`/`TwoPathExpander`/`DiamondExpander`: a tiny `MoveGenerator` enumerating successor configs along a line/diamond, `UniformCost` (or a custom `CostFn` for the nonuniform test), a site-reached `Goal`, and a manhattan `Heuristic`. Read the existing `LineExpander` (`frontier.rs:774`) to replicate its successor logic exactly.

- [ ] **Step 7: Port the must-port tests against `run_search`**

Re-express each must-port test using the v2 fixture + the appropriate `Frontier` (`PriorityFrontier::astar`, `Bfs`, `Dfs`, `IdsFrontier`), asserting the same invariant (goal found / optimal cost / expansion bound / no re-expansion). Run them incrementally: `cargo test -p bloqade-lanes-search <ported_test_name>`.

- [ ] **Step 8: Delete the legacy trio**

- Delete `crates/bloqade-lanes-search/src/drivers/astar.rs`; remove `pub mod astar;` from `drivers/mod.rs`.
- Delete `run_search_legacy` (`frontier.rs:70`) and the now-orphaned legacy test block in `frontier.rs`.
- Remove `Expander` from `frontier.rs:13`'s import; remove the `Expander` trait definition.
- Fix stale prose: the `drivers/mod.rs` doc comment (`:19-21`) and `generators/mod.rs:2` (`"[Expander]-based implementations"`).

- [ ] **Step 9: Verify nothing legacy remains and the suite is green**

Run: `rg "Expander|run_search_legacy|drivers::astar" crates/bloqade-lanes-search/src/` — Expected: no hits.
Run: `cargo test -p bloqade-lanes-search` — Expected: PASS (ported invariant tests included).
Run: `cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings` — Expected: clean (no remaining `#[allow(dead_code)]` for these items; remove the attributes too).

- [ ] **Step 10: Commit Phase B**

```bash
git add -A
git commit -m "refactor(search): remove legacy Expander/astar/run_search_legacy path

Port the unique A* invariant tests (closed-set, transposition, nonuniform
cost, strategy equivalence) to the v2 run_search trait API, then delete the
dead Expander trait, astar() shim, run_search_legacy loop, and astar.rs.

Dropped as redundant (covered by existing v2 integration tests):
<list each dropped test + the v2 test that covers it>

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final Verification (after all tasks)

- [ ] Run the full workspace build: `cargo build --workspace`
- [ ] Run all Rust tests: `cargo test -p bloqade-lanes-search -p bloqade-lanes-bytecode-python -p bloqade-lanes-bytecode-cli`
- [ ] Run the Python test suite (the python bindings consume this crate): `just test-python`
- [ ] Lint: `cargo clippy -p bloqade-lanes-search --all-targets -- -D warnings` and `cargo fmt --all --check`
- [ ] Confirm the dead/deprecated items called out in the review but *not* scoped here remain documented: `HopDistanceHeuristic::estimate`, `LaneIndex::triplets`, `MisplacedHeuristic` (out of scope — leave as-is unless a follow-up task is added).

## Notes on scope boundaries

- The `u8-discriminant triplet round-trip` emerging pattern (heuristic.rs:396/504/567) was flagged "ready to abstract" but **not** decided in this review cycle — it is intentionally out of scope. Revisit separately.
- Task 6 keeps the wide `lib.rs` facade (only fixes drift + removes the shim). Full encapsulation (making modules `pub(crate)` and migrating all 26 deep-path consumer imports) was explicitly declined.
- Tasks are independent; they can be executed in any order, but the listed order minimizes merge friction (mechanical fixes first, structural last).
