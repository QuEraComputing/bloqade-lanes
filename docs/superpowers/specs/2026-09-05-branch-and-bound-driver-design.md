# Branch and bound driver: a simpler search loop over the existing bound

**Status:** draft for review — branch `phil/branch-and-bound-search`, worktree
`.claude/worktrees/branch-and-bound`, based on `origin/main` at `b63334f5`.
**Date:** 2026-09-05
**Author:** Phillip Weinberg (with Claude)

## Summary

Add a third driver, `drivers::branch_and_bound::run_branch_and_bound`, beside
`frontier::run_search` and `entropy::entropy_search`. It is a textbook
branch and bound over the crate's existing pieces: the `SearchGraph` arena as
the state graph, `MoveGenerator` as the branching rule, `Objective` as `g`,
`CompletionBound` as `h`, and `Frontier` as the node-selection order. The
loop holds a mutable incumbent `C`, prunes every node with `g + h ≥ C`, and
runs until the frontier is empty or the expansion budget is spent.

It is **not** a modification of the entropy driver. The entropy driver's
control flow — per-node entropy counters, `reversion_steps` ancestry walks,
entropy-keyed candidate regeneration, and the scored resume buffer — predates
the completion bound and exists to escape plateaus without one. With an
admissible bound available, the hypothesis this driver tests is that ordinary
backtracking plus `g + h ≥ C` pruning finds plans at least as good on the
benchmark suite, with a driver an order of magnitude smaller and a cost model
that is a real `Objective` rather than a shot count with a duration tiebreak.

The reference configuration for evaluation is `WeightedDuration` as the
objective and `WeightedDistanceBound` (`h0`) as the bound, with the entropy
driver's own `generate_candidates` as the branching rule so the comparison
isolates the search loop.

## Why a separate driver

The August work (`2026-08-10-objective-completion-bound-design.md`) upgraded
the entropy driver's incumbent cut from `g ≥ C` to `g + h ≥ C`. It landed
soundly, but its measurements point at the control flow rather than the bound
as the remaining problem:

- `cuts_by_g` is zero on every measured case; `h0` does all the pruning. The
  bound is pulling its weight.
- Termination was deliberately left unchanged: a pruned root with an empty
  resume buffer does not end the search, so once ties are pruned the driver
  runs to its budget looking for strictly cheaper goals. The doc records the
  consequence — node counts are not monotone in the bound, and `adder_64`
  costs 4× wall time for 8 fewer moves.
- The bound is tested at three gates (resume, child, and inside
  `resume_buffer_pop_best`) with a per-node dedup bitmap so `BoundStats`
  counts each cut once. A driver with one open structure has one gate.

Separately, the entropy driver's `g` under the production `UniformCost` is
exactly depth, so its "objective" is lexicographic `(shot count, approximate
path time, deterministic tiebreaks)` and duration never enters pruning. The
`Objective` trait and `WeightedDuration` exist and are contract-tested but no
production path accumulates them. A driver whose incumbent comparison is the
objective, full stop, is the missing consumer.

`run_search` is the closer starting point — see §5 — but it returns on the
first goal and its `max_cost` is a static caller-supplied cap, so it is a
one-shot bounded search, not a B&B. The cascade (`Strategy::Cascade`) uses
exactly that: inner phase finds an incumbent, A\* refines under it. B&B folds
both phases into one loop.

## The algorithm

Notation: `g(n)` is the accumulated `Objective::edge_cost` along `n`'s parent
chain (stored on the node by `SearchGraph::insert`), `h(n)` is
`CompletionBound::estimate(config(n))`, and `C` is the incumbent cost,
`+∞` until a goal is found.

```
graph ← SearchGraph::new(root); C ← seed_incumbent or +∞; best ← None
frontier.receive_children([root])
while let Some(n) = frontier.select_next():
    if budget spent: break
    if closed[n] or graph.seen_id(config(n)) != n: continue      # expanded, or superseded
    closed[n] ← true
    if g(n) + h(n) ≥ C: record cut; continue                    # pop gate
    expand: generator.generate(config(n)) → candidates, already scored
    for each candidate in score order:
        g' ← g(n) + objective.edge_cost(shot)
        (c, is_new) ← graph.insert(n, shot, config', g')
        if !is_new: continue                                     # transposition at ≤ g'
        if goal.is_goal(config(c)):
            if g' < C: C ← g'; best ← c; observer.goal_found
            continue                                             # never expand a goal
        if g' + h(c) ≥ C: record cut; continue                   # push gate
        children.push(c)
    frontier.receive_children(children)
return SearchResult { goal: best, bound_stats, … }
```

Properties, each of which the tests in §7 pin:

- **Soundness of pruning.** A node is dropped only when `g + h ≥ C` with `h`
  admissible for the objective `g` accumulates, or when `h = +∞`. The
  objective/bound pairing is asserted at entry via `ObjectiveId`, exactly as
  `entropy_search_with_tables` does. Ties are cut: an equal-cost completion
  adds nothing once one is held.
- **Anytime.** `best` only improves. Every goal reported has `g < C` at the
  moment it is found, so the sequence of incumbents is strictly decreasing.
- **Termination is a proof.** If the loop exits because the frontier is
  empty (not the budget), `best` is optimal *over the generator's branching*.
  This is the termination the entropy driver declined; here it is the point.
  The optimality claim is qualified because `generate_candidates` truncates
  (`max_movesets_per_group`, positive-score filter): the bound is admissible
  for the true problem, but the search tree is a subgraph of it. §8 returns
  to this.
- **Transposition handling.** `SearchGraph::insert` already rejects a
  re-discovery at `≥ g` and, on a cheaper re-discovery, mints a new `NodeId`
  and repoints the table (lazy deletion). The superseded-node skip at pop
  (`graph.seen_id(config(n)) != n`) closes the gap that leaves: a stale id
  still on the frontier is never expanded at its worse `g`. `run_search`
  lacks this and relies on `closed` alone, which is correct for A\* with a
  consistent `h` but wastes expansions for DFS.
- **`h` is memoized per node** in a dense `Vec<Option<f64>>` indexed by
  `NodeId`, lifted from `entropy.rs::bound_estimate` unchanged. Under
  `NoBound` (`B::TRIVIAL`) the memo stays empty and the gates fold to
  `g ≥ C` at monomorphization, so the bound-disabled driver is plain DFS
  with an incumbent, not a driver with a disabled feature.
- **Goals are never expanded.** A goal's children are all costlier than the
  goal (C2, C4), so expanding one can only find worse goals. `run_search`
  with `check_goal_on_generate` behaves the same way by returning.

## Variants are frontiers

B&B is three orthogonal choices — branching, bounding, and node order — and
`Frontier` already abstracts the third. The driver is generic over it, and
the variants worth evaluating are existing or small frontiers:

| Variant | Frontier | Memory | Anytime | Notes |
|---|---|---|---|---|
| DFS B&B | `DfsFrontier<H>` | `O(depth × b)` | yes | The classic. First incumbent after one greedy dive; quality of that dive dominates total work. **First to evaluate.** |
| Greedy B&B | `IdsFrontier<H>` | open list | yes | h-primary, depth tiebreak, reversal penalty. The frontier currently strongest on the suite; with an incumbent prune it becomes a legitimate B&B. Second to evaluate. |
| LDS B&B | new `LdsFrontier<H>` | `O(depth × b)` per iteration | yes | Limited Discrepancy Search (Harvey & Ginsberg 1995; Korf's ILDS 1996): explores paths in increasing count of deviations from the heuristic's first choice, so a wrong choice near the root is revisited long before DFS would reach it. Matched to "good ordering, weak bound," which is this problem. Add if DFS shows the deep-doomed-subtree failure mode. |
| Anytime weighted A\* | `PriorityFrontier` on `g + w·h`, `w ↓ 1` | open list | yes | Prune by *unweighted* `g + h ≥ C`; `bounds.rs` already codifies that separation. Memory-heavy; deferred. |
| A\* / cascade | `PriorityFrontier` on `g + h` | open list | no | Already exists as `Strategy::AStar` / `Strategy::Cascade`. An incumbent prune adds nothing to A\* proper: before the optimal goal pops every open node has `f ≤ C*`. Not re-implemented here. |
| BFS / Dijkstra | `BfsFrontier` | worst | no | Under `WeightedDuration` depth is not cost order; the correct form is `h = 0` A\*, strictly worse than the row above. Not a candidate. |

**Ordering versus bounding.** The frontier's `Heuristic` orders; the
`CompletionBound` prunes. They are different objects on purpose: the
informative signal (the moveset score, or `h_sum`) is inadmissible, and the
admissible `h0` is a max over atoms and therefore plateau-heavy. Feeding the
bound to the frontier as its heuristic (`CompletionBound::as_heuristic`) is
allowed and is one of the configurations to measure, but the default ordering
for DFS is the generator's own score order, which the driver preserves by
pushing children in the order the generator returned them.

## What is reused, what changes

Reused unchanged: `SearchGraph`, `Config`, `MoveSet`, `LaneIndex`,
`SearchContext`, `Objective`/`CostFn` (`UniformCost`, `WeightedDuration`),
`CompletionBound` (`NoBound`, `MaxBound`, `WeightedDistanceBound`),
`BoundStats`, `assert_objective_contract`, `Frontier` and its three
implementations, `SearchResult` and therefore `extract`, `pick_best`, the
verify replay, and the Python result bindings. `SearchEvent::NodeExpanded`
and `SearchEvent::GoalFound` are the only observer events the driver emits;
`GoalFound` fires once per incumbent improvement.

Lifted from `entropy.rs` (moved to a shared location, not duplicated):
`Cut`, `bound_estimate`, `classify_cut`, `record_cut`. They depend only on
`SearchGraph`, `NodeId`, and `CompletionBound`. The entropy driver keeps
calling them from their new home; its behavior does not change.

Changes to shared types, each small and behavior-preserving for existing
callers:

1. **`MoveCandidate` gains `score: f64`.** `run_search` currently sorts by
   calling `CandidateScorer::score` inside the comparator, so each candidate
   is scored `O(log b)` times. `generate_candidates` already carries the
   score on `CandidateEntry`; `EntropyGenerator` copies it across, and
   `HeuristicGenerator` fills it from its own ranking. `run_search` is
   switched to sort by the field. The `CandidateScorer` trait stays for
   callers that re-rank.
2. **`EntropyGenerator` gains `tables: Option<&HeuristicTables>`** (or an
   `Arc`, decided at implementation by whether the lifetime reaches
   `run_with_components` cleanly). Today it passes `None` and recomputes
   blended distances per call, a handicap the entropy driver does not have.
   This is the one change without which the comparison in §7 is unfair.
3. **`Strategy::BranchAndBound { frontier: BnbFrontier }`** with
   `enum BnbFrontier { Dfs, Ids }` (LDS added when it exists).
4. **`restarts.rs`** grows an arm that threads the objective and the
   completion bound through to the new driver. The frontier arm today
   hardcodes `&UniformCost`, `&DistanceScorer`, and `max_cost: None`; the
   `objective` and `completion_bound` locals are already computed there and
   unused by it. The objective becomes selectable
   (`EntropyOptions::objective: ObjectiveKind { Uniform, WeightedDuration { tau } }`),
   defaulting to `Uniform` so every existing path is bit-identical. `tau`
   has no shipped default; the reference configuration passes
   `fastest_lane_duration_us()`, the choice `blended_distance` already makes
   for normalizing time.
5. **Parallel incumbent sharing (optional, second step).** Restarts already
   run under rayon with perturbed seeds. A shared
   `AtomicU64` holding the incumbent's `f64` bits, read at each gate and
   updated with a CAS-min on each goal, turns independent restarts into a
   parallel B&B where every worker prunes against the global best. It is a
   few lines inside the driver (an `Option<&AtomicU64>` parameter) and the
   restart dispatch. Deferred until the single-threaded driver has numbers,
   because it makes results depend on scheduling and complicates the
   monotonicity tests.

Not reused and not grown: `SearchState.entropy_map`. It exists so
`EntropyGenerator` can read a per-node entropy, which under this driver is
always the default. The driver's only per-node mutable state is the `h` memo
and the `closed` bitmap.

## Signature

```rust
pub fn run_branch_and_bound<G, O, B, Go, F, Ob>(
    root: Config,
    generator: &G,
    objective: &O,
    bound: &B,
    goal: &Go,
    frontier: &mut F,
    ctx: &SearchContext,
    state: &mut SearchState,
    observer: &mut Ob,
    max_expansions: Option<u32>,
    seed_incumbent: Option<f64>,
) -> SearchResult
where
    G: MoveGenerator,
    O: Objective,
    B: CompletionBound<Obj = O>,
    Go: Goal,
    F: Frontier,
    Ob: SearchObserver;
```

Differences from `run_search`'s signature and why: the `CostFn` is an
`Objective` so the pairing assertion has an `ObjectiveId` to check and
`min_shot_cost` is available to `record_cut`; there is no `CandidateScorer`
because ordering is the generator's (change 1 above); there is no
`max_depth` because a depth cap is not a cost cap under a non-uniform
objective (the comment above `reaches_cost_cap` in `frontier.rs` already
makes this argument); `seed_incumbent` replaces `max_cost` and is what lets a
cascade-style caller hand in a prior solution's cost.

## Testing

Unit tests in `drivers/branch_and_bound.rs`, on the example arch:

- **Optimality against brute force.** Tiny instances (1–3 atoms, small
  grids) under both `UniformCost` and `WeightedDuration`, with
  `ExhaustiveGenerator` as the branching rule so the generator is not
  truncating. Run to frontier exhaustion; the cost must equal a
  Dijkstra/BFS optimum computed independently. This is the `h0 ≤ optimum`
  test from `bounds.rs` turned around: the driver must *reach* the optimum,
  not just never cut below it.
- **`NoBound` collapses to DFS with an incumbent.** Same instance, `NoBound`
  vs `WeightedDistanceBound`: same final cost, `bound_stats` all-zero in the
  first case and `cuts_by_h > 0` in the second.
- **Incumbent monotonicity.** An observer collecting `GoalFound` events sees
  strictly decreasing `g`.
- **Superseded-node skip.** Construct a case where a config is re-discovered
  cheaper while its first id is on the stack; assert `nodes_expanded` does
  not count the stale id.
- **Pairing assertion.** Bound built against `WeightedDuration { tau: 1 }`,
  driver run with `tau: 5` → panics at entry.
- **Frontier exhaustion vs budget.** `SearchResult` must distinguish "proved
  optimal" from "ran out." Today it cannot: `goal: Some` with
  `nodes_expanded < budget` is only circumstantial. Add
  `BoundStats::frontier_exhausted: bool` (or a `SearchResult` field; decided
  at implementation) and pin it.

Property test, same shape as `h0_never_exceeds_brute_force_optimum_on_randomized_instances`:
randomized small instances, `ExhaustiveGenerator`, DFS and IDS frontiers,
both objectives, cost equals brute force whenever the frontier exhausts.

## Evaluation

The question is whether the simpler loop is at least as good, so the
comparison is against the bounded entropy driver (`rust_entropy_5_bounded` on
the logical suite), not against unbounded strategies. Same generator
(`generate_candidates` with tables), same bound, same budget, same restart
count.

Configurations: `{Dfs, Ids} × {UniformCost, WeightedDuration(τ = fastest lane)}`,
ordering by generator score, plus one run of Dfs ordered by `h0` via
`as_heuristic` to measure how much the inadmissible ordering signal is worth.

Reported per case: plan cost under the run's objective, shot count and
approximate duration (so `UniformCost` and `WeightedDuration` runs are
comparable on both axes), `nodes_expanded`, wall time, `BoundStats`
(`cuts_by_h`, depth ratio, `optimality_gap`), and whether the frontier
exhausted. The doc from August warns against asserting anytime dominance in
node counts; the criterion here is plan cost at equal budget, with node
count and wall time as secondary.

A `benches/branch_and_bound.rs` mirroring `benches/entropy.rs` for the
per-expansion cost, since the claim includes "faster" and a 4.5k-line driver
with three gates and observer payloads should lose to a 300-line one on
constant factors even before pruning differences.

## Open questions

1. **Generator truncation and the optimality claim.** `generate_candidates`
   filters to positive-score entries and truncates per triplet group. Under
   B&B that truncation is the difference between "optimal" and "optimal over
   what we looked at." Options: accept and report it (the gap statistic is
   still certified because `h(root)` is generator-independent); expose
   `max_movesets_per_group` as the knob it is; or add a completeness mode
   that widens the group cap when the frontier exhausts without proof. The
   first is right for the initial measurement.
2. **Which generator is the default branching rule.** `HeuristicGenerator`
   (the frontier path's) and `generate_candidates` (the entropy driver's)
   both implement `MoveGenerator`. The reference uses the latter to isolate
   the loop; the evaluation should include the former, since it has the
   `DeadlockPolicy` machinery the frontier strategies depend on and the
   entropy generator's deadlock breaker is a different mechanism.
3. **Deadlock policy under B&B.** The frontier strategies floor the policy at
   `MoveBlockers` because they have no jump-back. DFS B&B has ordinary
   backtracking, so `Skip` may be viable; IDS B&B has the same jump-back IDS
   does. Measure rather than decide.
4. **`tau`.** The reference uses `fastest_lane_duration_us()`. Whether that
   is the right normalization is the objective-policy decision the August
   doc deferred, and this work should not settle it by default — the
   `Uniform` default stands until a separate decision.
5. **Where `Cut`/`bound_estimate`/`classify_cut`/`record_cut` live.**
   `bounds.rs` is the natural home (they are the bound's consumer-side
   helpers), but they take a `SearchGraph`, which `bounds.rs` does not
   currently depend on. A `drivers/prune.rs` keeps the dependency direction
   as it is. Minor; decide at implementation.

## References

- `docs/superpowers/specs/2026-08-10-objective-completion-bound-design.md` —
  the bound, the `Objective` contract, and the measurements this work
  responds to.
- `docs/superpowers/specs/2026-08-10-backwards-search-findings.md` — the
  entropy asymmetry; B&B orientation should follow the same
  `backwards_search` flag.
- Harvey & Ginsberg, *Limited Discrepancy Search*, IJCAI 1995. Korf,
  *Improved Limited Discrepancy Search*, AAAI 1996.
- Dechter & Pearl, *Generalized best-first search strategies and the
  optimality of A\**, JACM 32(3), 1985 — why an incumbent prune adds nothing
  to A\* proper.
- Iterative Diving Search, arXiv:2512.13790 — the existing `IdsFrontier`.
