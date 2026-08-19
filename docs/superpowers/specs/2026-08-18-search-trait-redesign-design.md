# Search-crate interface redesign — working design (DRAFT)

**Date:** 2026-08-18
**Status:** DRAFT / in-progress. Converged in design discussion; **not yet
reviewed or approved.** Pairs with the evidence audit
[`2026-08-18-search-trait-inventory.md`](2026-08-18-search-trait-inventory.md);
read that first for the current-state citations this note builds on.
**Scope:** the target trait *shape* for `bloqade-lanes-search`. No code written;
no migration plan yet. Open questions are collected at the end.

This note is the "target interfaces" half of the interface-first, incremental
refactor. It does not yet say how to move implementations behind these seams.

---

## 1. The three layers

The crate conflates three levels of abstraction. The redesign separates them:

```
CzPlacement / StagePlacement   "given CZ pairs at a stage, decide WHERE atoms go, then route"   ← placement
      │  builds a Goal (+ heuristic guidance), orchestrates routing, picks best
      ▼
TargetSolver                   "route from a start config toward a goal, within a budget"        ← routing solver
      │  two independent realizations: search-based, and push-and-rotate (rule-based)
      ▼
search(gen, cost, bound, goal, traversal)   "move atoms until this Goal holds, min cost"         ← routing search core
```

The **seam between placement and routing is `Goal`**: placement's entire product
is a goal; routing's entire job is to reach one. The **routing solver contract is
independent of the search core**, which is what lets a non-search (push-and-rotate)
be a first-class peer of the search engine.

---

## 2. Tier 1 — generic, graph-agnostic search core

These speak only in `Config` (opaque state), `MoveSet` (opaque transition),
`f64`, `bool`, `NodeId`. No `LaneAddr`, no `LaneIndex`, no target placements.
Concrete implementors are arch-specific; the traits are not.

```rust
// The quantity minimized: g.
trait CostModel: Sync {
    fn edge_cost(&self, mv: &MoveSet, from: &Config, to: &Config) -> f64;
    fn id(&self) -> ObjectiveId;                 // instance identity, for bound pairing
}

// Successor function. Arch/problem data is captured in `&self` at construction
// (NOT threaded in via a lane-index-bearing SearchContext), keeping this generic.
trait MoveGenerator {
    fn generate(&self, state: &Config, out: &mut Vec<MoveCandidate>);
}
// MoveCandidate { move_set: MoveSet, new_config: Config } is the successor unit
// (edge + already-applied state). Its natural home is with this trait.

// Termination test. Pure predicate.
trait Goal { fn is_goal(&self, state: &Config) -> bool; }

// Order-only signal. NO contract: may be inadmissible / weighted / perturbed.
trait Heuristic { fn estimate(&self, state: &Config) -> f64; }
impl<F: Fn(&Config) -> f64> Heuristic for F {}

// Admissible, prune-capable lower bound. Tied to a CostModel; still config -> f64.
trait CompletionBound: Sync {
    type Obj: CostModel;                         // what it is admissible *w.r.t.*
    const TRIVIAL: bool = false;                 // NoBound<O> => prune compiles away
    fn objective_id(&self) -> ObjectiveId;
    fn estimate(&self, state: &Config) -> f64;   // promise: <= true cost-to-go under Obj
    fn as_heuristic(&self) -> impl Heuristic + Copy + '_ { move |s| self.estimate(s) } // one-way weaken
}
```

### Tier 2 seam — traversal / open-list discipline (generic)

```rust
trait Traversal {
    // select-next, receive-children, goal-check timing, prune-gate location ...
    fn best_reached(&self) -> Option<NodeId>;   // overridable; a default is supplied by the loop
}
// Frontier family (materialized open list): PriorityFrontier<H> | Bfs | Dfs<H> | Ids<H>
// Entropy/B&B family (implicit single path + resume buffer + entropy counters)
```

`H` (ordering heuristic) lives *inside* the frontier variants; the entropy
variant computes ordering internally (and statefully, which is why it is not a
plug-in `Heuristic`).

#### Best-reached tracking (feeds the resumable handoff, §6)

"Best" is a **fold over nodes seen** (`best = argmin(progress metric)`) and is
**independent of ordering** — so it is *not* per-traversal logic. The shared
search loop maintains it uniformly for every frontier traversal, at **push time**
(`receive_children` sees every *generated* node — a richer pool than only
expanded ones), using a **`MeasurableGoal` shortfall** as the metric (§3). A
traversal **overrides** the default only when it already tracks something richer
natively — entropy's resume buffer is the sole such case.

- **Two distinct "bests," unified.** *Incumbent* = best complete solution (a
  goal), what branch-and-bound prunes against; exists only after a goal is found.
  *Best-reached partial* = most-progressed config (goal or not), what P&R resumes
  from. `best_reached` returns the incumbent if one exists, else the most-
  progressed partial. `Solved` is just the case where `best_reached` satisfies
  the goal — so success and budget-exceed share one contract.
- **Availability is metric-gated, not traversal-gated.** `best_reached` is
  well-defined whenever a `MeasurableGoal` shortfall is supplied; the `Option`
  reflects "no progress metric," not "which traversal." **No BFS special case.**
- **Cost.** The fold is O(1)/node and needs no post-hoc graph scan. It is *free*
  for A\*/DFS/IDS/entropy (they already evaluate the ordering signal per node) and
  costs *one extra metric eval per node* for BFS (FIFO, otherwise never touches
  it). This is a non-issue: BFS is effectively **vestigial** for our graph sizes
  (too expensive to use in practice), so its extra per-node eval never matters —
  do not shape the design around BFS performance.

### The unified driver both traversals collapse to

```rust
fn search<G, O, B, Go, T>(
    root: Config,
    generator: &G,        // MoveGenerator   \
    objective: &O,        // CostModel (g)    |  all graph-agnostic
    bound:     &B,        // CompletionBound  |  B::Obj = O; NoBound<O> = pruning off
    goal:      &Go,       // Goal             /
    traversal: &mut T,    // <-- the only thing that differs between the two engines
) -> SearchResult
where O: CostModel, B: CompletionBound<Obj = O>, G: MoveGenerator, Go: Goal, T: Traversal;
```

---

## 3. Capability sub-traits — keeping the base traits minimal

Capabilities that only *some* consumers need drop out of the base traits into
sub-traits the consumer requires. Two motivations so far:

- **Admissibility support** for `WeightedDistanceBound`: `Objective::lane_weight`
  and `Goal::exact_targets` were this, bolted onto general traits. → `LaneAdditive`,
  `PointGoal`.
- **Progress signal** for best-reached tracking / the resumable handoff (§2, §6):
  a goal that can report *how far* a config is from satisfying it, not just a
  boolean. → `MeasurableGoal`.

```rust
// The cost model's decomposition over the lane graph. Only a lane-graph bound needs it.
trait LaneAdditive: CostModel {
    fn lane_weight(&self, lane: LaneAddr) -> f64;   // config-independent per-lane floor (C3)
    fn min_shot_cost(&self) -> f64;                 // scalar floor (C4)
}

// "I am point-valued; here is my required placement." Set-valued goals don't implement it.
trait PointGoal: Goal { fn required_placement(&self) -> &[(u32, LocationId)]; }

// Progress signal for best-reached tracking (§2). 0.0 == is_goal. Keeps Goal a pure predicate.
// e.g. AllAtTarget -> unresolved-qubit count; EntanglingConstraintGoal -> unsatisfied-pair count.
trait MeasurableGoal: Goal { fn shortfall(&self, config: &Config) -> f64; }

struct WeightedDistanceBound<O: LaneAdditive> { /* ... */ }
impl<O: LaneAdditive> WeightedDistanceBound<O> {
    fn for_point_goal<G: PointGoal>(objective: &O, goal: &G, /* index, blocked */) -> Self { /* ... */ }
}
```

Effect: `LaneAddr` and target placements now appear *only* where a lane-graph
distance bound is built. The current runtime `match goal.exact_targets() { Some
=> build, None => skip }` ([restarts.rs:227-233](../../..)) becomes a
compile-time fact — a set-valued goal can't even be offered to the bound
constructor.

---

## 4. Ordering vs pruning; admissible vs inadmissible

The lowest level is really: **one cost model, one admissible-bound-relative-to-it
(prune-capable), one unconstrained ordering heuristic (order-only), with a
one-way weakening from bound → heuristic.**

- **Contract lives in the trait; enforcement is the implementor's burden.** Rust
  cannot prove `h ≤ h*`. So the design encodes what it can in three tiers:
  1. **Types** — `CompletionBound::Obj: CostModel` makes "prune with a bound
     built for a different cost model" a compile error.
  2. **Construction assert** — `ObjectiveId` (hard assert, not debug) catches
     same-type/different-*instance* mismatch (`tau=1` vs `tau=5`) the types can't
     see.
  3. **Test** — `assert_objective_contract` checks C2/C3/C4 mechanically. C1 is
     structural (untestable) — the irreducible implementor-discipline residue.
- **Ordering-h and prune-h stay distinct even when derived from the same bound.**
  You may order by an inflated/inadmissible `h` (weighted A\*, `SumHop`,
  entropy perturbation) but must prune only on an admissible one. The prune slot
  accepts only `CompletionBound`; the ordering slot takes any `Heuristic`; you
  physically cannot pass the ordering-h into the prune slot.
- **Silent failure motivates the strong enforcement.** Break admissibility and
  you don't crash — you prune away the optimum. Hence a *hard* assert for the
  instance check.

---

## 5. Frontier vs branch-and-bound — and the cascade non-pruning finding

Formally the two drivers are the **same abstract search**, differing only on two
axes: (a) open-list discipline (materialized vs implicit/backtracking) and
(b) prune-against-incumbent on/off. A\* is itself a best-first branch-and-bound
(`f = g + h` *is* the bound). What the entropy driver has that the frontier lacks
is a real admissible `h` applied early enough to cut work.

**Finding (motivates wiring `CompletionBound` into the frontier):** the Cascade
strategy's A\* refinement does **not** meaningfully prune, and prunes **zero
memory**:

- Every generated child is `graph.insert`ed and pushed onto the heap
  **unconditionally** ([frontier.rs:653](../../..), [frontier.rs:681](../../..));
  `receive_children` has no cost gate. The cost cap is checked only at **pop**
  ([frontier.rs:614-618](../../..)) — the node is already resident by then. So the
  arena and the heap hold everything; the cap saves no memory.
- The cap is **`g`-only** (`g_score >= max_c`), the degenerate `h ≡ 0` bound: it
  cuts a branch only after its whole prefix cost is paid, i.e. at the bottom.
- When the inner incumbent is already optimal, the refinement (forbidden from
  accepting the equal-cost goal) explores essentially the entire `{f ≤ C}`
  region → the observed memory blow-up.

An admissible `h` would let you soundly prune `{f = g + h ≥ C}` **at push time**,
bounding memory to `{f < C}`. That capability exists and is proven admissible on
the entropy side (`WeightedDistanceBound`, C1–C4); the frontier just doesn't
consume it. Wiring it in is the single highest-leverage unification, and it is a
*memory* win, not only a speed one.

---

## 6. Tier 2 — the `TargetSolver` contract (resumable / composable)

`TargetSolver` is the **outer routing-solver contract**, deliberately independent
of the search-core traits. Both the search engine and push-and-rotate implement
it as peers.

```rust
trait TargetSolver {
    fn route(&self, start: &Config, goal: &impl PointGoal,
             blocked: &BlockedSet, budget: Budget) -> RouteOutcome;
}

struct RouteOutcome {
    status:  Completeness,   // Solved | Partial | Unsolvable(proof)  (only P&R asserts the proof)
    plan:    Vec<MoveSet>,   // moves applied *from `start`*
    reached: Config,         // config `plan` ends at — goal config if Solved, else best partial
    cost:    f64,
    stats:   SolverStats,    // nodes_expanded / deadlocks / bound_stats — solver-specific
}
```

Two departures from today's `SolveResult` make it composable:

1. **`start` is an explicit input** (today both engines always begin at the
   original `initial`).
2. **`reached` is always the config the plan ends at — the best partial, not the
   root.** Today `SolveResult::unsolved` returns the *root* on failure, discarding
   progress. `reached` is produced by the traversal best-tracking mechanism (§2)
   — an incremental push-time fold over a `MeasurableGoal` shortfall, not a
   post-hoc graph scan.

### Push-and-rotate is a *completion stage*, not a peer strategy

The intended pattern (a refinement over today's behaviour): if a search-based
solver exhausts its budget, hand its **best reached config** to P&R to finish the
stragglers. This is just **sequential composition of `TargetSolver`s** — chain
`a.reached → b.start`, concatenate plans.

> **Current-state correction (motivating this):** today's `fallback_push_rotate`
> **restarts P&R from the original `initial`**, discarding the search's progress
> ([target_solver.rs:324-341](../../..)) — it does *not* resume from the search's
> best config. It is also a hardcoded special case, not a general composition. The
> resumable contract turns it into "chain of routers" and lets P&R complete rather
> than restart.

### Capability boundary (must respect)

- **P&R is point-goal only** (routes to fixed target vertices; the crate already
  restricts it to the fixed-target path and substitutes A\* on loose goals). So
  the **composable trait is point-goal** (`goal: &impl PointGoal`). Loose-goal
  (set-valued) routing is a *search-only* capability used by the placement layer,
  not part of the shared trait and not implemented by P&R.
- **Flavor-2 handoff needs concretization.** If a *loose-goal* search stalls, you
  cannot hand the set-goal to P&R; you must first concretize the leftover pairs'
  sub-goal to specific entangling sites (a placement-layer step at the seam).
- **Budget is not one unit** — search = node expansions, P&R = emitted moves.
  `Budget` should be solver-specific, not a shared integer.
- **Chained plans stay executable for free** — P&R starts at exactly the config
  the partial plan ends at, so concatenation chains soundly through the replay
  verifier.

Open sub-question: is the shared trait strictly point-goal, or does it advertise
a capability so a caller can ask "can you take a set goal?" at runtime?

---

## 7. Tier 3 — the placement layer (`CzPlacement` / `StagePlacement`)

Placement sits *above* routing and is not modeled by the routing traits. Its job:
turn a CZ-stage spec into a goal (+ guidance), orchestrate routing, pick best.

```rust
trait StagePlacement {                        // ~ today's CzPlacement, honest about its job
    fn place_and_route(&self, initial: &[(u32, LocationAddr)], pairs: &[(u32, u32)],
                       blocked: &[LocationAddr], budget: Option<u32>,
                       future_layers: &[Vec<(u32, u32)>]) -> SolveResult;
}
```

- The two **flavors are which `Goal` placement builds**:
  - *Flavor 1 — select target then route:* enumerate candidate placements
    (`TargetGenerator`), each a **point goal** (`AllAtTarget` = `PointGoal`);
    route each. Placement and routing sequential/separable.
  - *Flavor 2 — loose goal:* one **set-valued goal** (`EntanglingConstraintGoal`)
    + Hungarian guidance; route once. Placement folded into routing via the
    goal's slack.
- **Composable pieces belong here, not in routing:** `TargetGenerator`
  (candidate placements), the goal-builder (point vs set = the flavor),
  routing-heuristic construction (Hungarian / `PairDistanceHeuristic` guidance),
  and multi-layer orchestration (receding-horizon `future_cz_layers` beam
  rollout).
- The trait needs the `future_layers` argument the current `CzPlacement::solve`
  can't express (which is why each impl has a richer inherent `solve_pairs`).

### The leak to lift (the core disentangling work)

Placement logic is currently smeared **down into the routing move generators**:

- `HeuristicGenerator::generate` branches on `ctx.cz_pairs.is_some()` to pick a
  `CzCoordination` policy ([generators/heuristic.rs:358](../../..)) — the routing
  generator changes behaviour by placement flavor.
- `LooseTargetGenerator` wraps `HeuristicGenerator` to inject loose-goal target
  assignment ([generators/loose_target.rs:6](../../..)).
- `SearchContext.cz_pairs` exists only to carry that signal down.

Disentangling means lifting pair-coordination and target-assignment **up** into
the placement layer (which builds the goal + guidance), so the routing generator
becomes goal-/pair-agnostic and `SearchContext.cz_pairs` disappears.

---

## 8. Open questions / next steps

1. **Best-partial extraction — mechanism now designed (§2), residual checks.**
   Resolved: it is an incremental shared-loop fold, not a post-hoc scan, so "cheap"
   is settled (O(1)/node, no scan; free wherever the ordering signal is already
   computed; BFS's extra eval is moot since BFS is vestigial). Residual to confirm:
   (a) that entropy's existing resume buffer maps cleanly onto the `best_reached`
   *override* rather than fighting the default fold; (b) that a `MeasurableGoal`
   shortfall is well-defined for the concrete goals (`AllAtTarget` →
   unresolved-qubit count; `EntanglingConstraintGoal` → unsatisfied-pair count);
   (c) whether best-reached should be tracked at push time (richer pool) given the
   metric-eval cost on generated-but-never-expanded nodes.
2. **P&R starting-state needs — RESOLVED: the handoff is just a `Config`.**
   Verified in `push_rotate/`. `plan_with` derives everything from the start
   placement + arch graph: occupancy is `occupancy(&start)`
   ([mod.rs:161-167](../../..)); the free-site/subgraph/empty-count structure is
   `Decomposition::build(graph, &occupancy(&start))` ([mod.rs:173](../../..)); the
   mutable state is `PlanState::new(graph, start)` ([mod.rs:253](../../..)). No
   carried/incremental state — P&R already runs fresh from any placement (today's
   fallback does exactly this). To resume, pass `reached` as `initial`;
   `target`/`blocked` are the routing instance, not "state." Caveats: (a) `blocked`
   must match — it is carved into the `LaneGraph`, so a different set is a different
   graph; (b) every start position must be on-graph — a location on a blocked site
   has no vertex, so `to_vertices` returns `None` and P&R reports unsolvable
   ([solver.rs:87-92](../../..)); search configs satisfy this except the
   pathological root-on-blocked case; (c) the ≥2-empties-per-moving-component
   regime gate ([mod.rs:179-186](../../..)) applies, but it is config-derived and
   empty counts are conserved under routing, so resuming adds no new regime risk.
3. **P&R fix-up quality — empirical, with a hypothesis to test.** Does resuming
   P&R from the search's best partial beat (a) restarting P&R from `initial`
   (today's behavior) and (b) letting the search continue?
   - **Expected default:** resume ≥ restart essentially always — it keeps the
     search's tighter prefix, and P&R's move count scales with distance-to-goal,
     so a more-solved start ⇒ shorter completion. Default to resume-from-partial.
   - **Risk 1 — metric mismatch:** `best_reached` picks min-`shortfall`, but P&R's
     cost-to-complete is *not* monotone in unresolved count (a few atoms in a
     congested region can cost more swaps than many in open space), so the search's
     "best" partial may not be P&R's cheapest resume point.
   - **Risk 2 — un-reoptimized seam:** `[search MoveSet layers] + [P&R batches]`
     come from two solvers with no cross-solver batching, so the boundary may add
     an AOD operation a unified plan would have merged. Mitigable by re-running
     `schedule`/`smooth` over the concatenated move list.
   - **Lever A — top-k handoff:** try P&R completion from the *k* best partials
     (buffer-backed `best_reached`, §2) and keep the cheapest — removes the bet
     that `shortfall` predicts completion cost (defuses Risk 1).
   - **Lever B — race, don't choose:** P&R is ~2 orders of magnitude faster
     ([options.rs `Strategy::PushRotate` doc](../../..)), so run P&R-from-best
     *and* continue the search, take the winner. The policy (single-best / top-k /
     race) lives at the **placement layer**, not in the routing contract — the
     contract only has to expose `best_reached` (or a best-set) and accept an
     arbitrary start.
   - **Where to measure:** which benchmark cases actually hit budget-exceed today,
     and how far from goal the best partial typically is — if budget-exceed configs
     are usually "almost solved," resume is a clear win; if "barely started,"
     resume ≈ restart.
4. **Shared trait goal-generality.** Strictly `PointGoal`, or capability-advertised?
5. **Other bound consumers.** Is `WeightedDistanceBound` the only thing needing
   `LaneAdditive`/`PointGoal`, or would a future zone-bus / phase bound want a
   different decomposition (affecting how those capability traits are shaped)?
6. **`Traversal` unification.** Can one trait host both a materialized `Frontier`
   and entropy's path+resume model, or do they stay two families over one shared
   Tier-1 core? (`IdsFrontier` is evidence a backtracking search already fits the
   `Frontier` shape.)

Safety net for whatever migration follows: the deterministic benchmark baselines
(`python/benchmarks/harness/latest_physical.csv` / `latest_logical.csv`,
zero-diff CI gate) plus the search-crate tests now run by `just test-rust` /
`just lint`.
