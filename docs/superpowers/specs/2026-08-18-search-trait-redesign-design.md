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
      │  realizations behind one polymorphic face: static search specializations + push-and-rotate
      ▼
SearchCore substrate  +  static specializations (frontier-family, entropy)                       ← routing search core
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

### Tier 2 — search as *static specializations over a shared `SearchCore`*

There is **no universal `Traversal` trait** that both engines implement, and **no
single `search()`** with a pluggable open-list. Tracing the entropy loop
([entropy.rs:2459-2854](../../..)) showed why: `Frontier`'s
`select_next`/`receive_children` bake in "pop a node → batch-generate *all* its
successors → push them to a *stored* open list," and entropy does the opposite on
three axes at once — **incremental** (one candidate per step, lazily generated),
**outcome-driven** (the loop decides descend/revert/resume from the *last*
result), and **bounded-memory** (a size-`k` resume buffer + parent chain, not a
materialized frontier). Forcing it under those two methods would push entropy's
rules into the shared loop and defeat the point.

So the model is: **each engine is a static (monomorphized) specialization of
search over a shared substrate.** What is shared is the substrate, not the loop:

- **`SearchCore` services** — the `SearchGraph` arena, node insertion, goal-check,
  the `CompletionBound` gate, and the `best_reached` fold (below).
- **The Tier-1 problem traits** (above) and the outward `SearchResult` /
  `RouteOutcome` contract (§6).

The specializations, each owning its own loop body:

- **`FrontierSearch<F: Frontier>`** — the materialized-open-list specialization.
  `Frontier` is a real, useful knob *here*: A\*/BFS/DFS/IDS differ *only* in it
  (`select_next` / `receive_children` + goal-check timing). `Frontier` is **scoped
  to this specialization** — it is *not* the cross-engine seam.
- **`EntropySearch`** — a sibling specialization, **not** a `Frontier` impl. Same
  substrate, different loop (single-path DFS + entropy backtracking + bounded
  resume buffer).
- **`PushRotate`** — a third realization with *no* search loop at all (rule-based
  router), behind the same outward contract.

Static specialization is what the type-level constraints already force anyway:
`CompletionBound` is non-object-safe (`const TRIVIAL`, RPIT `as_heuristic`),
`Heuristic + Copy` rules out `dyn`, and `NoBound::TRIVIAL` only compiles the prune
away because the loop is monomorphized. The **only** runtime-polymorphic seam is
`TargetSolver` (§6) — one coarse dispatch per solve, never inside a hot loop.

#### Best-reached — an optional field on `SearchResult` (feeds the resumable handoff, §6)

`best_reached` is **result data, not a trait method**: an optional field on the
driver's `SearchResult` (a `NodeId` in the result graph, exactly like `goal`),
which a specialization populates or leaves `None`.

```rust
struct SearchResult {
    goal: Option<NodeId>,          // the solution / incumbent (best *complete*)
    best_reached: Option<NodeId>,  // most-*progressed* node reached; None if not tracked
    graph: SearchGraph, nodes_expanded: u32, bound_stats: BoundStats, /* … */
}
```

*Computing* it is a **fold over nodes seen** (`argmin(progress metric)`),
independent of ordering, so it lives in the shared `SearchCore`: `FrontierSearch`
folds it at **push time** (as each generated node is inserted — a richer pool than
only expanded ones) using a **`MeasurableGoal` shortfall** (§3). `EntropySearch`
has a resume buffer + `found_goals` but does **not** track this today — producing
`best_reached` there is *new plumbing through its loop*, not a free read-out (an
earlier draft overstated it as "already implemented richer"). A specialization
that tracks nothing — BFS, or any run with no `MeasurableGoal` supplied — simply
leaves the field `None`. **No trait obligation, no default-impl gymnastics**; the
optionality is the whole point.

- **Two fields, two roles.** `goal` = best *complete* solution (the incumbent
  branch-and-bound prunes against); `best_reached` = most-*progressed* node (goal
  or not), what P&R resumes from. Consumers read `goal.or(best_reached)` to get
  "the config to report / resume from," so success and budget-exceed share one
  contract (§6 maps this to `RouteOutcome.reached`).
- **`None` is honest, not special-cased.** `best_reached` is `Some` whenever a
  `MeasurableGoal` shortfall was supplied and the specialization tracked it;
  otherwise `None`. BFS (or any metric-less run) leaves it `None` — the consumer
  falls back to `goal`, then root. No BFS branch anywhere in the code.
- **Cost — a real per-node eval, not free.** The fold is O(1)/node and needs no
  post-hoc graph scan, but it is *not* free: the frontier's per-node signal is the
  ordering `h` (heuristic/bound, `frontier.rs:153`), a **different function** from
  `MeasurableGoal::shortfall` (e.g. unresolved-qubit count), so computing
  `best_reached` at push time is a genuine extra evaluation per node — on the hot
  path of exactly the benchmark-gated strategies (A\*/DFS/IDS). (An earlier draft
  claimed it was free by "reusing the ordering signal"; that was wrong — the two
  metrics differ.) Mitigation: gate the fold on whether a `MeasurableGoal` was
  supplied, so runs that don't need a resumable partial pay nothing. BFS is moot
  either way — vestigial at our graph sizes.

### The shared substrate + the specializations (not one function)

```rust
// Shared, monomorphized services every specialization is written against.
struct SearchCore<'a> { /* graph arena, insert, goal-check, CompletionBound gate, best_reached fold, ctx */ }

// Specialization 1: the materialized-open-list family, parameterized by Frontier.
// This is today's `run_search`, minus the arch-bearing SearchContext.
fn frontier_search<G, O, B, Go, F>(core: &mut SearchCore, generator: &G, objective: &O,
                                   bound: &B, goal: &Go, frontier: &mut F) -> SearchResult
where O: CostModel, B: CompletionBound<Obj = O>, G: MoveGenerator, Go: Goal, F: Frontier;

// Specialization 2: entropy — its OWN loop over the same core; no Frontier.
fn entropy_search<G, O, B, Go>(core: &mut SearchCore, generator: &G, objective: &O,
                               bound: &B, goal: &Go, params: &EntropyParams) -> SearchResult
where O: CostModel, B: CompletionBound<Obj = O>, G: MoveGenerator, Go: Goal;
```

There is deliberately **no** single `search<…, T: Traversal>(…)`. The engines
share `SearchCore` and the Tier-1 traits; they do **not** share a loop body. They
are unified only at the `TargetSolver` face (§6) — the one polymorphic boundary.

### Two-tier dispatch — monomorphized fast path + a `dyn` fallback

Two requirements: (a) adding a new bound / objective / cost config should be
cheap, and (b) experimental, non-enumerated configs should be possible without
touching the fast dispatcher. Both fall out of writing each loop **once**,
generic, with two front doors.

**One generic loop.** `run<O: Objective, B: CompletionBound<Obj = O>, …>(core, obj,
bound, …)` is written a single time; everything below is *how you reach it*.

**Tier 1 (fast) — resolve one axis per match, let the compiler compose the
product.** Do not hand-write the `strategy × bound × objective` cartesian product.
Each axis gets a local match that resolves a concrete type and calls the next
stage generically (continuation- or builder-style); the compiler generates the
product of monomorphizations. Adding a bound = one arm in `with_bound`; adding an
objective = one arm in `with_objective`. (A `macro_rules!` over the axis lists can
generate the arms instead — same effect.) Every arm is fully monomorphized and
inlined. Cost is code size, bounded by the number of concrete axis values you
actually instantiate — keep the *runtime-open* axis set small (today: strategy ×
bound, with objective/scorer pinned).

**Tier 2 (flexible) — an object-safe shadow trait + wrapper, through the same
loop.** `CompletionBound` is non-object-safe (`const TRIVIAL`, RPIT
`as_heuristic`), so a boxed bound needs a shim:

```rust
// Object-safe: no const, no RPIT, no associated type.
trait DynBound: Sync { fn objective_id(&self) -> ObjectiveId; fn estimate(&self, c: &Config) -> f64; }
impl<T: CompletionBound> DynBound for T { /* blanket: every static bound is usable dynamically */ }

// Bridges dyn -> the non-object-safe generic trait.
struct ErasedBound<'a, O> { inner: &'a dyn DynBound, _o: PhantomData<O> }
impl<'a, O: Objective> CompletionBound for ErasedBound<'a, O> {
    type Obj = O;
    const TRIVIAL: bool = false;                                    // supplies the const the vtable can't
    fn objective_id(&self) -> ObjectiveId { self.inner.objective_id() }
    fn estimate(&self, c: &Config) -> f64 { self.inner.estimate(c) } // <- per-node vtable call
}
```

The experimental path is then `run::<O, ErasedBound<O>>(…)` — the **same loop**,
one extra monomorphization, with dynamic dispatch happening *inside* `estimate`.
An experimental bound impls `DynBound` (or gets it via the blanket) and is boxed;
the Tier-1 dispatcher is untouched. The same wrapper applies to any other axis you
want erasable (`ErasedObjective` over `dyn DynObjective`), so the fully-dynamic
entry is `run::<ErasedObjective, ErasedBound<ErasedObjective>>`.

**What Tier 2 trades (so keep it opt-in):** no inlining of `estimate` (a vtable
call per node); no `NoBound::TRIVIAL` compile-away; and the compile-time objective
pairing (`B::Obj = O`) degrades to the runtime `ObjectiveId` assert — which already
exists ([entropy.rs:2369](../../..)), so correctness is still checked, just later.
Gate it behind an explicit entry (`run_dynamic` / a `Strategy::Experimental` flag)
so production always takes Tier 1 and never accidentally pays the vtable cost.

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
distance bound is built.

> **Caveat (don't oversell this as free).** Making "only a `PointGoal` reaches the
> bound constructor" a *compile-time* fact is **not** a free consequence of the
> trait split. Today the bound is built at
> [restarts.rs:227-233](../../..) inside `run_with_components`, whose single
> monomorphization (`<Go, Gen, Hmax, Hsum, MkGen>` + 10 args,
> [restarts.rs:159](../../..)) is instantiated with **both** point goals
> (`AllAtTarget`) and set goals (`EntanglingConstraintGoal`, `PartialPlacementGoal`)
> flowing through the same dispatch. To enforce it statically you must either split
> that already-overloaded god-function by goal kind — multiplying its
> monomorphization breadth, which cuts against the inventory's monomorphization-cost
> constraint (#11) — or keep a runtime
> capability check. So this is a restructuring of the crate's most
> over-parameterized function, not a rename. A reasonable interim: keep the
> **runtime** `match goal.exact_targets()` gate and defer the compile-time version.

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
6. **`Traversal` unification — RESOLVED: no universal trait; static
   specializations over a shared `SearchCore` (see §2).** Tracing the entropy loop
   ([entropy.rs:2459-2854](../../..)) settled it, method-by-method vs `Frontier`:
   `check_goal_*` fits (entropy tests on generate); `best_reached` fits *as a result
   field* (though entropy would need new plumbing to populate it — it has a resume
   buffer + `found_goals` but no `best_reached` output today); but
   `select_next` / `receive_children` do **not** — they presume batch-expand into a
   *stored* open list, whereas entropy is incremental, outcome-driven, and
   bounded-memory. Resolution is **not** to widen the trait (a `step`/driver
   inversion would drag one engine's rules into a shared loop) but to share the
   **substrate** (`SearchCore` + Tier-1 + `best_reached` + `RouteOutcome`) and let
   each engine be its own monomorphized loop, unified only at `TargetSolver`.
   Correction to an earlier note: `IdsFrontier` shows *materialized* backtracking
   fits `Frontier` — it does **not** show entropy fits, because IDS keeps every
   node in its heap while entropy deliberately keeps only a bounded resume buffer.

---

## 9. Safety-net coverage & gaps (independent review, 2026-08-18)

The behaviour-preserving story leans on the deterministic benchmark baselines
(`python/benchmarks/harness/latest_physical.csv` / `latest_logical.csv`, zero-diff
CI gate) plus the search-crate tests (`just test-rust` / `just lint`). But the
coverage is **uneven — and thinnest on the paths this redesign most wants to
change:**

- **Gated (regression fails CI automatically):** the frontier strategies
  (`astar` / `bfs` / `dfs` / `ids` / `greedy`) and entropy variants
  (`entropy_{1,5,10,20}`, plus `entropy_5_bounded` on logical). Baseline columns
  include the bound stats, so **§5's frontier bound-wiring will trip the gate** —
  it is a behaviour change, not a pure refactor.
- **NOT gated (silent or unit-test-only):**
  - **Push-and-Rotate and Cascade are in *neither* baseline.** So the §6
    resumable-`TargetSolver` / P&R-completion work and the cascade leg touched by
    §5 are guarded only by the ~15 `target_solver.rs` tests, not the zero-diff
    gate. **Write dedicated resume-vs-restart / chained-plan tests before touching
    these.**
  - **Resumable-handoff *quality*** (§8.3) — no test asserts resume ≥ restart.
  - **The PyO3 adapter / string-label / dict-key ABI** — no Rust test crosses into
    Python; a `SolveResult`→`RouteOutcome` reshape is caught only if the Python
    integration tests exercise the changed getters.
  - **The `SearchEvent` / `EntropyTrace` viz transport** — the `entropy_tree`
    consumer is outside the Rust gate; a `SearchCore` extraction that perturbs
    entropy event-emission order can break the 1:1 viz contract un-gated.
- **Compiler-guarded (correctness only, no behaviour signal — fine):** trait
  renames, the capability split, dead-code removal.

## 10. Implementation sequencing (independent review, 2026-08-18)

Overall size (independent review): **L–XL, ~8–16 person-weeks, bimodal** — steps
1–5 are the cheap, safe ~40% (compiler-guarded, zero baseline move); steps 6–8
(P&R composition + placement lift) are the expensive, weakly-guarded ~60%. Order
"guarded-and-behaviour-preserving first, un-gated behavioural last."

1. **Dead-code + re-export hygiene** (§4). Delete `MaxHopHeuristic` /
   `SumHopHeuristic`; relocate `tests/public_bound_api.rs` to in-crate access (it
   exercises `run_search` / `entropy_search_*` / `MaxBound` / `WeightedDuration`,
   so those can't be demoted to `pub(crate)` until it does). Zero baseline move.
2. **Trait renames + capability split** (§3), *minus* static-`PointGoal`
   enforcement — keep the runtime `match goal.exact_targets()` gate for now.
   Compiler-guarded; add `MeasurableGoal` impls (unused yet). Zero baseline move.
3. **`SearchCore` extraction + scope `Frontier`** (§2) as pure code-motion; relies
   on frontier + entropy baselines staying bit-identical.
4. **`best_reached` as an additive, opt-in `SearchResult` field** (§2). Defaults
   `None`; no existing solve supplies a `MeasurableGoal` → zero baseline move. Do
   before step 5 (its `RouteOutcome.reached` depends on it).
5. **`RouteOutcome` / resumable `TargetSolver` as an *internal* type** (§6, part 1).
   `route()` returns best-partial, but `extract()` still maps back to today's
   `SolveResult` and the PyO3 surface is unchanged — isolates the DTO reshape from
   the ABI. Guarded by `target_solver.rs` tests.
6. **P&R-as-completion behaviour** (§6, part 2). First un-gated behavioural step —
   add resume-vs-restart + chained-plan-replay tests *before* landing.
7. **§5 frontier bound-wiring.** Explicitly a behaviour change: expect
   `nodes_explored` (and possibly which optimal-cost plan is returned) to shift on
   `astar` / `ids` / `cascade`; regenerate + inspect both baselines, confirm
   `success` unchanged. Do **not** bundle with steps 1–4.
8. **Placement lift** (§7) — last, highest behaviour risk, feeds the logical
   baseline via the loose-goal path; budget several baseline regen/inspect cycles.
9. **Two-tier `DynBound`** (§2) — optional; isolated and consumer-less (candidate
   to drop from scope until something needs a boxed experimental bound).
