# Objective / completion-bound design (B&B pruning for the entropy driver)

Status: **reviewed 2026-08-10.** **Steps 1–5 landed** (§7); step 6 (brute-force
admissibility property test) not started.

Measured outcome with the bound enabled: move counts never worse, `adder_64`
better (1540 → 1532 events), and `bv_70` / `qpe_9` / `steane_logical_5` returned
**provably optimal** plans (`h(root)` equals the incumbent). `cuts_by_g` is zero
on every case measured — the pre-existing `g >= C` test never fires, so `h0`
does all of the pruning. The optimality gap elsewhere is 0.33–0.79, so the
composable bounds left out of scope have real headroom.

Two deviations from the plan as written, both deliberate:

- **Termination is unchanged** (reviewed decision): a pruned root with an empty
  resume buffer does *not* end the search. The bound decides which branches are
  worth exploring, never when the search is over. The consequence is that once
  ties are pruned every goal after the first must be strictly cheaper, so the
  search often runs to its budget — buying better plans at more nodes. Node
  counts are therefore **not** monotone, and the §7 caveat's warning against
  asserting anytime dominance applies to node counts too.
- **`rust_entropy_5_bounded` is tracked on the logical suite only.** On
  `adder_64` the bound costs 4× wall time for 8 fewer moves, which would roughly
  double physical CI runtime; physical coverage stays ad hoc.

Goal: upgrade the entropy driver's incumbent cut from `g >= C` to
`g + h(config) >= C` with an admissible completion bound `h`, without touching
the entropy heuristic (generation, reweighting, restart perturbation) and
without hardcoding either the objective or the bound.

---

## 0. Audit of `g` (invariant 3) — read this first

The task statement assumes the driver's current objective is
`cost = (number of movesets) + (total move duration) / tau`. **It is not.**
What the driver actually accumulates and compares:

| Question | Answer | Evidence |
|---|---|---|
| What does a shot cost? | Exactly `1.0`, hardcoded | [`entropy.rs:1489`](../../../crates/bloqade-lanes-search/src/drivers/entropy.rs:1489) (`cost: 1.0` on every `CandidateEntry`) |
| How does `g` accumulate? | `g(child) = g(parent) + cost` | [`entropy.rs:2141`](../../../crates/bloqade-lanes-search/src/drivers/entropy.rs:2141); both fallbacks also add `1.0` ([1827](../../../crates/bloqade-lanes-search/src/drivers/entropy.rs:1827), [1903](../../../crates/bloqade-lanes-search/src/drivers/entropy.rs:1903)) |
| So what is `g`? | **`g(node) == depth(node)` exactly**, for every node in the entropy graph | uniform `1.0` per insert, no exceptions |
| What does the incumbent cut compare? | **`depth`, not `g`** | `best_goal_depth` ([1997](../../../crates/bloqade-lanes-search/src/drivers/entropy.rs:1997)); cuts at [2011](../../../crates/bloqade-lanes-search/src/drivers/entropy.rs:2011), [2330](../../../crates/bloqade-lanes-search/src/drivers/entropy.rs:2330), and inside `resume_buffer_pop_best` ([584](../../../crates/bloqade-lanes-search/src/drivers/entropy.rs:584)) |
| Are ties pruned? | Yes — `depth >= depth_cap`, and `best_goal_depth` updates on strict `<` | same sites |
| Where does duration enter? | Only as a **lexicographic tiebreak among equal-depth goals**, never in `g` | `select_best_goal_with_tiebreak` → `approx_path_time_us` ([599–649](../../../crates/bloqade-lanes-search/src/drivers/entropy.rs:599)) |

**Verdict for invariant 3: `g` is exact — for the objective "number of
movesets" (uniform cost 1.0 per shot).** It is not an approximation of
anything, so pruning from the `g` side is sound today. The driver's real
objective is lexicographic: `(moveset count, approximate path time, …
deterministic tiebreaks)`.

### Why this changes the plan

`cost` (= `g_score(goal)`) is load-bearing outside the driver, and both
consumers read it as a moveset count:

- `pick_best` ranks restart results by `cost`
  ([`restarts.rs:73`](../../../crates/bloqade-lanes-search/src/search/restarts.rs:73)).
- `Strategy::Cascade` converts it to a **depth** budget for the A\* refinement:
  `max_depth = inner_result.cost.ceil() as u32`
  ([`restarts.rs:256`](../../../crates/bloqade-lanes-search/src/search/restarts.rs:256)).

So making `1 + dur/tau` the *default* objective would silently change restart
selection and shrink/inflate the cascade depth bound — a behavior change with
the bound switched off, violating invariant 4, and a benchmark-baseline churn
unrelated to bounding.

**Decision (Q1, confirmed in review):** the default — and the only objective on
any production path — is the existing `UniformCost` (uniform `1.0`), so `g` is
bit-identical to today. The duration-weighted objective ships as a public type
with a **required** `tau` constructor argument, exercised only by the
swappability test that satisfies the acceptance criterion: no `EntropyOptions`
or Python plumbing, and no shipped `tau` default.

`tau` earns its own note, because **it does not exist in this repo.** The only
`tau` anywhere is `math.tau` (2π) in the gate rewrites. Promoting the weighted
objective to a default would mean inventing a time-normalization constant; the
principled arch-derived choice would be `tau = fastest_lane_duration_us()`
(dimensionless, no magic number, and the precedent `blended_distance` already
sets by normalizing time against `fastest_lane_us`). That is a deliberate
objective-policy decision for its own PR, not a side effect of adding pruning.

Pleasant consequence: under `UniformCost`, `w(lane) ≡ 1`, so `h0` degenerates to
**max over unresolved atoms of hop distance on the blocked-excluded lane
graph** — i.e. real pruning power lands with the default objective, using the
sibling of the existing `HopDistanceHeuristic::estimate_max` (tighter, because
blocked sites are excluded from the graph).

---

## 1. The objective as a first-class abstraction

`CostFn::edge_cost(move_set, from, to)` already is the driver-facing per-shot
cost hook (the frontier driver routes `g` through it; the entropy driver
bypasses it with the hardcoded `1.0`). Extend it rather than introduce a rival:

```rust
/// The quantity the search minimizes.
///
/// # Stated constraints of the bound framework
/// C1. **Per-shot additive.** `g(path) = Σ_{shot ∈ path} edge_cost(shot)`.
///     A multiplicative objective (e.g. fidelity) must be pre-transformed
///     (−log) by the impl; the framework does not do it.
/// C2. **Non-negative.** `edge_cost(..) >= 0`.
/// C3. **Lane floor.** For every shot `s` and every lane `l ∈ s`:
///     `edge_cost(s, ..) >= lane_weight(l)`.
///
/// C1–C3 are exactly what makes a weighted-distance bound admissible
/// (§6). They are the framework's contract, not incidental properties of
/// today's cost model.
pub trait Objective: CostFn + Sync {
    /// A floor on the cost of *any* shot that contains `lane`.
    /// Returning `0.0` is the explicit "I cannot certify a floor" opt-out:
    /// bounds built from it are trivial (h ≡ 0), never unsound.
    fn lane_weight(&self, lane: LaneAddr) -> f64;

    /// A positive floor on any shot's cost. Used to convert a cost budget
    /// into a depth budget (§5, §8 — this is what `Cascade` needs) and for
    /// the pruning-depth instrumentation (§7 step 5).
    fn min_shot_cost(&self) -> f64;

    /// Identity of this objective *instance*, parameters included.
    fn id(&self) -> ObjectiveId;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectiveId { pub kind: &'static str, pub params: u64 }
```

Why extend rather than define a standalone trait (Q2): a standalone `Objective`
with a blanket `impl<O: Objective> CostFn for O` fails twice over. Its only
motivation was giving `shot_cost` an `&LaneIndex` parameter, but `edge_cost`
has none to forward, so the objective must own its arch data anyway; and the
blanket impl overlaps the existing concrete `impl CostFn for UniformCost`
([`cost.rs:10`](../../../crates/bloqade-lanes-search/src/cost.rs:10)) the moment
`UniformCost` implements `Objective` — a coherence error. Extending also makes
the frontier driver's `g`, which already routes through `CostFn`, the same
source of truth for free. `Sync` is required because restarts borrow one
objective across `rayon::into_par_iter` (§7).

Concrete objectives own whatever arch data they need, so `CostFn`'s signature
stays unchanged:

```rust
// The existing UniformCost *is* the moveset-count objective. It gains the
// Objective impl under its own name — no rename, and run_frontier's existing
// `&UniformCost` becomes a valid Objective for free.
impl Objective for UniformCost { /* lane_weight = 1.0, min_shot_cost = 1.0 */ }

/// Owns an f64-per-lane weight map built from `&LaneIndex` at construction, so
/// there is no lifetime parameter to cascade into `CompletionBound::Obj`, and
/// `edge_cost` is a hash lookup rather than a per-expansion recomputation.
/// `tau` is a required argument — the framework ships no default (§0).
pub struct WeightedDuration { weights: HashMap<u64, f64>, tau: f64, min_shot: f64 }
```

`WeightedDuration::edge_cost` = `1.0 + max_{l ∈ shot} dur(l) / tau`
(max, not sum: a shot's lanes transport in parallel — this mirrors
`approx_layer_time_us`), and `lane_weight(l)` = `1.0 + dur(l) / tau`. C3 holds
because the max over a shot's lanes dominates each individual lane — including
the empty AOD *filler* lanes that `build_aod_grids` adds, which only push shot
cost up. Missing-duration lanes use `unwrap_or(1.0)` in **both** methods; the
fallback must agree or C3 breaks.

**Driver changes (step 1, behavior-identical):** `CandidateEntry.cost` becomes
`objective.edge_cost(..)`; the incumbent becomes `best_cost: Option<f64>` over
`g_score(goal)`; the three depth-cap sites and `select_best_goal_with_tiebreak`'s
primary key become cost-based. Under `UniformCost`, `g ≡ depth`, so every one
of those comparisons yields the identical decision. The duration tiebreak stays
a tiebreak (it is not folded into `g`).

## 2. Admissibility is a relationship — how the pairing is enforced

A bound is admissible *relative to* an objective. Options considered:

| Option | Catches kind mismatch | Catches param mismatch (same type, `tau=1` vs `tau=5`) | Cost |
|---|---|---|---|
| (a) Doc comment + care | no | no | free |
| (b) Bound parameterized by `&O`, weights read from the objective | no (two `WeightedDuration` instances still typecheck) | no | free |
| (c) `type Obj` associated type on the bound; driver generic over both | **yes, compile time** | no | free |
| (d) Objective is the only factory (`O::bind(ctx, spec) -> O::Bound`); driver constructs, never accepts | yes, unrepresentable | **yes** | forces bound *selection* into the driver (breaks req. 3) |
| (e) (c) + `ObjectiveId` equality assert at driver entry | yes, compile time | **yes, construction time (panic)** | one comparison per solve |

**Pick (e)** (Q3, confirmed in review). (c) alone leaves the parameter hole.
The decisive argument against (d) is *not* performance — a bound build is one
Dijkstra per unique target, comparable to `HeuristicTables`' documented "roughly
one node expansion", and `restarts` defaults to 1 — it is that **(d) forces
bound *selection* into the driver**: assembling `MaxBound(h0, cut_bound)` would
mean the driver matching on a bound-spec enum, so every future bound edits the
driver, against requirement 3. Under (e) the caller composes and the driver
never learns a bound's name.

The `ObjectiveId` check is a **hard** `assert_eq!`, not a `debug_assert`. The
neighbouring `debug_assert_tables_match`
([`entropy.rs:1125`](../../../crates/bloqade-lanes-search/src/drivers/entropy.rs:1125))
guards the same class of by-convention coupling for `w_t`, but the severities
differ: a `w_t` mismatch only perturbs *ordering* and still yields a valid plan,
whereas an objective mismatch makes *pruning unsound* — silently discarding
correct and possibly better solutions. That is not a debug-only concern, and it
costs one comparison per solve.

```rust
/// `Sync` for the same reason `Objective` is: the bound is built once per
/// solve and borrowed by every restart closure under `rayon` (§7 step 4).
pub trait CompletionBound: Sync {
    type Obj: Objective;
    /// `true` only for `NoBound`. Lets the driver monomorphize the bound
    /// test away entirely when disabled (invariant 4).
    const TRIVIAL: bool = false;

    fn objective_id(&self) -> ObjectiveId;

    /// Admissible lower bound on remaining cost. `+INFINITY` = infeasible.
    ///
    /// This value is **unweighted, always**. Ordering may scale a heuristic
    /// (`PriorityFrontier::astar(h, weight)`); pruning may not (invariant 6).
    /// There is deliberately no weight parameter anywhere in this trait.
    fn estimate(&self, config: &Config) -> f64;
}

pub struct NoBound<O>(PhantomData<O>);   // estimate ≡ 0.0, TRIVIAL = true
```

Driver entry asserts `bound.objective_id() == objective.id()` once per solve.

**Contract test, not just prose.** Ship
`assert_objective_bound_contract(&objective, &index)`: enumerate lanes and
representative shots from the arch spec, assert C2 and C3
(`edge_cost(s) >= max_{l ∈ s} lane_weight(l)`) for each. Every objective impl
gets one test line. This is what gives requirement 2 teeth — the property
admissibility rests on is checked per objective, not argued per review.

## 3. Bounds compose by max

```rust
/// Fields are **private** and construction goes through `new`. Public tuple
/// fields would let `MaxBound(a, b)` skip the `ObjectiveId` check below —
/// reintroducing, one level up, exactly the instance-level mismatch §2 exists
/// to prevent. The type parameters agree; the *instances* must be checked.
pub struct MaxBound<A, B> { a: A, b: B }

impl<A, B> MaxBound<A, B>
where A: CompletionBound, B: CompletionBound<Obj = A::Obj>
{
    pub fn new(a: A, b: B) -> Self {
        assert_eq!(a.objective_id(), b.objective_id(),
                   "composed bounds must target the same objective instance");
        Self { a, b }
    }
}

impl<A, B> CompletionBound for MaxBound<A, B>
where A: CompletionBound, B: CompletionBound<Obj = A::Obj>
{
    type Obj = A::Obj;
    fn estimate(&self, c: &Config) -> f64 { self.a.estimate(c).max(self.b.estimate(c)) }
    fn objective_id(&self) -> ObjectiveId { self.a.objective_id() }   // == b's, asserted in `new`
}
```

`B: CompletionBound<Obj = A::Obj>` makes "combined bounds must share an
objective" a compile-time fact, and `new`'s assert extends that to the instance
level. Max of admissible bounds is admissible, so the out-of-scope zone-bus-cut
and phase-decomposition bounds compose in without the driver learning anything
new. Nesting handles n > 2.

## 4. One bound, both drivers

The frontier side wants `Heuristic` (`fn estimate(&Config) -> f64`), which
already has a blanket impl for `Fn(&Config) -> f64`. A blanket
`impl<C: CompletionBound> Heuristic for C` would collide with it, so expose an
adapter instead:

```rust
pub trait CompletionBound {
    // … as above, plus a provided method (RPITIT):
    fn as_heuristic(&self) -> impl Heuristic + Copy + '_
    where
        Self: Sized,
    {
        move |c: &Config| self.estimate(c)
    }
}
```

Zero new impls, and the returned closure is `Copy`, which the
`Hmax: Heuristic + Copy` call sites in `restarts.rs` require. So
`PriorityFrontier::astar(bound.as_heuristic(), weight)` works today. Verdict on
requirement 4: **the existing `Heuristic` trait serves for consumption; a new
trait is warranted for construction/pairing** — `Heuristic` has no notion of
which objective it is admissible for, which is the whole point of
`CompletionBound`. `HopDistanceHeuristic` stays as-is (used by the Hungarian
pipeline and frontier strategies); the new bound is its blocked-excluded,
weight-generic sibling, and frontier callers may migrate later — **§8 works
through exactly what that migration absorbs and what it costs.**

## 5. Sensitivity analysis — what changes per future objective

| Future objective | What changes | What does **not** |
|---|---|---|
| (a) Uniform moveset count | Nothing — this *is* the default. `lane_weight ≡ 1`, `h0` = blocked-excluded hop max | driver, bound machinery |
| (b) Pure duration sum | New `impl Objective`: `edge_cost = max lane dur`, `lane_weight = dur(l)`, `min_shot_cost = fastest_lane_us`. Bound is the same Dijkstra table at a different weight function | driver, `MaxBound`, `h0` code |
| (c) Displacement / move-metric | New impl. C3 needs a *certified floor*: if the metric discounts co-moved atoms so a lane's standalone cost can exceed its shot's cost, `lane_weight` must be the min over shots containing that lane — or `0.0` (weak, still sound). The contract test tells you which | driver, bound machinery |
| (d) Extra additive per-shot terms (`1 + dur/tau + κ·n_lanes`) | New impl; floor becomes `1 + dur/tau + κ` (every shot has ≥ 1 lane). **A negative term (a credit for moving many atoms) can violate C2/C3** — the contract test catches it; the fix is to fold the credit into `lane_weight` or opt out with `0.0` | driver, bound machinery |
| (e) Fidelity (multiplicative) | Requires an explicit −log transform inside the impl. Called out because C1 is a **stated constraint of the framework**, not a hidden assumption | — |

Per-shot additivity (C1) is the load-bearing assumption and is documented as
such on the trait. Non-additive objectives (a schedule-history-dependent cost,
say) are out of the framework and would need a different bound argument, not a
different bound impl. Two known couplings that also read `g` as a moveset
count, to be made explicit rather than left implicit: `pick_best` (fine — it
compares costs under one objective) and `Cascade`'s
`max_depth = cost.ceil()`, which is only valid when `min_shot_cost == 1`;
it becomes `floor(cost / objective.min_shot_cost())` (identical for
`UniformCost`; see §8 for why that conversion is integer-valued at all).

## 6. First bound: `h0` = weighted-distance max

Precompute, per solve: Dijkstra on the **reversed** lane graph from each unique
target, edge weight `objective.lane_weight(lane)`, **blocked sites excluded as
graph nodes** (blocked never frees up within a solve — see
[`movement.py:84`](../../../python/bloqade/lanes/heuristics/physical/movement.py:84),
"treated as blocked obstacles for that solve rather than free atoms to route" —
so exclusion tightens and stays sound). Mirrors `DistanceTable`'s
interface/encoding (flat `n_loc × n_loc`, compact location interning) as a
separate type — `WeightedDistanceTable`. Then

```
h0(config) = max over atoms i with location(i) != target(i) of  wdist(location(i), target(i))
h0 = +INFINITY if any unresolved atom's target is unreachable
```

**Admissibility.** Fix any completion and any unresolved atom `i`; let
`l_1..l_k` be the lanes `i` traverses, in shots `s_1..s_k`. The shots are
distinct: a `MoveSet` assigns each qubit at most one destination, so an atom
moves at most one lane hop per shot (true for generated candidates, the
deadlock breaker, and both fallbacks, which emit single-lane shots). Then

```
remaining cost = Σ_{shots s in completion} edge_cost(s)      [C1]
              >= Σ_{j=1..k} edge_cost(s_j)                   [C2, distinct shots]
              >= Σ_{j=1..k} lane_weight(l_j)                 [C3]
              >= wdist(location(i), target(i))               [l_1..l_k is a path; wdist is the min]
```

Max over `i` of individually-valid floors is a valid floor and is the tightest
of them. **Not sum** — shots move many atoms in parallel, so summing
double-counts shared shots and breaks admissibility.

**Blocked exclusion is load-bearing, and it is why there is no cross-solve
cache (Q4).** `blocked` is not static arch data and is rarely empty: the
physical pipeline rebuilds it every solve from live atom state,
`blocked_native = [loc._inner for loc in state.occupied] + spectator_native`
([`movement.py:306`](../../../python/bloqade/lanes/heuristics/physical/movement.py:306))
— every un-routed atom plus every spectator. Two consequences:

- Excluding that many nodes lengthens or severs many paths, which *raises*
  `wdist`. The bound is materially tighter than the arch-only hop table, and
  the tightening grows with congestion — exactly the hard/dense instances where
  pruning is supposed to pay.
- The [`BlendedColumnCache`](../../../crates/bloqade-lanes-search/src/drivers/entropy.rs:819)
  pattern **cannot** be reused. That cache is valid precisely because
  `DistanceTable` columns ignore occupancy and `blocked` entirely, so a column
  is arch-pure and eternal. A weighted column is a function of
  `(arch, blocked, objective)`; with `blocked` changing nearly every solve the
  hit rate collapses and a set-hash key grows unbounded. Inapplicable, not
  merely premature. Building over the full graph instead would restore
  cacheability but surrender most of the tightening above — a bad trade.

**Invariant 2 (no ordering artifacts).** `WeightedDistanceTable` is a distinct
type with a single constructor taking `(&O, &SearchContext)`. It has no access
to and no code path reachable from the generator's contested-destination
penalty, pair-coordination boost, or seeded perturbation, and does not use
`blended_distance` / `HeuristicTables` (those mix `w_t`-weighted *time* for
ordering and are not costs). Enforced structurally, not by convention.

## 7. Landing order and instrumentation

1. **Step 1** — `Objective` trait + the `UniformCost` impl; route
   `CandidateEntry.cost`, `g`, the incumbent, and goal selection through it.
   Depth-cap → cost-cap, and `Cascade`'s `cost.ceil() as u32` →
   `floor(cost / min_shot_cost())`. Tests pin `g` semantics (`g == depth` under
   `UniformCost`, on a small instance) and byte-identical traces.
2. **Step 2** — `WeightedDistanceTable` (Dijkstra, blocked-excluded). Tests:
   weights match arch durations; blocked exclusion; agrees with the hop-count
   table when all weights are 1.
3. **Step 3** — `CompletionBound`, `NoBound`, `MaxBound`,
   `WeightedDistanceBound`, `as_heuristic`, contract-test helper. Tests: 0 when
   resolved, `+inf` when unreachable, exact single-atom value. Plus the
   swappability test: same driver, `WeightedDuration` + its paired bound.
4. **Step 4** — wire in behind `EntropyOptions::completion_bound: Option<BoundKind>`
   (default `None`). The bound is built **once per solve beside
   `entropy_tables`** in `run_with_components`
   ([`restarts.rs:159`](../../../crates/bloqade-lanes-search/src/search/restarts.rs:159)),
   under the same "is this an entropy strategy" gate, and borrowed by every
   restart closure — hence `Objective: Sync` and `CompletionBound: Sync`.
   Mechanically the option is a two-arm match at that one site, each arm calling
   the generic driver with a different `B` (`NoBound` or the real bound), so
   `NoBound::TRIVIAL` monomorphizes the test away when off rather than
   branching per node. `h == +inf` is an unconditional infeasibility cut.

   **Pre-agreed side effect:** with a large `blocked` set, `h0(root) = +inf`
   will fire for genuinely unroutable target assignments. Today those burn the
   whole expansion budget and return `BudgetExceeded`; with the bound on they
   return `Unsolvable` almost immediately. Sound — both fallbacks also carve
   out blocked (`LaneGraph::build(index, blocked)` and `find_path_occupied`),
   so neither could have reached an unreachable target — and a large node-count
   win, but it moves the `status` and `nodes_explored` columns for the
   suite's known-failing large cases. Call it out in the step-5 baseline review
   so it doesn't read as a regression.
5. **Step 5** — instrumentation on `SearchResult` → `SolveResult` → benchmark
   row (`harness/models.py::BenchmarkRow`):
   - counters: nodes expanded, cuts by `g` alone, cuts by `g + h0`;
   - per cut, `cut_depth` vs the earliest depth `g` alone could have reached `C`
     (`depth + ceil((C − g) / min_shot_cost)`) — the depth ratio measuring
     nodes saved (well-defined for any objective, which is what
     `min_shot_cost` is for);
   - per episode, `h0(root)` — a certified global lower bound on the optimum,
     valid despite sampled branch generation because `h0` does not depend on
     generation — and the final incumbent, giving a certified optimality gap.
   New CSV columns ⇒ **both benchmark baselines must be regenerated in this
   step's PR** (deterministic columns only; see AGENT.md).
6. **Step 6** — property test: `h0(start) <= brute-force optimum` over
   randomized small instances via `generators/exhaustive.rs`; flag-off trace
   identity; flag-on solution quality on the suite.

### One honest caveat on the anytime-dominance criterion

Pruning is sound: a cut branch provably contains no solution strictly better
than `C`, so no improving solution is ever lost. But the driver has a
**finite resume buffer** (`max_goal_candidates - 1`) and a node budget, so
cutting earlier changes buffer eviction and the expansion order. "Flag on ⇒
incumbent never worse at equal node budget" is therefore *expected and to be
measured on the suite*, not a theorem I can offer. If a regression shows up,
the fix belongs in the resume-buffer policy, not in weakening the bound —
and I would report it rather than tune the bound to hide it.

## 8. Frontier reuse — what the abstraction absorbs

Requirement 4 asked whether the existing heuristic trait can serve. Answering it
against the actual code: the frontier stack is *already* a single coherent
instantiation of this design, expressed implicitly in five places.

**The frontier's objective is the same one, unnamed.** Every frontier call site
passes `&UniformCost`
([`restarts.rs:103`](../../../crates/bloqade-lanes-search/src/search/restarts.rs:103)),
and every admissible heuristic is an integer hop count cast to `f64` —
`HopDistanceHeuristic::estimate_max` (`max_dist as f64`),
`PairDistanceHeuristic::estimate_max`, `MisplacedHeuristic` (`1.0`/`0.0`). Each
counts *shots*. So `g` in shots, `h` in shots, `f = g + h` integral. Nothing is
broken; the objective is just restated per call site rather than named once.

**Why `Cascade` uses integers.** `run_search`'s `max_depth` is a literal
tree-depth cutoff, not a cost cutoff:
`let depth = graph.depth(node_id); if depth >= max_d { continue }`
([`frontier.rs:571`](../../../crates/bloqade-lanes-search/src/drivers/frontier.rs:571)),
so `u32` counts tree levels. `inner_result.cost.ceil() as u32` converts the
inner solver's `f64` cost into that integer depth budget, and `.ceil()` is an
identity op on an integer-valued cost — defensive float→int hygiene, not
rounding intent. The cascade never prunes by cost; it says "don't look deeper
than the solution I hold", which coincides with "don't look for a costlier
solution" **only** under unit cost. Under the abstraction it becomes
`floor(cost / objective.min_shot_cost())`: identical for `UniformCost`, and
*correct* rather than accidentally correct for any future objective. Giving
`run_search` a genuine cost cutoff alongside `max_depth` is the deeper fix and
is out of scope here.

**The ordering/bounding split already exists here — by hand, untyped.**
`estimate_sum` is documented "Not admissible (overestimates because of bus
parallelism), but gives much better ordering for IDS/DFS"
([`distance.rs:357`](../../../crates/bloqade-lanes-search/src/primitives/distance.rs:357)),
and `IdsFrontier` mixes `IDS_REVERSAL_PENALTY` directly into its `h_score`
([`frontier.rs:373`](../../../crates/bloqade-lanes-search/src/drivers/frontier.rs:373))
— a pure ordering artifact. `run_with_components` takes `h_max` and `h_sum` as
separate generic parameters and it is on the author to route each correctly;
nothing in the types stops `h_sum` reaching `PriorityFrontier::astar` and
silently forfeiting optimality. Same hazard as invariants 2 and 6, already live.
(IDS itself is best-first-with-diving, not `f`-thresholded, so a non-unit
objective would not break it — only the cascade's depth conversion would.)

### Three reuse steps, in increasing order of behavior risk

1. **Free, zero behavior change.** `Objective: CostFn` and `run_search` already
   takes `&impl CostFn`, so `UniformCost`-as-`Objective` needs no frontier
   change at all. Step 1 makes both drivers' `g` the same named abstraction by
   construction. Included in step 1.
2. **Typed admissibility, still zero metric change.** Wrap the `estimate_max`
   family as `CompletionBound` impls for `UniformCost`, and add a *sibling*
   constructor `PriorityFrontier::astar_bounded(bound, weight)` requiring a
   bound paired to the objective — making "inadmissible `h` passed to A\*" a
   compile error. It must be a sibling, not a replacement: many tests pass bare
   closures to `astar`
   ([`frontier.rs:123`](../../../crates/bloqade-lanes-search/src/drivers/frontier.rs:123)),
   and `estimate_sum` plus the IDS penalty legitimately remain plain
   `Heuristic`s. Weighting stays where it belongs — the frontier may scale for
   ordering, the entropy prune consumes the same object unweighted (invariant 6).
   Follow-up, own PR.
3. **A tighter A\* heuristic — behavior change, own step.**
   `WeightedDistanceBound` under `UniformCost` *is*
   `HopDistanceHeuristic::estimate_max` plus blocked exclusion: strictly ≥ the
   current value, still admissible, so A\* stays optimal while expanding fewer
   nodes — and §6 argues the gap may be substantial given how large `blocked` is
   in the physical pipeline. But it changes which nodes A\*/Cascade expand, so it
   needs its own baseline regeneration and must not ride along with step 1.
   Follow-up, own PR.

## Resolved in review (2026-08-10)

1. **Default objective (§0).** `UniformCost` stays the default and the only
   objective on any production path. `WeightedDuration` is a public type with a
   required `tau`, wired only into the swappability test — no options/Python
   plumbing, no shipped `tau`. Promoting it is a separate objective-policy PR.
2. **Trait shape (§1).** `trait Objective: CostFn + Sync`; `UniformCost` gains
   the impl under its existing name. Standalone-trait variants rejected
   (no index to forward, plus a coherence overlap).
3. **Pairing enforcement (§2).** Associated `type Obj` for compile-time
   agreement, plus a **hard** `assert_eq!` on `ObjectiveId` at driver entry.
   Driver-constructs-the-bound rejected: it would drag bound selection into the
   driver, against requirement 3.
4. **Caching (§6).** No cross-solve cache. `blocked` varies per solve, so the
   `BlendedColumnCache` pattern is inapplicable rather than premature; revisit
   only if step 5's numbers show build cost matters.
5. **Frontier reuse (§8).** Reuse step 1 (shared `Objective`, no frontier
   change) rides along with step 1 here. Typed admissibility via
   `astar_bounded`, and swapping A\*'s `h` for the tighter blocked-excluded
   bound, are separate follow-up PRs — the latter moves benchmark baselines.
   The cascade's `cost.ceil() as u32` becomes
   `floor(cost / min_shot_cost())` in step 1 (identical under `UniformCost`).

Two items deliberately carried forward as *measured, not asserted*: the
anytime-dominance criterion (§7 caveat) and the `Unsolvable`-instead-of-
`BudgetExceeded` status shift on unroutable instances (step 4).
