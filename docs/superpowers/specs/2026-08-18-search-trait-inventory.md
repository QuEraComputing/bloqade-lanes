# Search-crate trait & interface inventory

**Date:** 2026-08-18
**Scope:** `crates/bloqade-lanes-search/` — read-only evidence audit of the trait
layer, the entry points, and the result/context DTOs, as the first step toward
an interface-first redesign of the crate's public surface.
**Status:** inventory only. No API is proposed here; no code was changed. The
redesign is the next artifact, authored after this is reviewed.

> **Citation convention.** Unless a path is given in full, every `file:line`
> below is relative to `crates/bloqade-lanes-search/src/`. Citations into the
> PyO3 crate are written `bloqade-lanes-bytecode-python/src/…`; into the CLI
> crate `bloqade-lanes-bytecode-cli/src/…`; into integration tests
> `crates/bloqade-lanes-search/tests/…`; into the Python layer `python/…`.
> "Production" means a non-`#[cfg(test)]` caller; "test-only" means the caller
> lives in a `#[cfg(test)] mod tests`, a `tests/` integration file, or a bench.

## How the evidence was gathered

Every `trait …` definition under `src/` was enumerated by grep (16 total), then
each definition, impl site, and consumer site was read. External reachability
(PyO3 crate, Python layer, CLI crate, benchmark harness, workspace `Cargo.toml`
graph) and the seven peripheral traits were mapped by two parallel read-only
sub-agents; their citations were spot-checked against the source. Reachability
claims ("no production caller", "no external caller") were confirmed by grep
across the whole workspace.

Two crates depend on `bloqade-lanes-search`:
`bloqade-lanes-bytecode-cli/Cargo.toml:16` and
`bloqade-lanes-bytecode-python/Cargo.toml:13`. (`bloqade-lanes-dsl-core` does
**not** depend on it — the string match there is a version-pin comment.) The
benchmark harness (`python/benchmarks/`) never touches the crate directly; it
drives it transitively through the Python heuristics layer.

---

## 1. The 16 traits at a glance

| # | Trait | Def | Supertraits / marker bounds | Dispatch in production | Prod impls | Test-only impls | First-cut class |
|---|-------|-----|-----------------------------|------------------------|-----------:|----------------:|-----------------|
| 1 | `MoveGenerator` | `traits.rs:10` | — | generic (`run_search`, `run_with_components`) | 4 | 3 | real interface (w/ diagnostic wart) |
| 2 | `CandidateScorer` | `traits.rs:28` | — | generic (`run_search` only) | 1 used¹ | 1 | real interface, seam unused |
| 3 | `CostFn` | `traits.rs:34` | — | generic (`run_search`; base of `Objective`) | 1 used² | 3 | real interface (as `Objective` base) |
| 4 | `Objective` | `traits.rs:85` | `CostFn + Sync` | generic (`&impl`/`<O>`) | 1 used² | — | real interface, single prod impl |
| 5 | `Goal` | `traits.rs:108` | — | generic | 3 | 1 | real interface |
| 6 | `Heuristic` | `traits.rs:128` | — | generic + blanket `Fn` impl | blanket³ | — | real interface (via closures) |
| 7 | `CompletionBound` | `bounds.rs:150` | `Sync` | generic only (non-object-safe) | 3 | — | real interface |
| 8 | `Frontier` | `drivers/frontier.rs:44` | — | generic | 4 | — | real interface (A\*-family only) |
| 9 | `SearchObserver` | `observer.rs:148` | — | `&mut dyn` | 2 | 1 | real seam / accreted payload |
| 10 | `CzPlacement` | `placement/cz_placement.rs:32` | — | concrete (monomorphic) | 4 | — | real interface, seam unused |
| 11 | `TargetGenerator` | `placement/target_generator.rs:32` | `Send + Sync` | `Box<dyn>` / `&dyn` | 1 | — | real interface, single impl |
| 12 | `CzCoordination` | `generators/cz_coordination.rs:30` | — (`pub(crate)`) | `Box<dyn>` | 2 | — | real interface (internal) |
| 13 | `PlanHeuristics` | `push_rotate/heuristics.rs:49` | — | `&dyn` | 2 | 1 | real interface (internal) |
| 14 | `BatchPolicy` | `push_rotate/schedule.rs:86` | — | `&dyn` | 1 (trivial) | — | unclear (speculative) |
| 15 | `MoveKernelObserver` | `dsl/move_policy_dsl/observer.rs:43` | — | `&mut dyn` | 3 | 1 | real seam / serialization payload |
| 16 | `TargetKernelObserver` | `dsl/target_generator_dsl/observer.rs:29` | — | `&mut dyn` | 3 | — | real seam / serialization payload |

¹ `CandidateScorer` has two impls (`DistanceScorer`, `EntropyScorer`) but the
only production consumer, `run_search`, is always handed `&DistanceScorer`
(`search/restarts.rs:139`). `EntropyScorer` is never called inside the crate; it
is surfaced independently to Python as `PyEntropyScorer` for the visualization
layer.
² `CostFn`/`Objective` have two impls (`UniformCost`, `WeightedDuration`) but
`WeightedDuration` is documented as **not wired into any production path**
(`cost.rs:65-68`); production always uses `UniformCost`
(`search/restarts.rs:139,191`).
³ `Heuristic`'s production workhorse is the blanket `impl<F: Fn(&Config)->f64>`
(`traits.rs:133`); the two named impls `MaxHopHeuristic`/`SumHopHeuristic` have
**zero callers anywhere** (see §5).

---

## 2. Trait-by-trait detail

### 2.1 `MoveGenerator` — `traits.rs:10`

- **Definition.** `generate(&self, config: &Config, node_id: NodeId, ctx: &SearchContext, state: &mut SearchState, out: &mut Vec<MoveCandidate>)` (`traits.rs:11-18`), plus a defaulted `deadlock_count(&self) -> u32 { 0 }` (`traits.rs:21-23`). No supertraits, no associated items, no marker bounds.
- **Semantic contract.** "Produces candidate move sets from a configuration." `deadlock_count` is "Number of deadlock occurrences tracked by this generator" — a diagnostic counter, not part of generation.
- **Implementors.** Production: `ExhaustiveGenerator` (`generators/exhaustive.rs:49`), `GreedyGenerator` (`generators/greedy.rs:28`), `HeuristicGenerator` (`generators/heuristic.rs:340`), `LooseTargetGenerator` (`generators/loose_target.rs:176`), `EntropyGenerator` (`generators/entropy.rs:25`). Test-only: `LineGen`/`TwoPathGen`/`DiamondGen` (`drivers/frontier.rs:776,801,845`). No blanket impls.
- **Consumers.** Generic bound `G: MoveGenerator` in `run_search` (`drivers/frontier.rs:536`) and `run_with_components`'s generator factory `MkGen: Fn(u64, DeadlockPolicy) -> Gen` (`search/restarts.rs:173,176`). `deadlock_count()` is read in production only to populate `SolveResult.deadlocks` (`search/restarts.rs:243,249,360,401`).
- **Object-safety / dispatch.** Object-safe, but used only via generics/monomorphization. In production only `HeuristicGenerator` (fixed-target) and `LooseTargetGenerator` (loose-goal, wraps the former) are instantiated; `EntropyGenerator` feeds the entropy driver.
- **Coupling smell.** The core method is clean domain. The wart is `deadlock_count()`: a result-DTO diagnostic bolted onto the generation trait so `extract` can read it back out. It is the one place the generator trait knows about the shape of the result.

### 2.2 `CandidateScorer` — `traits.rs:28`

- **Definition.** `score(&self, candidate: &MoveCandidate, config: &Config, ctx: &SearchContext) -> f64` (`traits.rs:29`). "Higher score = better candidate. Used to sort before graph insertion."
- **Implementors.** Production: `DistanceScorer` (`scorers/distance.rs:18`), `EntropyScorer` (`scorers/entropy.rs:29`). Test-only: `ZeroScorer` (`drivers/frontier.rs:899`).
- **Consumers.** Only `run_search` (`S: CandidateScorer`, `drivers/frontier.rs:537`). Its sole production caller `run_frontier` always passes `&DistanceScorer` (`search/restarts.rs:139`). `EntropyScorer` has **no in-crate caller** (only re-exports at `lib.rs:72`, `scorers/mod.rs:7`); it is consumed via PyO3 `PyEntropyScorer` (`bloqade-lanes-bytecode-python/src/search_python.rs:574-580`) for the entropy-tree viz.
- **Object-safety / dispatch.** Object-safe; used via generics only. **The variation seam is not exercised in production** — the frontier always scores with `DistanceScorer`, and the entropy driver does not route candidate scoring through this trait at all.
- **Coupling smell.** None in the type. The smell is architectural: a scorer abstraction whose only real polymorphic user is a Python-facing standalone scoring endpoint, not the search loop.

### 2.3 `CostFn` — `traits.rs:34`

- **Definition.** `edge_cost(&self, move_set: &MoveSet, from: &Config, to: &Config) -> f64` (`traits.rs:35`). "Computes edge cost for g-score accumulation … this affects A\* optimality guarantees."
- **Implementors.** Production: `UniformCost` (`cost.rs:21`), `WeightedDuration` (`cost.rs:159`). Test-only: `TwoPathCost`/`DiamondCost`/`FrontLoadedCost` (`drivers/frontier.rs:825,870,1526`).
- **Consumers.** As a standalone bound, only `run_search` (`C: CostFn`, `drivers/frontier.rs:538`), whose production caller hardcodes `&UniformCost` (`search/restarts.rs:139`). Its real role is as the supertrait of `Objective`.
- **Object-safety / dispatch.** Object-safe; generics only.
- **Coupling smell.** None. But like `CandidateScorer`, the standalone frontier seam is single-valued in production; the trait earns its keep as `Objective`'s base.

### 2.4 `Objective` — `traits.rs:85` (`: CostFn + Sync`)

- **Definition.** Extends `CostFn` with `lane_weight(&self, lane: LaneAddr) -> f64` (`traits.rs:90`), `min_shot_cost(&self) -> f64` (`traits.rs:101`), `id(&self) -> ObjectiveId` (`traits.rs:104`). `ObjectiveId { kind: &'static str, params: u64 }` is a `Copy`/`Eq` instance identity (`traits.rs:45-51`).
- **Semantic contract (the crate's most explicit).** C1 per-shot additive, C2 non-negative, C3 lane floor (`edge_cost(s) >= lane_weight(l)` for each lane `l∈s`), C4 shot floor (`min_shot_cost() > 0` finite, and `edge_cost(s) >= min_shot_cost()` for *every* shot incl. empty) — `traits.rs:59-81`. Mechanically checked by `bounds::assert_objective_contract` (`bounds.rs:404`, `test-util` feature). `ObjectiveId` promises: "Two objectives with equal ids must agree on `edge_cost` and `lane_weight` for every input" (`traits.rs:38-44`).
- **Implementors.** `UniformCost` (`cost.rs:27`, all constants), `WeightedDuration` (`cost.rs:179`). No test-only impls of `Objective` itself.
- **Consumers.** `&impl Objective` / `<O: Objective>` in the entropy entry chain (`drivers/entropy.rs:2287,2321,2360`), in `WeightedDistanceBound::new`/`NoBound::for_objective` (`bounds.rs:189,306`), and in `WeightedDistanceTable::new` (`primitives/weighted_distance.rs:84`). `id()` is checked at `entropy.rs:2369` and `MaxBound::new` (`bounds.rs:236`).
- **Object-safety / dispatch.** Object-safe in isolation, but paired with the non-object-safe `CompletionBound` via its associated type, so used via generics/monomorphization only. `Sync` is required for the rayon restart fan-out (`search/restarts.rs:216,322`).
- **Coupling smell.** None — this is domain-shaped and carefully specified. The caveat is **under-exercise**: the only production `Objective` is `UniformCost`, whose methods all return `1.0`; `WeightedDuration` exists so "the driver's objective-swappability is exercised by tests" (`cost.rs:65-68`). The abstraction is real but its payoff is latent.

### 2.5 `Goal` — `traits.rs:108`

- **Definition.** `is_goal(&self, config: &Config) -> bool` (`traits.rs:109`), plus defaulted `exact_targets(&self) -> Option<&[(u32, u64)]> { None }` (`traits.rs:121-123`).
- **Semantic contract.** `exact_targets` is the admissibility gate for target-distance bounds: it returns the exactly-required `(qubit, target)` set, or `None` for a set-valued goal, because "the distance to any one member is not a lower bound on the distance to the *nearest* member" (`traits.rs:111-123`). Default is the conservative `None` — a new goal opts in.
- **Implementors.** `AllAtTarget` (`goals.rs:24`, overrides `exact_targets` → `Some`, `goals.rs:36`), `PartialPlacementGoal` (`goals.rs:62`, default `None`), `EntanglingConstraintGoal` (`goals.rs:110`, default `None`). Test-only: `SiteGoal` (`drivers/frontier.rs:886`).
- **Consumers.** Generic `Go: Goal` in `run_search` (`drivers/frontier.rs:539`) and `Go: Goal + Sync` in `run_with_components` (`search/restarts.rs:172`); `&impl Goal` throughout the entropy driver (`drivers/entropy.rs:2248,…`); `exact_targets()` consumed to decide bound construction at `search/restarts.rs:229`.
- **Object-safety / dispatch.** Object-safe; generics only. `Sync` demanded by the restart fan-out.
- **Coupling smell.** None. `exact_targets` is a clean domain replacement for what used to be inferred from `SearchContext.cz_pairs` (see §4, `SearchContext`).

### 2.6 `Heuristic` — `traits.rs:128`

- **Definition.** `estimate(&self, config: &Config) -> f64` (`traits.rs:129`), with a blanket `impl<F: Fn(&Config) -> f64> Heuristic for F` (`traits.rs:133-137`). "Must be admissible (never overestimates) for A\* optimality."
- **Implementors.** The blanket closure impl is the production workhorse — every frontier construction passes a closure, e.g. `|cfg: &Config| h.estimate_max(cfg)` (`search/restarts.rs:347`, `placement/receding_horizon.rs:572`). Named struct impls `MaxHopHeuristic`/`SumHopHeuristic` (`heuristics.rs:25,46`) exist and are re-exported but have **no caller anywhere** (§5). `CompletionBound::as_heuristic` returns `impl Heuristic + Copy` (`bounds.rs:172`).
- **Consumers.** `Hmax/Hsum: Heuristic + Copy + Sync` in `run_with_components` (`search/restarts.rs:174-175`); `PriorityFrontier<H>`/`DfsFrontier<H>`/`IdsFrontier<H>` are generic over `H: Heuristic` (`drivers/frontier.rs:145,228,341`); `run_inner_rollout<…, Hsum: Heuristic + Copy + Sync>` (`placement/receding_horizon.rs:489`).
- **Object-safety / dispatch.** Object-safe, **but the `+ Copy` bound at the frontier call sites rules out `dyn Heuristic`** (a trait object is not `Copy`). The closure-based blanket impl is what satisfies `Copy`; this is load-bearing and documented at `bounds.rs:166-172`.
- **Coupling smell.** None. Dead named impls aside (§5), this is a clean, well-used seam.

### 2.7 `CompletionBound` — `bounds.rs:150` (`: Sync`)

- **Definition.** `type Obj: Objective` (`bounds.rs:152`); `const TRIVIAL: bool = false` (`bounds.rs:157`); `objective_id(&self) -> ObjectiveId` (`bounds.rs:160`); `estimate(&self, config: &Config) -> f64` (`bounds.rs:163`); defaulted `as_heuristic(&self) -> impl Heuristic + Copy + '_ where Self: Sized` (`bounds.rs:172-177`).
- **Semantic contract.** `estimate` is a lower bound on remaining cost, admissible **relative to `Obj`**; `f64::INFINITY` asserts infeasibility; `0.0` is always sound (`bounds.rs:143-149`). Admissibility "is a relationship, not a property" — the associated type makes mismatched bounds a compile error, and `ObjectiveId` catches same-type/different-instance mismatch at construction (`bounds.rs:8-24`). `TRIVIAL` "lets a driver monomorphize the bound test away entirely" (`bounds.rs:154-157`); const-asserted at `bounds.rs:700-701`.
- **Implementors.** `NoBound<O>` (`bounds.rs:198`, `TRIVIAL = true`), `WeightedDistanceBound<O>` (`bounds.rs:332`), `MaxBound<A, B>` (`bounds.rs:246`, `TRIVIAL = A::TRIVIAL && B::TRIVIAL`). All production-defined; `MaxBound` has only test callers (§5).
- **Consumers.** `<B: CompletionBound>` / `B: CompletionBound<Obj = O>` in the entropy driver (`drivers/entropy.rs:1319,1351,1399,662,2322,2361`) and in `MaxBound` bounds (`bounds.rs:230-231,248-249`). Reached in production when `EntropyOptions.completion_bound = Some(BoundKind::WeightedDistance)` and the goal is point-valued (`search/restarts.rs:227-233`).
- **Object-safety / dispatch.** **Non-object-safe by design** — the associated `const TRIVIAL`, the RPIT `as_heuristic`, and `where Self: Sized` all forbid `dyn CompletionBound`. Callers must monomorphize; the driver emits two arms (`Some(bound)` / `None`) so "bounding off" compiles to the same code as no bounding (`search/restarts.rs:274-301`).
- **Coupling smell.** None. This is the crate's best-designed piece.

### 2.8 `Frontier` — `drivers/frontier.rs:44`

- **Definition.** `select_next(&mut self) -> Option<NodeId>` (`:46`); `receive_children(&mut self, children: &[NodeId], graph: &SearchGraph)` (`:50`); defaulted `check_goal_on_pop(&self) -> bool { false }` (`:54`) and `check_goal_on_generate(&self) -> bool { true }` (`:60`). The #427 traversal abstraction.
- **Implementors.** Production: `PriorityFrontier<H>` (`:145`, A\*/greedy), `BfsFrontier` (`:197`), `DfsFrontier<H>` (`:228`), `IdsFrontier<H>` (`:341`). All constructed in `search/restarts.rs` (`:241,247,347,437,450,463,482`) and `placement/receding_horizon.rs:572`.
- **Consumers.** `F: Frontier` in `run_search` (`:540`) and `run_with_components`'s helpers.
- **Object-safety / dispatch.** Object-safe, but `PriorityFrontier/Dfs/Ids` are generic over `H`, and used monomorphically.
- **Coupling smell.** None. The important structural fact: **the entropy driver does not implement or use `Frontier`** — it is a separate hand-rolled single-path DFS (`drivers/entropy.rs:2240-2241`). So `Frontier` is the interface of the A\*/BFS/DFS/IDS *family*, not "the search's" interface. Push-and-Rotate is likewise off the `Frontier` path (`push_rotate/solver.rs:8-14`).

### 2.9 `SearchObserver` — `observer.rs:148`

- **Definition.** `on_event(&mut self, event: SearchEvent<'_>)` (`:149`); defaulted `wants_events(&self) -> bool { true }` (`:158`), which `NoOpObserver` overrides to `false` (`:171`) to gate eager payload construction.
- **Semantic contract.** Dispatch is `&mut dyn`, so the compiler cannot devirtualize; the event payload (moveset clones, buffer ranking) is built eagerly and therefore gated on `wants_events` (`observer.rs:9-16`). `SearchEvent<'a>` borrows driver-owned state; observers must copy out (`observer.rs:26-30,142-147`).
- **Implementors.** Production: `NoOpObserver` (`:166`), `EntropyTrace` (`drivers/entropy.rs:99`). Test-only: `LabelObserver` (`observer.rs:187`).
- **Consumers.** `O: SearchObserver` in `run_search` (`drivers/frontier.rs:541`); `&mut dyn SearchObserver` throughout the entropy driver (`drivers/entropy.rs:2042,2254,…`) and at the strategy dispatch (`search/restarts.rs:266`).
- **Object-safety / dispatch.** Object-safe; used as `&mut dyn` in the entropy driver and as a generic in the frontier.
- **Coupling smell — significant, in the payload not the trait.** `SearchEvent` (`observer.rs:32-140`) is a 7-variant enum whose six entropy variants carry ~13 fields each, shaped verbatim by the Python trace. Its only non-noop production impl, `EntropyTrace`, exists solely to serialize those events into `EntropyTraceStep` — tuple-typed fields like `Vec<(u8, u8, u32, u32, u32, u32)>` (`drivers/entropy.rs:81-82`) explicitly "preserving the legacy step-record shape consumed by the Python visualization layer" (`drivers/entropy.rs:94-98`). The observability *seam* is fine; the *event vocabulary* is a Python-viz transport format.

### 2.10 `CzPlacement` — `placement/cz_placement.rs:32`

- **Definition.** Single method `solve(&self, initial: &[(u32, LocationAddr)], controls: &[u32], targets: &[u32], blocked: &[LocationAddr], max_expansions: Option<u32>) -> Result<SolveResult, ConfigError>` (`:55-62`). No supertraits/associated items.
- **Semantic contract.** Precondition `controls.len() == targets.len()`, stated but not trait-enforced (`:47-49`); implementors check it inconsistently — hard `assert_eq!` in `LooseGoalCzPlacement` (`placement/loose_goal.rs:124`), `debug_assert_eq!` in RH/NoHome (`placement/receding_horizon.rs:1210`, `placement/nohome.rs:434`), unchecked in `SingleHeuristicCzPlacement` (`placement/single_heuristic.rs:90-107`).
- **Implementors (all production).** `SingleHeuristicCzPlacement` (`single_heuristic.rs:89`), `LooseGoalCzPlacement` (`loose_goal.rs:115`), `RecedingHorizonCzPlacement` (`receding_horizon.rs:1201`), `NoHomeCzPlacement` (`nohome.rs:425`).
- **Consumers.** No production code takes `CzPlacement` polymorphically. PyO3 wrappers hold the concrete type and call `solve` directly (`bloqade-lanes-bytecode-python/src/search_python.rs:1719,1847,1952,2046`). Only tests use `&dyn CzPlacement` (`single_heuristic.rs:250`, `loose_goal.rs:359`).
- **Object-safety / dispatch.** Object-safe; production is pure monomorphization on the concrete type. It is documented as a "user-facing seam" (`cz_placement.rs:14-15`) but is not currently used as one.
- **Coupling smell.** The trait signature is clean domain. The coupling is one level down: it returns `SolveResult`, which carries Python-shaped instrumentation (§4). Note also each impl has a richer inherent `solve_pairs(...)` taking a `future_cz_layers` lookahead the trait cannot express (`loose_goal.rs:90-93`, `receding_horizon.rs:1176-1178`, `nohome.rs:400-403`) — the trait method always passes `&[]` — so the trait is a lossy projection of the real capability.

### 2.11 `TargetGenerator` — `placement/target_generator.rs:32` (`: Send + Sync`)

- **Definition.** `generate(&self, ctx: &TargetContext) -> Vec<Vec<(u32, LocationAddr)>>` (`:33`). `TargetContext<'a>` = `{ placement, controls, targets, index: &LaneIndex }` (`:15-24`).
- **Semantic contract.** Each candidate is a full placement; candidates are tried in order, first successful solve wins (`:28-36`). Correctness checked externally by `validate_candidate` (`:129-184`), not by the trait.
- **Implementors.** `DefaultTargetGenerator` only (`:48`, `#[derive(Copy)]`). The Target-Generator DSL produces the same `Vec<Vec<(u32, LocationAddr)>>` shape via an inherent `TargetPolicyRunner::generate` (`dsl/target_generator_dsl/kernel.rs:86`) but **does not** implement this trait.
- **Consumers.** `Box<dyn TargetGenerator>` field + `&dyn TargetGenerator` in `SingleHeuristicCzPlacement` (`single_heuristic.rs:37,43,56,121`; call at `:138`); PyO3 constructs `Box::new(DefaultTargetGenerator)` (`bloqade-lanes-bytecode-python/src/search_python.rs:1697`).
- **Object-safety / dispatch.** Object-safe; used as trait objects.
- **Coupling smell.** Documentary only: `TargetContext` is "analogous to Python's TargetContext" (`:14`) and `DefaultTargetGenerator` "Mirrors the Python DefaultTargetGenerator" (`:43`). The Rust types mirror Python counterparts rather than a PyO3 type leaking into the signature. That the DSL runner reproduces the shape without implementing the trait is a hint the trait is not the true seam.

### 2.12 `CzCoordination` — `generators/cz_coordination.rs:30` (`pub(crate)`)

- **Definition.** Three all-defaulted methods: `contested_penalty(&self) -> i32 { 0 }`, `fallback_width(&self) -> usize { 1 }`, `boost_coordinated_pairs(&self, _selected: &mut Vec<(TripletKey, ScoredTriple)>) {}` (`:30-56`).
- **Semantic contract.** Each method "mirrors exactly one of the three formerly-inline `ctx.cz_pairs.is_some()` decision sites in `HeuristicGenerator::generate`" (`:28-29`); behavior pinned by tests (`:112-218`).
- **Implementors (production).** `FixedTargetCoordination` (`:61`, all defaults), `EntanglingCoordination<'_>` (`:69`, overrides all three).
- **Consumers.** `Box<dyn CzCoordination>` selected once per `HeuristicGenerator::generate` (`generators/heuristic.rs:358`).
- **Object-safety / dispatch.** Object-safe; `Box<dyn>`.
- **Coupling smell.** None. A clean internal strategy extraction. This is the one place where the `cz_pairs.is_some()` mode branch is legitimately consumed.

### 2.13 `PlanHeuristics` — `push_rotate/heuristics.rs:49`

- **Definition.** Four all-defaulted methods (`agent_order`, `score_step`, `rank_clear_target`, `rank_swap_vertex`) over `PlanCtx`/`PlanState`/`VertexId` returning ranking keys (`:49-122`).
- **Semantic contract (correctness-preserving by construction).** "Every method returns a *ranking key* instead of making the choice … it can reorder equally-good options but cannot select an illegal one or lengthen a path. A heuristic that returns a constant is exactly the default." (`:20-26`). Directionality documented per method (`score_step` higher-is-better, ranks lower-is-better).
- **Implementors.** Production: `DefaultHeuristics` (`:132`, all defaults), `AlignmentHeuristics` (`:251`, overrides `score_step`, holds `RefCell<Cache>` so not `Copy`). Test-only: `ContraryHeuristics` (`tests/push_rotate.rs:674`).
- **Consumers.** `&dyn PlanHeuristics` in `PlanCtx` (`push_rotate/context.rs:59,70`), `plan_with` (`push_rotate/mod.rs:136`), `solve_push_rotate_with` (`push_rotate/solver.rs:73`); call sites `mod.rs:272,316,342`, `ops.rs:65,347`.
- **Object-safety / dispatch.** Object-safe; `&dyn`. `AlignmentHeuristics` uses interior mutability to cache under `&self`.
- **Coupling smell.** None at the type level. Docs reference "benchmark numbers" (`:18,128,164`) but no benchmark/CSV type is in any signature. Not surfaced in the `lib.rs` prelude (reachable as `push_rotate::heuristics::*`).

### 2.14 `BatchPolicy` — `push_rotate/schedule.rs:86`

- **Definition.** One defaulted method `score_batch(&self, moves: &[Move], lanes: &[LaneAddr], index: &LaneIndex) -> f64 { moves.len() as f64 }` (`:86-96`).
- **Semantic contract.** "Higher is better"; strict comparison so a tie keeps the default's group ordering (`:84-85`, enforced at `schedule.rs:216-221`).
- **Implementors.** `LargestBatch` only (`:103`, all defaults). **No non-default impl exists anywhere, and no caller passes anything but `&LargestBatch`** (`schedule.rs:120`).
- **Consumers.** `&dyn BatchPolicy` in `schedule_with` (`schedule.rs:142`); `schedule` is the sole production caller.
- **Object-safety / dispatch.** Object-safe; `&dyn`.
- **Coupling smell.** None. But this is an **extensibility seam with zero realized variation** — a `dyn` indirection whose single implementation is the trivial default.

### 2.15 `MoveKernelObserver` — `dsl/move_policy_dsl/observer.rs:43`

- **Definition.** Four all-default-empty hooks: `on_init(&mut self, &PolicyGraphSnapshot)`, `on_step(&mut self, u64, u32, &MoveAction, &GraphDelta)`, `on_builtin(&mut self, u64, &str, bool)`, `on_halt(&mut self, &PolicyStatus)` (`:43-51`).
- **Semantic contract.** `GraphDelta` reflects tick-end kernel state, not per-action delta (`:21-32`); observers needing per-action granularity record actions themselves.
- **Implementors.** Production: `NoOpMoveObserver` (`:55`), `JsonMoveTraceObserver<W: Write>` (`:114`), `HumanMoveTraceObserver<W: Write>` (`bloqade-lanes-bytecode-cli/src/policy/trace.rs:162`). Test-only: `RecordingObserver` (`:160`).
- **Consumers.** `&mut dyn MoveKernelObserver` in `solve_with_policy` (`dsl/move_policy_dsl/kernel.rs:181`); callers pass `NoOpMoveObserver` (PyO3 `policy_runner_python.rs:302`, CLI `eval.rs:84`) or the JSON/Human tracers (CLI `trace.rs:90,94`).
- **Object-safety / dispatch.** Object-safe; `&mut dyn`. Writer-generic impls erased to `dyn` at call sites.
- **Coupling smell — serialization format.** Every payload type is `serde::Serialize`: `PolicyGraphSnapshot` (`:14`), `GraphDelta` (`:33`, `last_builtin: Option<String>` stringly-typed), `MoveAction` (`dsl/move_policy_dsl/actions.rs:20`), `PolicyStatus` (`dsl/move_policy_dsl/kernel.rs:114`). With `const SCHEMA_VERSION: u32 = 1` (`:90`) and versioned NDJSON envelopes, the trait's shape is dictated by a JSON trace format — the CLI/JSON boundary rather than PyO3, but transport nonetheless.

### 2.16 `TargetKernelObserver` — `dsl/target_generator_dsl/observer.rs:29`

- **Definition.** Two all-default-empty hooks: `on_invoke(&mut self, u64, &TargetContextSnapshot)`, `on_result(&mut self, u64, &CandidateSummary, bool)` (`:29-32`).
- **Semantic contract.** One `on_invoke`/`on_result` pair per CZ stage; no per-step loop (`:1-6`).
- **Implementors.** Production: `NoOpTargetObserver` (`:35`), `JsonTargetTraceObserver<W: Write>` (`:70`), `HumanTargetTraceObserver<W: Write>` (`bloqade-lanes-bytecode-cli/src/policy/trace.rs:203`).
- **Consumers.** `&mut dyn TargetKernelObserver` in `TargetPolicyRunner::generate` (`dsl/target_generator_dsl/kernel.rs:95`) and `run_target_policy` (`:218`); PyO3 passes `NoOpTargetObserver` (`target_generator_dsl_python.rs:60`), CLI passes JSON/Human.
- **Object-safety / dispatch.** Object-safe; `&mut dyn`.
- **Coupling smell — serialization format (same as 2.15).** `TargetContextSnapshot` and `CandidateSummary` are `serde::Serialize` (`:11,23`); `const SCHEMA_VERSION` + envelope structs (`:37-53`). `CandidateSummary` is a **flattened count-only projection** (`num_candidates`, `first_candidate_size`) of the real `Vec<Vec<(u32, LocationAddr)>>` — a serialization-shaped summary, not the domain value.

---

## 2.5 How the traits compose — interaction graph

The flat per-trait tables above don't show that the traits are wired into **two
disjoint search engines plus two side subsystems**, and that several traits are
consumed *inside* other traits' implementations. This section maps that.

**Notation.**
- `Impl <: Trait` — `Impl` implements `Trait`.
- `Consumer ──▶ Trait [Impl]` — `Consumer` takes `Trait` (as generic bound,
  `dyn`, or field); `[Impl]` is the concrete type filling it **in production**.
- `[inert]` = impl exists but is never wired in production; `[test]` = test-only;
  `[only value]` = the seam is real but never varies in production.

### Top level: `CzPlacement` peers compose the whole stack

```
CzPlacement            (placement seam; every impl returns SolveResult)
├─ SingleHeuristicCzPlacement <: CzPlacement
│     ├─ owns TargetSolver
│     └─ owns Box<dyn TargetGenerator>  ──▶ TargetGenerator [DefaultTargetGenerator]
├─ LooseGoalCzPlacement        <: CzPlacement ──▶ solve_loose_goal
├─ NoHomeCzPlacement           <: CzPlacement ──▶ solve_nohome
└─ RecedingHorizonCzPlacement  <: CzPlacement ──▶ RH driver
        (all four bottom out in run_with_components)

TargetSolver  ── owns ─▶ SearchEngine + MoveSearch
      └─ solve → solve_with_engine → run_with_components   (strategy dispatch)
                                          │
                    ┌─────────────────────┴──────────────────────┐
              ENGINE A: frontier loop                    ENGINE B: entropy driver
              run_search<G,S,C,Go,F,O>                   entropy_search_with_tables<O,B>
```

### Engine A — `run_search` consumes six traits at once

`run_search<G,S,C,Go,F,O>` (`drivers/frontier.rs:521-542`); production fill via
`run_frontier` (`search/restarts.rs:136-149`):

```
run_search ──▶ MoveGenerator   G  [HeuristicGenerator]
                                    └─ owns Box<dyn CzCoordination>
                                         ──▶ CzCoordination [FixedTargetCoordination | EntanglingCoordination]
                                  [LooseTargetGenerator] ── wraps ─▶ HeuristicGenerator
           ──▶ CandidateScorer  S  [DistanceScorer] [only value]     (EntropyScorer <: CandidateScorer [inert])
           ──▶ CostFn           C  [UniformCost]     [only value]     (UniformCost <: Objective too)
           ──▶ Goal             Go [AllAtTarget | EntanglingConstraintGoal | PartialPlacementGoal]
           ──▶ Frontier         F  [PriorityFrontier<H> | BfsFrontier | DfsFrontier<H> | IdsFrontier<H>]
                                     └─ Priority/Dfs/Ids ──▶ Heuristic  H  [closure via blanket Fn impl,
                                                                            e.g. |c| h.estimate_max(c)]
           ──▶ SearchObserver   O  [NoOpObserver]     [only value]
```

### Engine B — `entropy_search_with_tables` consumes a different four

`entropy_search_with_tables<O,B>` (`drivers/entropy.rs:2346-2362`):

```
entropy_search ──▶ Goal            [AllAtTarget | EntanglingConstraintGoal | …]
               ──▶ Objective   O   [UniformCost]           (UniformCost <: CostFn)
               ──▶ CompletionBound B where B::Obj = O
                        [NoBound<UniformCost> | WeightedDistanceBound<UniformCost>]
                        WeightedDistanceBound ── built from ─▶ Objective + Goal::exact_targets()
                        CompletionBound::as_heuristic() ── produces ─▶ Heuristic  (ordering only, never pruning)
                        MaxBound<A,B> ── composes ─▶ two CompletionBounds  [test]
               ──▶ &mut dyn SearchObserver  [EntropyTrace | NoOpObserver]
                        EntropyTrace ── consumes ─▶ SearchEvent (emitted by the driver)
```

**Engine B consumes neither `MoveGenerator`, `CandidateScorer`, nor `Frontier`.**
It reimplements candidate generation + scoring *inline* — the code comment says
so: "Mirrors the Python `HeuristicMoveGenerator.generate()` + `CandidateScorer`"
(`drivers/entropy.rs:801`). `EntropyGenerator <: MoveGenerator`
(`generators/entropy.rs:25`) and `EntropyScorer <: CandidateScorer`
(`scorers/entropy.rs:29`) are parallel trait impls of that same logic but are
never wired into the driver (test/PyO3-viz only).

### Trait-to-trait edges (structural dependencies)

| Edge | Kind | Site |
|------|------|------|
| `Objective : CostFn` | supertrait | `traits.rs:85` |
| `CompletionBound::Obj : Objective` | associated type binds bound↔cost model | `bounds.rs:152` |
| `CompletionBound ──▶ Heuristic` | `as_heuristic()` produces one | `bounds.rs:172` |
| `MoveGenerator(HeuristicGenerator) ──▶ CzCoordination` | `Box<dyn>` per `generate` | `generators/heuristic.rs:358` |
| `MoveGenerator(LooseTargetGenerator) ──▶ MoveGenerator(HeuristicGenerator)` | wraps (`inner` field) | `generators/loose_target.rs:6` |
| `Frontier(Priority/Dfs/Ids) ──▶ Heuristic` | generic `<H>` field | `drivers/frontier.rs:145,228,341` |
| `CzPlacement(SingleHeuristic) ──▶ TargetGenerator` | `Box<dyn>` field | `placement/single_heuristic.rs:37` |
| `SearchObserver(EntropyTrace) ──▶ SearchEvent` | pattern-matches the event | `drivers/entropy.rs:100` |

### Side subsystem 1 — Push & Rotate (own engine, shared `SolveResult`)

```
solve_push_rotate_with ──▶ PlanHeuristics [DefaultHeuristics | AlignmentHeuristics]   (push_rotate/solver.rs:73)
      └─ schedule       ──▶ BatchPolicy    [LargestBatch]                             (push_rotate/schedule.rs:142)
```
Reached from Engine-A's dispatch via `Strategy::PushRotate` and the
`fallback_push_rotate` net (`search/target_solver.rs:262,324`), so it produces
the same `SolveResult` contract but shares none of the search traits.

### Side subsystem 2 — DSL sidecar

```
solve_with_policy               ──▶ MoveKernelObserver   [NoOpMoveObserver | JsonMoveTraceObserver | HumanMoveTraceObserver]
run_target_policy / …::generate ──▶ TargetKernelObserver [NoOpTargetObserver | JsonTargetTraceObserver | HumanTargetTraceObserver]
```
Independent of the search traits; observers are the only seam.

### What the graph reveals for the redesign

- **`Goal` is the one trait both engines share.** `SearchObserver` is nominally
  shared but each engine emits a disjoint set of `SearchEvent` variants.
- **Engine B is a parallel universe.** Its generate/score logic is inline, so the
  `MoveGenerator`/`CandidateScorer`/`Frontier` abstractions do not span the crate
  — they describe Engine A only. Any "unify the search interface" move must
  reconcile these two, or accept two families.
- **The deepest real composition chain is**
  `CzPlacement ▸ TargetSolver ▸ run_with_components ▸ {Engine A: MoveGenerator ▸ CzCoordination}` and, on the entropy branch,
  `▸ {Engine B: Objective ◁ CompletionBound ▸ Heuristic}`.
  These are the edges a redesign must keep coherent.
- **Inert edges to prune:** `EntropyGenerator`/`EntropyScorer` (mirror inline
  code), `MaxBound` (composition never used), and the never-varying `S`/`C` seams
  in Engine A.

---

## 3. Entry points and DTOs

### 3.1 Entry points

| Entry point | Def | Visibility | Production callers | External callers | First-cut class |
|-------------|-----|-----------|--------------------|------------------|-----------------|
| `TargetSolver::solve` | `search/target_solver.rs:74` | pub | (Python) | `search_python.rs:1646` (`PyTargetSolver`) | real interface (primary) |
| `SearchEngine::from_*` | `search/engine.rs:80,94,112` | pub | placement + PyO3 | `search_python.rs:1406-1438` | real interface |
| `MoveSearch` (+ factories) | `search/move_search.rs:19` | pub | PyO3, tests | `search_python.rs:1457+` | real interface |
| `CzPlacement` peers `solve`/`solve_pairs`/`solve_with_attempts` | `placement/*` | pub | PyO3 | `search_python.rs:1719-2046` | real interface |
| `solve_with_engine` | `search/target_solver.rs:127` | `pub(crate)` | placement, target_solver | — | real internal (shared impl) |
| `run_with_components` | `search/restarts.rs:159` | `pub(crate)` | `target_solver.rs:308`, `loose_goal.rs:264`, `restarts.rs:573` | — | real internal, over-parameterized (5 generics, 10 args, `too_many_arguments`) |
| `run_search` | `drivers/frontier.rs:521` | **pub** | only via `run_frontier` (`restarts.rs:136`) + `receding_horizon.rs:584` | **none** | over-exposed internal engine |
| `entropy_search` / `_with_objective` / `_with_bound` | `drivers/entropy.rs:2246,2275,2308` | **pub** | **none** (production uses `_with_tables`) | **none** | over-exposed convenience wrappers |
| `entropy_search_with_tables` | `drivers/entropy.rs:2346` | `pub(crate)` | `restarts.rs:275,288` | — | real internal (the actual entropy entry) |
| `solve_push_rotate` / `_with` | `push_rotate/solver.rs:56,67` | pub, re-exported | reached in-crate via `Strategy::PushRotate` (`target_solver.rs:263,325`) | **none by name** | real interface; the free fn itself has no external caller |
| `solve_with_policy` | `dsl/move_policy_dsl/kernel.rs:175` | pub | PyO3 `PolicyRunner`, CLI | `policy_runner_python.rs:294`, `cli/policy/eval.rs:86`, `trace.rs:91` | real interface (DSL sidecar) |
| `run_target_policy` / `TargetPolicyRunner::generate` | `dsl/target_generator_dsl/kernel.rs:218,86` | pub | PyO3, CLI | `target_generator_dsl_python.rs`, `cli/policy/*` | real interface (DSL sidecar) |

Notes:
- **`run_search` and the `entropy_search*` trio are `pub` with no external and no in-crate production caller.** Production reaches the frontier through the private `run_frontier` wrapper and the entropy loop through the private `entropy_search_with_tables`. The public wrappers are documented as being "for tests, benches, and direct callers" (`drivers/entropy.rs:2343-2344`) — but there are none. Candidates for `pub(crate)`.
- Python selects Push-and-Rotate through `Strategy::PushRotate` on `SolveOptions`, not through `solve_push_rotate`; the re-exported free function has no external consumer.
- `run_with_components` is the true dispatch core but is a god-function: `<Go, Gen, Hmax, Hsum, MkGen>` plus ten value arguments, `#[allow(clippy::too_many_arguments)]` (`search/restarts.rs:158-177`).

### 3.2 DTOs

| DTO | Def | Re-exported at `lib.rs`? | Notes / coupling | First-cut class |
|-----|-----|--------------------------|------------------|-----------------|
| `SolveResult` | `search/result.rs:56` | no (only via `search::result`) | Core plan contract (`status`, `move_layers`, `goal_config`, `cost`, `nodes_expanded`) **plus** three riders: `deadlocks` diagnostic, `entropy_trace: Option<EntropyTrace>` (viz, `:81`), `bound_stats: BoundStats` (instrumentation, `:83`, doc: "The Python surface reports an unbounded run as an *empty* dict"). No serde/pyclass derive. | real interface DTO, bloated with instrumentation |
| `SolveStatus` | `search/result.rs:13` | no | `as_label()` "for status reporting (PyO3 wrappers, logs)" (`:41-42`); Python compares against the **string** `"solved"` (`python/…/movement.py:404`), not the enum. | real interface DTO w/ string-ABI coupling |
| `SearchResult` | `drivers/result.rs:13` | **yes** (`lib.rs:48`) | Raw driver output (`goal: Option<NodeId>`, `graph: SearchGraph`, `bound_stats`). Consumed only by `extract` → `SolveResult`; no external caller. | real internal DTO, mildly over-exposed |
| `SearchContext<'a>` | `primitives/context.rs:19` | yes (`lib.rs:67`) | `index`/`dist_table`/`blocked`/`targets` are core; `cz_pairs: Option<&[(u32,u32)]>` (`:24-26`) is a **mode-marker** read only by `HeuristicGenerator` to pick `CzCoordination` (`generators/heuristic.rs:358`). It formerly also drove admissibility; that role moved to `Goal::exact_targets()` (`search/restarts.rs:218-226`). | real interface DTO w/ residual mode-marker |
| `SearchState` | `primitives/context.rs:38` | yes (`lib.rs:67`) | Only field is `entropy_map: HashMap<NodeId, EntropyNodeState>` — entropy-driver state threaded through the *generic* `run_search`/generator signature that the frontier drivers never touch. | accreted (entropy-specific field on a "shared" type) |
| `MoveCandidate` | `primitives/context.rs:13` | yes (`lib.rs:67`) | Generator output `{ move_set, new_config }`. Domain. | real internal DTO |
| `MultiSolveResult` | `search/result.rs:165` | yes (`lib.rs:76`) | `single_heuristic` multi-candidate result. | real interface DTO |
| `CandidateAttempt` | `search/result.rs:151` | yes (`lib.rs:76`) | Thin debug info; flattened into a `PyDict` per attempt (`search_python.rs:1317-1331`). | real interface DTO, serialized to Python |
| `BoundStats` | `bounds.rs:51` | via `CompletionBound` re-export path | Threaded through **both** `SearchResult` and `SolveResult`; serialized to a `PyDict` (`search_python.rs:284-306`); **no external caller by name**. This is the memory-flagged "`bound_stats` threaded through result types" example. | accreted instrumentation transport |
| `EntropyTrace` / `EntropyTraceStep` | `drivers/entropy.rs:49,74` | no (via `drivers::entropy`) | Tuple-shaped fields "preserving the legacy step-record shape consumed by the Python visualization layer" (`:94-98`); wrapped 1:1 as getter-only pyclasses; consumed by `python/…/visualize/entropy_tree/*`. | accreted transport (viz) |
| `SearchEvent<'a>` | `observer.rs:32` | yes (`lib.rs:55`) | 7-variant enum; entropy variants shaped by the trace format (see §2.9). No external caller by name. | accreted transport (viz vocabulary) |
| `SolveOptions` / `Strategy` / `InnerStrategy` / `EntropyOptions` / `EntanglingOptions` / `BoundKind` | `search/options.rs` | yes (`lib.rs:75`) | Config bundles; each mirrored 1:1 into a `Py*` pyclass constructed from Python (`search_python.rs:99,743,828,943,1051,…`). | real interface (config), shaped partly for the Python constructor surface |
| `RecedingHorizonOptions` / `default_weight_grid` | `placement/receding_horizon.rs:68,146` | yes (`lib.rs:59`) | RH-specific config; mirrored as `PyRecedingHorizonOptions`. | real interface (config) |
| DSL result DTOs: `PolicyResult`/`PolicyStatus`, target candidate lists | `dsl/move_policy_dsl/kernel.rs`, `dsl/target_generator_dsl/kernel.rs` | via `dsl::*` | `PolicyStatus::as_label` "for … PyO3 wrappers, CLI output" (`kernel.rs:146-151`); copied into a flat `PyPolicySolveResult` (`policy_runner_python.rs:95-207`). | real interface DTO w/ string-ABI + transport copy |

Domain **primitives** re-exported from `lib.rs` (not part of the trait layer, listed for completeness): `Config`/`ConfigError` (`:66`), `MoveSet`/`NodeId`/`SearchGraph` (`:69`), `LaneIndex` (`:70`), `PairDistanceHeuristic` (`:68`, a standalone struct with inherent `estimate_max`/`estimate_sum` — **not** a `Heuristic` impl), `DistanceTable`.

### 3.3 `pub` items with no production / no non-test caller (dead or over-exposed surface)

Confirmed by workspace-wide grep. These are re-exported from `lib.rs` (or `pub` at module scope) yet reach no production consumer:

| Item | Re-export | Reachability | Verdict |
|------|-----------|--------------|---------|
| `MaxHopHeuristic`, `SumHopHeuristic` | `lib.rs:54` | **zero callers anywhere** (only own def/tests) | dead |
| `GreedyGenerator` | `lib.rs:51` | never instantiated in production; only `greedy.rs` self-tests | dead in production |
| `ExhaustiveGenerator` | `lib.rs:51` | only test/brute-force-oracle callers (`bounds.rs`, `tests/conveyor_1d.rs`, generator tests) | test/reference only |
| `PartialPlacementGoal` | `lib.rs:53` | only `#[cfg(test)]` (`goals.rs` tests, `restarts.rs:732`) | test-only |
| `MaxBound` | `lib.rs:43` | only test callers (`bounds.rs` tests); speculative bound composition | speculative, no prod use |
| `WeightedDuration` | `lib.rs:47` | test-only, documented "not wired into any production path" (`cost.rs:65-68`) | test-only by design |
| `LooseTargetGenerator` | `lib.rs:51` | production-internal only (`receding_horizon.rs:520`); no external caller | over-exposed re-export |
| `run_search` | (via `drivers::frontier`) | no external, prod only via private `run_frontier` | over-exposed |
| `entropy_search` / `_with_objective` / `_with_bound` | (via `drivers::entropy`) | no external, no prod caller | over-exposed |
| `SearchResult`, `SearchEvent`, `SearchState`, `MoveCandidate`, `SearchObserver` (by name), `solve_push_rotate` (by name), `BoundStats` (by name) | `lib.rs:48,55,67` etc. | not referenced by name in PyO3/CLI; internal or transport-only | over-exposed / transport |

---

## 4. Preliminary classification summary

**Real interfaces (the salvageable core).**
- `Objective` + `ObjectiveId` + `CompletionBound` — the strongest design in the
  crate: admissibility encoded as a type relationship, instance identity checked
  at construction, `TRIVIAL` monomorphization. Keep the contract (C1–C4), the
  associated-type pairing, and the `ObjectiveId` check intact. Caveat: only one
  production `Objective` (`UniformCost`) exercises it non-trivially.
- `Goal` — clean predicate with the `exact_targets()` admissibility gate. Keep.
- `Heuristic` — real, but *as the blanket closure impl*; the named struct impls
  are dead. The `+ Copy` requirement at the frontier is load-bearing.
- `Frontier` — real, but scoped to the A\*/BFS/DFS/IDS family; it is **not** the
  search's universal interface (entropy and push-rotate bypass it).
- `MoveGenerator` — real, minus the `deadlock_count()` result-DTO wart.
- `CzPlacement`, `TargetGenerator` — real user-facing seams by design, but
  currently monomorphic/single-impl in production; each is a *lossy* projection
  (`CzPlacement::solve` drops the `future_cz_layers` lookahead; the DSL target
  runner reproduces `TargetGenerator`'s shape without implementing it).
- `CzCoordination`, `PlanHeuristics` — clean internal strategy seams. Keep.
- `solve_with_engine` / `run_with_components` / `entropy_search_with_tables` —
  the real internal entry points; `run_with_components` needs decomposition, not
  removal.
- `TargetSolver`, `SearchEngine`, `MoveSearch`, and the option bundles —
  legitimate public entry points.

**Accreted transport (shaped by a boundary, not the domain).**
- `SearchEvent` + `EntropyTrace`/`EntropyTraceStep` — a Python-viz trace format
  wearing the `SearchObserver` seam. The seam is fine; the event vocabulary is
  transport.
- `BoundStats` — instrumentation threaded through both result DTOs and existing
  to become a `PyDict`. The prior-session example, confirmed.
- `MoveKernelObserver` / `TargetKernelObserver` payloads — serde/NDJSON trace
  format (CLI boundary); `CandidateSummary` is a serialization-shaped count
  projection of the real target result.
- `SolveResult`'s `entropy_trace`/`bound_stats`/`deadlocks` riders and
  `SolveStatus::as_label` / `PolicyStatus::as_label` string-ABI helpers.
- `CandidateScorer`/`CostFn` as *standalone frontier seams* — real traits whose
  variation is never exercised (`run_search` always gets `DistanceScorer` +
  `UniformCost`); `CandidateScorer`'s second impl exists only for the Python
  scorer endpoint.

**Unclear / speculative.**
- `BatchPolicy` — a `dyn` extensibility seam with a single trivial impl and no
  varying caller.
- `SearchState` — an entropy-specific field masquerading as shared mutable state
  on the generic search signature.
- `SearchContext.cz_pairs` — a residual mode-marker now that admissibility moved
  to `Goal::exact_targets()`; its only remaining reader is
  `HeuristicGenerator`'s `CzCoordination` selection.

**Working hypothesis — verdict.** Largely **supported**. The trait layer *is*
the better-designed part: the `Objective`/`CompletionBound`/`Goal` cluster is
genuinely interface-first, and the internal strategy traits
(`CzCoordination`, `PlanHeuristics`) are clean. The accretion is concentrated in
(a) the result/context/event **DTOs** (`SolveResult` riders, `BoundStats`,
`SearchEvent`/`EntropyTrace`, `SearchState`), (b) the **flat over-exposed
re-exports** (`lib.rs` exports several dead or internal-only symbols), and (c) a
few **public entry points with no caller** (`run_search`, `entropy_search*`).
Two nuances qualify the hypothesis: some traits are real-but-inert
(`CandidateScorer`/`CostFn`/`BatchPolicy` seams never vary; `CzPlacement`/
`TargetGenerator` are never used polymorphically), and two "seam" traits
(`SearchObserver`, the DSL observers) are well-shaped but carry transport-format
payloads.

---

## 5. Constraints any redesign must respect

Ordered roughly by how sharply they will bite if ignored.

1. **`CompletionBound` is non-object-safe on purpose.** `const TRIVIAL`, the
   RPIT `as_heuristic`, and `where Self: Sized` all forbid `dyn`. The driver
   relies on `TRIVIAL` to make "bounding off" compile to *the same code* as no
   bounding (`search/restarts.rs:274-301`; const asserts `bounds.rs:700-701`).
   Any move to `dyn CompletionBound` deletes that property.
2. **Admissibility is a typed relationship, checked twice.** `CompletionBound::
   Obj: Objective` pairs a bound to a cost model at compile time; the runtime
   `ObjectiveId` equality (`entropy.rs:2369`, `MaxBound::new` `bounds.rs:236`)
   catches same-type/different-instance mismatch. Dropping either makes a
   mismatched bound prune away the optimum silently.
3. **`Objective`/`CompletionBound`/`Goal` must stay `Send + Sync`.** They are
   shared by reference across the rayon restart fan-out
   (`search/restarts.rs:172-176,216,322`). Boxing them must preserve `Sync`.
4. **`Heuristic + Copy` at the frontier call sites forbids `dyn Heuristic`.**
   `PriorityFrontier<H>`/`Dfs`/`Ids` take `H` by value and clone per restart;
   `as_heuristic` returns `impl Heuristic + Copy`. The closure blanket impl is
   what satisfies `Copy`; a trait-object heuristic will not compile here.
5. **The `Objective` contract C1–C4 is the admissibility foundation** and is
   partly prose, partly `assert_objective_contract` (test-util). Any new
   objective must pass it; keep the mechanical check reachable
   (`bounds.rs:404`, `traits.rs:59-81`).
6. **`Goal::exact_targets()` is the sole admissibility gate for target-distance
   bounds.** Only point-valued goals return `Some`; set-valued goals must return
   `None` or `h0` becomes inadmissible (`search/restarts.rs:227-233`,
   `traits.rs:111-123`). Preserve this when reshaping `Goal`.
7. **There are two disjoint search engines, not one.** The `Frontier` generic
   loop (`run_search`) and the entropy driver (hand-rolled, `&mut dyn
   SearchObserver`) share almost nothing; push-rotate is a third path off both.
   A single unifying "search interface" cannot be assumed — `Frontier` covers
   only the A\*-family.
8. **`SearchObserver` dispatch is `&mut dyn` gated by `wants_events()`**, and
   `SearchEvent<'a>` borrows driver-owned state. The gate exists because dynamic
   dispatch blocks devirtualization and the payload is built eagerly
   (`observer.rs:9-16`). A redesigned event type must keep the borrow + the gate
   or pay per-event allocation on the hot path.
9. **`SolveResult` is a shared contract across search, push-rotate, and
   mirroring**, and its core plan fields are read positionally by the replay
   verifier (`search/verify.rs`, `search/restarts.rs:49`) and by Python. Splitting
   the DTO must keep the core plan (`move_layers`, `goal_config`, `cost`,
   `nodes_expanded`) stable; the `bound_stats`/`entropy_trace`/`deadlocks` riders
   are the separable part.
10. **Python couples to string labels and dict shapes, not Rust types.**
    `result.status` is compared to `"solved"` (`SolveStatus::as_label`),
    `bound_stats` and `attempts` are consumed as dicts, `EntropyTrace` as a
    getter-only pyclass. Until the Python adapter is rebuilt, the label strings
    and dict keys are a de-facto ABI — renaming an enum variant or a dict key
    breaks Python silently. The redesign should treat the PyO3 crate as the
    place to *absorb* this, keeping the Rust DTOs clean.
11. **Monomorphization cost is already high.** `run_search` (`<G,S,C,Go,F,O>`)
    and `run_with_components` (`<Go,Gen,Hmax,Hsum,MkGen>`) are
    `too_many_arguments` god-functions; each added trait parameter multiplies
    generated code. Decomposition should reduce, not add, generic breadth.
12. **`ArchSpec` validation is a load-time precondition the search assumes.**
    `SearchEngine::from_json_validated`/`from_arch_spec` reject cyclic buses;
    the search layers treat per-bus acyclicity as *given* rather than checking it
    (`search/engine.rs:85-115`). Any new constructor/entry point must not become
    an unvalidated back door.

---

## 6. Next step (not this artifact)

With the inventory reviewed, the follow-up artifact proposes the redesigned
crate-level interfaces — separating the salvageable trait cluster from the
accreted DTOs, collapsing the dead/over-exposed re-exports, and specifying the
thin PyO3 adapter that absorbs the string-label/dict ABI — validated against the
deterministic benchmark baselines
(`python/benchmarks/harness/latest_physical.csv` /
`latest_logical.csv`, zero-diff CI gate) and the search-crate tests now run by
`just test-rust` / `just lint`.
