# Critique: `bloqade-lanes-search` interface redesign

**Date:** 2026-08-20 (independent review), re-verified against `main` 2026-09-23.
**Reviews:** [`2026-08-18-search-trait-redesign-design.md`](2026-08-18-search-trait-redesign-design.md)
and [`../plans/2026-08-18-search-refactor-epics.md`](../plans/2026-08-18-search-refactor-epics.md)
as they stood at `fc6500da`.
**Status:** findings F1–F4 adopted; F5–F9 adopted as scope/wording amendments. The
critique led directly to the **binding-first** revision of the epics plan (2026-09-23).
Sections 1–4 are the review as written (line numbers are from `b823c308`, PR #925);
§5 records what changed when the claims were re-checked against current `main`.

---

## 1. Verdict

**Sound-with-amendments.** The load-bearing decisions — `Goal` as the placement/routing
seam, no universal `Traversal` trait, engines as static specializations, `Frontier`
demoted to the materialized family, the §5 cascade-pruning finding — all check out
against the code, and several are verifiably *correct rejections* of tempting-but-wrong
unifications. But the middle of the design is weaker than its edges: the
**`best_reached` / `MeasurableGoal` mechanism is the single weakest element** — an eager
push-time fold plus a new capability trait plus admitted new entropy plumbing, for
information the returned `SearchGraph` already contains and a lazy failure-path scan
would deliver at zero hot-path cost. Two of the §2 sketch signatures do not compile
against the real drivers (entropy's generator parameter; the entropy loop's
`min_shot_cost` dependence), and the `RouteOutcome` contract has an unexamined collision
with `backwards_search` mirroring. None of this invalidates the layering; all of it
should be resolved before Epic 2's steps 3–5.

## 2. Findings (by severity)

### F1. DESIGN FLAW / over-engineering: `best_reached` as an eager push-time fold + a new `MeasurableGoal` trait

The design (§2) insists the fold be incremental — "O(1)/node, no post-hoc graph scan" —
acknowledges it is a *genuine extra per-node evaluation* on the benchmark-gated hot
paths, mitigates that with a gating flag, and then concedes entropy needs "new plumbing
through its loop" (§8.1a). All of this machinery is solving a problem the code doesn't
have:

- **The full node arena is already in the result.** `SearchResult` carries
  `graph: SearchGraph` out of both engines (`drivers/frontier.rs:690-697`,
  `drivers/entropy.rs:2856+`), and `extract()` already consumes it
  (`search/restarts.rs:37-78`). Every node the push-time fold would see is inserted into
  that arena first — the pools are identical. An `argmin(shortfall)` scan over the arena
  **in `extract`/`route`, on the non-`Solved` path only**, costs zero on successful
  solves (the vast majority), does the same total number of shortfall evaluations as the
  fold when it does run, requires no change to either loop body, and makes §8.1(a)
  vanish: entropy needs *no* plumbing because its graph is returned too.
- **The codebase already does exactly this.** The receding-horizon rollout extracts its
  best partial by a post-hoc scan (`extract_best_leaf`, `placement/receding_horizon.rs:555`)
  and lives with it fine.
- **`MeasurableGoal` is redundant for its only named consumer.** §6 restricts the
  resumable handoff to `PointGoal`. For point goals, an "unresolved-qubit count" already
  exists as a free function (`unresolved_count`, `drivers/entropy.rs:1435-1445`), and
  `WeightedDistanceBound::estimate` is already a progress measure with `0 ⇔ is_goal`
  (`bounds.rs:339-372`) *(false — see §5)*. A third capability trait whose set-valued
  impls (`EntanglingConstraintGoal`) feed nothing (set-goal partials can't be handed to
  P&R without the concretization step §6 itself requires) is speculative surface. It
  also gets its own example wrong: "unsatisfied-pair count" for `EntanglingConstraintGoal`
  violates the stated `0.0 == is_goal` contract, because `is_goal` also rejects
  accidental spectator CZs (`goals.rs:111-139`) — a config with all pairs placed but an
  accidental CZ would report shortfall 0 while not being a goal.

**Instead:** compute `best_reached` lazily from `result.graph` on the failure path, keyed
on an ordinary function (unresolved count / bound estimate), deterministically
tie-broken by `(shortfall, g, NodeId)`. Defer the trait until a consumer needs shortfall
on a set-valued goal.

### F2. DESIGN FLAW: `RouteOutcome` is unrepresentable under `backwards_search`, and the resumed-`Unsolvable` proof is unstated

`RouteOutcome` promises `plan` = "moves applied from `start`" and `reached` = the config
the plan ends at (§6). But the mirroring path solves `target → initial` and inverts
(`search/target_solver.rs:165-251`). A *failed* mirror's partial plan, inverted, is a
**suffix** (`X → target`), not a prefix from `start` — there is no
`(plan-from-start, reached)` pair to report, which is precisely why today's code returns
the root on mirror failure (`target_solver.rs:184-191`). The design never mentions
`backwards_search`. Consequence: on exactly the instances mirroring exists for (hard,
constrained-target ones), `reached` must silently degrade to `start` and the P&R-resume
premise ("keep the search's progress") evaporates — or the contract needs a
suffix-composition variant nobody has designed. This must be specified before Epic 2
step 5.

Secondly, `Completeness::Unsolvable(proof)` from a *resumed* P&R is a proof about
`start`, not the caller's `initial`. It does transfer — the prefix plan's move sets are
invertible, so `start→goal` unsolvable implies `initial→goal` unsolvable *given the
identical blocked carve* — but that argument is nowhere in the design, and P&R's proof
is already documented as blocked-relative (`push_rotate/solver.rs:44-48`). Python
callers treat `Unsolvable` as a verdict about their instance; the composition helper
must state (and assert) the blocked-set-identity precondition, which §8.2 lists only as
a caveat about *feasibility*, not about *proof transfer*.

### F3. DESIGN FLAW (internal incoherence): the §2 `entropy_search` sketch takes a Tier-1 `MoveGenerator` it cannot use

`fn entropy_search<G, O, B, Go>(core, generator: &G, …) where G: MoveGenerator` (design
§2) contradicts both the design's own analysis and the code. Entropy generation is
parameterized by **per-node dynamic state owned by the loop**: the entropy level feeds
the score formula (`score = (w_d/e_eff)·d̂ + w_m·e_eff·m̂ + perturbation`,
`drivers/entropy.rs:1651`), the RNG is seeded by `seed ⊕ hash(config) ⊕ entropy`
(`entropy.rs:1518-1528`), and candidates are drawn incrementally through per-node
`tried_moves`/`failed_candidates`/`candidate_cache` (`get_next_candidate`,
`entropy.rs:2882-2935`). A Tier-1 `generate(&self, &Config, &mut Vec<MoveCandidate>)`
cannot express any of that — which is exactly why the inventory correctly reports
Engine B consumes no `MoveGenerator` (inventory §2.5) and why `EntropyGenerator` is
inert. The `G` parameter is a leftover that will send Epic-2 implementers down a dead
end. Either delete it (entropy owns generation, driven by `EntropyParams`) or admit a
wider, stateful generator seam for that specialization only.

### F4. Claim (c) is false against the code: the entropy *loop* needs `LaneAdditive`, not just the bound

The design asserts capability traits are needed "only by `WeightedDistanceBound` and the
best-reached fold." But the entropy driver calls `objective.min_shot_cost()` at six sites
inside its loop for the `BoundStats` depth-ratio accounting (`drivers/entropy.rs:2488,
2495, 2678, 2825, 2840, 2848`; consumed by `record_cut` at `entropy.rs:1399-1433` as
`ceil((C−g)/min_shot_cost)`). So the sketched `entropy_search<O: CostModel>` will not
compile. Fixes are cheap — bound the entropy specialization at `O: LaneAdditive`, or
move the depth conversion behind the bound/stats type — but the error shows the
capability split was drawn from the trait inventory, not checked against driver
internals. UNDERSPECIFIED → fix the sketch before Epic 2 step 2.

### F5. UNDERSPECIFIED: the three-layer stack does not describe RecedingHorizon, which consumes the raw graph

RH bypasses `TargetSolver` entirely: it calls `run_search` directly with a `max_depth`
*layer horizon* (`placement/receding_horizon.rs:576-600`), runs its own greedy beam
first, and — critically — **reads the returned `SearchGraph` to extract best leaves per
tier** (`receding_horizon.rs:555`). `RouteOutcome` carries no graph and `route()` has no
horizon parameter, so RH cannot be expressed through Tier 2 as sketched. That is fine —
but the design must say that a placement is licensed to reach past `TargetSolver` into
the search-core specializations, or Epic 3's migration will try to force RH through
`route()` and either bloat `Budget`/`RouteOutcome` (graph in a DTO — bad) or stall.
Related consistency nit: §1's diagram implies all placement flows through
`TargetSolver`; the inventory's own composition graph (§2.5) shows three of four
placements don't.

### F6. UNDERSPECIFIED: §7's "goal-agnostic generator" overclaims what the placement lift achieves

Lifting `cz_pairs`/`CzCoordination`/target assignment removes the *mode branch*
(`generators/heuristic.rs:358-361`) — but `HeuristicGenerator` contains a second block of
entangling-domain logic that runs regardless of `cz_pairs`: accidental-CZ spectator
detection and escape-move nomination (`heuristic.rs:386-421` and step 2b at
`heuristic.rs:503-522`), which reads arch CZ partners and exists purely to serve
`EntanglingConstraintGoal`'s spectator constraint. After the lift as scoped, the
generator is still goal-aware. Either the accidental-CZ nomination lifts too (a bigger,
riskier change than §7 budgets — it changes what candidates exist, on the baseline-gated
loose path), or the claim should be softened to "pair-coordination lifted; spectator
handling stays." Pick one before Epic 3B.

### F7. §5's prune is sound but materially narrower than the section implies

Soundness: verified. With admissible `h`, every node on a plan of cost `< C` satisfies
`g + h < C`, so a push-time `f ≥ C` prune removes no strictly-cheaper plan — and this
survives the transposition/lazy-deletion insert (`primitives/graph.rs:118-157`: a
cheaper rediscovery creates a new node, so a once-pruned config re-arriving cheaper is
re-tested at its new `f`), the closed set (pruned children never get ids), and path
reconstruction (only goal ancestors, all inserted). Weighted A* is fine as long as the
prune uses unweighted `h`, which §4 already mandates.

Scope: the `g + h ≥ C` gate fires only when an incumbent exists — i.e., the **cascade
refinement** (`max_cost` at `search/restarts.rs:345`). Plain A*/IDS carry no incumbent;
their `nodes_explored` can shift only via the `h = ∞` infeasibility cut, which the
design never separates out (yet §10 step 7 predicts astar/ids shifts — say why). And the
bound requires `exact_targets()` (`restarts.rs:227-233`), so the **loose-goal cascade
keeps its memory blow-up untouched** — `solve_loose_goal` dispatches `Strategy::Cascade`
with a set-valued goal (`placement/loose_goal.rs:264-275`) *(narrower — see §5)*. The
"single highest-leverage unification" is really "fixed-target cascade memory fix plus an
infeasibility cut"; state that so the payoff is measured against the right expectation.

### F8. NITPICK: `ErasedBound` misreports trivial bounds; the whole dyn tier is consumer-less

`ErasedBound::TRIVIAL = false` means a boxed `NoBound` reports `bound_enabled = true`
with `root_lower_bound = 0.0` published as a measurement — exactly the failure mode
`MaxBound`'s `TRIVIAL` override exists to prevent (`bounds.rs:251-258`). Stats-level, not
prune-level, but it contradicts the crate's own carefully-argued instrumentation
honesty. Combined with zero consumers, §10 step 9's "candidate to drop" should be
exercised: drop it.

### F9. NITPICKS: contract details left dangling

- Type drift: `PointGoal::required_placement -> &[(u32, LocationAddr)]` (§3) vs today's
  `exact_targets -> &[(u32, u64)]` (`traits.rs:121-123`) — pick one encoding.
- `Budget` semantics: `max_expansions` is **per restart** today
  (`restarts.rs:317-328`); a `Budget` handed to `route()` must define restart
  multiplication or the contract is ambiguous.
- Chained `cost`: P&R prices plans as layer count "matching `UniformCost`"
  (`push_rotate/solver.rs:153-156`); a chain under any future non-uniform objective sums
  incommensurate costs. One sentence in §6 fixes it.
- Who assembles the completion chain for bare fixed-target users?
  `fallback_push_rotate` lives *inside* `solve_with_engine` (`target_solver.rs:324-341`)
  and is a Python-visible flag (`search_python.rs:835-845`); §8.3 puts chain policy "at
  the placement layer," but `PyTargetSolver` has no placement above it. Name the
  combinator and its home.
- Epics: the bespoke Epic-1 Rust harness is justified (plan digests, resume semantics
  the CSV can't express), but the cheaper alternative — adding P&R/cascade rows to the
  existing benchmark registry to close the §9 gate gap immediately — is never weighed.
  Doing both, cheap one first, would de-risk Epic 1 itself.

## 3. What survives scrutiny

- **No universal `Traversal`; entropy as a sibling loop.** Verified method-by-method:
  `Frontier` is batch-expand-into-stored-list (`frontier.rs:44-63, 643-688`); entropy is
  one-candidate-per-iteration, outcome-driven, bounded resume buffer
  (`entropy.rs:2459-2854`). The correction that `IdsFrontier` (materialized,
  `frontier.rs:341-383`) doesn't license entropy is exactly right. This is the design's
  best call.
- **§5's diagnosis of the cascade non-prune.** Unconditional insert+push
  (`frontier.rs:653-654, 685-687`), `g`-only cap checked at pop (`frontier.rs:613-618`),
  equal-cost goal excluded (`frontier.rs:588-604, 664-667`). All confirmed; the memory
  framing is accurate.
- **P&R resume-is-just-a-Config (§8.2).** Confirmed: `plan_with` derives everything from
  the start placement (`push_rotate/mod.rs:161-173, 253`); the blocked-carve and
  off-graph-vertex caveats are real (`push_rotate/solver.rs:82-92`).
- **The current-state correction that `fallback_push_rotate` restarts from `initial`.**
  Confirmed (`target_solver.rs:324-341`). And the composition it proposes has in-crate
  precedent: `solve_loose_goal` already chains a second solve and concatenates layers
  through the replay verifier (`loose_goal.rs:309-327`) *(each leg is replayed
  separately — see §5)*.
- **The `LaneAdditive`/`PointGoal` split's shape and the §3 caveat.** The honesty about
  `run_with_components` (`restarts.rs:159-177`) blocking compile-time `PointGoal`
  enforcement, and the interim runtime gate, is the right call. `ObjectiveId`
  hard-assert and `TRIVIAL` preservation constraints are respected throughout.
- **The epic structure.** Phase A/checkpoint/Phase B with zero-drift verification, and
  Epic-1-first given the §9 gating gaps, is disciplined and matches where the coverage
  actually is thin.

## 4. The one question to force before Epic 2

**"`SearchResult` already returns the complete node arena to `extract()` — name the
consumer that needs `best_reached` before that point. If there is none, replace the
push-time fold + `MeasurableGoal` (Epic 2 steps 2 and 4) with a lazy failure-path scan
over `result.graph`, and show the revised `entropy_search`/`frontier_search` signatures
that actually compile against the real drivers (inline stateful generation,
`min_shot_cost`-consuming stats) before any code motion begins."**

The answer either deletes roughly a third of the design's new machinery (fold, trait,
entropy plumbing, the per-node cost debate, §8.1a) or surfaces a real consumer nobody
has written down — and Epic 2 steps 3–5 are built directly on whichever it is.

**Answer (2026-08-20):** there is no such consumer. The lazy scan is adopted; the fold
and `MeasurableGoal` are dropped.

---

## 5. Re-verification against `main` (2026-09-23)

Between `b823c308` and `27db773e`, the search crate was touched by #1000 (exhaustive
generator made exact; `Termination`, `SolveResult::{proven, termination}`,
`bound_terminates`, `AodCapacity`), #1002 (entropy stops when the root can no longer
generate), #937 (`MotionModel`), #978 (PyO3 `FromPyObject` annotations) and #947 (CI line
endings). No trait signature changed; `traits.rs` gained only the documented objective
contract C5.

### Claims that still hold (current locations)

Paths relative to `crates/bloqade-lanes-search/src/`.

| Claim | Now at |
|---|---|
| Frontier inserts + pushes every child; no cost/bound gate (dedup only) | `drivers/frontier.rs:655-690`; `receive_children` `150-165` |
| Cascade cap is `g`-only, checked at pop; equal-cost goal rejected | `reaches_cost_cap` `frontier.rs:516-518`; pop gate `616-620`; goal reject `589-591`, `666-668` |
| Frontier never consumes `CompletionBound`; `bound_stats` always default | `frontier.rs:551, 603, 680, 698` |
| Bound built in `run_with_components`, runtime-gated on `exact_targets()` | `search/restarts.rs:236-263`; `run_with_components` `189-207` (5 generics, 10 args) |
| Cascade `max_cost`; `max_expansions` is per restart | `restarts.rs:377, 388`; per restart `351, 356, 439, 444` |
| Mirror failure returns the root, empty layers | `search/target_solver.rs:168-255`, failure `187-193` |
| Fallback restarts P&R from the caller's `initial` | `target_solver.rs:328-347` |
| Both engines return the full `graph` | `drivers/result.rs:41`; entropy `drivers/entropy.rs:3065-3072`; frontier `545, 598, 675, 693` |
| Entropy owns generation (per-node caches, entropy-seeded RNG, score formula) | loop `entropy.rs:2544-3022`; `get_next_candidate` `3080-3134`; score `1731`; seed `1598-1607` |
| Six `min_shot_cost()` sites feeding `record_cut` | `entropy.rs:2573, 2580, 2807, 2973, 2998, 3006`; `record_cut` `1479-1513` |
| `unresolved_count` free function | `entropy.rs:1515-1524` (callers: observer/trace only) |
| RH bypasses `TargetSolver`, scans the graph | `run_search` `placement/receding_horizon.rs:590-604`; `extract_best_leaf` `620`; hand-rolled `beam_rollout` `379` |
| `HeuristicGenerator` mode branch + separate spectator handling | `generators/heuristic.rs:358-361`; spectators `386-421`, step 2b `503-521` |
| `EntanglingConstraintGoal::is_goal` rejects spectator CZs | `goals.rs:110-139` |
| P&R proof is blocked-relative; layer-count pricing | `push_rotate/solver.rs:45-48`, `159-162` |
| Python compares status strings | `movement.py:454`, `move_synthesis.py:47`, `_no_return_base.py:290`, `policy_movement.py:75` |

### Corrections

- **F1's "`WeightedDistanceBound::estimate` is `0 ⇔ is_goal`" is false** (and was at
  `b823c308`): an unresolved atom parked on a blocked site contributes 0
  (`bounds.rs:364-365`). The lazy scan must key on the unresolved-atom count, not on the
  bound.
- **F7's loose-goal claim is narrower.** `solve_loose_goal` runs the *caller's*
  strategy (`placement/loose_goal.rs:265-276`; the `SolveOptions` default is A*), not a
  hard-wired cascade. The loose-goal memory blow-up applies only when a caller selects
  cascade. The conclusion stands that set-valued goals are never bounded.
- **§3's "chains through the replay verifier" is imprecise.** The accidental-CZ cleanup
  leg (`loose_goal.rs:310-327`) is replayed on its own in `extract`, and the layers are
  concatenated with `extend`. The chain as a whole is never replayed, and the cleanup
  leg's `proven`/`termination`/`bound_stats` are not merged.
- **Inventory: `EntropyScorer` is not PyO3-consumed.** It has zero callers anywhere; the
  Python-visible `PyEntropyScorer` wraps `compute_moveset_metrics`
  (`bloqade-lanes-bytecode-python/src/search_python.rs:609-620`).
- **Inventory: `run_search` has two production callers:** `run_frontier`
  (`restarts.rs:166`) and RH's IDS fallback (`receding_horizon.rs:590`).

### New since #925 that bears on the plan

- **`proven` merges two different proofs**: plan-optimal (entropy root certificate,
  `entropy.rs:721, 3054`) and no-plan-exists (P&R `Unsolvable`). It is also a stored copy
  of `matches!(termination, Exhausted { proof: true })`. On the Python side,
  `rust_proven_total` counts both, and the `termination == "exhausted_proof"` docstring
  says "optimal".
- **Precedent for the F2 proof question.** RH's `merge_fallback`
  (`receding_horizon.rs:1092`) *downgrades* a fallback's proof whenever a committed prefix
  exists.
- **The fallback now promotes on `fallback.proven`** instead of `status == Unsolvable`. A
  returned fallback reports `nodes_expanded = 0` and default `bound_stats`, so the search
  counters are lost.
- **`AodCapacity`** adds two public fields, `SearchContext.capacity` and
  `SolveOptions.aod_capacity`. Neither is exposed to Python (hard-coded `None`,
  `search_python.rs:910`). The exhaustive generator also carries its own cap, combined
  via `tighten`, so capacity now comes from two sources. PyO3 builds `SearchContext` with
  a struct literal (`search_python.rs:729`), so every new context field breaks the
  bindings.
- **`bound_terminates`** is documented as an A/B measurement knob but is threaded
  through core, PyO3 and the Python traversal dataclass.
- **`BoundStats.bound_enabled`** exists to drive the Python "empty dict when disabled"
  behaviour. It is confirmed as a Python-facing field carried in core.
- **Dangling API:** `SearchEngine::exhaustive_preconditions()` (`search/engine.rs:152-155`)
  has no callers, and `ConfigError::UnsupportedArchitecture` is never constructed.
