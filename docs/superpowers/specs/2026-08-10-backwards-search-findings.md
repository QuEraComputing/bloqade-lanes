# Backwards Search — Findings

**Status:** empirical findings
**Date:** 2026-08-10
**Branch:** `phil/reverse-solve`

## Summary

Atom routing is easier in one direction than the other, and the asymmetry is not
symmetric in the endpoints — it depends on how *constrained* each endpoint is.
Searching from a CZ pairing configuration toward a scattered one costs 2.2–4.7×
fewer node expansions than the reverse on synthetic instances.

Because AOD moves are invertible, a solver can exploit this by solving the
mirrored instance and turning the plan around. That is
`SolveOptions::backwards_search`.

On the committed benchmark suite the gain does **not** appear, for a reason that
turns out to be a property of the suite rather than of the option — see
[Why the suite cannot see it](#why-the-suite-cannot-see-it). The one congested
case in the suite does improve materially.

## The mechanism

`MoveSet::inverse()` flips `Direction` on every lane. Lane identity
(`move_type`, `zone_id`, `word_id`, `site_id`, `bus_id`) is
direction-independent — `site_id`/`word_id` always encode the forward-direction
source, and `Direction` only selects which endpoint `ArchSpec::lane_endpoints`
returns as src versus dst — so the flip maps `(src, dst)` to `(dst, src)`
exactly, and the flipped lane always exists because `LaneIndex::new` registers
both directions.

A plan `[m1, …, mk]` for `target → initial` therefore becomes a plan for
`initial → target` as `[mk⁻¹, …, m1⁻¹]`: the list reversed **and** every element
inverted. Doing only one of the two produces a plan that is often still
executable but lands somewhere else, which is why the transformed plan is
replayed through `assert_move_layers_executable` before it is returned.

That replay covers lane endpoints and occupancy; it does **not** cover AOD
geometry, and cannot. `ArchSpec::check_lane_group_geometry`
(`arch/query.rs`) builds its grid from each lane's raw
`(zone_id, word_id, site_id)` — the forward source by convention — and never
calls `lane_endpoints`, so a lane and its inverse always receive the identical
verdict, and an inverted group is checked at its drop side rather than its
pickup side. What makes the one-sided check sound is an unstated property of the
architecture: a bus's src → dst coordinate map must be **separable** — dst *x* a
function of src *x* alone, dst *y* of src *y* alone — which is what lets a
rectangle on one side certify a rectangle on the other. All 22 buses of the
bundled Gemini physical spec (3 site buses, 19 word buses, 0 zone buses) are
separable, so on that arch the one-sided check is equivalent to a two-sided one.
Nothing enforces this: arch build time validates that each bus's *full* src and
dst word/site sets form AOD rectangles (`_validate_aod_rectangle`, called from
`add_site_bus`, `add_word_bus` and `ArchBuilder.connect`), which is strictly
weaker — full rectangles on both sides say nothing about an arbitrary lane
*subset* mapping to a rectangle. This is pre-existing behaviour that
`backwards_search` does not change, and a follow-up worth taking up separately:
either assert separability at arch-load time, or make the geometry check
endpoint-aware.

## Direction alone is not the cause

Three things differ between a mirrored solve and a forward one in the
configuration originally measured: direction, heuristic aggregation, and
deadlock policy. Isolating direction by mirroring **random scattered → random
scattered** instances shows no effect at all:

| 20 atoms, 30 instances | solved | median layers |
|---|---|---|
| mirrored (goal→start) | 30/30 | 45.0 |
| forward (start→goal) | 30/30 | 43.0 |
| `greedy` (h_max) | 21/30 | 41.0 |
| `ids` (h_sum) | 28/30 | 43.0 |

Two conclusions. Direction is irrelevant when both endpoints are equally
disordered. And the robustness gap in that table (30/30 vs 21/30) comes from
`DeadlockPolicy::AllMoves` plus `top_c=None`, **not** from direction — a
separate, independently actionable finding about the existing strategies that is
out of scope for this branch.

## The asymmetry is an entropy effect

Random-to-random instances cannot exhibit an entropy asymmetry, because neither
endpoint is more constrained than the other. The real workload is not
random-to-random: a CZ layer target places atom *pairs* on both words of an
entangling pair at a common site — a highly constrained, low-entropy
configuration — while the resting layout is comparatively scattered.

Re-running with that instance class (scattered ↔ paired on the bundled Gemini
physical spec: 10 entangling word pairs, 8 sites):

| atoms | solver | scattered → paired | paired → scattered |
|---|---|---|---|
| 8 | greedy | 30/30 solved | 29/30 solved |
| 16 | — | 50.0 median nodes | **35.0** median nodes |
| 24 | greedy | **12/30 solved** | **19/30 solved** |
| 24 | — | 102.0 median nodes | **53.0** median nodes |

The effect is absent at 8 atoms and grows with density, which is what a
constraint-counting argument predicts: there are exponentially many scattered
configurations and comparatively few paired ones, so a search moving toward the
scattered set has many admissible ways to make progress, while one moving toward
the paired set is squeezed into a narrowing funnel and stalls on plateaus.

## Effect on the production solver

Per-instance paired comparison, 20 instances per row, budget 3000,
scattered → paired:

| atoms | solver | fwd solved | mirrored solved | median fwd nodes | median mirrored nodes | speedup | mirrored better on |
|---|---|---|---|---|---|---|---|
| 12 | entropy | 20/20 | 20/20 | 608.0 | **129.0** | **4.71×** | 14/20 |
| 16 | entropy | 20/20 | 20/20 | 1128.5 | **315.5** | **3.58×** | 17/20 |
| 24 | entropy | 20/20 | 20/20 | 1611.5 | **721.0** | **2.24×** | 14/20 |
| 24 | ids | 7/20 | **14/20** | 59.0 | 47.0 | 1.26× | 4/5 |
| 24 | greedy | 5/20 | **13/20** | 334.0 | 50.0 | 6.68× | 2/3 |

Plan quality holds — the speedup is not traded away:

| atoms | median layers fwd → mirrored | median lanes fwd → mirrored | mirrored shorter |
|---|---|---|---|
| 12 | 25.5 → 25.0 | 64.0 → 65.5 | 11/20 |
| 16 | 35.0 → **33.0** | 81.5 → 89.5 | 12/20 |
| 24 | 50.0 → **45.5** | 126.0 → 125.0 | 12/20 |

Layer counts favour mirroring; lane counts are mixed — worse at 16 atoms
(89.5 vs 81.5), level at 24. Since `estimated_fidelity` depends on both, the
fidelity effect is not established by these synthetic runs.

## Palindrome makes the gradient structural

The obvious objection is that a multi-layer circuit would erase the gradient:
layer *k+1* starting from layer *k*'s paired positions means both endpoints are
constrained. That does **not** apply under `PalindromePlacementStrategy`.

`PalindromePlacementStrategy._unwrap` rebuilds the home `ConcreteState` from
`ExecuteCZReturn.initial_layout` before delegating
(`analysis/placement/strategy.py`), and its docstring states the invariant:
*"Palindrome moves always return atoms to the pre-CZ home before the next CZ."*
Every CZ solve therefore starts from the resting layout and targets a CZ
pairing.

Consequences:

1. The synthetic scattered → paired instances model *every* layer of a
   palindrome-scheduled circuit, not just the first.
2. **Mirroring is categorically the right orientation for that path**, fixed by
   the scheduler's design rather than being a per-workload gamble. This is why
   `make_physical_placement_strategy` ties `backwards_search` to its
   `return_moves` flag.
3. On the two-phase no-home path it applies to exactly **one** routing call.
   `solve_nohome` skips its return phase whenever every atom already sits on a
   home site (`has_returners` in `placement/nohome.rs`), which palindrome
   guarantees — leaving a single fixed-target entangling solve, precisely the
   call with the gradient.

### Corollary: palindrome makes the no-home tuning inert

Point 3 has a consequence beyond this option. Under palindrome,
`NoHomePlacementStrategy` reduces to a plain fixed-target router, so everything
feeding its return phase never runs: `k_candidates`, `gamma`,
`top_bus_signatures`, `bus_reward_rho`.

Confirmed empirically: with `return_moves=True` and a home starting layout
(atoms on the even/home words), sweeping `move_solutions_per_layer` — which the
factory maps to `k_candidates` — over 1, 3, 8 and 20 produces byte-identical
move layers at 4, 6 and 8 atoms. The factory docstring now carries a warning to
that effect.

## Why the suite cannot see it

Run against the physical benchmark suite, mirroring produced no node-expansion
win anywhere, six of nine cases byte-identical, one regression, and one strong
improvement:

| case | metric | forward | mirrored |
|---|---|---|---|
| `steane_physical_35` (entropy) | fidelity | 0.00192 | **0.00688** (3.6×) |
| `steane_physical_35` (entropy) | lanes | 206 | **158** |
| `steane_physical_35` (ids) | success | **fail** | **solve** |
| `trotter_rand_35` (entropy) | lanes | 1548 | 1464 (fidelity 0 either way) |
| `ghz_6` (entropy) | fidelity | **0.74482** | 0.71714 (regression) |

The reason is structural. In the committed baseline, `nodes_explored` is exactly
`move_count_events / 2` for every single-restart dive strategy:

| strategy | cases | exactly 2.00 | range |
|---|---|---|---|
| `rust_entropy_1` | 9 | 9/9 | 2.00–2.00 |
| `rust_greedy` | 7 | 7/7 | 2.00–2.00 |
| `rust_ids` / `rust_dfs` | 8 | 7/8 | 1.86–2.00 |
| `rust_astar` / `rust_bfs` | 6 | 4/6, 1/6 | 0.75–2.00 |

Palindrome emits each plan twice, so `events = 2 × layers`; a ratio of exactly 2
therefore means `nodes == layers` — **one expansion per emitted layer, i.e. the
router never backtracks on these kernels.** (The `entropy_5/10/20` ratios of
0.02–0.40 are restart multiples, not backtracking.)

The entropy asymmetry is a cost of plateaus and backtracking. With no
backtracking there is nothing for it to remove, and `nodes_explored` on this
suite is a plan-length proxy rather than a measure of search effort.

**Evaluating this option — or any routing-search change — needs congested
kernels where backtracking actually occurs.** `steane_physical_35` is the only
case in the suite that shows signs of that regime, and it is where mirroring
helps.

## A correctness hazard, and the guard for it

`blocked` is **direction-asymmetric**. No move may land on a blocked location,
but nothing prevents the *root* placement from sitting on one — the generators
never check the root. An instance whose target overlaps `blocked` is therefore
correctly unsolvable forward, while its mirror starts on the blocker and moves
away, yielding a plan that parks an atom on top of an external atom.

`assert_move_layers_executable` **cannot** catch this: blocked atoms are not in
the `Config` it replays. `backwards_search` therefore declines to mirror when
either endpoint intersects `blocked` (`mirroring_breaks_blocked` in
`search/target_solver.rs`).

On the one path that enables the option, the guard is a safety net rather than a
live case. `backwards_search` reaches only `NoHomePlacementStrategy`, whose
`blocked` list is plain `state.occupied` (`_no_return_base.py`).
`block_spectators` belongs to `PhysicalPlacementStrategy`, which builds its
`SolveOptions` in `_move_search_from_traversal` and never sets
`backwards_search`, so that flag is irrelevant to this option. And
`ConcreteState.__post_init__` (`analysis/placement/lattice.py`) asserts
`occupied.isdisjoint(layout)`, so `initial ∩ blocked` cannot occur on this
path — only `target ∩ blocked` can trip the guard, and such an instance is
unsolvable *forward* too, since no move may land on a blocked location and no
atom starts there. The guard therefore never de-mirrors a layer that was
solvable in the first place. It stays because the
failure it prevents (a fabricated plan parking an atom on an external one) is
invisible to the replay verifier, and because `SolveOptions` is a public surface
that callers other than this one path can set.

## An unexploited performance opportunity

`solve_with_engine` builds a fresh `DistanceTable` per solve, keyed on the
target locations. Under palindrome the *forward* target changes every layer (a
new CZ pairing) while the *mirrored* target is always the resting layout — so
the mirrored path rebuilds the same table on every CZ layer.

An architecture-level cached table (built once, covering every location) would
remove that per-solve build entirely on this path. Unmeasured; it trades one
BFS-per-target-location per solve for a one-time build, so it should favour
circuits with many CZ layers.

## Open questions

1. **Does it help where it should?** The synthetic gains are real but the suite
   cannot observe them. This needs a congested kernel before the option's value
   is settled either way.
2. **Fidelity.** Layer counts favour mirroring, lane counts are mixed, and the
   only non-zero-fidelity cases in the suite disagree (`steane_physical_35`
   better, `ghz_6` worse). Unresolved.
3. **Should it be on by default off the palindrome path?** No evidence either
   way. It is opt-in, and `make_physical_placement_strategy` enables it only
   where the gradient is structural.

## Reproducing

Probe scripts are not committed. Each builds a `TargetSolver` over
`bloqade.lanes.arch.gemini.physical.spec.get_arch_spec()`, generates endpoint
pairs with a fixed seed — `scattered` as uniformly random locations, `paired` as
atom pairs on both words of an entangling pair at a common site — and compares
`result.nodes_expanded` and `len(result.move_layers)` between `solve(a, b)` and
`solve(b, a)`.
