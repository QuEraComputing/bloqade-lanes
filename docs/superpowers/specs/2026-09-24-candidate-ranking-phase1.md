# Candidate ranking, phase 1: re-measure with real candidate sets

**Date:** 2026-09-24. **Plan:** Epic 4, phase 1 of
[`plans/2026-08-18-search-refactor-epics.md`](../plans/2026-08-18-search-refactor-epics.md).
**Harness:** `python/benchmarks/ranking.py` (`uv run --no-sync python -m benchmarks.ranking`,
from `python/`), on top of the Epic 2B stack.

## Question and bar

Does choosing among a CZ stage's candidate target placements by a cheap Push-and-Rotate plan
beat the choice the pipeline makes today?

**Go / no-go, fixed before the run:** summed over the physical benchmark kernels, the
Push-and-Rotate-ranked pick routes in **at least 5% fewer operations** than today's pick,
with **no success regressions**, judged per candidate space.

## Method

A per-stage counterfactual along today's own compile trajectory:

1. Compile every physical benchmark kernel with the pipeline as it is, recording each CZ
   stage exactly as the router sees it: the starting placement, the blocked sites, and the
   router's own search configuration and budget.
2. For each stage, enumerate the candidate targets of a space.
3. Route every candidate twice from the same start: with the pipeline's router, to cost it
   (operations = move layers), and with Push and Rotate, to rank it.
4. Compare today's pick with the Push-and-Rotate pick (the candidate its plan is cheapest
   for, ties to the earlier candidate), both costed by the router. "Best" is the cheapest
   candidate by the router — an oracle, not a policy.

Two candidate spaces:

- **`nohome` — the default pipeline's per-pair mover choice.** `pipeline_default` runs
  NoHome under palindrome, which routes each stage once, to targets from a fixed per-pair
  rule (`resolve_cz_targets`): the control moves to the target's partner site if both atoms
  share a word, else the target moves if it sits on a home site, else the control. The
  candidates are the rule's assignment and the other assignments of "which atom moves" —
  all of them up to 64, otherwise the rule, every single-pair flip and seeded random
  samples. The harness re-derives the rule in Python and checks it against NoHome's own
  placement on every solved stage: **409 of 409 match.**
- **`generator:*` — the Python target generators** (`CongestionAwareTargetGenerator`,
  `AODClusterTargetGenerator`, `LookaheadCongestionAwareTargetGenerator`) under
  `PhysicalPlacementStrategy`, whose candidate list is the generator's output plus the
  default; today's pick is the first candidate that routes.

## Results

Operations summed over all 9 physical kernels and 409 stages:

| space | stages with a choice | today | ranked | best | improvement | captured | regressions | verdict |
|---|---|---|---|---|---|---|---|---|
| `nohome` | 409 | 1,625 | 1,543 | 1,461 | **5.05%** | 50% | 0 | **GO (marginal)** |
| `generator:congestion_aware` | 52 | 1,657 | 1,651 | 1,638 | 0.4% | 32% | 0 | NO-GO |
| `generator:aod_cluster` | 76 | 1,560 | 1,580 | 1,555 | −1.3% | — | 0 | NO-GO |
| `generator:lookahead_congestion_aware` | 52 | 1,669 | 1,663 | 1,650 | 0.4% | 32% | 0 | NO-GO |

`nohome` per kernel (today → ranked, best):

| kernel | stages | today | ranked | best | improvement |
|---|---|---|---|---|---|
| trotter_rand_35 | 56 | 552 | 501 | 456 | 9.2% |
| adder_64 | 257 | 772 | 741 | 712 | 4.0% |
| ghz_4 | 2 | 6 | 5 | 5 | 16.7% |
| ghz_6 | 3 | 10 | 9 | 9 | 10.0% |
| steane_physical_35 | 3 | 48 | 50 | 42 | −4.2% |
| adder_4, bv_70, qpe_9, steane_logical_5 | 88 | 237 | 237 | 237 | 0% |

## Verdict and reading (per stage)

- **NoHome's mover choice: GO, by the letter — 5.05% against a 5% bar.** Ranking by a
  Push-and-Rotate plan captures half of what an oracle over the same candidates would. The
  win is concentrated where routing is hard (trotter_rand_35, adder_64); on the easy
  kernels every candidate costs the same. It is not uniform: on steane_physical_35 the
  ranking picks a worse candidate than the rule.
- **The Python generators: NO-GO.** They rarely offer a real choice (52–76 of 409 stages),
  and when they do, Push-and-Rotate does not rank them better than their own order. This
  is consistent with the generators already encoding congestion heuristics that the
  cheap plan does not see.

## End to end

The per-stage figures cost every stage from today's placement. To follow the
trajectory a different pick leads to, `--end-to-end` compiles each kernel whole with
NoHome's mover choice replaced at every stage, and reads the benchmark row's metrics.

- `rule`: the rule's candidate, routed by the router. This is the control; it
  reproduces `pipeline_default` exactly on all 9 kernels (events, lanes, fidelity).
- `ranked`: the Push-and-Rotate-ranked candidate.
- `oracle`: every candidate routed, the cheapest kept, stage by stage. Greedy, and
  implementable (it is what `PhysicalPlacementStrategy`'s candidate loop does), but it
  routes every candidate.

No stage fell back to NoHome in any run, and all runs succeed on all 9 kernels.

| kernel | rule events | ranked | oracle | ranked picks changed | fidelity, ranked ÷ rule |
|---|---|---|---|---|---|
| trotter_rand_35 | 1,104 | 1,008 (−8.7%) | 900 (−18.5%) | 35 / 56 | ×36 |
| adder_64 | 1,544 | 1,478 (−4.3%) | 1,422 (−7.9%) | 28 / 257 | ×95 |
| ghz_4 | 12 | 10 (−16.7%) | 10 | 1 / 2 | ×1.009 |
| ghz_6 | 20 | 18 (−10.0%) | 18 | 1 / 3 | ×1.013 |
| steane_physical_35 | 96 | 98 (+2.1%) | 84 (−12.5%) | 3 / 3 | ×0.93 |
| adder_4, bv_70, qpe_9, steane_logical_5 | 474 | 474 | 474 | 1 / 88 | ×1 |
| **total** | **3,250** | **3,086 (−5.05%)** | **2,908 (−10.5%)** | | |

- **The per-stage result holds end to end**, to the event: 5.05% fewer events, the
  same kernels gaining and the same one losing. Following the changed trajectory
  neither compounds nor erodes the win.
- **Lanes barely move** (3,868 → 3,884 ranked, 3,844 oracle): ranking trades the number
  of move operations, not the atoms moved.
- **Fidelity follows events.** It rises on every kernel whose events fall, and drops on
  steane_physical_35 (0.0106 → 0.0098).
- **Wall time** (single run, both of the runner's compiles, all kernels): rule 38 s,
  ranked 40 s (+4%), oracle 74 s (about 2×; trotter_rand_35 4×).

## Caveats

- **Per-stage figures are settled by the end-to-end run** above, which agrees with them.
- **The margin is thin.** The run is deterministic, but 5.05% clears the bar by 0.05
  points, and one kernel (trotter_rand_35) contributes 62% of the saving (51 of 82 ops).
- **Cost was not measured separately.** The whole `nohome` pass, which routes every
  candidate with both the router and Push and Rotate, took 38 s over all kernels; the
  ranking's own share, which phases 2–3 would pay per stage, is not broken out.

## Recommendation for phases 2–3

Choose **inside `NoHomeCzPlacement`, over the per-pair mover choice**: the only space that
passed, and the one that reaches `pipeline_default` directly (decision 2's open half). The
Python-generator route is not worth pursuing on this evidence.

The end-to-end run adds a second option. Ranking by Push and Rotate recovers half the
available saving for about 4% more compile time. Routing every candidate and keeping the
cheapest recovers all of it (10.5%, and no kernel regresses) for about twice the compile
time. Which one ships, and whether it becomes the default, is a speed-versus-quality call.
