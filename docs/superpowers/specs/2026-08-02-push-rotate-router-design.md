# Push and Rotate: a complete router as a reliability net

**Status:** implemented — branch `feat/push-rotate-router`, stacked on
`feat/move-feasibility` (which it depends on for the decomposition).
**Date:** 2026-08-02
**Author:** Phillip Weinberg (with Claude)

## Summary

`crates/bloqade-lanes-search/src/push_rotate/` implements Push and Rotate
(de Wilde, ter Mors & Witteveen, *Push and Rotate: a Complete Multi-agent
Pathfinding Algorithm*, JAIR 51 (2014) 443–492). It is **complete** at two or
more empty locations: it finds a solution whenever one exists.

It is exposed two ways:

- `Strategy::PushRotate` — selectable, so it can be benchmarked next to the
  search strategies.
- `SolveOptions::fallback_push_rotate` — **default off**. When on, a search
  returning anything other than `Solved` is retried with the planner.

**It is a reliability net, not a primary strategy.** It produces more AOD
operations than the search strategies on instances they can solve, and is
roughly two orders of magnitude faster. The measurements and the reason it
cannot be made competitive on operation count are below.

## Why this exists

The search strategies fail outright on some instances. On
reachable-by-construction instances — atoms scattered, then displaced by a
random walk of legal single-atom slides, so a solution provably exists:

| group | astar | ids | entropy | push-rotate |
|---|---|---|---|---|
| physical/k8 | **0/5** | 5/5 | 5/5 | **5/5** |
| physical/k16 | **0/5** | 5/5 | 5/5 | **5/5** |
| logical/k8 | **3/5** | 5/5 | 5/5 | **5/5** |
| logical/k16 | **1/5** | **4/5** | **4/5** | **5/5** |

Wall time, per instance on physical/k16:

| | median | max |
|---|---|---|
| ids | 1.76 ms | 1.97 ms |
| entropy | 10.0 ms | 100.0 ms |
| push-rotate | **0.20 ms** | **0.21 ms** |

Its runtime is essentially flat in atom count (0.16 ms at k=1, 0.20 ms at
k=16) and its max equals its median — it is rule-based, so there is no search
variance. That is what makes it viable as a fallback: trying it costs nothing.

Note that **IDS, not entropy, is the incumbent to beat**. It is the only
search strategy that neither blows up nor degrades, and it is ~9x slower than
the planner rather than 50x.

## Why it is not a primary strategy

Operation count, on the instances the searches *can* solve:

| physical/k16 | ops | xfer_us |
|---|---|---|
| ids | 110 | 65266 |
| entropy | 103 | 62098 |
| push-rotate | 163 | 97506 |

This is structural, not a tuning gap. The diagnosis, in the order it was
established:

1. **The dependency DAG is not the constraint.** Its longest path at k=16 is
   **32** against 180 moves — a 5.6x reduction is available in the ordering —
   but the scheduler only reaches 163. The `headroom` example reproduces this.
2. **Geometry is the constraint, and specifically bus membership.**
   Instrumenting the ready set: with ~6.8 moves ready simultaneously, the
   largest subset sharing a bus group averages **1.57**, and the largest
   same-row subset within a group averages 1.18. Rectangle *shape* was never
   binding; concurrent moves are simply on different buses.
3. **The move set, not the move order, is at fault.** The scheduler already
   reorders freely across agents. What it cannot do is re-*choose*. Push and
   Rotate routes atoms one at a time, so a plan is the union of *k*
   independently selected shortest paths, and independently selected paths
   through a bus-structured graph rarely share a bus.

A tie-break between equally short paths is therefore too weak a lever: the
paths of different atoms are chosen in separate BFS calls that never see each
other. `AlignmentHeuristics` implements the best available version of it —
score a step by how many other unfinished atoms could also progress on that
bus — and buys 163 → 159 on physical/k16 and 257 → 240 on logical/k16. Real,
but not the win.

Closing the gap needs parallelism in the **move alphabet**, not in a
heuristic. That is what the open entropy generator already does by proposing
whole AOD rectangles as candidates, and it is why a fairly blunt search
reaches 2.3 atoms per operation. Push and Rotate's `push`, `swap` and `rotate`
are all defined over single-atom motion and its completeness proof rests on
that, so changing the alphabet means a different algorithm — approximately the
one entropy already is.

## Design

### Where it plugs in

`solve_with_engine` in `search/target_solver.rs`, which is where a concrete
target placement exists. `Strategy::PushRotate` bypasses the frontier
machinery entirely; the fallback runs after `run_with_components` returns a
non-`Solved` result.

`Strategy::PushRotate` is honoured only on the **fixed-target** path. The
loose-goal path deliberately leaves the target open for the Hungarian
assignment, so there is nothing for a fixed-target router to aim at; it
substitutes A* there rather than reaching the `unreachable!` in
`run_strategy_v2`.

### The three-way answer

Because the planner is complete, its `Unsolvable` is a **proof**, unlike a
search whose frontier merely drained. With the fallback on:

| outcome | meaning |
|---|---|
| search succeeds | normal path |
| search fails, planner succeeds | recovered; worse ops, but a schedule where there was none |
| both fail | provably impossible on this device |

On the double-failure path the planner's verdict is returned for exactly that
reason.

### Structure

| module | contents |
|---|---|
| `mod.rs` | Algorithms 7–8: feasibility check and the `solve` loop |
| `ops.rs` | Algorithms 4, 6, 10–13: push, swap, rotate and auxiliaries |
| `state.rs` | placement + move log, with checkpoint/rollback |
| `smooth.rs` | Algorithm 9: redundant-move removal |
| `schedule.rs` | DAG list scheduler into AOD operations (the paper's `condense` is unspecified) |
| `context.rs` | read-only context, including per-edge bus group and position |
| `heuristics.rs` | the four §5 decision points, as a swappable trait |
| `solver.rs` | `SolveResult` adapter |
| `instances.rs` | reachable-by-construction instance generator for tests and benchmarks |

The Kornhauser decomposition (Algorithms 1–3) is **not** here — it lives in
`feasibility/`, shared with the infeasibility checker, because it is the same
computation.

### Heuristic seam

`PlanHeuristics` covers the four places §5 identifies as free choices: agent
order, shortest-path tie-break, clear target, swap vertex. `BatchPolicy`
covers the scheduler's one. Methods return a *ranking key* rather than making
the choice, and callers apply it as a tie-break on top of the existing rule —
so a heuristic can reorder equally-good options but cannot select an illegal
one or lengthen a path. A bad heuristic is a quality problem, never a
correctness one.

`DefaultHeuristics` is the trait defaults and is what every number above was
measured with.

## Deviations from the paper

Four of these were bugs found by tests rather than by reading, and all four
only reproduce on instances the physical spec rarely produces.

1. **`swap` and `rotate` reverse only Π′.** In `swap` that is multipush+clear
   (the exchange is appended separately); in `rotate` it is `clear_vertex`
   alone. Reversing to the end of the log undoes the operation itself and
   lands agents on non-adjacent vertices.
2. **`rotate`'s cycle loop runs `len−1` times**, not `len−2` as a literal
   reading gives. The final step vacates `v` for the reversal to refill.
3. **`rotate` verifies the cycle is closed.** `q` in Algorithm 8 accumulates
   vertices across successive agents, so a repeated vertex does not by itself
   mean the suffix is a closed walk.
4. **`smooth` deletes `NV(π)` inclusively.** Line 8's `while π′ ≠ NV(π)` stops
   one short, leaving an agent moving to a vertex it never left; the prose and
   Figure 16 both include the return.
5. **`smooth` adds virtual start arrivals**, so a round trip back to an
   agent's *initial* vertex is detectable. Algorithm 9 only sees moves, and an
   initial position is not one.

`smooth` turns out to remove ~1% on Gemini instances (2 moves of 182 at
k=16). Push and Rotate's round trips are mostly *productive*: the swap
reversal is what accomplishes the swap.

## Testing

`tests/push_rotate.rs`. Correct rearrangement and AOD validity are checked
through `AtomStateData::apply_moves` and `ArchSpec::check_lanes` rather than a
hand-rolled replay — a bug in a reimplemented simulator could mask exactly the
bug it exists to catch. `AtomStateData` also earns its place by recording
collisions (both qubits are dropped from the location maps, so asserting
`collision` is empty catches any operation that would crash two atoms) and by
*silently skipping* a lane whose source is empty, which is why total
`move_count` is asserted against lanes issued.

Coverage spans displacement instances, **pure permutations** (occupied sites
unchanged, atoms shuffled among them — the harder case, forcing in-place
exchange rather than pushing into free space), two-atom transpositions, both
shipped specs, and the solver-level surface.

## Two cautions worth carrying forward

**"The incumbent failed" is not evidence the instance is hard.** An early
finding that A* could not move a single atom one hop looked like a solver bug;
it was a degenerate fixture — `examples/arch/full.json` gives all three of its
words identical grid coordinates, so no well-defined AOD rectangle exists
across them. Filed as
[#859](https://github.com/QuEraComputing/bloqade-lanes/issues/859). Attribute
every tier-1 win before claiming it.

**Lane count is not a quality metric.** One operation moves a whole rectangle,
so packing more atoms per operation *raises* the lane count while lowering
real cost. Optimising for fewer lanes optimises for serialisation. Use `ops`
and transport time.

## Open questions

- **Trigger rate on real circuits.** The failure rates above are on synthetic
  instances. The fallback's value depends on how often the pipeline's actual
  workload hits them, and IDS is more robust than the astar/entropy figures
  suggest.
- **Should the fallback default on?** It is off so this could land without
  moving the committed baselines. Flipping it changes results for every case
  the incumbents currently fail, so it wants its own reviewed PR with
  regenerated baselines.
- **Seeding rather than replacing.** The planner is fast and complete, so it
  could hand a rectangle-aware search a valid witness to improve rather than a
  blank start. Speculative; measure the fallback on real workloads first.

## References

- de Wilde, ter Mors & Witteveen (2014), JAIR 51:443–492.
  <https://jair.org/index.php/jair/article/view/10913>
- Reference C++ implementation, useful as a cross-check on the unspecified
  `condense` pass:
  <https://github.com/PathPlanning/Push-and-Rotate--CBS--PrioritizedPlanning>
- `feasibility/` in this crate — the shared decomposition, and why atom
  rearrangement reduces exactly to pebble motion here.
