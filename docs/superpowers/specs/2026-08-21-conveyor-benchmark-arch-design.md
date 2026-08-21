# Conveyor-capable benchmark ArchSpec

Design for #939 (part of #887, last scope bullet): add a conveyor-capable
ArchSpec to the benchmark registry so chain assembly is measured and protected
by CI.

**Status: implemented.** The suite measures the *architecture*; chain assembly
itself is pinned by tests in `crates/bloqade-lanes-search` rather than by
benchmark metrics — see "Attribution problem" below for why that split is
necessary. No packed-corridor kernel was added, so the shipped baselines are
untouched.

| Artifact | Path |
|---|---|
| Generator (with `--check` drift guard) | `scripts/gen_conveyor_arch.py` |
| Generated spec | `examples/arch/gemini-conveyor.json` |
| Suite recipe | `just benchmark-conveyor` |
| Baseline | `python/benchmarks/harness/latest_conveyor.csv` |
| Spec guards | `python/tests/arch/test_conveyor_arch.py` |
| Harness fix + regressions | `python/benchmarks/harness/runner.py`, `python/tests/benchmarks/test_runner.py` |

## Measured result

Conveyor baseline vs shipped physical baseline, all 81 rows (71 where both
succeed):

| metric | better | same | worse |
|---|---|---|---|
| `move_count_events` | 71 | 0 | **0** |
| `move_count_lanes` | 67 | 4 | **0** |
| `estimated_fidelity` | 58 | 13 | **0** |
| `nodes_explored` | 59 | 3 | **9** |

Plus **4 new successes, 0 lost** — `steane_physical_35` under `rust_dfs`,
`rust_greedy`, `rust_ids`, and `trotter_rand_35` under `rust_greedy`. Mean
−18.9% move events.

### Fidelity is the headline, and it compounds

`estimated_fidelity` is a product of per-gate fidelities, so it is roughly
exponential in operation count — a linear move-count reduction shows up as a
much larger fidelity gain, and the effect is biggest on the deepest circuits:

| case | rows | builtin fid | conveyor fid | relative |
|---|---|---|---|---|
| `steane_physical_35` | 4 | 0.006066 | 0.015584 | **+157%** |
| `qpe_9` | 9 | 0.011876 | 0.016108 | **+36%** |
| `adder_4` | 9 | 0.606020 | 0.642523 | +6% |
| `ghz_6` | 9 | 0.744816 | 0.788346 | +6% |
| `steane_logical_5` | 9 | 0.789301 | 0.830047 | +5% |
| `ghz_4` | 9 | 0.870604 | 0.881078 | +1% |
| `adder_64`, `bv_70`, `trotter_rand_35` | 22 | 0.0 | — | see below |

Over the 49 rows with a nonzero baseline: mean **+30.2%**, median **+5.8%**, max
+650%. The mean is skewed by the low-fidelity cases, so quote the median for
"typical" and the per-case table for anything load-bearing.

`steane_physical_35` is the clearest illustration: −19% move events became a
**2.6× fidelity improvement**. And of the 22 rows whose baseline fidelity
underflows to `0.0`, **9 are lifted off the floor** by the conveyor spec — from
unmeasurable to merely bad. The remaining 13 (`adder_64`, `bv_70`,
`trotter_rand_35` at ~1000-1500 move events) stay at `0.0` under both specs;
those rows cannot report a fidelity delta at all, which is worth knowing before
reading the column.

Note this metric is only trustworthy *because* of the harness fix: fidelity was
previously computed against bundled Gemini regardless of the target arch, so on a
custom spec it was measuring the wrong machine.

### The cost lands entirely on entropy

Output quality never regresses, but search effort does, and it splits cleanly by
strategy family:

| family | rows | mean `nodes_explored` delta | worst |
|---|---|---|---|
| non-entropy (astar/bfs/dfs/ids/greedy) | 35 | **−23.7%** | −2% (every row cheaper) |
| entropy (1/5/10/20) | 36 | −2.8% | **+139%** |

All nine more-expensive rows are `rust_entropy_*` on the three large cases, and
every one of them still produced a *better* plan — so this is a genuine
effort-for-quality trade, not a regression:

```
steane_physical_35  rust_entropy_5    nodes 1808 -> 4320  (+139%)   events 108 -> 88  (-19%)
steane_physical_35  rust_entropy_20   nodes 2849 -> 5184   (+82%)   events 104 -> 86  (-17%)
trotter_rand_35     rust_entropy_10   nodes 35538 -> 54576 (+54%)   events 1136 -> 1028 (-10%)
adder_64            rust_entropy_5    nodes 15395 -> 20349 (+32%)   events 1540 -> 1492  (-3%)
```

The asymmetry is the mechanism worth remembering. The distance-guided searches
get *cheaper* because more lanes shorten graph distances: the plan is shorter and
the heuristic that guides them is correspondingly tighter, so they win on both
axes. Entropy instead enumerates candidate move **sets** per node, and richer
buses inflate that per-node combinatorics (scaled further by
`max_goal_candidates`); on large packed instances the branching growth outruns
the path shortening.

So "more lanes per bus improves performance" holds for output quality here, and
is a free lunch for most strategies — but it is not free for the candidate-
enumerating ones.

### Why this motivates conveyor buses as a hardware extension

Read forward rather than defensively, this suite is the first quantitative case
for overlapping site buses as a *hardware* direction, and it is a strong one:

- **Fidelity improves on every row that can move**, by a median +6% and up to
  2.6× on the deepest circuit — and it lifts 9 rows off the underflow floor.
  Fidelity is the metric that decides whether a circuit is runnable at all, so
  moving it is worth real hardware cost.
- **Nothing regresses.** Across 81 rows there is no case, strategy, or metric
  where the conveyor spec loses on output quality, and 4 previously-failing rows
  start solving.
- **The compiler already exploits it.** No solver work is required to collect
  this: chain assembly (#896/#919) and the existing generators pick the shorter
  plans up for free. The extension would land against a router that is already
  ready for it.
- **The cost is understood and bounded** — a larger per-node candidate set for
  the entropy strategies (worst observed +139% nodes), with distance-guided
  strategies getting strictly *cheaper*.

So the case to make upstream is not "the router got faster" but "a modest change
to site-bus topology buys a fidelity improvement that compounds with circuit
depth, on a compiler that can already use it."

### What it does not establish

The spec is physically hypothetical, so the hardware question is genuinely open.
Its 100 new transport paths were synthesized in the bundled spec's waypoint
idiom, whereas the real Gemini lane paths were derived from qlue kernels (#753).
What a feasibility study would have to answer:

- Can the AOD realize a stride-1 overlapping site bus at all, given the
  clearance geometry the synthesized waypoints assumed?
- Does a new lane cost what an existing one costs? Every metric here assumes so.
  If overlapping lanes carry a duration or heating penalty, the fidelity gain
  shrinks — and `MoveMetricCalculator` derives durations from path segment
  lengths, so a revised path set changes these numbers.
- Is the win robust to real lane durations rather than idiom-matched synthetic
  ones?

Until those are answered these numbers describe a *compiler* result on a
plausible architecture, not a validated hardware proposal.

## Problem

Chain assembly is dead code on both shipped Gemini specs. A site bus is an
elementwise hop map `src[i] -> dst[i]`, and every shipped bus is
*endpoint-disjoint* (`set(src) & set(dst) == {}`). `vacating_lane()`
(`crates/bloqade-lanes-search/src/ops/aod_grid.rs:103`) looks for an outgoing
lane from a blocked destination on the same bus; on a disjoint bus that is
necessarily `None`, so every chain path — the #896 grid-layer repair closure and
the #919 selection-time `close_chain_entries` — is unreachable. The benchmark
suites were byte-identical before and after both PRs, and CI cannot see a chain
regression.

Measured evidence to date is ad hoc: #896 measured roughly a 20% move-count
reduction on a 98-atom Ising kernel on an overlapping-bus spec, and #919 added
`tests/conveyor_1d.rs` (correctness plus optimality against a brute-force
oracle). Nothing in `python/benchmarks` exercises a chain-capable spec.

## Rejected: scale up `line_arch_json`

#939 suggests scaling the search crate's `chain_arch_json()` /
`line_arch_json(n)` fixtures. Those cannot serve as benchmark archs. The harness
threads the custom spec only into the *placement* stage; the layout stage
(`BenchmarkRunner._build_layout_heuristic`,
`python/benchmarks/harness/runner.py:263`) constructs
`PhysicalLayoutHeuristicGraphPartitionCenterOut()` with no arguments, which
defaults to the bundled Gemini spec and derives home words from zone entangling
pairs. A bare one-word conveyor line has no entangling pairs and no word buses,
so the layout stage cannot place anything and the existing kernels cannot route.
A benchmark conveyor spec has to stay Gemini-shaped.

## Design: Gemini physical with every hypercube dimension in conveyor form

Take `_physical_spec.json` (20 words x 8 sites, 1 zone, 32x5 grid, entangling
pairs `[[0,1],...,[18,19]]`, 1120 transport paths) and convert each of the three
hypercube site buses to conveyor form. Dimension `d` has stride `s = 2^d`; the
conveyor bus is `i -> i + s` for every valid `i`:

| dim | stride | builtin lanes | conveyor lanes | src & dst overlap |
|---|---|---|---|---|
| 0 | 1 | `0->1, 2->3, 4->5, 6->7` | `0->1, 1->2, ... 6->7` | `{1..6}` — one 8-long chain |
| 1 | 2 | `0->2, 1->3, 4->6, 5->7` | `0->2, 1->3, 2->4, 3->5, 4->6, 5->7` | `{2,3,4,5}` — two 4-long chains |
| 2 | 4 | `0->4, 1->5, 2->6, 3->7` | identical | `{}` — already disjoint, fixed point |

Everything else in the spec is untouched: words, grid, entangling pairs, word
buses, `words_with_site_buses` (odd words 1..19), modes.

### Why this shape

**Each conveyor bus is a strict lane superset of the hypercube bus it
replaces.** Verified for all three dimensions. Traversing a longer distance
along a strided conveyor now takes several hops where the hypercube did one bit
flip, which is the accepted cost. The consequences that matter:

- The conveyor spec can do everything builtin can, plus chains — so *plan
  existence* is monotone. This is weaker than "results can only improve": every
  strategy is a bounded heuristic search, so a strictly larger space can make it
  find worse plans or exhaust its budget. What the property does buy is that a
  conveyor regression can never be blamed on lost connectivity.
- Geometry is byte-identical to builtin, so the layout heuristic, entangling
  structure and every existing kernel work unchanged. A conveyor row differs
  from its builtin row only by bus structure — a controlled A/B.
- Dimension 2 is a fixed point, so the conversion is not uniform in effect: dim
  0 gains the long chain, dim 1 gains two, dim 2 gains nothing.

All three conveyor buses remain acyclic with unique `src` and unique `dst`, so
`check_bus_relation` accepts them. Wrap-around (`n-1 -> 0`) would be rejected as
`CyclicBus` and is not used.

### Transport paths are additive

A `LaneAddr` is keyed by *source site id*, not by an index into the bus `src`
array (`crates/bloqade-lanes-bytecode-core/src/arch/addr.rs:124`). So converting
a bus leaves every existing path entry valid and byte-identical, and only *adds*
entries for the new lanes. The waypoint idiom in the bundled spec is exactly
uniform — for a lane `i -> j` in word `w`:

```
[[xi, yw], [xi, yw + 5], [xj, yw + 5], [xj, yw]]
```

with `y_clearance = 5.0` in the `+y` direction for all ten site-bus words. New
lanes are synthesized in the same idiom, so path generation is exact rather than
approximate. New lanes per word: 3 on dim 0 (`1->2, 3->4, 5->6`), 2 on dim 1
(`2->4, 3->5`), 0 on dim 2 — 5 lanes x 2 directions x 10 words = **100 added
entries, 1120 preserved, 1220 total**.

Intermediate-site clearance is not a new concern: the existing dim-1 and dim-2
paths already arc over intervening sites via the same `+5` detour.

### Generation, not duplication

The spec is generated from the bundled one by a committed script rather than
checked in as a ~144KB near-duplicate JSON. This keeps it honest — if
`_physical_spec.json` changes, regeneration tracks it, and the diff between the
two specs stays reviewable as "three bus rewrites plus 100 paths".

A complementary option is a `ConveyorSiteTopology` in
`python/bloqade/lanes/arch/build/topology.py`. The `SiteTopology` protocol slot
is vacant for this — `HypercubeSiteTopology`, `AllToAllSiteTopology` and
`TransversalSiteTopology` exist and none produces an overlapping bus. Worth
adding for reuse, but the benchmark spec still wants to derive from the bundled
Gemini geometry, so the script is the primary path.

## Harness changes required

Two hardcoded-Gemini spots in `python/benchmarks/harness/runner.py` make the
custom spec disagree with the pipeline. Both currently *warn* rather than fail,
and the observable symptom is a confusing downstream error
(`SSAValue <ResultValue[State] stmt: fill, uses: 2> not found`) rather than the
mismatch itself:

1. **`runner.py:245`** — `arch_spec = get_physical_arch_spec()` is passed to
   `PhysicalPipeline` and `MoveToSquinPhysical` while the placement strategy
   carries the custom spec. Use `placement_strategy.arch_spec`.
2. **`runner.py:263`** — `_build_layout_heuristic` constructs
   `PhysicalLayoutHeuristicGraphPartitionCenterOut()` with no arguments. Pass
   the job's arch spec. This is safe: `get_physical_layout_arch_spec()` is
   literally `return get_arch_spec()`
   (`python/bloqade/lanes/arch/gemini/physical/spec.py:29`), so there is no
   separate layout spec to preserve.

Both are latent bugs for *any* `--arch-spec` use, not just this one — the flag
is effectively broken today for the physical-initialize path. Worth noting the
mismatch is currently a `UserWarning`; consider raising it to an error so the
next caller gets the real message.

`estimated_fidelity` is computed through the same `arch_spec` local, so fixing
(1) also fixes fidelity being measured against the wrong architecture. Fidelity
on the conveyor arch is a different noise integral than builtin, so its column
is *not* comparable across arch ids — only within one.

## Suite wiring: own opt-in suite

`--arch-spec` *replaces* the builtin spec rather than adding to it, and row
identity already includes `arch_spec_id`. So a conveyor-only baseline compares
cleanly against a conveyor-only run with **no change to `cli.py`**:

```
just benchmark-conveyor
  -> python -m benchmarks.cli --architecture physical \
       --arch-spec <spec> --compare python/benchmarks/harness/latest_conveyor.csv
```

The CI benchmark job is already parameterized by `architecture`, so enabling it
was a one-word matrix change — `just benchmark-<arch>` and
`latest_<arch>.csv` both fall out of the existing pattern.

One sharp edge, found by running the recipe: `--output` is **required**. The
default output path is derived from `--architecture` (`physical` here), not from
`--compare`, so omitting it overwrites `latest_physical.csv` with conveyor rows
while still reporting "no differences found" against the conveyor baseline. That
would have silently replaced the shipped baseline in CI.

Otherwise the shipped baselines are structurally untouchable by this suite, which
directly satisfies #939's "any diff outside the new rows is a red flag".

## Attribution problem: the superset property conflates two effects

This is the finding that most shapes the kernel work. Measured on `ghz_6`,
builtin vs conveyor, all nine strategies:

```
strategy            builtin ev/ln   conveyor ev/ln   delta
rust_astar                  20/26            14/20   -30% ev, -23% ln
rust_greedy                 20/26            14/20   -30% ev, -23% ln
   (identical for bfs, dfs, ids, entropy_1/5/10/20)
```

A real and consistent win at unchanged `success` — but `rust_greedy` shows the
*same* win, and per #938 `GreedyGenerator` is structurally chain-incapable
(`find_path_occupied` treats every atom as a wall, so neither the #896 repair
closure nor #919's `close_chain_entries` can fire). So on `ghz_6` the win is the
richer lane set, **not** chain assembly.

The packed case `steane_physical_35` confirms this is not a `ghz_6` artifact:

```
strategy            builtin ev/ln   conveyor ev/ln   delta
rust_entropy_5            108/166           88/138   -19% ev, -17% ln
rust_greedy                  FAIL           86/138   new success, best events
rust_ids                     FAIL           90/134   new success
rust_astar                   FAIL             FAIL
```

Two things worth separating. The move-count delta on `rust_entropy_5` (-19%)
reproduces #896's ad-hoc ~20% measurement, which is reassuring. And the superset
property pays off in `success`: greedy and IDS go from failing to solving. But
chain-incapable greedy posts the *best* event count of all four strategies, so
connectivity — not chain assembly — dominates the metric on this workload too.

A sweep of the five fast cases settles it. Every case improves, and in every
case **greedy's delta is identical to the chain-capable strategies'**:

| case | builtin ev/ln | conveyor ev/ln | delta (all four strategies) |
|---|---|---|---|
| `steane_logical_5` | 18/26 | 12/20 | -33% ev, -23% ln |
| `ghz_6` | 20/26 | 14/20 | -30% ev, -23% ln |
| `adder_4` | 38/50 | 30/42 | -21% ev, -16% ln |
| `ghz_4` | 12/14 | 10/12 | -17% ev, -14% ln |
| `qpe_9` | 220/224 | 196/200 | -11% ev, -11% ln |

Sweep covered `rust_astar`, `rust_entropy_5`, `rust_greedy`, `rust_ids`. The only
divergence anywhere is `ghz_4`/`rust_entropy_5` landing 10/14 where the other
three get 10/12 — entropy slightly *worse* on lanes, which is noise-level and not
a chain effect.

So: **no existing kernel isolates chain assembly.** The whole -11% to -33% band
is the connectivity superset. This is a genuine architectural win worth recording
in a baseline, but it is not what #939 set out to quantify.

The superset property is therefore double-edged: it makes `success` monotone and
the comparison clean, but it conflates "more lanes" with "chains". Since
chain-capability *is* the `src`/`dst` overlap, there is no spec pair with an
identical lane set and chains toggled — the effects cannot be separated by spec
alone.

**Use `rust_greedy` as the built-in control.** It is chain-incapable by
construction, so on the conveyor arch it measures the connectivity-only effect.
Chain assembly is the *gap* between the chain-capable strategies and greedy,
relative to their gap on builtin. That makes the greedy row load-bearing rather
than incidental, and it should be called out in the harness docs so nobody
"optimizes" it away.

This also gives the acceptance criterion real teeth: #939 expects fewer
`move_count_events`/`move_count_lanes`, but on both cases measured so far a bare
improvement is satisfiable by connectivity alone. The kernel must produce a
**greedy-vs-chain-capable gap**.

### Resolution: benchmarks measure the arch, tests pin the chains

Rather than engineer a kernel whose only purpose is to make greedy deadlock, the
split adopted is:

- the **conveyor suite** records what the architecture buys (connectivity plus
  chains, not separable) with current success semantics and no new kernel; and
- **chain assembly is pinned by explicit solver tests**, which can name the
  mechanism directly instead of inferring it from aggregate metrics.

`rust_greedy` stays in the matrix as the chain-incapable control, so the
connectivity share of any future delta remains readable.

The alternative below is recorded because it is still the sharpest *benchmark*
gate available, should one ever be wanted.

### Alternative considered: gate on `success`, not move count

Move counts on a chain-capable spec are a blend of two effects and greedy can
win them outright, so they make a weak CI gate. The sharp signal is the one
#887's scope bullet 4 already names: a workload where a chain is the *only* way
to make progress — a fully-packed corridor whose sole free site is at the head
of the chain. There, chain-incapable greedy **deadlocks** while the chain-capable
strategies solve.

That turns the gate into a boolean per row: `success=False` for `rust_greedy`,
`success=True` for the heuristic-generator-backed strategies and entropy. A
regression in chain assembly flips a `True` to `False`, which no amount of
connectivity change or metric noise can mask, and which the compare gate already
treats as a hard difference. Move-count deltas stay in the CSV as secondary
evidence.

This is a stronger property than #939 asks for, and it is the thing the ad-hoc
#896 measurement could never protect.

## Kernel work: none

No new kernel was added. Its only purpose was the greedy-deadlock gate, which
the tests-not-metrics split replaces. Because kernel discovery is global (any
`.py` under `benchmarks/kernels/{small,medium,large}/` with exactly one
`ir.Method`), skipping it is what keeps the shipped physical and logical
baselines byte-identical — the conveyor suite runs the existing nine cases only.

## Chain-assembly tests

The coverage survey found #887's two "unchecked" scope boxes are in fact already
covered — `search/target_solver.rs:407` runs `MoveSearch::entropy()` and asserts
`move_layers.len() == 1`, and `tests/conveyor_1d.rs:276` case `(10, 9, 1)` is a
fully-packed chain-only corridor with no escape route, asserted for astar, ids
and entropy. The real gaps were elsewhere:

- **A stale `#[ignore]`.** `drivers/entropy.rs`
  `generate_candidates_allows_follow_moves_into_moving_occupants` was disabled
  for a reason ("greedy_init ... fail to seed valid single-element rects") that
  #896's repair path obsoleted. Un-ignored; it passes, and it now asserts
  `move_set.len() == 3` so it pins one operation rather than just the placement.
- **Entropy preference was untestable as posed.** On a pure conveyor instance
  entropy emits the chain as the *sole* candidate — the serialized rival is
  pruned before scoring, so a score comparison passes vacuously. Replaced with
  `entropy_offers_only_the_chain_when_it_is_strictly_better`, asserting no
  serialized candidate exists.
- **A no-op guard for the shipped specs.** `vacating_lane` is the primitive every
  chain path funnels through, so `vacating_lane_is_always_none_on_an_endpoint_disjoint_bus`
  proves the feature is unreachable on Gemini exhaustively over the bus, rather
  than spot-checking one site.

### Which layer assembles a chain — the discriminator for writing these tests

When every atom in a run is targeted, all are nominated by scoring and the
**grid-layer repair** (#896 `rect_outcome`/`Repairable`) assembles the chain
alone; stubbing `close_chain_entries` changes nothing. The selection-time closure
only becomes load-bearing when a follower is **untargeted**, since an unnominated
spectator is invisible to the repair (which can only pull in cells whose source
is already an entry). Verified: with only the leader targeted and the closure
stubbed, entropy generates *zero* candidates — cause 3 of #910, a silent
deadlock. Any test meant to pin `close_chain_entries` must therefore target only
the leader, or it will pass with the closure removed.

Remaining known gap, not addressed: the entropy **deadlock-breaker** chain path
(`drivers/entropy.rs:488`) has no coverage, in particular the documented
invariant that followers are exempt from the `target_movers` cap.

Gotcha: kernel discovery is global (any `.py` under
`benchmarks/kernels/{small,medium,large}/` with exactly one `ir.Method`), so a
new kernel adds rows to the shipped physical **and** logical baselines too, not
just the conveyor suite. All three baselines need regenerating, and the
determinism re-run per AGENT.md applies to each.

## Open questions

- ~~Does any existing kernel isolate chain assembly?~~ **Answered: no.** Six of
  the nine cases measured (five fast plus `steane_physical_35`); greedy's delta
  matches the chain-capable strategies' in every one. The packed-corridor kernel
  is the only possible witness, so #939 cannot be closed without it. The three
  unmeasured large cases (`adder_64`, `bv_70`, `trotter_rand_35`) are unlikely to
  differ in kind but should be run when the baseline is generated anyway.
- `rust_astar` fails on `steane_physical_35` under *both* specs. Unrelated to
  this work (it is a pre-existing expected failure in the physical suite), but
  confirm it stays put rather than being masked.
- Should the connectivity win be split out as its own result? The -11% to -33%
  band is a real finding about Gemini's site-bus topology — conveyor buses
  strictly dominate hypercube buses on every workload measured — and it is
  arguably more valuable than the chain measurement #939 asked for. It may
  deserve its own issue rather than being buried in a chain-assembly baseline.
- Sizing: 8 sites per word caps a dim-0 chain at 8 and dim-1 chains at 4. If
  that is too short for a decisive gap, the alternative is a wider word — but
  that changes geometry and forfeits the controlled A/B against builtin. Prefer
  keeping 8 and letting the corridor kernel span words.
- Should the conveyor spec also convert *word* buses (19 column-pair shifts) to
  conveyor form? Out of scope here; site buses are where `conveyor_1d.rs` and
  #896 established the win.

## Non-goals

- Fixing #940 (`pack_aod_rectangles` missing the chain closure, so Starlark
  policies still serialize chains). The DSL strategy is deliberately excluded
  from the benchmark matrix, so this suite cannot see it.
- Deciding #938 (greedy chain-incapability). This design *depends* on greedy
  staying chain-incapable and uses it as the control, which is an argument for
  #938's "document it" outcome.
- Changing the shipped Gemini specs.

## Appendix: reference generator

The script used to produce and validate the spec measured above. Kept here rather
than in `scripts/` until the design is accepted. Verified with
`cargo run -p bloqade-lanes-bytecode-cli -- arch validate` (passes) and through
the harness (results in the tables above).

```python
import json, copy

SRC = "python/bloqade/lanes/arch/gemini/physical/_physical_spec.json"
Y_CLEAR = 5.0

spec = json.load(open(SRC))
zone = spec["zones"][0]
grid = zone["grid"]

xs = [grid["x_start"]]
for d in grid["x_spacing"]:
    xs.append(xs[-1] + d)
ys = [grid["y_start"]]
for d in grid["y_spacing"]:
    ys.append(ys[-1] + d)

n_sites = len(spec["words"][0]["sites"])

def word_coords(w):
    return [(xs[i], ys[j]) for i, j in spec["words"][w]["sites"]]

def enc(direction, move_type, zone_id, word_id, site_id, bus_id):
    d0 = ((word_id & 0xFFFF) << 16) | (site_id & 0xFFFF)
    d1 = ((direction & 1) << 31) | ((move_type & 3) << 29) \
         | ((zone_id & 0xFF) << 21) | (bus_id & 0xFFFF)
    return d0 | (d1 << 32)

# 1. convert each hypercube site bus to conveyor form (stride 2^d)
old_buses = copy.deepcopy(zone["site_buses"])
new_buses = []
for d in range(len(old_buses)):
    stride = 1 << d
    src = list(range(0, n_sites - stride))
    new_buses.append({"src": src, "dst": [i + stride for i in src]})
zone["site_buses"] = new_buses

# invariant: every conveyor bus is a superset of the hypercube bus it replaces
for d, (old, new) in enumerate(zip(old_buses, new_buses)):
    o = set(zip(old["src"], old["dst"]))
    n = set(zip(new["src"], new["dst"]))
    assert o <= n, f"bus {d} not a superset: missing {o - n}"

# 2. add paths for the new lanes only; existing entries are keyed by source
#    site id and stay byte-identical
existing = {int(p["lane"], 16) for p in spec["paths"]}
for w in zone["words_with_site_buses"]:
    coords = word_coords(w)
    for bus_id, bus in enumerate(new_buses):
        for s, t in zip(bus["src"], bus["dst"]):
            (x0, y0), (x1, y1) = coords[s], coords[t]
            wp = [[x0, y0], [x0, y0 + Y_CLEAR], [x1, y1 + Y_CLEAR], [x1, y1]]
            for direction, path in ((0, wp), (1, list(reversed(wp)))):
                lane = enc(direction, 0, 0, w, s, bus_id)
                if lane not in existing:
                    spec["paths"].append(
                        {"lane": f"0x{lane:016x}", "waypoints": path}
                    )
                    existing.add(lane)

json.dump(spec, open("gemini_conveyor.json", "w"), indent=1)
```
