# Congestion-aware target generator

Status: Design — pending review
Date: 2026-04-17
Module: `python/bloqade/lanes/heuristics/physical/target_generator.py`
Depends on: [2026-04-17-target-generator-plugin-design.md](./2026-04-17-target-generator-plugin-design.md)

## Summary

Introduce `CongestionAwareTargetGenerator`, a concrete
`TargetGeneratorABC` plugin that, for each CZ pair in a stage, decides
whether to move the *control* or the *target* to complete the CZ
blockade pairing. The decision is made jointly across pairs in a
sorted, sequential loop that re-scores direction choices against a
working placement and a running congestion record — giving direct
preference to schedules that avoid opposite-direction reuse of lanes.

The generator produces exactly one candidate. The plugin framework
(defined in the target-generator-plugin spec) auto-appends
`DefaultTargetGenerator` as a fallback, so the default remains a
guaranteed last resort when the congestion-aware candidate's search
fails.

## Motivation

`DefaultTargetGenerator` encodes one fixed rule: move the control to
the CZ-blockade partner of the target's current location. That rule is
cheap and correct but ignores:

1. Cases where moving the *target* is strictly cheaper (e.g. the
   control sits deep inside a cluster of other atoms while the target
   has an open lane to the control's partner).
2. Inter-pair interactions. When two pairs both must move, the moves
   share the physical lane graph and the hardware's move scheduler —
   two paths that use the same lane in opposite directions are
   particularly costly to schedule, whereas two paths that simply
   cross at a shared site (different lanes) are easy for the scheduler
   to sequence.

The goal of this heuristic is a target placement that is cheap in
schedule-time cost *and* minimizes adversarial congestion between pairs
so that downstream search (Python `ConfigurationTree` traversal or
Rust `MoveSolver`) converges faster.

## Non-goals

- Modifying the plugin interface. `CongestionAwareTargetGenerator` is a
  consumer of the contract defined in the target-generator-plugin
  spec; all new symbols are local to the new class.
- Replacing `DefaultTargetGenerator`. The default stays the guaranteed
  fallback.
- Lookahead across CZ stages. `TargetContext.lookahead_cz_layers` and
  `cz_stage_index` are ignored by this heuristic. Current-stage
  optimization only.
- Rust FFI surface. Entirely Python-level; reuses the existing Python
  `layout.PathFinder`.
- Global-optimum assignment. This is a greedy heuristic (sorted
  sequential commit), not a joint optimization solver. Exploring ILP
  or beam-search variants is explicitly out of scope.
- Composition machinery. No combinator / ladder / parametric family
  classes — those are deferred until a second heuristic ships and a
  concrete need for composition exists.

## Interface

Colocated with `DefaultTargetGenerator` in
`python/bloqade/lanes/heuristics/physical/target_generator.py` (the
target-generator plugin interface landed in its own module in PR #533,
separate from `movement.py` which still hosts `PhysicalPlacementStrategy`
and the traversal classes):

```python
@dataclass(frozen=True)
class CongestionAwareTargetGenerator(TargetGeneratorABC):
    """Joint, longest-first, congestion-aware target generator.

    For each CZ pair, picks whether to move the control or the target
    based on schedule-time cost computed against a working placement
    that reflects all prior pairs' committed moves and a running
    directional congestion record.

    Per-lane weighting uses a single ``direction_factor`` raised to the
    signed net of same-direction minus opposite-direction prior commits
    on that lane: ``weight = base * direction_factor ** (N - M)`` where
    ``N`` counts prior same-direction commits and ``M`` counts opposite.
    With ``direction_factor < 1`` the factor rewards net-same traffic,
    penalises net-opposite, and is neutral at ``N == M``. A separate
    ``shared_site_factor`` applies to lanes that don't overlap prior
    commits but touch a previously traversed site.

    ``direction_factor`` must be strictly positive (``0 ** -M`` is
    undefined); ``shared_site_factor`` must be ``>= 0``. Both are
    validated in ``__post_init__``.
    """

    direction_factor: float = 0.5
    shared_site_factor: float = 1.1

    def generate(
        self, ctx: TargetContext
    ) -> list[dict[int, LocationAddress]]: ...
```

Defaults are placeholders — empirical tuning is a follow-up (see
"Open questions").

### Return shape

A one-element list containing the full target placement
`{qid: LocationAddress}` for all qubits in `ctx.placement`. Non-CZ
qubits are carried through unchanged.

An empty list is returned when the generator cannot produce a
candidate (see "Edge cases"); the framework then falls through to the
appended default. No in-class fallback — the framework's default-append
is the sole fallback path.

## Algorithm

```text
placement = ctx.placement                    # {qid: LocationAddress}
working   = dict(placement)                  # mutated as pairs commit
committed_lanes: dict[LaneKey, Direction] = {}
committed_sites: set[LocationAddress]     = set()
pf = PathFinder(ctx.arch_spec)

# 1. Initial sort score (empty congestion state → pure lane-duration cost).
#    Observation: because lanes are bidirectional and CZ partnering is
#    a translation, the uncongested costs of the two directions are
#    equal by symmetry. The min() below is therefore well-defined but
#    degenerate; Python's stable sort preserves original pair order on
#    ties.
def score(pair):
    ctrl, tgt = pair
    ctrl_loc, tgt_loc = placement[ctrl], placement[tgt]
    ctrl_partner = arch_spec.get_cz_partner(tgt_loc)
    tgt_partner  = arch_spec.get_cz_partner(ctrl_loc)
    occ_ctrl = frozenset(l for q, l in placement.items() if q != ctrl)
    occ_tgt  = frozenset(l for q, l in placement.items() if q != tgt)
    p_ctrl = pf.find_path(ctrl_loc, ctrl_partner, occ_ctrl,
                          edge_weight=pf.metrics.get_lane_duration_cost)
    p_tgt  = pf.find_path(tgt_loc,  tgt_partner,  occ_tgt,
                          edge_weight=pf.metrics.get_lane_duration_cost)
    c_ctrl = _sum_base(p_ctrl)  if p_ctrl else inf
    c_tgt  = _sum_base(p_tgt)   if p_tgt  else inf
    return min(c_ctrl, c_tgt)

pairs = sorted(zip(ctx.controls, ctx.targets), key=score, reverse=True)
# LONGEST-FIRST: hardest pairs commit first, easier pairs adapt.

# 2. Walk sorted pairs; re-score against working + committed state.
for ctrl, tgt in pairs:
    ctrl_loc, tgt_loc = working[ctrl], working[tgt]
    ctrl_partner = arch_spec.get_cz_partner(tgt_loc)
    tgt_partner  = arch_spec.get_cz_partner(ctrl_loc)

    occ_ctrl = frozenset(l for q, l in working.items() if q != ctrl)
    occ_tgt  = frozenset(l for q, l in working.items() if q != tgt)

    weight = _make_weight_fn(pf, committed_lanes, committed_sites, self)
    path_ctrl = pf.find_path(ctrl_loc, ctrl_partner, occ_ctrl,
                             edge_weight=weight)
    path_tgt  = pf.find_path(tgt_loc,  tgt_partner,  occ_tgt,
                             edge_weight=weight)

    cost_ctrl = _sum_weighted(path_ctrl, weight) if path_ctrl else inf
    cost_tgt  = _sum_weighted(path_tgt,  weight) if path_tgt  else inf

    if cost_ctrl == inf and cost_tgt == inf:
        return []                             # fall through to default

    # Tiebreak hierarchy: cost, then path-lane-count, then prefer control.
    chose_control = _choose_control(
        cost_ctrl, cost_tgt,
        len(path_ctrl[0]) if path_ctrl else inf,
        len(path_tgt[0])  if path_tgt  else inf,
    )

    if chose_control:
        mover, new_loc, chosen = ctrl, ctrl_partner, path_ctrl
    else:
        mover, new_loc, chosen = tgt,  tgt_partner,  path_tgt

    working[mover] = new_loc
    for lane in chosen[0]:
        committed_lanes[_lane_key(lane)] = lane.direction
    committed_sites.update(chosen[1])         # all sites the path visits

return [working]
```

### Tiebreak hierarchy at commit

`_choose_control(cost_c, cost_t, len_c, len_t)` returns `True` when
control-moves should be chosen:

1. If `cost_c < cost_t`: control.
2. If `cost_t < cost_c`: target.
3. Else if `len_c < len_t`: control (fewer physical moves).
4. Else if `len_t < len_c`: target.
5. Else: control (parity with `DefaultTargetGenerator`; deterministic).

### Reused building blocks

- `layout.PathFinder(arch_spec)` — already exists; Dijkstra-based
  shortest-path on the physical lane graph. Built once per `generate()`
  call; caching across calls deferred (see "Open questions").
- `PathFinder.find_path(start, end, occupied, edge_weight)` — returns
  `(tuple[LaneAddress, ...], tuple[LocationAddress, ...])` or `None`.
  `edge_weight` is the callable injection point for congestion
  costs. **Explicit deviation from default**: `find_path` defaults
  `edge_weight` to `get_lane_duration_us` (microseconds). We pass
  `get_lane_duration_cost` (normalized) so the congestion-cost floats
  remain commensurate with the base term.
- `PathFinder.get_endpoints(lane) -> (LocationAddress, LocationAddress)`
  — used inside the edge-weight closure. The underlying
  `end_points_cache` is populated for every lane produced during
  `PathFinder.__post_init__`, so lanes returned by `find_path` are
  guaranteed to round-trip through `get_endpoints` with non-`None`
  values. (`get_endpoints` returns `(None, None)` for unknown lanes,
  which is structurally unreachable here.)
- `MoveMetricCalculator.get_lane_duration_cost(lane)` — normalized
  base weight; same metric the logical heuristic uses. Distinct from
  `get_lane_duration_us` (absolute microsecond duration).
- `ArchSpec.get_cz_partner(loc)` — partner lookup.

### Local helpers

All module-private:

- `_sum_base(path) -> float` — sum of
  `pf.metrics.get_lane_duration_cost(lane)` over `path[0]` (the
  `tuple[LaneAddress, ...]` component). Returns `0.0` for the
  zero-length path `((), (start,))`.
- `_sum_weighted(path, weight_fn) -> float` — sum of `weight_fn(lane)`
  over `path[0]`. Returns `0.0` for zero-length paths. Kept separate
  from `_sum_base` so the initial sort (which uses base weights) and
  the commit loop (which uses the full congestion-aware closure) each
  have a clearly labeled helper.
- `_make_weight_fn(pf, committed_lanes, committed_sites, gen)` — see
  "Cost function and congestion encoding".
- `_lane_key(lane) -> _LaneKey` — see "Lane canonical key".
- `_choose_control(cost_c, cost_t, len_c, len_t) -> bool` — tiebreak
  comparator; returns `True` when control-moves wins under the
  (cost, path-lane-count, prefer-control) hierarchy.

## Cost function and congestion encoding

### Edge weight closure

```python
def _make_weight_fn(pf, committed_lanes, committed_sites, gen):
    df = gen.direction_factor

    def weight(lane: LaneAddress) -> float:
        base = pf.metrics.get_lane_duration_cost(lane)
        key = _lane_key(lane)
        counts = committed_lanes.get(key)  # Counter[Direction] | None
        if counts:
            same = counts.get(lane.direction, 0)
            opposite = counts.get(_opposite(lane.direction), 0)
            net = same - opposite
            if net != 0:
                return base * (df ** net)
            return base
        src, dst = pf.get_endpoints(lane)
        if src in committed_sites or dst in committed_sites:
            return base * gen.shared_site_factor
        return base
    return weight
```

``committed_lanes`` maps each lane's canonical key to a per-direction
``Counter``. Reward and penalty compose into one exponent of
``direction_factor``:

| ``N`` same | ``M`` opposite | Factor (with ``df = 0.5``) | Interpretation |
|---|---|---|---|
| 0 | 0 | 1.0 (base) | Falls through to ``shared_site_factor`` or base. |
| 1 | 0 | 0.5 | Reward for joining one prior same-direction atom. |
| 2 | 0 | 0.25 | Stronger reward — larger AOD cluster. |
| 3 | 0 | 0.125 | Reward compounds with cluster size. |
| 0 | 1 | 2.0 | Penalty for crossing one opposite-direction commit. |
| 0 | 2 | 4.0 | Stronger penalty. |
| 1 | 1 | 1.0 (neutral) | Balanced traffic — local signal is ambiguous. |
| 2 | 1 | 0.5 | Net ``+1`` reward. |

``direction_factor`` must be strictly positive (the negative exponent is
otherwise undefined); ``shared_site_factor`` must be ``>= 0``. Dijkstra
requires non-negative edge weights; ``__post_init__`` validates both.
A path's total weighted cost is the sum of ``weight(lane)`` over its
lanes — re-evaluated by the caller (`_sum_weighted`) against the same
closure the pathfinder used.

### Lane canonical key

`LaneAddress` encodes direction in its identity (`Direction.FORWARD`
vs `Direction.BACKWARD`), but forward and backward on the physical
arch are the same lane resource. To detect "same lane in opposite
direction" we need a direction-agnostic key:

```python
_LaneKey = tuple[MoveType, int, int, int, int]

def _lane_key(lane: LaneAddress) -> _LaneKey:
    return (lane.move_type, lane.word_id, lane.site_id,
            lane.bus_id, lane.zone_id)
```

The five fields above are exactly the direction-independent subset of
`LaneAddress` (which also exposes `direction`). Verified against
`python/bloqade/lanes/layout/encoding.py`. If a second caller wants
this canonical-key concept, promoting `_lane_key` to
`LaneAddress.canonical()` is a small follow-up — but a private helper
is adequate for this first cut.

### Shared-site accounting

After a pair commits, every location its path traversed (start,
intermediates, end) is added to `committed_sites`. Subsequent pairs'
candidate lanes whose endpoint is in that set pay
`shared_site_factor` — but only when the lane itself is not reused
(lane-reuse factors dominate; shared-site is the lower tier).

### Bidirectional-cost symmetry note

On the physical arch, lanes are bidirectional and their
`get_lane_duration_cost` is direction-symmetric. CZ partnering is a
fixed translation on the lattice. Consequently, in the **ideal
unconstrained lane graph**, the uncongested path cost of "move
control to partner(target)" equals the uncongested path cost of "move
target to partner(control)" — by the symmetry argument
`distance(L, partner(M)) == distance(M, partner(L))`.

**Caveat — the occupied-set is asymmetric.** The two `find_path`
calls within `score(pair)` use different `occupied` frozensets:
`occ_ctrl` excludes only `ctrl` (so `tgt` blocks), `occ_tgt` excludes
only `tgt` (so `ctrl` blocks). When a candidate shortest path would
need to pass through the pair's *other* atom, one direction can be
infeasible (`find_path` returns `None`) while the other is fine, or
both paths are feasible but forced onto different detours of
different cost. The symmetry argument therefore holds only modulo
which partnered endpoint sits on the shortest-path geodesic — in the
typical case of two partners with an unobstructed direct route
between them, the two directions tie; in degenerate cases they
diverge.

Implication for the **initial sort score**: `min(c_ctrl, c_tgt)` is
well-defined either way. Most pairs tie (symmetric case); a minority
may have an asymmetric score due to endpoint-blocking. Direction
choice gains meaningful signal at commit time once congestion from
earlier pairs enters the weighting.

This is why **move-count is a tiebreaker only at commit time, not
during the initial sort**: at initial sort, the typical symmetric
case offers no move-count asymmetry to exploit, and the degenerate
asymmetric case is already resolved by the primary cost comparison.

## Edge cases

1. **Empty stage** — `controls=() and targets=()`: loop is a no-op;
   return `[placement]`. Framework's candidate validation passes (no
   pairs to check); search runs once with zero expansions.

2. **Already-partnered pair** — `ctrl_loc` and `tgt_loc` are already
   CZ blockade partners: both `find_path` calls return `((), (start,))`
   with cost 0 and length 0. The pair no-ops; tiebreak picks control;
   `working[ctrl]` is reassigned to the same location;
   `committed_lanes` is unchanged (no lanes traversed) and
   `committed_sites` gains `ctrl_loc` (the single site in
   `chosen[1]`). The latter is benign: the pair's atom already sits
   there and the lane-reuse penalties always dominate the
   shared-site tier, so this accounting costs nothing observable to
   later pairs.

3. **Both directions infeasible** — `path_ctrl is None and path_tgt is
   None`: `return []`. The framework's auto-appended default is the
   sole fallback; no exception raised. Rationale: infeasibility under
   this heuristic's all-atoms-as-blockers model is not a plugin
   contract violation — the default rule may still be expressible.

4. **Infeasible in one direction only** — the other direction
   succeeds; take it. Cost `inf` for the infeasible side is handled by
   the tiebreak comparator.

5. **`get_cz_partner(loc) is None`** — means an atom in the current
   placement sits at a location with no CZ partner. This is a
   configuration / upstream contract violation (the same case the
   default generator asserts on). Raise `AssertionError` with the
   offending `qid` and `LocationAddress`, mirroring the existing
   default behavior.

6. **Destination occupied by a non-participating atom** — routing
   fails because the target node is in `occupied`; that direction
   yields `None`. Treated as direction-infeasible per case 4. Only
   both-None triggers case 3.

7. **Lookahead ignored** — `ctx.lookahead_cz_layers` and
   `ctx.cz_stage_index` are unused. Documented so reviewers /
   downstream users do not expect lookahead behavior.

8. **Validation interplay** — because both directions are constructed
   using `arch_spec.get_cz_partner`, the emitted candidate satisfies
   one of the two valid directions of the framework's
   "pair-CZ-blockade-partnered" check (see target-generator-plugin
   spec, "Validation"). The generator never produces a candidate that
   the framework rejects as malformed.

9. **Stable determinism** — `sorted(...)` is stable; `PathFinder` is
   deterministic; `frozenset` iteration order is deterministic for
   `LocationAddress` given hashable content. Given identical inputs,
   `generate(ctx)` produces identical outputs on repeated calls — a
   property the unit tests assert.

## Type safety

- `CongestionAwareTargetGenerator` is a frozen dataclass with three
  `float` fields and a single method; pyright-clean.
- `generate` returns `list[dict[int, LocationAddress]]`, matching the
  ABC.
- Helper functions (`_make_weight_fn`, `_lane_key`, `_sum_weighted`,
  `_sum_base`, `_choose_control`) are module-private with explicit
  annotations.
- `PathFinder.find_path` may return `None`; every call site handles
  `None` via the `inf` sentinel on cost and an explicit `None` guard
  on path indexing.

## Testing

### Unit — new file `python/tests/heuristics/test_congestion_aware_target_generator.py`

Each test constructs a minimal `TargetContext` directly and asserts on
`generate(ctx)` output.

1. **Single pair, already partnered** → `[placement]`.
2. **Single pair, one-step move, no other atoms** → control-direction
   chosen by tiebreak; cost and length tie.
3. **Single pair, target-cheaper by obstacles** — non-participating
   atom blocks control's would-be path only → target direction
   chosen.
4. **Two pairs, longest-first sort order** — construct inputs where
   only longest-first commit order produces a valid candidate; verify.
5. **Opposite-direction congestion avoided** — two pairs where a
   naive order reuses a lane in opposite directions; with
   `direction_factor=1e-6` (very strong reward/penalty per net atom)
   verify the second pair picks the direction that does not reuse in
   opposite sense.
6. **Same-direction reuse preferred to opposite** — both directions
   reuse a committed lane, one same-dir, one opposite-dir; same-dir is
   chosen even when slightly longer in lane count.
7. **Shared-site factor tier** — two directions, one crosses a prior
   path's site, the other fully fresh but longer in raw duration;
   verify the decision depends on `shared_site_factor` magnitude as
   expected.
8. **Neutral-factor config** — all three factors = 1.0 → identical
   output to a greedy-shortest-cost joint heuristic with no congestion
   awareness (regression reference).
9. **Both directions infeasible** → `[]` (not raise).
10. **Empty stage** — `controls=(), targets=()` → `[placement]`.
11. **Move-count commit tiebreaker** — craft `cost_ctrl == cost_tgt`
    with `len(path_ctrl) < len(path_tgt)`; verify control chosen.
    Mirror case with target-shorter; verify target chosen. All-tied
    case: control (parity with default).
12. **Stable determinism** — two consecutive `generate(ctx)` calls on
    identical inputs produce identical outputs.
13. **`get_cz_partner` returning `None`** — raises `AssertionError`
    with qid and LocationAddress in the message.

### Integration — extend `python/tests/heuristics/test_physical_placement.py`

14. **End-to-end**: `PhysicalPlacementStrategy(
    target_generator=CongestionAwareTargetGenerator())` produces a
    valid move program for a representative circuit. Reuse the
    existing move-program validators from neighbor tests.
15. **Dedup with default** — for a trivially-symmetric stage where the
    heuristic's candidate equals the default's, framework dedup
    ensures exactly one search runs. Assertion mechanism depends on
    the traversal path:
    - **Rust path**: assert on
      `PhysicalPlacementStrategy.rust_nodes_expanded_total` — the
      existing strategy-level counter already sums across
      candidates.
    - **Python path**: no strategy-level counter exists
      (`SearchResult.nodes_expanded` is per-call only). Instrument by
      installing a counting `on_search_step` hook on the
      `EntropyPlacementTraversal`, or by monkey-patching / wrapping
      `self.traversal.path_to_target_config` to count calls. A call
      count of `1` (not `2`) after the dedup step is the signal.
16. **Rust-path parity** — same circuit under
    `EntropyPlacementTraversal` and `RustPlacementTraversal`; plugin
    applies identically; both terminate with `ExecuteCZ`.

### Out of scope

- **Performance benchmarks** — the existing `bench/` suite exercises
  `PhysicalPlacementStrategy`; a congestion-aware entry there is a
  follow-up PR, not a merge gate.
- **Penalty tuning** — defaults are illustrative; empirical sweep is
  its own follow-up (open question below).
- **Lookahead-aware variants** — separate class, separate spec.

## Migration and compatibility

- Strictly additive. `DefaultTargetGenerator` and the rest of the
  plugin framework are untouched.
- No Rust API changes.
- Opt-in: existing callers not passing `target_generator=...` to
  `PhysicalPlacementStrategy` see no behavior change.
- This spec depends on the target-generator plugin spec's
  implementation landing first. If the plugin PR merges in a state
  where `_cz_counter` is only incremented on the Python path (pending
  the parity fix noted in that spec), this heuristic does not depend
  on `cz_stage_index` for correctness — it just ignores that field —
  so the ordering of the two PRs is flexible.

## Open questions

1. **Factor defaults.** The shipped defaults (`opposite=10.0`,
   `same=0.25`, `shared_site=1.1`) reflect the canonical tuning —
   penalise opposite-direction reuse, reward same-direction reuse
   (AOD-parallelism proxy), mild shared-site penalty. Empirical
   retuning — sweeping factors on a representative workload and
   measuring `rust_nodes_expanded_total` or wall-time — is a follow-up.
   Tracking issue recommended.

2. **`LaneAddress.canonical()` helper.** If more callers benefit from
   a direction-stripped lane key, promote `_lane_key` from private
   helper to a method on `LaneAddress`. Defer until a second caller
   exists.

3. **PathFinder caching across `generate()` calls.** The dataclass is
   frozen, so any cache must use the `object.__setattr__` /
   `__post_init__` pattern (as `PathFinder` itself does). Defer
   until profiling shows construction as a measurable cost.

4. **Non-participating-qubit motion.** This heuristic never moves a
   non-CZ qubit. Heuristics that *do* (one of the spec's motivating
   examples) are a separate design and will live in their own class.
