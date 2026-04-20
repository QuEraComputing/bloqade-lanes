# Congestion-aware target generator — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `CongestionAwareTargetGenerator`, a concrete `TargetGeneratorABC` that chooses control-vs-target direction per CZ pair using a joint, longest-first, direction-aware congestion model on the physical lane graph.

**Spec:** `docs/superpowers/specs/2026-04-17-congestion-aware-target-generator-design.md`

**Depends on:** target-generator plugin interface (`TargetContext`, `TargetGeneratorABC`, `DefaultTargetGenerator`, `_validate_candidate`, `_build_candidates`, strategy-level shared-budget loop) landing via `docs/superpowers/plans/2026-04-17-target-generator-plugin.md` on branch `spec/target-generator-plugin`. **Do not start execution until that branch is merged to main** — every task below imports symbols that only exist after that work lands.

**Architecture:** Add one new public class plus five module-private helpers to `python/bloqade/lanes/heuristics/physical/target_generator.py`. The class is a frozen dataclass with three penalty-weight floats and a `generate()` method that (a) builds a `layout.PathFinder` once, (b) scores all pairs with base lane-duration cost on the current placement, (c) sorts longest-first, (d) walks the sorted list, re-scoring both directions against a running `committed_lanes` / `committed_sites` record, committing the cheaper direction per pair with a `(cost, path-length, prefer-control)` tiebreak. No Rust surface changes; no modifications to the plugin framework.

**Tech Stack:** Python 3.10+ (dataclasses, typing), `layout.PathFinder`, `MoveMetricCalculator`, pytest, pyright (clean required), pre-commit (black/isort/ruff).

**Branch:** `spec/congestion-aware-target-generator` (this plan is committed on that branch; implementation continues on it).

---

## File Structure

| Path | Change | Responsibility |
|---|---|---|
| `python/bloqade/lanes/heuristics/physical/target_generator.py` | modify | Add `_LaneKey`, `_lane_key`, `_sum_base`, `_sum_weighted`, `_choose_control`, `_make_weight_fn`, and `CongestionAwareTargetGenerator`. New public export colocated with `DefaultTargetGenerator`. |
| `python/bloqade/lanes/heuristics/physical/__init__.py` | modify | Re-export `CongestionAwareTargetGenerator`. |
| `python/tests/heuristics/test_congestion_aware_target_generator.py` | create | Unit tests 1–13 from spec (helpers + `generate()` behavior). |
| `python/tests/heuristics/test_physical_placement.py` | modify | Integration tests 14–16 (end-to-end strategy, dedup, Rust-path parity). |

---

## Pre-flight verification

Before starting any task, confirm the plugin interface is merged:

- [ ] **Confirm dependency symbols are importable**

```bash
uv run python -c "
from bloqade.lanes.heuristics.physical.target_generator import (
    TargetContext, TargetGeneratorABC, DefaultTargetGenerator,
)
print('plugin interface ready')
"
```

Expected: `plugin interface ready`. If `ImportError`, stop and wait for `spec/target-generator-plugin` to merge to main.

- [ ] **Confirm physical arch + the fixture-pair helper discovers a CZ-partnered pair**

```bash
uv run python -c "
from bloqade.lanes.arch.gemini.physical import get_arch_spec
arch = get_arch_spec()
# _pick_cz_pair pattern (see 'Shared test helpers' below): iterate
# home_sites and return the first atom-location with a CZ partner.
pair = None
for s in arch.home_sites:
    p = arch.get_cz_partner(s)
    if p is not None and p != s:
        pair = (s, p)
        break
assert pair is not None, 'arch has no CZ-partnered pair'
print(f'fixture pair: {pair}')
"
```

Expected: non-None pair printed. `ArchSpec.cz_zone_addresses` is a `frozenset[ZoneAddress]` (not a subscriptable sequence of `LocationAddress`), so it cannot be used directly to seed fixtures — always go through `home_sites` + `get_cz_partner`.

## Shared test helpers

All tests in this plan use a module-level helper to obtain a valid CZ-partnered pair of `LocationAddress` values from the physical arch. Add this near the top of `python/tests/heuristics/test_congestion_aware_target_generator.py`:

```python
import pytest
from bloqade.lanes import layout
from bloqade.lanes.arch.gemini.physical import get_arch_spec


@pytest.fixture(scope="module")
def arch() -> layout.ArchSpec:
    return get_arch_spec()


def _pick_cz_pair(
    arch: layout.ArchSpec,
) -> tuple[layout.LocationAddress, layout.LocationAddress]:
    """Return the first CZ-partnered (loc, partner) pair from arch.home_sites."""
    for s in arch.home_sites:
        p = arch.get_cz_partner(s)
        if p is not None and p != s:
            return s, p
    raise AssertionError(
        "fixture prerequisite failed: arch has no CZ-partnered home site"
    )
```

Integration tests in `test_physical_placement.py` use the same helper (copy-paste or import it from the unit-test module — the latter is cleaner; add `_pick_cz_pair` to a small shared utility if the integration tests diverge).

---

## Chunk 1: Private helpers (pure functions)

### Task 1: `_LaneKey` alias and `_lane_key` function

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/target_generator.py`
- Test: `python/tests/heuristics/test_congestion_aware_target_generator.py`

- [ ] **Step 1: Create the test file with a failing test**

Create `python/tests/heuristics/test_congestion_aware_target_generator.py`:

```python
from __future__ import annotations

from bloqade.lanes.heuristics.physical.target_generator import _LaneKey, _lane_key
from bloqade.lanes.layout import Direction, LaneAddress, MoveType


def test_lane_key_strips_direction():
    forward = LaneAddress(MoveType.SITE, 1, 2, 3, Direction.FORWARD, 4)
    backward = LaneAddress(MoveType.SITE, 1, 2, 3, Direction.BACKWARD, 4)
    assert _lane_key(forward) == _lane_key(backward)


def test_lane_key_distinguishes_different_lanes():
    a = LaneAddress(MoveType.SITE, 1, 2, 3, Direction.FORWARD, 4)
    b = LaneAddress(MoveType.SITE, 1, 2, 3, Direction.FORWARD, 5)  # different zone
    assert _lane_key(a) != _lane_key(b)


def test_lane_key_tuple_shape():
    lane = LaneAddress(MoveType.SITE, 1, 2, 3, Direction.FORWARD, 4)
    key = _lane_key(lane)
    assert isinstance(key, tuple)
    assert len(key) == 5
    assert key == (MoveType.SITE, 1, 2, 3, 4)
```

- [ ] **Step 2: Run test to confirm failure**

```bash
uv run pytest python/tests/heuristics/test_congestion_aware_target_generator.py -v
```

Expected: `ImportError: cannot import name '_LaneKey'` or equivalent.

- [ ] **Step 3: Add `_LaneKey` and `_lane_key` to `target_generator.py`**

Insert near the bottom of the file, after all existing strategy/traversal classes and before the `CongestionAwareTargetGenerator` class (which Task 5 will add). A reasonable anchor: immediately after `PhysicalPlacementStrategy.measure_placements`. For now, just add:

```python
_LaneKey = tuple[MoveType, int, int, int, int]


def _lane_key(lane: layout.LaneAddress) -> _LaneKey:
    """Direction-independent canonical key for a lane.

    A physical lane used FORWARD and BACKWARD is the same resource;
    `_lane_key` strips `direction` so the two map to the same key.
    """
    return (
        lane.move_type,
        lane.word_id,
        lane.site_id,
        lane.bus_id,
        lane.zone_id,
    )
```

Also add `MoveType` to the existing `from bloqade.lanes.layout import (...)` block if it is not already imported. (It is — verify at write time; if missing, add.)

- [ ] **Step 4: Run tests + pyright**

```bash
uv run pytest python/tests/heuristics/test_congestion_aware_target_generator.py -v
uv run pyright python/bloqade/lanes/heuristics/physical/target_generator.py python/tests/heuristics/test_congestion_aware_target_generator.py
```

Expected: 3 passed; pyright: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/target_generator.py python/tests/heuristics/test_congestion_aware_target_generator.py
git commit -m "feat(heuristics): add _LaneKey canonical key for congestion-aware generator"
```

---

### Task 2: `_sum_base` and `_sum_weighted`

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/target_generator.py`
- Modify: `python/tests/heuristics/test_congestion_aware_target_generator.py`

- [ ] **Step 1: Add failing tests**

Append to `test_congestion_aware_target_generator.py`:

```python
from bloqade.lanes.heuristics.physical.target_generator import _sum_base, _sum_weighted
from bloqade.lanes.layout import PathFinder


def test_sum_base_empty_path(arch):
    pf = PathFinder(arch)
    loc, _ = _pick_cz_pair(arch)
    path = pf.find_path(loc, loc)
    assert path is not None
    assert _sum_base(path, pf) == 0.0


def test_sum_weighted_empty_path(arch):
    pf = PathFinder(arch)
    loc, _ = _pick_cz_pair(arch)
    path = pf.find_path(loc, loc)
    assert path is not None
    assert _sum_weighted(path, lambda lane: 42.0) == 0.0


def test_sum_weighted_sums_per_lane(arch):
    pf = PathFinder(arch)
    src, dst = _pick_cz_pair(arch)
    path = pf.find_path(src, dst)
    assert path is not None
    lane_count = len(path[0])
    assert _sum_weighted(path, lambda lane: 1.0) == float(lane_count)
```

- [ ] **Step 2: Run test to confirm failure**

```bash
uv run pytest python/tests/heuristics/test_congestion_aware_target_generator.py -v
```

Expected: `ImportError` on `_sum_base` / `_sum_weighted`.

- [ ] **Step 3: Add helpers to `target_generator.py`**

Immediately after `_lane_key`:

```python
def _sum_base(
    path: tuple[
        tuple[layout.LaneAddress, ...],
        tuple[layout.LocationAddress, ...],
    ],
    pf: layout.PathFinder,
) -> float:
    """Sum of base (no-penalty) lane-duration cost over a path's lanes."""
    return sum(pf.metrics.get_lane_duration_cost(lane) for lane in path[0])


def _sum_weighted(
    path: tuple[
        tuple[layout.LaneAddress, ...],
        tuple[layout.LocationAddress, ...],
    ],
    weight_fn: Callable[[layout.LaneAddress], float],
) -> float:
    """Sum of `weight_fn(lane)` over a path's lanes.

    Returns 0.0 for zero-length paths (empty lane tuple).
    """
    return sum(weight_fn(lane) for lane in path[0])
```

Ensure `PathFinder` is re-exported from `bloqade.lanes.layout`. Check with `grep PathFinder python/bloqade/lanes/layout/__init__.py`; add to exports if not present. Similarly, confirm `bloqade.lanes.layout.PathFinder` is the intended public path (as opposed to `bloqade.lanes.layout.path.PathFinder`).

- [ ] **Step 4: Run tests + pyright**

```bash
uv run pytest python/tests/heuristics/test_congestion_aware_target_generator.py -v
uv run pyright python/bloqade/lanes/heuristics/physical/target_generator.py python/tests/heuristics/test_congestion_aware_target_generator.py
```

Expected: 6 passed; pyright: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/target_generator.py python/bloqade/lanes/layout/__init__.py python/tests/heuristics/test_congestion_aware_target_generator.py
git commit -m "feat(heuristics): add _sum_base and _sum_weighted path-cost helpers"
```

---

### Task 3: `_choose_control` tiebreak comparator

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/target_generator.py`
- Modify: `python/tests/heuristics/test_congestion_aware_target_generator.py`

- [ ] **Step 1: Add failing tests**

```python
import math
from bloqade.lanes.heuristics.physical.target_generator import _choose_control


def test_choose_control_lower_cost_wins():
    assert _choose_control(cost_c=1.0, cost_t=2.0, len_c=10, len_t=1) is True
    assert _choose_control(cost_c=2.0, cost_t=1.0, len_c=1, len_t=10) is False


def test_choose_control_cost_tie_uses_length():
    # Equal cost → shorter path wins
    assert _choose_control(cost_c=1.0, cost_t=1.0, len_c=2, len_t=5) is True
    assert _choose_control(cost_c=1.0, cost_t=1.0, len_c=5, len_t=2) is False


def test_choose_control_all_tied_prefers_control():
    assert _choose_control(cost_c=1.0, cost_t=1.0, len_c=3, len_t=3) is True


def test_choose_control_inf_handled():
    # Target infeasible → control wins
    assert _choose_control(cost_c=5.0, cost_t=math.inf, len_c=1, len_t=0) is True
    # Control infeasible → target wins
    assert _choose_control(cost_c=math.inf, cost_t=5.0, len_c=0, len_t=1) is False
```

- [ ] **Step 2: Run test to confirm failure**

```bash
uv run pytest python/tests/heuristics/test_congestion_aware_target_generator.py -v
```

Expected: `ImportError` on `_choose_control`.

- [ ] **Step 3: Add `_choose_control` to `target_generator.py`**

After `_sum_weighted`:

```python
def _choose_control(
    cost_c: float, cost_t: float, len_c: float, len_t: float
) -> bool:
    """Tiebreak comparator. Returns True iff control-direction wins.

    Hierarchy:
      1. lower cost wins
      2. on cost tie, shorter path wins (fewer physical moves)
      3. on both-tie, prefer control (parity with DefaultTargetGenerator)
    """
    if cost_c < cost_t:
        return True
    if cost_t < cost_c:
        return False
    if len_c < len_t:
        return True
    if len_t < len_c:
        return False
    return True
```

- [ ] **Step 4: Run tests + pyright**

```bash
uv run pytest python/tests/heuristics/test_congestion_aware_target_generator.py -v
uv run pyright python/bloqade/lanes/heuristics/physical/target_generator.py
```

Expected: 10 passed; pyright: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/target_generator.py python/tests/heuristics/test_congestion_aware_target_generator.py
git commit -m "feat(heuristics): add _choose_control tiebreak comparator"
```

---

## Chunk 2: Penalty closure

### Task 4: `_make_weight_fn`

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/target_generator.py`
- Modify: `python/tests/heuristics/test_congestion_aware_target_generator.py`

This closure is the heart of the congestion model. It is tested in isolation here so the final `generate()` tests don't need to probe implementation details.

- [ ] **Step 1: Add failing tests**

```python
from bloqade.lanes.heuristics.physical.target_generator import _make_weight_fn


class _WeightCtx:
    """Minimal stand-in for the generator's penalty-weight fields."""

    def __init__(self, opposite: float, same: float, site: float) -> None:
        self.opposite_direction_penalty = opposite
        self.same_direction_penalty = same
        self.shared_site_penalty = site


def _first_lane(pf: "layout.PathFinder") -> "layout.LaneAddress":
    # Pick any lane in the physical graph for tests.
    return next(iter(pf.end_points_cache))


def test_weight_fn_no_congestion_returns_base(arch):
    pf = layout.PathFinder(arch)
    weight = _make_weight_fn(pf, {}, set(), _WeightCtx(10.0, 1.0, 0.1))
    lane = _first_lane(pf)
    base = pf.metrics.get_lane_duration_cost(lane)
    assert weight(lane) == base


def test_weight_fn_same_direction_adds_same_penalty(arch):
    pf = layout.PathFinder(arch)
    lane = _first_lane(pf)
    committed_lanes = {_lane_key(lane): lane.direction}
    weight = _make_weight_fn(pf, committed_lanes, set(), _WeightCtx(10.0, 1.0, 0.1))
    base = pf.metrics.get_lane_duration_cost(lane)
    assert weight(lane) == base + 1.0


def test_weight_fn_opposite_direction_adds_opposite_penalty(arch):
    pf = layout.PathFinder(arch)
    lane = _first_lane(pf)
    reversed_lane = lane.reverse()
    # Mark the reversed direction as committed; now `lane` is opposite.
    committed_lanes = {_lane_key(lane): reversed_lane.direction}
    weight = _make_weight_fn(pf, committed_lanes, set(), _WeightCtx(10.0, 1.0, 0.1))
    base = pf.metrics.get_lane_duration_cost(lane)
    assert weight(lane) == base + 10.0


def test_weight_fn_shared_site_without_lane_reuse(arch):
    pf = layout.PathFinder(arch)
    lane = _first_lane(pf)
    src, dst = pf.get_endpoints(lane)
    assert src is not None and dst is not None
    weight = _make_weight_fn(pf, {}, {src}, _WeightCtx(10.0, 1.0, 0.1))
    base = pf.metrics.get_lane_duration_cost(lane)
    assert weight(lane) == base + 0.1


def test_weight_fn_lane_reuse_dominates_shared_site(arch):
    """When a lane is committed AND an endpoint is in committed_sites,
    the lane-reuse penalty applies, not the shared-site penalty.
    """
    pf = layout.PathFinder(arch)
    lane = _first_lane(pf)
    src, dst = pf.get_endpoints(lane)
    assert src is not None
    committed_lanes = {_lane_key(lane): lane.direction}
    weight = _make_weight_fn(
        pf, committed_lanes, {src}, _WeightCtx(10.0, 1.0, 0.1)
    )
    base = pf.metrics.get_lane_duration_cost(lane)
    assert weight(lane) == base + 1.0
```

- [ ] **Step 2: Run test to confirm failure**

```bash
uv run pytest python/tests/heuristics/test_congestion_aware_target_generator.py -v -k weight_fn
```

Expected: `ImportError` on `_make_weight_fn`.

- [ ] **Step 3: Implement `_make_weight_fn`**

After `_choose_control`:

```python
from typing import Protocol


class _HasPenalties(Protocol):
    opposite_direction_penalty: float
    same_direction_penalty: float
    shared_site_penalty: float


def _make_weight_fn(
    pf: layout.PathFinder,
    committed_lanes: dict[_LaneKey, layout.Direction],
    committed_sites: set[layout.LocationAddress],
    gen: _HasPenalties,
) -> Callable[[layout.LaneAddress], float]:
    """Closure over the running congestion state; passed to `find_path`.

    Penalty precedence (largest to smallest):
      1. lane reused in opposite direction -> opposite_direction_penalty
      2. lane reused in same direction    -> same_direction_penalty
      3. lane not reused, but an endpoint is in committed_sites
                                          -> shared_site_penalty
      4. no overlap                        -> 0 (base only)
    """

    def weight(lane: layout.LaneAddress) -> float:
        base = pf.metrics.get_lane_duration_cost(lane)
        key = _lane_key(lane)
        prior_dir = committed_lanes.get(key)
        if prior_dir is not None:
            if prior_dir != lane.direction:
                return base + gen.opposite_direction_penalty
            return base + gen.same_direction_penalty
        src, dst = pf.get_endpoints(lane)
        if src in committed_sites or dst in committed_sites:
            return base + gen.shared_site_penalty
        return base

    return weight
```

Ensure `Protocol` is imported from `typing` at the top of the module.

- [ ] **Step 4: Run tests + pyright**

```bash
uv run pytest python/tests/heuristics/test_congestion_aware_target_generator.py -v
uv run pyright python/bloqade/lanes/heuristics/physical/target_generator.py
```

Expected: 15 passed; pyright: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/target_generator.py python/tests/heuristics/test_congestion_aware_target_generator.py
git commit -m "feat(heuristics): add _make_weight_fn penalty closure"
```

---

## Chunk 3: `CongestionAwareTargetGenerator` class

### Task 5: Class skeleton + degenerate cases

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/target_generator.py`
- Modify: `python/tests/heuristics/test_congestion_aware_target_generator.py`

Start with: empty stage returns `[placement]`, already-partnered pair returns a candidate equal to current placement.

- [ ] **Step 1: Add failing tests**

Append a test-fixture helper and the first two generator tests:

```python
from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.heuristics.physical.target_generator import (
    CongestionAwareTargetGenerator,
    TargetContext,
)


def _ctx(
    arch: layout.ArchSpec,
    layout_tup: tuple[layout.LocationAddress, ...],
    controls: tuple[int, ...],
    targets: tuple[int, ...],
) -> TargetContext:
    state = ConcreteState(
        occupied=frozenset(),
        layout=layout_tup,
        move_count=(0,) * len(layout_tup),
    )
    return TargetContext(
        arch_spec=arch,
        state=state,
        controls=controls,
        targets=targets,
        lookahead_cz_layers=(),
        cz_stage_index=0,
    )


def test_generate_empty_stage_returns_current_placement(arch):
    loc0, loc1 = _pick_cz_pair(arch)
    ctx = _ctx(arch, (loc0, loc1), controls=(), targets=())
    out = CongestionAwareTargetGenerator().generate(ctx)
    assert out == [{0: loc0, 1: loc1}]


def test_generate_already_partnered_pair_is_noop(arch):
    loc0, loc1 = _pick_cz_pair(arch)
    ctx = _ctx(arch, (loc0, loc1), controls=(0,), targets=(1,))
    out = CongestionAwareTargetGenerator().generate(ctx)
    assert out == [{0: loc0, 1: loc1}]
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest python/tests/heuristics/test_congestion_aware_target_generator.py -v -k generate
```

Expected: `ImportError: cannot import name 'CongestionAwareTargetGenerator'`.

- [ ] **Step 3: Add class skeleton**

After `_make_weight_fn`:

```python
@dataclass(frozen=True)
class CongestionAwareTargetGenerator(TargetGeneratorABC):
    """Joint, longest-first, congestion-aware target generator.

    For each CZ pair, picks whether to move the control or the target
    based on schedule-time cost computed against a working placement
    that reflects all prior pairs' committed moves and a running
    directional congestion record.

    Penalty weights are additive on top of
    ``MoveMetricCalculator.get_lane_duration_cost(lane)`` and are tuned
    relative to that base-duration scale. Defaults are illustrative;
    empirical tuning is a documented follow-up.
    """

    opposite_direction_penalty: float = 10.0
    same_direction_penalty: float = 1.0
    shared_site_penalty: float = 0.1

    def generate(
        self, ctx: TargetContext
    ) -> list[dict[int, layout.LocationAddress]]:
        placement = ctx.placement
        if not ctx.controls:
            return [dict(placement)]

        pf = layout.PathFinder(ctx.arch_spec)
        working: dict[int, layout.LocationAddress] = dict(placement)
        committed_lanes: dict[_LaneKey, layout.Direction] = {}
        committed_sites: set[layout.LocationAddress] = set()

        pairs = self._sort_pairs_longest_first(ctx, pf)

        for ctrl, tgt in pairs:
            result = self._commit_pair(
                ctx.arch_spec, pf, working, committed_lanes,
                committed_sites, ctrl, tgt,
            )
            if result is None:
                return []  # both directions infeasible -> fallback to default
            mover, new_loc, chosen = result
            working[mover] = new_loc
            for lane in chosen[0]:
                committed_lanes[_lane_key(lane)] = lane.direction
            committed_sites.update(chosen[1])

        return [working]

    def _sort_pairs_longest_first(
        self, ctx: TargetContext, pf: layout.PathFinder
    ) -> list[tuple[int, int]]:
        # Placeholder: return in original order.
        # Task 6 replaces this with real sorting.
        return list(zip(ctx.controls, ctx.targets))

    def _commit_pair(
        self,
        arch_spec: layout.ArchSpec,
        pf: layout.PathFinder,
        working: dict[int, layout.LocationAddress],
        committed_lanes: dict[_LaneKey, layout.Direction],
        committed_sites: set[layout.LocationAddress],
        ctrl: int,
        tgt: int,
    ) -> (
        tuple[
            int,
            layout.LocationAddress,
            tuple[
                tuple[layout.LaneAddress, ...],
                tuple[layout.LocationAddress, ...],
            ],
        ]
        | None
    ):
        # Placeholder: picks control direction, no path computation.
        # Task 7 replaces this with real direction choice.
        ctrl_loc = working[ctrl]
        tgt_loc = working[tgt]
        partner = arch_spec.get_cz_partner(tgt_loc)
        assert partner is not None, (
            f"No CZ partner for qid={tgt} at {tgt_loc}"
        )
        if ctrl_loc == partner:
            # Already partnered; trivial zero-length path at ctrl_loc.
            return (ctrl, partner, ((), (ctrl_loc,)))
        # Walk a real path so the already-partnered case works end-to-end
        # even before Task 7 lands.
        occ = frozenset(l for q, l in working.items() if q != ctrl)
        path = pf.find_path(ctrl_loc, partner, occupied=occ)
        if path is None:
            return None
        return (ctrl, partner, path)
```

`TargetGeneratorABC` is defined at the top of `target_generator.py` (added by PR #533); the imports we need (`abc`, `dataclass`, `layout`, `ConcreteState`, `LocationAddress`, `Callable`) are already present. Task 4 introduces `import math` and `from typing import Protocol` for the new helpers.

- [ ] **Step 4: Run tests**

```bash
uv run pytest python/tests/heuristics/test_congestion_aware_target_generator.py -v
uv run pyright python/bloqade/lanes/heuristics/physical/target_generator.py
```

Expected: 17 passed; pyright: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/target_generator.py python/tests/heuristics/test_congestion_aware_target_generator.py
git commit -m "feat(heuristics): add CongestionAwareTargetGenerator skeleton"
```

---

### Task 6: Implement `_sort_pairs_longest_first`

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/target_generator.py`
- Modify: `python/tests/heuristics/test_congestion_aware_target_generator.py`

- [ ] **Step 1: Add a failing test that requires actual sort behavior**

```python
def _find_two_distinct_cost_pairs(arch):
    """Scan home_sites for two CZ-partnered pairs with different
    uncongested path costs. Returns ((short_pair, short_cost),
    (long_pair, long_cost)) or None if arch doesn't support this.
    """
    pf = layout.PathFinder(arch)
    candidates: list[tuple[layout.LocationAddress, layout.LocationAddress, float]] = []
    seen: set[frozenset[layout.LocationAddress]] = set()
    for s in arch.home_sites:
        p = arch.get_cz_partner(s)
        if p is None or p == s:
            continue
        key = frozenset({s, p})
        if key in seen:
            continue
        seen.add(key)
        path = pf.find_path(s, p)
        if path is None:
            continue
        candidates.append((s, p, _sum_base(path, pf)))
    if len(candidates) < 2:
        return None
    costs = sorted(candidates, key=lambda t: t[2])
    if costs[0][2] == costs[-1][2]:
        return None
    return costs[0], costs[-1]


def test_sort_longest_first_orders_by_descending_uncongested_min_cost(arch):
    """Construct two pairs with different uncongested shortest-path
    cost; verify the longer pair sorts first.
    """
    picked = _find_two_distinct_cost_pairs(arch)
    if picked is None:
        pytest.skip(
            "physical arch has no two CZ-partnered pairs with distinct path costs; "
            "replace with a tiny arch_builder fixture if this test is re-enabled "
            "(follow-up issue)"
        )
    (loc_short, partner_short, _), (loc_long, partner_long, _) = picked
    ctx = _ctx(
        arch,
        (loc_short, partner_short, loc_long, partner_long),
        controls=(0, 2),
        targets=(1, 3),
    )
    sorted_pairs = CongestionAwareTargetGenerator()._sort_pairs_longest_first(
        ctx, layout.PathFinder(arch)
    )
    assert sorted_pairs[0] == (2, 3), (
        f"longest-first expected (2,3) first, got {sorted_pairs}"
    )
```

- [ ] **Step 2: Run test to confirm failure**

```bash
uv run pytest python/tests/heuristics/test_congestion_aware_target_generator.py -v -k sort_longest
```

Expected: fails because the placeholder returns input order.

- [ ] **Step 3: Replace `_sort_pairs_longest_first` implementation**

```python
def _sort_pairs_longest_first(
    self, ctx: TargetContext, pf: layout.PathFinder
) -> list[tuple[int, int]]:
    import math

    placement = ctx.placement
    arch = ctx.arch_spec

    def score(pair: tuple[int, int]) -> float:
        ctrl, tgt = pair
        ctrl_loc = placement[ctrl]
        tgt_loc = placement[tgt]
        ctrl_partner = arch.get_cz_partner(tgt_loc)
        tgt_partner = arch.get_cz_partner(ctrl_loc)
        assert ctrl_partner is not None, (
            f"No CZ partner for qid={tgt} at {tgt_loc}"
        )
        assert tgt_partner is not None, (
            f"No CZ partner for qid={ctrl} at {ctrl_loc}"
        )
        occ_ctrl = frozenset(l for q, l in placement.items() if q != ctrl)
        occ_tgt = frozenset(l for q, l in placement.items() if q != tgt)
        p_ctrl = pf.find_path(ctrl_loc, ctrl_partner, occupied=occ_ctrl)
        p_tgt = pf.find_path(tgt_loc, tgt_partner, occupied=occ_tgt)
        c_ctrl = _sum_base(p_ctrl, pf) if p_ctrl is not None else math.inf
        c_tgt = _sum_base(p_tgt, pf) if p_tgt is not None else math.inf
        return min(c_ctrl, c_tgt)

    pairs = list(zip(ctx.controls, ctx.targets))
    pairs.sort(key=score, reverse=True)
    return pairs
```

Add `import math` near the top of `target_generator.py` (it's not currently imported there).

- [ ] **Step 4: Run tests + pyright**

```bash
uv run pytest python/tests/heuristics/test_congestion_aware_target_generator.py -v
uv run pyright python/bloqade/lanes/heuristics/physical/target_generator.py
```

Expected: 18 passed; pyright: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/target_generator.py python/tests/heuristics/test_congestion_aware_target_generator.py
git commit -m "feat(heuristics): implement longest-first sort in CongestionAwareTargetGenerator"
```

---

### Task 7: Implement `_commit_pair` with direction choice

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/target_generator.py`
- Modify: `python/tests/heuristics/test_congestion_aware_target_generator.py`

This replaces the placeholder with real direction-choice logic using `_make_weight_fn` and `_choose_control`.

- [ ] **Step 1: Add two concrete tests and two explicitly-deferred tests**

Two tests for which the fixture construction is mechanical go in full; two
tests for which the fixture needs a synthetic arch (a follow-up) are
marked `@pytest.mark.skip(reason=...)` with a tracking-issue hook.

```python
def _find_blocker_scenario(arch):
    """Return (loc_ctrl, loc_tgt, blocker) where the UNCONSTRAINED
    shortest control-direction path passes through `blocker`, but the
    UNCONSTRAINED target-direction path does not.

    Reason this is non-tautological: we compute both paths with
    `occupied=frozenset()` and compare the location sequences.
    """
    pf = layout.PathFinder(arch)
    home = list(arch.home_sites)
    for a in home:
        pa = arch.get_cz_partner(a)
        if pa is None or pa == a:
            continue
        # control direction from b to pa: need a path through some site.
        for b in home:
            if b in (a, pa):
                continue
            pb = arch.get_cz_partner(b)
            if pb is None or pb in (a, pa):
                continue
            path_ctrl = pf.find_path(b, pa)  # control moves b -> partner(a)
            path_tgt = pf.find_path(a, pb)   # target  moves a -> partner(b)
            if path_ctrl is None or path_tgt is None:
                continue
            ctrl_locs = set(path_ctrl[1])
            tgt_locs = set(path_tgt[1])
            # A non-participating site that sits on the control path but
            # not the target path is the one we want as a blocker; ensure
            # it's free to place an atom there (not in {a, b, pa, pb}).
            candidates = ctrl_locs - tgt_locs - {a, b, pa, pb}
            if candidates:
                return (b, a, next(iter(candidates)))
    return None


def test_target_direction_chosen_when_target_path_cheaper(arch):
    scenario = _find_blocker_scenario(arch)
    if scenario is None:
        pytest.skip(
            "physical arch has no asymmetric-blocker scenario; "
            "re-enable with synthetic fixture (follow-up issue)"
        )
    loc_ctrl, loc_tgt, blocker = scenario
    # Layout: qid 0 = control, qid 1 = target, qid 2 = blocker atom
    ctx = _ctx(
        arch,
        (loc_ctrl, loc_tgt, blocker),
        controls=(0,),
        targets=(1,),
    )
    out = CongestionAwareTargetGenerator().generate(ctx)
    assert out, "generator returned empty"
    plan = out[0]
    partner_of_tgt = arch.get_cz_partner(loc_tgt)
    partner_of_ctrl = arch.get_cz_partner(loc_ctrl)
    # Target-direction chosen: control stays, target moves to partner(ctrl).
    assert plan[0] == loc_ctrl
    assert plan[1] == partner_of_ctrl, (
        f"expected target moved to {partner_of_ctrl}, got {plan[1]}; "
        f"control would have moved to {partner_of_tgt}"
    )
    assert plan[2] == blocker


def test_penalty_zero_reproduces_default_on_symmetric_stage(arch):
    """With all penalties = 0 and a single-pair symmetric stage, the
    congestion-aware heuristic reduces to the default (control-moves)
    by symmetry + tiebreak. Sanity check the reduction.
    """
    from bloqade.lanes.heuristics.physical.target_generator import DefaultTargetGenerator

    loc0, loc1 = _pick_cz_pair(arch)
    # Seed a fresh pair of qubits at non-partnered storage locations so
    # both directions have non-trivial paths (not already-partnered).
    scenario = _find_blocker_scenario(arch)
    if scenario is None:
        pytest.skip("arch has no suitable non-partnered pair; see follow-up issue")
    loc_ctrl, loc_tgt, _blocker = scenario
    ctx = _ctx(arch, (loc_ctrl, loc_tgt), controls=(0,), targets=(1,))
    gen_zero = CongestionAwareTargetGenerator(0.0, 0.0, 0.0)
    gen_default = DefaultTargetGenerator()
    out_zero = gen_zero.generate(ctx)
    out_default = gen_default.generate(ctx)
    assert out_zero == out_default, (
        f"penalty=0 should match DefaultTargetGenerator on symmetric stages.\n"
        f"zero:    {out_zero}\ndefault: {out_default}"
    )


@pytest.mark.skip(
    reason="requires synthetic multi-lane arch fixture; tracked as follow-up issue"
)
def test_multi_pair_avoids_opposite_direction_reuse(arch):
    """Two pairs whose uncongested shortest paths would traverse the
    same lane in opposite directions. With opposite_direction_penalty
    large enough, the second-committed pair picks its more-expensive
    direction to avoid the conflict.
    """
    # Fixture: a minimal arch with at least one lane that serves two
    # CZ-partnered pairs in opposite directions (not available on the
    # current physical Gemini arch without custom construction). See
    # `python/tests/layout/test_arch.py:35-68` for a `from_components`
    # pattern. Implementation plan: build a 2-word 4-site arch with
    # explicit site-bus edges linking pair A's path and pair B's path.
    ...


@pytest.mark.skip(
    reason="covered by Task 3 unit test of _choose_control; end-to-end "
    "version requires arch-specific fixture, tracked as follow-up"
)
def test_move_count_tiebreak_at_commit_end_to_end(arch):
    """End-to-end verification that move-count breaks cost ties at
    commit. Unit test `test_choose_control_cost_tie_uses_length`
    already covers the logic directly.
    """
    ...
```

- [ ] **Step 2: Run tests; verify pass + skipped counts**

```bash
uv run pytest python/tests/heuristics/test_congestion_aware_target_generator.py -v
```

Expected: the two non-skipped tests may still fail (implementation in Step 3 below replaces the placeholder). Actual pre-Step-3 state: both fail because the placeholder `_commit_pair` always picks control. Note this number explicitly as the baseline.

- [ ] **Step 3: Replace `_commit_pair` with full direction-choice logic**

```python
def _commit_pair(
    self,
    arch_spec: layout.ArchSpec,
    pf: layout.PathFinder,
    working: dict[int, layout.LocationAddress],
    committed_lanes: dict[_LaneKey, layout.Direction],
    committed_sites: set[layout.LocationAddress],
    ctrl: int,
    tgt: int,
) -> (
    tuple[
        int,
        layout.LocationAddress,
        tuple[
            tuple[layout.LaneAddress, ...],
            tuple[layout.LocationAddress, ...],
        ],
    ]
    | None
):
    ctrl_loc = working[ctrl]
    tgt_loc = working[tgt]
    ctrl_partner = arch_spec.get_cz_partner(tgt_loc)
    tgt_partner = arch_spec.get_cz_partner(ctrl_loc)
    assert ctrl_partner is not None, (
        f"No CZ partner for qid={tgt} at {tgt_loc}"
    )
    assert tgt_partner is not None, (
        f"No CZ partner for qid={ctrl} at {ctrl_loc}"
    )

    occ_ctrl = frozenset(l for q, l in working.items() if q != ctrl)
    occ_tgt = frozenset(l for q, l in working.items() if q != tgt)

    weight = _make_weight_fn(pf, committed_lanes, committed_sites, self)
    path_ctrl = pf.find_path(
        ctrl_loc, ctrl_partner, occupied=occ_ctrl, edge_weight=weight
    )
    path_tgt = pf.find_path(
        tgt_loc, tgt_partner, occupied=occ_tgt, edge_weight=weight
    )

    cost_ctrl = _sum_weighted(path_ctrl, weight) if path_ctrl else math.inf
    cost_tgt = _sum_weighted(path_tgt, weight) if path_tgt else math.inf

    if cost_ctrl == math.inf and cost_tgt == math.inf:
        return None

    len_ctrl = float(len(path_ctrl[0])) if path_ctrl else math.inf
    len_tgt = float(len(path_tgt[0])) if path_tgt else math.inf

    if _choose_control(cost_ctrl, cost_tgt, len_ctrl, len_tgt):
        assert path_ctrl is not None
        return (ctrl, ctrl_partner, path_ctrl)
    assert path_tgt is not None
    return (tgt, tgt_partner, path_tgt)
```

- [ ] **Step 4: Run all tests + pyright**

```bash
uv run pytest python/tests/heuristics/test_congestion_aware_target_generator.py -v
uv run pyright python/bloqade/lanes/heuristics/physical/target_generator.py
```

Expected: `test_target_direction_chosen_when_target_path_cheaper` and `test_penalty_zero_reproduces_default_on_symmetric_stage` now pass (or `pytest.skip` cleanly if the arch's geometry has no suitable blocker scenario — follow-up issue then tracks the synthetic fixture). Two `@pytest.mark.skip`-decorated tests remain deferred. Pyright: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/target_generator.py python/tests/heuristics/test_congestion_aware_target_generator.py
git commit -m "feat(heuristics): implement direction choice in CongestionAwareTargetGenerator"
```

---

### Task 8: Determinism + deferred infeasibility test

**Files:**
- Modify: `python/tests/heuristics/test_congestion_aware_target_generator.py`

Covers spec unit tests 9 (infeasibility → `[]`) and 12 (determinism).

Spec test 13 (partner-None → `AssertionError`) is **dropped**: on the physical Gemini arch every `home_sites` location has a CZ partner (all words are entangling-paired per `arch/gemini/physical/spec.py`), so the scenario is unreachable without a synthetic arch fixture. The assertion already lives in `_sort_pairs_longest_first` and `_commit_pair`; it runs in production for any misconfigured arch. The test adds no meaningful coverage on the reachable arch and is deferred to a follow-up alongside the synthetic-arch fixtures from Task 7.

- [ ] **Step 1: Add determinism test**

```python
def test_generate_is_deterministic_across_calls(arch):
    loc0, loc1 = _pick_cz_pair(arch)
    ctx = _ctx(arch, (loc0, loc1), controls=(0,), targets=(1,))
    gen = CongestionAwareTargetGenerator()
    assert gen.generate(ctx) == gen.generate(ctx)
```

- [ ] **Step 2: Add deferred infeasibility test**

```python
@pytest.mark.skip(
    reason="requires a blocker set that provably isolates both partners "
    "from the full physical lane graph; deferred to a synthetic arch "
    "fixture follow-up (same issue as opposite-direction-reuse test)"
)
def test_both_directions_infeasible_returns_empty_list():
    """Surround both endpoints of a pair by blockers so that no path
    exists; verify generate() returns [].

    Fixture construction sketch for the follow-up: use arch_builder to
    build a 2-word arch where both partners of the test pair are
    reachable only through a single lane, then place atoms on that
    lane's two endpoints. The all-paths-infeasible property follows
    from the arch's topology.
    """
    ...
```

- [ ] **Step 3: Run tests + pyright**

```bash
uv run pytest python/tests/heuristics/test_congestion_aware_target_generator.py -v
uv run pyright python/bloqade/lanes/heuristics/physical/target_generator.py
```

Expected: determinism test passes; infeasibility test shows `SKIPPED`. Pyright: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add python/tests/heuristics/test_congestion_aware_target_generator.py
git commit -m "test(heuristics): add determinism test and defer infeasibility to synthetic arch fixture"
```

---

## Chunk 4: Integration tests + wiring

### Task 9: Re-export from package `__init__`

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/__init__.py`

- [ ] **Step 1: Identify existing re-export pattern**

```bash
grep -n "^from .target_generator" python/bloqade/lanes/heuristics/physical/__init__.py
grep -n "__all__" python/bloqade/lanes/heuristics/physical/__init__.py
```

- [ ] **Step 2: Add re-export**

Append `CongestionAwareTargetGenerator` to whatever import line currently re-exports `DefaultTargetGenerator` (added by the plugin-interface PR), and add it to `__all__` if that pattern is used.

- [ ] **Step 3: Verify public import path**

```bash
uv run python -c "
from bloqade.lanes.heuristics.physical import CongestionAwareTargetGenerator
print(CongestionAwareTargetGenerator)
"
```

Expected: prints the class.

- [ ] **Step 4: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/__init__.py
git commit -m "feat(heuristics): re-export CongestionAwareTargetGenerator"
```

---

### Task 10: End-to-end integration test

**Files:**
- Modify: `python/tests/heuristics/test_physical_placement.py`

The existing test module exercises `PhysicalPlacementStrategy`. Add a parametrized block using `CongestionAwareTargetGenerator`.

- [ ] **Step 1: Locate a representative existing test**

```bash
grep -n "PhysicalPlacementStrategy" python/tests/heuristics/test_physical_placement.py | head
grep -n "target_generator" python/tests/heuristics/test_physical_placement.py | head
```

Identify a test that (a) runs a full `cz_placements` call and (b) asserts that output is `ExecuteCZ`. Use it as a template.

- [ ] **Step 2: Add the new integration test**

```python
def _pick_cz_pair_integration(arch):
    for s in arch.home_sites:
        p = arch.get_cz_partner(s)
        if p is not None and p != s:
            return s, p
    raise AssertionError("arch has no CZ-partnered home site")


def test_cz_placements_with_congestion_aware_generator_produces_execute_cz():
    """Smoke test: PhysicalPlacementStrategy wired with
    CongestionAwareTargetGenerator completes cz_placements and returns
    an ExecuteCZ result for a simple stage.
    """
    from bloqade.lanes.arch.gemini.physical import get_arch_spec
    from bloqade.lanes.heuristics.physical import (
        CongestionAwareTargetGenerator, PhysicalPlacementStrategy,
    )
    from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ

    arch = get_arch_spec()
    loc0, loc1 = _pick_cz_pair_integration(arch)

    strategy = PhysicalPlacementStrategy(
        arch_spec=arch,
        target_generator=CongestionAwareTargetGenerator(),
    )
    state = ConcreteState(occupied=frozenset(), layout=(loc0, loc1), move_count=(0, 0))
    result = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(result, ExecuteCZ), (
        f"expected ExecuteCZ, got {type(result).__name__}"
    )
```

- [ ] **Step 3: Run the integration test + full test suite**

```bash
uv run pytest python/tests/heuristics/test_physical_placement.py -v -k congestion_aware
uv run pytest python/tests/heuristics -v
```

Expected: new test passes; no regression elsewhere.

- [ ] **Step 4: Commit**

```bash
git add python/tests/heuristics/test_physical_placement.py
git commit -m "test(heuristics): integration smoke test for CongestionAwareTargetGenerator"
```

---

### Task 11: Dedup-with-default integration test

**Files:**
- Modify: `python/tests/heuristics/test_physical_placement.py`

- [ ] **Step 1: Add the test**

```python
def test_congestion_aware_dedups_with_default_on_symmetric_stage():
    """For a stage where the congestion-aware candidate equals what
    DefaultTargetGenerator would produce, the framework's dedup drops
    the default and search runs exactly once.

    Rust path: assert on strategy.rust_nodes_expanded_total.
    Python path: instrument by counting calls to
                 traversal.path_to_target_config.
    """
    from unittest.mock import patch
    from bloqade.lanes.arch.gemini.physical import get_arch_spec
    from bloqade.lanes.heuristics.physical import (
        CongestionAwareTargetGenerator, EntropyPlacementTraversal,
        PhysicalPlacementStrategy,
    )
    from bloqade.lanes.analysis.placement import ConcreteState

    arch = get_arch_spec()
    # Single-pair stage: both generators produce the same candidate.
    loc0, loc1 = _pick_cz_pair_integration(arch)

    strategy = PhysicalPlacementStrategy(
        arch_spec=arch,
        traversal=EntropyPlacementTraversal(),
        target_generator=CongestionAwareTargetGenerator(),
    )
    state = ConcreteState(occupied=frozenset(), layout=(loc0, loc1), move_count=(0, 0))

    call_count = {"n": 0}
    real = EntropyPlacementTraversal.path_to_target_config

    def counting(self, *args, **kwargs):
        call_count["n"] += 1
        return real(self, *args, **kwargs)

    with patch.object(EntropyPlacementTraversal, "path_to_target_config", counting):
        _ = strategy.cz_placements(state, controls=(0,), targets=(1,))

    assert call_count["n"] == 1, (
        f"expected 1 search call after dedup, got {call_count['n']}"
    )
```

Note: if the framework's dedup comparison is strict dict equality and the congestion-aware candidate always returns a *copy* of `placement` (which may be a new dict object but equal as a dict), dedup should still match. Verify empirically; if it doesn't match for a trivial reason, refine the dedup check or adjust the test.

- [ ] **Step 2: Run test**

```bash
uv run pytest python/tests/heuristics/test_physical_placement.py -v -k dedups_with_default
```

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add python/tests/heuristics/test_physical_placement.py
git commit -m "test(heuristics): verify dedup with default on symmetric stages"
```

---

### Task 12: Rust-path parity

**Files:**
- Modify: `python/tests/heuristics/test_physical_placement.py`

- [ ] **Step 1: Add the test**

```python
def test_congestion_aware_applies_to_rust_traversal():
    """Plugin applies identically under RustPlacementTraversal; both
    traversals complete with ExecuteCZ for a simple stage.
    """
    from bloqade.lanes.arch.gemini.physical import get_arch_spec
    from bloqade.lanes.heuristics.physical import (
        CongestionAwareTargetGenerator, EntropyPlacementTraversal,
        PhysicalPlacementStrategy, RustPlacementTraversal,
    )
    from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ

    arch = get_arch_spec()
    loc0, loc1 = _pick_cz_pair_integration(arch)

    for traversal in (
        EntropyPlacementTraversal(),
        RustPlacementTraversal(),
    ):
        strategy = PhysicalPlacementStrategy(
            arch_spec=arch,
            traversal=traversal,
            target_generator=CongestionAwareTargetGenerator(),
        )
        state = ConcreteState(
            occupied=frozenset(), layout=(loc0, loc1), move_count=(0, 0)
        )
        result = strategy.cz_placements(state, controls=(0,), targets=(1,))
        assert isinstance(result, ExecuteCZ), (
            f"{type(traversal).__name__} did not produce ExecuteCZ"
        )
```

- [ ] **Step 2: Run test**

```bash
uv run pytest python/tests/heuristics/test_physical_placement.py -v -k applies_to_rust
```

Expected: pass.

- [ ] **Step 3: Run full suite + pyright + format**

```bash
uv run pytest python/tests -v
uv run pyright python/bloqade/lanes/heuristics/physical/target_generator.py python/tests/heuristics/
uv run black python/bloqade/lanes/heuristics/physical/target_generator.py python/tests/heuristics/
uv run isort python/bloqade/lanes/heuristics/physical/target_generator.py python/tests/heuristics/
uv run ruff check python/bloqade/lanes/heuristics/physical/target_generator.py python/tests/heuristics/
```

Expected: 0 failures, 0 pyright errors, no formatting diff.

- [ ] **Step 4: Final commit**

```bash
git add python/tests/heuristics/test_physical_placement.py
git commit -m "test(heuristics): rust-path parity for CongestionAwareTargetGenerator"
```

---

## Post-implementation

- [ ] **Open follow-up issues**

Create GitHub issues (every issue gets the `A-Lane` label):

1. **Penalty tuning** — empirical sweep on a representative workload; report which weights win. Labels: `A-Lane`, performance.
2. **`LaneAddress.canonical()`** — promote `_lane_key` to a layout-level helper if a second caller appears. Labels: `A-Lane`, refactor.
3. **`PathFinder` caching across `generate()` calls** — frozen-dataclass-compatible cache if profiling flags construction as hot. Labels: `A-Lane`, performance.
4. **Synthetic arch fixture for deferred tests** — build a minimal `arch_builder`-constructed arch that exercises (a) opposite-direction lane reuse between two pairs, (b) all-paths-infeasible isolation, (c) a home site with no CZ partner. Un-skips `test_multi_pair_avoids_opposite_direction_reuse`, `test_move_count_tiebreak_at_commit_end_to_end`, `test_both_directions_infeasible_returns_empty_list`, and (if desired) a partner-None assertion test. Labels: `A-Lane`, test-infrastructure.

- [ ] **Push the branch**

```bash
git push -u origin spec/congestion-aware-target-generator
```

- [ ] **Open PR**

PR description references:
- Spec: `docs/superpowers/specs/2026-04-17-congestion-aware-target-generator-design.md`
- Plan: `docs/superpowers/plans/2026-04-17-congestion-aware-target-generator.md`
- Depends on: `spec/target-generator-plugin` (link the companion PR)

Labels: `A-Lane`, `S-backport`, `backport v0.7` (non-breaking, additive).
