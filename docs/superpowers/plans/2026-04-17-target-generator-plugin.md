# Target-generator plugin — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pluggable target-generation heuristic to `PhysicalPlacementStrategy` with a guaranteed default fallback and shared-budget multi-candidate search.

**Spec:** `docs/superpowers/specs/2026-04-17-target-generator-plugin-design.md`

**Architecture:** Introduce `TargetContext`, `TargetGeneratorABC`, `DefaultTargetGenerator` (and a private callable adapter) in `physical/movement.py`. Add an optional `target_generator` field to `PhysicalPlacementStrategy`. Thread a shared-budget candidate loop through both `cz_placements` (Python path) and `_cz_placements_rust`. Validation rejects malformed candidates; orchestration dedups plugin output and always appends the default as a guaranteed last candidate.

**Tech Stack:** Python 3.10+ (dataclasses, abc, typing unions), pytest, pyright (all new code must type-check clean), pre-commit hooks (black/isort/ruff).

**Branch:** `spec/target-generator-plugin` (continue on the branch that holds the spec commit).

---

## File Structure

| Path | Change | Responsibility |
|---|---|---|
| `python/bloqade/lanes/heuristics/physical/movement.py` | modify | Add `TargetContext`, `TargetGeneratorABC`, `DefaultTargetGenerator`, `_CallableTargetGenerator`, `_validate_candidate`, `_build_candidates`; add `target_generator` field on `PhysicalPlacementStrategy`; thread shared-budget loop through `cz_placements` and `_cz_placements_rust`; add `_cz_counter` increment to Rust path. |
| `python/bloqade/lanes/heuristics/physical/__init__.py` | modify | Re-export `TargetContext`, `TargetGeneratorABC`, `DefaultTargetGenerator`. |
| `python/tests/heuristics/test_target_generator.py` | create | Unit tests for `TargetContext`, `DefaultTargetGenerator`, callable adapter, `_validate_candidate`. |
| `python/tests/heuristics/test_physical_placement.py` | modify | Integration tests: `target_generator=None` regression guard, plugin `[]` behaves like `None`, cheaper candidate wins, shared-budget invariant, dedup, Rust path, plugin-raises. |

---

## Chunk 1: Core interface + defaults

### Task 1: Add `TargetContext` dataclass

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/movement.py` (new import + dataclass near `PlacementTraversalABC`)
- Test: `python/tests/heuristics/test_target_generator.py`

- [ ] **Step 1: Create the test file with a failing test**

Create `python/tests/heuristics/test_target_generator.py`:

```python
from __future__ import annotations

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.heuristics.physical.movement import TargetContext


def _make_state() -> ConcreteState:
    return ConcreteState(
        occupied=frozenset(),
        layout=(
            layout.LocationAddress(0, 0),
            layout.LocationAddress(1, 0),
        ),
        move_count=(0, 0),
    )


def test_target_context_placement_derives_from_state_layout():
    state = _make_state()
    ctx = TargetContext(
        arch_spec=logical.get_arch_spec(),
        state=state,
        controls=(0,),
        targets=(1,),
        lookahead_cz_layers=(),
        cz_stage_index=0,
    )
    assert ctx.placement == {0: state.layout[0], 1: state.layout[1]}
```

- [ ] **Step 2: Run test to confirm failure**

```bash
uv run pytest python/tests/heuristics/test_target_generator.py -v
```

Expected: `ImportError` — `TargetContext` not defined.

- [ ] **Step 3: Add `TargetContext` to `movement.py`**

Add after the `OnSearchStep` type alias (around line 43 in the current file) and before `class PlacementTraversalABC`:

```python
@dataclass(frozen=True)
class TargetContext:
    """Signals passed to a TargetGenerator.

    Composes ConcreteState to avoid duplicating lattice state fields.
    """

    arch_spec: layout.ArchSpec
    state: ConcreteState
    controls: tuple[int, ...]
    targets: tuple[int, ...]
    lookahead_cz_layers: tuple[
        tuple[tuple[int, ...], tuple[int, ...]], ...
    ]
    cz_stage_index: int

    @property
    def placement(self) -> dict[int, LocationAddress]:
        return dict(enumerate(self.state.layout))
```

- [ ] **Step 4: Run test to confirm pass + pyright clean**

```bash
uv run pytest python/tests/heuristics/test_target_generator.py -v
uv run pyright python/bloqade/lanes/heuristics/physical/movement.py python/tests/heuristics/test_target_generator.py
```

Expected: 1 passed; pyright: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/movement.py python/tests/heuristics/test_target_generator.py
git commit -m "feat(heuristics): add TargetContext for target-generator plugin"
```

---

### Task 1.5: Pre-flight fixture sanity check

**Files:** none (verification only)

The fixture used in `test_physical_placement.py` (`_make_state`) places qubits at `LocationAddress(0, 0)` and `LocationAddress(1, 0)`. Confirm these are CZ-blockade-partnered on `logical.get_arch_spec()` before proceeding, so every downstream test in Chunks 1–3 has a valid arch fixture.

- [ ] **Step 1: Verify fixture validity**

```bash
uv run python -c "
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes import layout
arch = logical.get_arch_spec()
l0 = layout.LocationAddress(0, 0)
l1 = layout.LocationAddress(1, 0)
p0 = arch.get_cz_partner(l0)
p1 = arch.get_cz_partner(l1)
assert p0 == l1, f'partner({l0}) == {p0}, expected {l1}'
assert p1 == l0, f'partner({l1}) == {p1}, expected {l0}'
print('fixture ok')
"
```

Expected: `fixture ok`. If this fails, switch to a fixture drawn from `arch.cz_zone_addresses` before proceeding.

---

### Task 2: Add `TargetGeneratorABC` + `DefaultTargetGenerator`

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/movement.py`
- Test: `python/tests/heuristics/test_target_generator.py`

- [ ] **Step 1: Add failing test for `DefaultTargetGenerator.generate`**

Append to `test_target_generator.py`:

```python
from bloqade.lanes.heuristics.physical.movement import (
    DefaultTargetGenerator,
    TargetGeneratorABC,
)


def test_default_target_generator_matches_current_rule():
    state = _make_state()
    arch_spec = logical.get_arch_spec()
    ctx = TargetContext(
        arch_spec=arch_spec,
        state=state,
        controls=(0,),
        targets=(1,),
        lookahead_cz_layers=(),
        cz_stage_index=0,
    )
    candidates = DefaultTargetGenerator().generate(ctx)
    assert len(candidates) == 1
    target = candidates[0]
    # Target qubit stays put
    assert target[1] == state.layout[1]
    # Control qubit moves to the CZ partner of target's current location
    assert target[0] == arch_spec.get_cz_partner(state.layout[1])


def test_default_target_generator_is_target_generator_abc():
    assert isinstance(DefaultTargetGenerator(), TargetGeneratorABC)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest python/tests/heuristics/test_target_generator.py -v
```

Expected: `ImportError` — `DefaultTargetGenerator`, `TargetGeneratorABC` not defined.

- [ ] **Step 3: Add ABC and default implementation to `movement.py`**

Add immediately after `TargetContext`:

```python
class TargetGeneratorABC(abc.ABC):
    """Plugin interface for choosing the target configuration of a CZ stage.

    Implementations return an *ordered* list of candidate target
    placements. The strategy framework appends the default candidate
    (``DefaultTargetGenerator``) as a guaranteed last-resort, so a plugin
    may return ``[]`` to defer entirely to the default.
    """

    @abc.abstractmethod
    def generate(
        self, ctx: TargetContext
    ) -> list[dict[int, LocationAddress]]: ...


@dataclass(frozen=True)
class DefaultTargetGenerator(TargetGeneratorABC):
    """Current rule: control qubit moves to the CZ partner of the target's location."""

    def generate(
        self, ctx: TargetContext
    ) -> list[dict[int, LocationAddress]]:
        target = dict(ctx.placement)
        for control_qid, target_qid in zip(ctx.controls, ctx.targets):
            target_loc = target[target_qid]
            partner = ctx.arch_spec.get_cz_partner(target_loc)
            assert partner is not None, (
                f"No CZ blockade partner for {target_loc}"
            )
            target[control_qid] = partner
        return [target]
```

- [ ] **Step 4: Run tests + pyright**

```bash
uv run pytest python/tests/heuristics/test_target_generator.py -v
uv run pyright python/bloqade/lanes/heuristics/physical/movement.py python/tests/heuristics/test_target_generator.py
```

Expected: 3 passed; pyright: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/movement.py python/tests/heuristics/test_target_generator.py
git commit -m "feat(heuristics): add TargetGeneratorABC + DefaultTargetGenerator"
```

---

### Task 3: Add `_validate_candidate` helper

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/movement.py`
- Test: `python/tests/heuristics/test_target_generator.py`

Validation covers three cases:

1. **Missing qid**: candidate must contain every qid from the current placement.
2. **Unknown location**: every location value must be recognized by the `arch_spec`, via the existing `arch_spec.check_location_group(...)` Rust-backed helper.
3. **Invalid CZ pair**: for each `(control_qid, target_qid)`, the pair must be CZ-blockade-partnered in either direction.

- [ ] **Step 1: Add failing tests**

Append to `test_target_generator.py`:

```python
import pytest

from bloqade.lanes.heuristics.physical.movement import _validate_candidate


def _make_valid_ctx() -> TargetContext:
    state = _make_state()
    return TargetContext(
        arch_spec=logical.get_arch_spec(),
        state=state,
        controls=(0,),
        targets=(1,),
        lookahead_cz_layers=(),
        cz_stage_index=0,
    )


def test_validate_candidate_accepts_default():
    ctx = _make_valid_ctx()
    candidate = DefaultTargetGenerator().generate(ctx)[0]
    _validate_candidate(ctx, candidate)  # no raise


def test_validate_candidate_rejects_missing_qid():
    ctx = _make_valid_ctx()
    candidate = DefaultTargetGenerator().generate(ctx)[0]
    del candidate[1]
    with pytest.raises(ValueError, match="missing"):
        _validate_candidate(ctx, candidate)


def test_validate_candidate_rejects_non_cz_pair():
    ctx = _make_valid_ctx()
    # (0,0) and (2,0) are NOT CZ partners on logical.get_arch_spec()
    # — partner((0,0))=(1,0); partner((2,0))=(3,0). Both checks fail.
    non_partner = layout.LocationAddress(2, 0)
    candidate = {0: ctx.state.layout[0], 1: non_partner}
    with pytest.raises(ValueError, match="blockade"):
        _validate_candidate(ctx, candidate)


def test_validate_candidate_accepts_reverse_pair_direction():
    """The 'either direction' OR-check in _validate_candidate must work
    when partner(control)==target (not just partner(target)==control)."""
    ctx = _make_valid_ctx()
    arch = ctx.arch_spec
    # Build a candidate where the control (qid 0) stays put at its
    # current location, and the target (qid 1) moves to control's partner.
    c_loc = ctx.state.layout[0]
    candidate = {0: c_loc, 1: arch.get_cz_partner(c_loc)}
    _validate_candidate(ctx, candidate)  # no raise


def test_validate_candidate_rejects_unknown_location():
    ctx = _make_valid_ctx()
    # LocationAddress with a wildly out-of-range word_id will fail
    # arch_spec.check_location_group
    bogus = layout.LocationAddress(999, 999)
    candidate = {0: bogus, 1: ctx.state.layout[1]}
    with pytest.raises(ValueError):
        _validate_candidate(ctx, candidate)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest python/tests/heuristics/test_target_generator.py -v
```

Expected: `ImportError` on `_validate_candidate`.

- [ ] **Step 3: Add `_validate_candidate` to `movement.py`**

Add immediately after `DefaultTargetGenerator`:

```python
def _validate_candidate(
    ctx: TargetContext,
    candidate: dict[int, LocationAddress],
) -> None:
    """Raise ValueError if the candidate is not a legal CZ target.

    Checks:
    1. Every qid from ``ctx.placement`` appears in ``candidate``.
    2. Each ``(control_qid, target_qid)`` pair is CZ-blockade-partnered
       in either direction (matching the convention at
       ``python/bloqade/lanes/analysis/placement/lattice.py:134-135``).
    """
    placement = ctx.placement
    missing = set(placement.keys()) - set(candidate.keys())
    if missing:
        raise ValueError(
            f"target-generator candidate missing qubits: {sorted(missing)}"
        )
    # Reuse the Rust-backed location-group validator.
    loc_errors = ctx.arch_spec.check_location_group(list(candidate.values()))
    if loc_errors:
        raise ValueError(
            f"target-generator candidate contains invalid locations: "
            f"{list(loc_errors)}"
        )
    for control_qid, target_qid in zip(ctx.controls, ctx.targets):
        c_loc = candidate[control_qid]
        t_loc = candidate[target_qid]
        if (
            ctx.arch_spec.get_cz_partner(c_loc) != t_loc
            and ctx.arch_spec.get_cz_partner(t_loc) != c_loc
        ):
            raise ValueError(
                f"target-generator candidate CZ pair "
                f"(control={control_qid}@{c_loc}, target={target_qid}@{t_loc}) "
                f"is not blockade-partnered"
            )
```

- [ ] **Step 4: Run tests + pyright**

```bash
uv run pytest python/tests/heuristics/test_target_generator.py -v
uv run pyright python/bloqade/lanes/heuristics/physical/movement.py python/tests/heuristics/test_target_generator.py
```

Expected: 8 passed; pyright: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/movement.py python/tests/heuristics/test_target_generator.py
git commit -m "feat(heuristics): add target-generator candidate validation"
```

---

### Task 4: Add `_CallableTargetGenerator` adapter

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/movement.py`
- Test: `python/tests/heuristics/test_target_generator.py`

- [ ] **Step 1: Add failing test for callable wrapping**

Append to `test_target_generator.py`:

```python
from bloqade.lanes.heuristics.physical.movement import (
    _CallableTargetGenerator,
    _coerce_target_generator,
)


def test_callable_target_generator_wraps_function():
    ctx = _make_valid_ctx()
    expected = DefaultTargetGenerator().generate(ctx)

    def fn(c: TargetContext) -> list[dict[int, layout.LocationAddress]]:
        assert c is ctx
        return expected

    gen = _CallableTargetGenerator(fn)
    assert isinstance(gen, TargetGeneratorABC)
    assert gen.generate(ctx) == expected


def test_coerce_target_generator_passthrough_for_abc():
    abc_gen = DefaultTargetGenerator()
    assert _coerce_target_generator(abc_gen) is abc_gen


def test_coerce_target_generator_wraps_callable():
    def fn(c: TargetContext) -> list[dict[int, layout.LocationAddress]]:
        return []

    gen = _coerce_target_generator(fn)
    assert isinstance(gen, _CallableTargetGenerator)


def test_coerce_target_generator_returns_none_for_none():
    assert _coerce_target_generator(None) is None
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest python/tests/heuristics/test_target_generator.py -v
```

Expected: `ImportError` on `_CallableTargetGenerator` / `_coerce_target_generator`.

- [ ] **Step 3: Add adapter + coercion helper to `movement.py`**

Add immediately after `_validate_candidate`:

```python
TargetGeneratorCallable = Callable[
    [TargetContext], list[dict[int, LocationAddress]]
]


@dataclass(frozen=True)
class _CallableTargetGenerator(TargetGeneratorABC):
    """Private adapter that lifts a bare callable to TargetGeneratorABC."""

    fn: TargetGeneratorCallable

    def generate(
        self, ctx: TargetContext
    ) -> list[dict[int, LocationAddress]]:
        return self.fn(ctx)


def _coerce_target_generator(
    value: TargetGeneratorABC | TargetGeneratorCallable | None,
) -> TargetGeneratorABC | None:
    """Normalize the public union down to TargetGeneratorABC | None."""
    if value is None or isinstance(value, TargetGeneratorABC):
        return value
    return _CallableTargetGenerator(value)
```

- [ ] **Step 4: Run tests + pyright**

```bash
uv run pytest python/tests/heuristics/test_target_generator.py -v
uv run pyright python/bloqade/lanes/heuristics/physical/movement.py python/tests/heuristics/test_target_generator.py
```

Expected: 12 passed; pyright: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/movement.py python/tests/heuristics/test_target_generator.py
git commit -m "feat(heuristics): add callable adapter for target-generator plugin"
```

---

## Chunk 2: Strategy integration

### Task 5: Add `target_generator` field + `__post_init__` narrowing

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/movement.py` (PhysicalPlacementStrategy class)
- Test: `python/tests/heuristics/test_target_generator.py`

- [ ] **Step 1: Add failing tests**

Append to `test_target_generator.py`:

```python
from bloqade.lanes.heuristics.physical.movement import PhysicalPlacementStrategy


def test_strategy_default_target_generator_is_none():
    s = PhysicalPlacementStrategy()
    # The internal normalized form, not the public field
    assert s._resolved_target_generator is None


def test_strategy_accepts_abc_target_generator():
    gen = DefaultTargetGenerator()
    s = PhysicalPlacementStrategy(target_generator=gen)
    assert s._resolved_target_generator is gen


def test_strategy_wraps_callable_target_generator():
    def fn(ctx: TargetContext) -> list[dict[int, layout.LocationAddress]]:
        return []

    s = PhysicalPlacementStrategy(target_generator=fn)
    assert isinstance(s._resolved_target_generator, _CallableTargetGenerator)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest python/tests/heuristics/test_target_generator.py -v -k target_generator
```

Expected: 3 failures — `target_generator` kwarg not accepted; `_resolved_target_generator` not defined.

- [ ] **Step 3: Modify `PhysicalPlacementStrategy`**

In `movement.py`, around the existing `@dataclass class PhysicalPlacementStrategy` (line 180):

1. Add field after `traversal`:

```python
    target_generator: (
        TargetGeneratorABC | TargetGeneratorCallable | None
    ) = None
```

2. Add private normalized attribute after the other `field(..., init=False, ...)` lines:

```python
    _resolved_target_generator: TargetGeneratorABC | None = field(
        default=None, init=False, repr=False
    )
```

3. **Replace the existing `__post_init__` entirely** (currently ends at line 222) with the following. `PhysicalPlacementStrategy` is not frozen, so plain attribute assignment works:

```python
    def __post_init__(self) -> None:
        assert_single_cz_zone(self.arch_spec, type(self).__name__)
        if not isinstance(
            self.traversal, (PlacementTraversalABC, RustPlacementTraversal)
        ):
            raise TypeError(
                "traversal must implement PlacementTraversalABC or be a "
                "RustPlacementTraversal instance"
            )
        if self.target_generator is not None and not (
            isinstance(self.target_generator, TargetGeneratorABC)
            or callable(self.target_generator)
        ):
            raise TypeError(
                "target_generator must be a TargetGeneratorABC, a callable, "
                "or None"
            )
        self._resolved_target_generator = _coerce_target_generator(
            self.target_generator
        )
```

- [ ] **Step 4: Run tests + pyright**

```bash
uv run pytest python/tests/heuristics/test_target_generator.py -v
uv run pyright python/bloqade/lanes/heuristics/physical/movement.py python/tests/heuristics/test_target_generator.py
```

Expected: 15 passed; pyright: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/movement.py python/tests/heuristics/test_target_generator.py
git commit -m "feat(heuristics): add target_generator field to PhysicalPlacementStrategy"
```

---

### Task 6: Add `_build_candidates` helper

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/movement.py` (method on `PhysicalPlacementStrategy`)
- Test: `python/tests/heuristics/test_target_generator.py`

This helper owns the entire "plugin → validate → dedup → append default" sequence. Having it as a single method makes the `cz_placements` / `_cz_placements_rust` changes minimal and easy to test.

- [ ] **Step 1: Add failing tests**

Append to `test_target_generator.py`:

```python
def _make_strategy_with_generator(
    gen: TargetGeneratorABC | TargetGeneratorCallable | None,
) -> PhysicalPlacementStrategy:
    return PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        target_generator=gen,
    )


def test_build_candidates_none_returns_default_only():
    strategy = _make_strategy_with_generator(None)
    ctx = _make_valid_ctx()
    candidates = strategy._build_candidates(ctx)
    assert candidates == DefaultTargetGenerator().generate(ctx)


def test_build_candidates_empty_plugin_appends_default():
    def fn(c):
        return []

    strategy = _make_strategy_with_generator(fn)
    ctx = _make_valid_ctx()
    candidates = strategy._build_candidates(ctx)
    assert candidates == DefaultTargetGenerator().generate(ctx)


def test_build_candidates_dedups_default():
    """Plugin returning the default verbatim should yield one candidate."""
    def fn(c):
        return DefaultTargetGenerator().generate(c)

    strategy = _make_strategy_with_generator(fn)
    ctx = _make_valid_ctx()
    candidates = strategy._build_candidates(ctx)
    assert len(candidates) == 1


def test_build_candidates_preserves_plugin_order_with_default_last():
    arch_spec = logical.get_arch_spec()
    ctx = _make_valid_ctx()
    default = DefaultTargetGenerator().generate(ctx)[0]
    # Construct a second valid candidate by swapping control/target
    alt = dict(default)
    alt[0], alt[1] = alt[1], alt[0]

    def fn(c):
        return [alt]

    strategy = _make_strategy_with_generator(fn)
    candidates = strategy._build_candidates(ctx)
    assert candidates == [alt, default]


def test_build_candidates_raises_on_malformed():
    def fn(c):
        # Both qubits stay put — not CZ-partnered, should raise
        return [{0: c.state.layout[0], 1: c.state.layout[1]}]

    strategy = _make_strategy_with_generator(fn)
    ctx = _make_valid_ctx()
    with pytest.raises(ValueError, match="blockade"):
        strategy._build_candidates(ctx)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest python/tests/heuristics/test_target_generator.py -v -k build_candidates
```

Expected: 5 failures — `_build_candidates` not defined.

- [ ] **Step 3: Add `_build_candidates` method to `PhysicalPlacementStrategy`**

Add as a method on `PhysicalPlacementStrategy` (in `movement.py`, between `_target_from_stage_controls_only` and `_run_search`):

```python
    def _build_candidates(
        self,
        ctx: TargetContext,
    ) -> list[dict[int, LocationAddress]]:
        """Build the ordered candidate list: plugin output + default-as-fallback.

        Dedups plugin candidates by dict equality (preserving order) and
        appends the default candidate only if it is not already present.
        Validates every candidate against ``_validate_candidate`` before
        returning; a malformed candidate raises ``ValueError``.
        """
        plugin = self._resolved_target_generator
        plugin_candidates: list[dict[int, LocationAddress]] = (
            [] if plugin is None else list(plugin.generate(ctx))
        )

        deduped: list[dict[int, LocationAddress]] = []
        for candidate in plugin_candidates:
            _validate_candidate(ctx, candidate)
            if candidate not in deduped:
                deduped.append(candidate)

        default = DefaultTargetGenerator().generate(ctx)[0]
        if default not in deduped:
            deduped.append(default)
        return deduped
```

- [ ] **Step 4: Run tests + pyright**

```bash
uv run pytest python/tests/heuristics/test_target_generator.py -v
uv run pyright python/bloqade/lanes/heuristics/physical/movement.py python/tests/heuristics/test_target_generator.py
```

Expected: 20 passed; pyright: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/movement.py python/tests/heuristics/test_target_generator.py
git commit -m "feat(heuristics): add _build_candidates with plugin + default dedup"
```

---

## Chunk 3: Wire into search paths

### Task 7: Shared-budget loop in `cz_placements` (Python path)

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/movement.py` (`cz_placements` method)
- Test: `python/tests/heuristics/test_physical_placement.py`

**Approach:** keep `_run_search` as the single-target primitive (its signature is unchanged — existing monkeypatched tests keep working). The candidate loop lives inline in `cz_placements`. When `target_generator is None`, `_build_candidates` returns `[default]` and the loop runs a single search — behavior is identical to today.

**Preliminary: declare `max_expansions` on `PlacementTraversalABC`.** pyright needs to see the field on the base class so `base_traversal.max_expansions` type-checks after the `isinstance(base_traversal, PlacementTraversalABC)` narrow. All three existing subclasses already define this field as a dataclass attribute, so declaring it on the ABC is zero-behavior-change:

```python
class PlacementTraversalABC(abc.ABC):
    """Placement-facing traversal API for target-configuration search."""

    max_expansions: int | None

    @abc.abstractmethod
    def path_to_target_config(
        self, *, tree: ConfigurationTree, target: dict[int, layout.LocationAddress],
    ) -> SearchResult:
        """Run search and return one or more goal nodes."""
        ...
```

Apply this edit **before** adding the loop below.

- [ ] **Step 1: Add failing tests**

Append to `test_physical_placement.py`:

```python
from bloqade.lanes.heuristics.physical.movement import (
    DefaultTargetGenerator,
    TargetContext,
)


def test_target_generator_none_matches_today_behavior(monkeypatch):
    """Regression guard: None plugin path is functionally identical to today."""
    strategy = PhysicalPlacementStrategy(arch_spec=logical.get_arch_spec())
    state = _make_state()
    seen_targets: list[dict] = []

    def fake_run_search(self, tree, target, traversal=None):
        _ = self, tree, traversal
        seen_targets.append(dict(target))
        return SearchResult(
            goal_node=tree.root, nodes_expanded=0, max_depth_reached=0
        )

    monkeypatch.setattr(PhysicalPlacementStrategy, "_run_search", fake_run_search)
    strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert len(seen_targets) == 1


def test_target_generator_empty_plugin_behaves_like_none(monkeypatch):
    strategy = PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        target_generator=lambda ctx: [],
    )
    state = _make_state()
    count = 0

    def fake_run_search(self, tree, target, traversal=None):
        nonlocal count
        count += 1
        _ = self, target, traversal
        return SearchResult(
            goal_node=tree.root, nodes_expanded=0, max_depth_reached=0
        )

    monkeypatch.setattr(PhysicalPlacementStrategy, "_run_search", fake_run_search)
    strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert count == 1  # only the default candidate runs


def test_target_generator_cheaper_candidate_wins(monkeypatch):
    """Plugin returns a candidate that solves on first attempt; default never runs."""
    arch_spec = logical.get_arch_spec()
    state = _make_state()
    default_target = {
        0: arch_spec.get_cz_partner(state.layout[1]),
        1: state.layout[1],
    }
    # Alt candidate swaps the roles: target moves to control's partner.
    alt_target = {
        0: state.layout[0],
        1: arch_spec.get_cz_partner(state.layout[0]),
    }
    strategy = PhysicalPlacementStrategy(
        arch_spec=arch_spec,
        target_generator=lambda ctx: [alt_target],
    )
    targets_tried: list[dict] = []

    def fake_run_search(self, tree, target, traversal=None):
        _ = self, traversal
        targets_tried.append(dict(target))
        # First attempt succeeds
        return SearchResult(
            goal_node=tree.root, nodes_expanded=0, max_depth_reached=0
        )

    monkeypatch.setattr(PhysicalPlacementStrategy, "_run_search", fake_run_search)
    strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert targets_tried == [alt_target]  # default not tried


def test_target_generator_shared_budget(monkeypatch):
    """Sum of per-candidate nodes_expanded cannot exceed configured max."""
    arch_spec = logical.get_arch_spec()
    state = _make_state()
    alt_target = {
        0: state.layout[0],
        1: arch_spec.get_cz_partner(state.layout[0]),
    }
    strategy = PhysicalPlacementStrategy(
        arch_spec=arch_spec,
        traversal=EntropyPlacementTraversal(max_expansions=10),
        target_generator=lambda ctx: [alt_target],
    )
    budgets_seen: list[int | None] = []
    consumed_per_call = 4

    def fake_path_to_target_config(self, **kwargs):
        _ = kwargs
        budgets_seen.append(self.max_expansions)
        return SearchResult(
            goal_node=None,
            nodes_expanded=consumed_per_call,
            max_depth_reached=0,
        )

    monkeypatch.setattr(
        EntropyPlacementTraversal,
        "path_to_target_config",
        fake_path_to_target_config,
    )
    strategy.cz_placements(state, controls=(0,), targets=(1,))
    # First call uses full 10; second call uses 10 - 4 = 6.
    assert budgets_seen == [10, 6]


def test_target_generator_raises_propagates():
    def boom(ctx):
        raise RuntimeError("plugin exploded")

    strategy = PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        target_generator=boom,
    )
    state = _make_state()
    with pytest.raises(RuntimeError, match="plugin exploded"):
        strategy.cz_placements(state, controls=(0,), targets=(1,))
```

Add `import pytest` at the top of the file if missing.

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest python/tests/heuristics/test_physical_placement.py -v -k target_generator
```

Expected: 5 failures — shared-budget not plumbed, plugin-path not implemented.

- [ ] **Step 3: Modify `cz_placements`**

Replace the body of `cz_placements` (around lines 259–313) to use the candidate loop. New structure:

```python
    def cz_placements(
        self,
        state: AtomState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
    ) -> AtomState:
        if len(controls) != len(targets) or state == AtomState.bottom():
            return AtomState.bottom()
        if not isinstance(state, ConcreteState):
            return AtomState.top()

        if isinstance(self.traversal, RustPlacementTraversal):
            return self._cz_placements_rust(
                state, controls, targets, lookahead_cz_layers
            )

        ctx = TargetContext(
            arch_spec=self.arch_spec,
            state=state,
            controls=controls,
            targets=targets,
            lookahead_cz_layers=lookahead_cz_layers,
            cz_stage_index=self._cz_counter,
        )
        candidates = self._build_candidates(ctx)

        tree = ConfigurationTree.from_initial_placement(
            self.arch_spec,
            ctx.placement,
            blocked_locations=state.occupied,
        )

        should_trace = (
            self._trace_cz_index is None
            or self._cz_counter == self._trace_cz_index
        )
        base_traversal = self.traversal
        assert isinstance(base_traversal, PlacementTraversalABC)
        if (
            isinstance(base_traversal, EntropyPlacementTraversal)
            and not should_trace
        ):
            base_traversal = replace(base_traversal, on_search_step=None)
        if should_trace:
            self._traced_tree = tree
            self._traced_target = dict(candidates[0])

        remaining = base_traversal.max_expansions
        best_goal = None
        for candidate in candidates:
            if remaining is not None and remaining <= 0:
                break
            per_call_traversal = (
                base_traversal
                if remaining == base_traversal.max_expansions
                else replace(base_traversal, max_expansions=remaining)
            )
            result = self._run_search(tree, candidate, per_call_traversal)
            if remaining is not None:
                remaining -= int(result.nodes_expanded)
            if result.goal_nodes:
                best_goal = result.goal_nodes[0]
                break

        self._cz_counter += 1

        if best_goal is None:
            return AtomState.bottom()

        move_program = best_goal.to_move_program()
        goal_layout_map = best_goal.configuration
        goal_layout = tuple(
            goal_layout_map[qid] for qid in range(len(state.layout))
        )
        move_count = tuple(
            mc + int(src != dst)
            for mc, src, dst in zip(state.move_count, state.layout, goal_layout)
        )
        return ExecuteCZ(
            occupied=state.occupied,
            layout=goal_layout,
            move_count=move_count,
            active_cz_zones=self.arch_spec.cz_zone_addresses,
            move_layers=move_program,
        )
```

**Notes:**
- `base_traversal.max_expansions` is declared on the ABC (see "Preliminary" above) so pyright sees it after the `isinstance` narrow.
- If `remaining == base_traversal.max_expansions` (first iteration), we skip the `replace(...)` allocation as a micro-opt. When `remaining is None`, the comparison is `None == None` which is `True`, so we also skip — correct, since unlimited budget has no per-iteration override.
- Tracing captures `candidates[0]` (the first-attempted candidate) in `_traced_target`. This is a behavior change from today only when a plugin is supplied — without a plugin, `candidates[0]` is the default, matching today's `_traced_target`.

- [ ] **Step 4: Run all affected tests + pyright**

```bash
uv run pytest python/tests/heuristics/test_physical_placement.py python/tests/heuristics/test_target_generator.py -v
uv run pyright python/bloqade/lanes/heuristics/physical/movement.py python/tests/heuristics/test_target_generator.py python/tests/heuristics/test_physical_placement.py
```

Expected: all previous tests + 5 new pass; pyright: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/movement.py python/tests/heuristics/test_physical_placement.py
git commit -m "feat(heuristics): thread shared-budget candidate loop through cz_placements"
```

---

### Task 8: Shared-budget loop in `_cz_placements_rust`

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/movement.py` (`_cz_placements_rust` + its call site)
- Test: `python/tests/heuristics/test_physical_placement.py`

Also fixes the `_cz_counter` parity gap called out in the spec: the Rust path did not increment `_cz_counter`. After this task it does.

- [ ] **Step 1: Update `cz_placements` to pass `lookahead_cz_layers` through to Rust**

(Already in place if Task 7 added it; verify the Rust branch receives the argument.)

- [ ] **Step 2: Add failing tests**

Append to `test_physical_placement.py`:

```python
def test_rust_path_target_generator_shared_budget(monkeypatch):
    strategy = PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        traversal=RustPlacementTraversal(max_expansions=10),
        target_generator=lambda ctx: [
            # A plausible alt — swap control's destination
            {
                0: ctx.state.layout[0],
                1: ctx.arch_spec.get_cz_partner(ctx.state.layout[0]),
            }
        ],
    )
    state = _make_state()
    budgets_seen: list[int | None] = []
    consumed = 4

    class _FakeResult:
        def __init__(self):
            self.status = "unsolvable"
            self.nodes_expanded = consumed

    class _FakeSolver:
        def solve(self, *args, **kwargs):
            _ = args
            budgets_seen.append(kwargs.get("max_expansions"))
            return _FakeResult()

    monkeypatch.setattr(
        PhysicalPlacementStrategy,
        "_get_rust_solver",
        lambda _self: _FakeSolver(),
    )
    strategy.cz_placements(state, controls=(0,), targets=(1,))
    # alt candidate first with full 10; default candidate second with 6.
    assert budgets_seen == [10, 6]


def test_rust_path_cz_counter_increments():
    """Parity fix: _cz_counter must increment on the Rust path too."""
    strategy = PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        traversal=RustPlacementTraversal(),
    )
    state = _make_state()
    strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert strategy._cz_counter == 1
```

- [ ] **Step 3: Run tests to confirm failure**

```bash
uv run pytest python/tests/heuristics/test_physical_placement.py -v -k "rust_path_target_generator or rust_path_cz_counter"
```

Expected: 2 failures.

- [ ] **Step 4: Modify `_cz_placements_rust`**

Replace the method body (around lines 328–386) with the candidate loop:

```python
    def _cz_placements_rust(
        self,
        state: ConcreteState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        lookahead_cz_layers: tuple[
            tuple[tuple[int, ...], tuple[int, ...]], ...
        ] = (),
    ) -> AtomState:
        assert isinstance(self.traversal, RustPlacementTraversal)
        ctx = TargetContext(
            arch_spec=self.arch_spec,
            state=state,
            controls=controls,
            targets=targets,
            lookahead_cz_layers=lookahead_cz_layers,
            cz_stage_index=self._cz_counter,
        )
        candidates = self._build_candidates(ctx)

        initial = [
            (qid, loc.zone_id, loc.word_id, loc.site_id)
            for qid, loc in ctx.placement.items()
        ]
        blocked = [
            (loc.zone_id, loc.word_id, loc.site_id)
            for loc in state.occupied
        ]
        solver = self._get_rust_solver()

        remaining = self.traversal.max_expansions
        winning_result = None
        for candidate in candidates:
            if remaining is not None and remaining <= 0:
                break
            target_tuples = [
                (qid, loc.zone_id, loc.word_id, loc.site_id)
                for qid, loc in candidate.items()
            ]
            result = solver.solve(
                initial,
                target_tuples,
                blocked,
                max_expansions=remaining,
                strategy=self.traversal.strategy,
                top_c=self.traversal.top_c,
                max_movesets_per_group=self.traversal.max_movesets_per_group,
            )
            self._rust_nodes_expanded_total += int(result.nodes_expanded)
            if remaining is not None:
                remaining -= int(result.nodes_expanded)
            if result.status == "solved":
                winning_result = result
                break

        self._cz_counter += 1

        if winning_result is None:
            return AtomState.bottom()

        move_layers = tuple(
            tuple(
                LaneAddress(
                    self._MT_MAP[mt], word, site, bus,
                    self._DIR_MAP[d], zone,
                )
                for d, mt, zone, word, site, bus in step
            )
            for step in winning_result.move_layers
        )

        goal_map = {
            qid: LocationAddress(w, s, z)
            for qid, z, w, s in winning_result.goal_config
        }
        goal_layout = tuple(
            goal_map[qid] for qid in range(len(state.layout))
        )
        move_count = tuple(
            mc + int(src != dst)
            for mc, src, dst in zip(state.move_count, state.layout, goal_layout)
        )
        return ExecuteCZ(
            occupied=state.occupied,
            layout=goal_layout,
            move_count=move_count,
            active_cz_zones=self.arch_spec.cz_zone_addresses,
            move_layers=move_layers,
        )
```

Also update `cz_placements`'s Rust dispatch call to pass `lookahead_cz_layers` (should be done in Task 7; verify here).

- [ ] **Step 5: Run all tests + pyright**

```bash
uv run pytest python/tests/heuristics/test_physical_placement.py python/tests/heuristics/test_target_generator.py -v
uv run pyright python/bloqade/lanes/heuristics/physical/movement.py python/tests/heuristics/test_target_generator.py python/tests/heuristics/test_physical_placement.py
```

Expected: all previous + 2 new pass; pyright: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/movement.py python/tests/heuristics/test_physical_placement.py
git commit -m "feat(heuristics): thread target-generator through Rust path + fix _cz_counter parity"
```

---

## Chunk 4: Exports + full-suite verification

### Task 9: Re-export public symbols

**Files:**
- Modify: `python/bloqade/lanes/heuristics/physical/__init__.py`

- [ ] **Step 1: Update the subpackage `__init__.py`**

Edit `python/bloqade/lanes/heuristics/physical/__init__.py`:

```python
from bloqade.lanes.heuristics.physical.layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical.movement import (
    BFSPlacementTraversal,
    DefaultTargetGenerator,
    EntropyPlacementTraversal,
    GreedyPlacementTraversal,
    PhysicalPlacementStrategy,
    PlacementTraversalABC,
    RustPlacementTraversal,
    TargetContext,
    TargetGeneratorABC,
)

__all__ = [
    "BFSPlacementTraversal",
    "DefaultTargetGenerator",
    "EntropyPlacementTraversal",
    "GreedyPlacementTraversal",
    "PhysicalLayoutHeuristicGraphPartitionCenterOut",
    "PhysicalPlacementStrategy",
    "PlacementTraversalABC",
    "RustPlacementTraversal",
    "TargetContext",
    "TargetGeneratorABC",
]
```

- [ ] **Step 2: Sanity-check the new imports resolve**

```bash
uv run python -c "
from bloqade.lanes.heuristics.physical import (
    TargetContext,
    TargetGeneratorABC,
    DefaultTargetGenerator,
)
print('imports ok')
"
```

Expected: `imports ok`.

- [ ] **Step 3: Commit**

```bash
git add python/bloqade/lanes/heuristics/physical/__init__.py
git commit -m "feat(heuristics): re-export target-generator public API from physical subpackage"
```

---

### Task 10: Full-suite regression pass

**Files:** none modified — verification only.

- [ ] **Step 1: Run the full Python test suite**

```bash
uv run pytest python/tests -x -q
```

Expected: all tests pass. Any failure here is a regression introduced by the refactor and must be fixed before proceeding.

- [ ] **Step 2: Run all linters + type check**

```bash
uv run ruff check python
uv run black python --check
uv run isort python --check
uv run pyright python
```

Expected: all clean, 0 pyright errors.

- [ ] **Step 3: Push + open PR**

```bash
git push -u origin spec/target-generator-plugin
gh pr create --title "feat(heuristics): target-generator plugin for PhysicalPlacementStrategy" --body "$(cat <<'EOF'
Closes the design at \`docs/superpowers/specs/2026-04-17-target-generator-plugin-design.md\`.

## Summary
- Add \`TargetContext\`, \`TargetGeneratorABC\`, \`DefaultTargetGenerator\` plus a private callable adapter in \`physical/movement.py\`.
- Add optional \`target_generator\` field on \`PhysicalPlacementStrategy\`.
- Shared-budget candidate loop in both \`cz_placements\` (Python path) and \`_cz_placements_rust\`.
- Incidental parity fix: \`_cz_counter\` now increments on the Rust path.

## API surface
- **Python** — additive only. \`target_generator=None\` (default) preserves current behavior bit-for-bit.
- **Rust** — no changes.
- **C** — no changes.

## Test plan
- [x] \`uv run pytest python/tests\` — full suite green
- [x] \`uv run pyright python\` — 0 errors
- [x] New unit tests in \`test_target_generator.py\` exercise the ABC, default generator, validator, callable adapter, and \`_build_candidates\` dedup/append-default behavior.
- [x] New integration tests in \`test_physical_placement.py\` cover the \`None\` regression guard, empty-plugin passthrough, cheaper-candidate-wins, shared-budget invariant, plugin-raises, Rust path, and the \`_cz_counter\` parity fix.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Add `S-backport` + `backport v0.7` labels** (non-breaking change, per project convention)

```bash
PR_NUM=$(gh pr view --json number --jq .number)
gh api -X POST repos/QuEraComputing/bloqade-lanes/issues/$PR_NUM/labels -f "labels[]=S-backport" -f "labels[]=backport v0.7"
```

---

## Notes for the implementer

- **Do not batch tasks.** Commit after each task completes. This makes it trivial to bisect if a regression appears.
- **Preserve `_run_search` signature.** Existing tests monkeypatch it; adding a per-call `max_expansions` override via `replace(traversal, …)` keeps the signature unchanged.
- **`PhysicalPlacementStrategy` is not frozen.** Plain attribute assignment in `__post_init__` works; no `object.__setattr__` needed.
- **Tracing captures `candidates[0]`.** The first-attempted candidate is stored in `_traced_target`. Deeper per-candidate tracing is out of scope.
- **If a test in Task 7 or 8 accidentally breaks an existing test** (e.g. because `cz_placements`'s signature changed), fix the existing test first — the change should be additive (new optional `lookahead_cz_layers` kwarg) and should not require updates elsewhere.
