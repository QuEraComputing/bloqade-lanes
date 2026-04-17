# Target-generator plugin for `PhysicalPlacementStrategy`

Status: Design — pending review
Date: 2026-04-17
Module: `python/bloqade/lanes/heuristics/physical/movement.py`

## Summary

`PhysicalPlacementStrategy` currently hardcodes a single rule for choosing
where atoms should move before each CZ layer: the control qubit goes to
the CZ-blockade partner of the target qubit's current location. This
spec introduces a plugin interface that lets callers supply an
alternative (or augmenting) target-selection heuristic while preserving
the current behavior as a guaranteed fallback.

## Motivation

The existing rule in `_target_from_stage_controls_only` is one valid
policy among many. Other heuristics we may want to try include:

- Moving the *target* to the control's CZ partner instead of moving the control.
- Per-pair optimization so each atom travels less.
- Lookahead-aware placement that picks partners favorable for *future* CZ layers.
- Fully custom heuristics that may reassign qubits other than the CZ pair.

These cannot be expressed by tuning the traversal (search) plugin
because the search is given a target to reach; the heuristic for *which*
target is a separate, upstream concern.

The Python search path is being retired in favor of the Rust solver, so
the interface must be neutral to which traversal executes the search.

## Non-goals

- Replacing the `traversal` plugin or changing search semantics.
- Adding new Rust FFI surface. The plugin is Python-level and the
  existing `MoveSolver.solve(target, max_expansions=...)` API is
  sufficient.
- Shipping specific new heuristics. Only the interface and the default
  (current-behavior) implementation are in scope here.
- Performance benchmarking of new heuristics.

## Interface

All new public symbols colocate with `PlacementTraversalABC` in
`python/bloqade/lanes/heuristics/physical/movement.py`.

### `TargetContext`

A frozen dataclass carrying all signals a heuristic might need. Composes
`ConcreteState` rather than duplicating its fields, so future additions
to the lattice state flow through automatically.

```python
@dataclass(frozen=True)
class TargetContext:
    arch_spec: layout.ArchSpec
    state: ConcreteState
    controls: tuple[int, ...]
    targets: tuple[int, ...]
    lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    cz_stage_index: int

    @property
    def placement(self) -> dict[int, LocationAddress]:
        return dict(enumerate(self.state.layout))
```

`controls` and `targets` are the CZ qubit pairs for the current stage
(parallel indexing: `controls[i]` is the control for `targets[i]`).
`cz_stage_index` is a per-strategy counter equal to `_cz_counter` at
call time.

### `TargetGeneratorABC`

```python
class TargetGeneratorABC(abc.ABC):
    @abc.abstractmethod
    def generate(
        self, ctx: TargetContext
    ) -> list[dict[int, LocationAddress]]: ...
```

Returns an **ordered list** of candidate target placements. Earlier
candidates are tried first; the framework appends the default as the
last candidate automatically.

### `DefaultTargetGenerator`

A concrete `TargetGeneratorABC` that encapsulates today's
`_target_from_stage_controls_only` behavior. Exposed publicly so plugin
authors can compose or extend it.

```python
@dataclass(frozen=True)
class DefaultTargetGenerator(TargetGeneratorABC):
    def generate(self, ctx: TargetContext) -> list[dict[int, LocationAddress]]:
        # control -> CZ partner of target's current location; targets stay put
        ...
```

### Callable form

A plain callable with the signature
`Callable[[TargetContext], list[dict[int, LocationAddress]]]` is also
accepted. The strategy's `__post_init__` wraps it in a private
`_CallableTargetGenerator` adapter so the internal loop always sees a
`TargetGeneratorABC`. pyright sees the union on the public field; the
adapter narrows it at construction time.

### Strategy field

```python
@dataclass
class PhysicalPlacementStrategy(PlacementStrategyABC):
    ...
    target_generator: (
        TargetGeneratorABC
        | Callable[[TargetContext], list[dict[int, LocationAddress]]]
        | None
    ) = None
```

- `None` — the strategy behaves byte-for-byte identically to today
  (single search, no plugin overhead).
- Any other value — plugin path: plugin produces candidates, default is
  appended, shared-budget loop tries each in order.

## Orchestration

Both `cz_placements` (Python traversal) and `_cz_placements_rust` are
updated to the same pattern:

1. If `target_generator is None`: compute the default target once, run
   the existing single-search path. No behavior change.
2. Otherwise:
   - Build `TargetContext` from the current state and stage info.
   - `candidates = list(plugin.generate(ctx))` (or callable-adapter).
   - Validate each candidate (see "Validation" below). Invalid →
     `ValueError`.
   - Dedup candidates by dict equality.
   - Append `DefaultTargetGenerator().generate(ctx)` output if not
     already present.
   - Loop with shared budget.

### Shared-budget loop

```python
remaining = configured_max_expansions   # from self.traversal
for candidate in candidates:
    if remaining is not None and remaining <= 0:
        break
    result = run_one(candidate, max_expansions=remaining)
    if remaining is not None:
        remaining -= int(result.nodes_expanded)
    if succeeded(result):
        return build_execute_cz(...)
return AtomState.bottom()
```

- **Python path** — `run_one` does
  `replace(self.traversal, max_expansions=remaining).path_to_target_config(tree, candidate)`
  (frozen dataclass `replace` produces a per-iteration copy).
- **Rust path** — `run_one` calls
  `solver.solve(..., target=candidate, max_expansions=remaining)`.
  `_rust_nodes_expanded_total` accumulates across candidates.
- `None` budget (unlimited) passes through untouched; no decrementing.

### Data flow summary

```
cz_placements(state, controls, targets, lookahead)
  → narrow to ConcreteState
  → (if plugin) build TargetContext
  → candidates = plugin.generate(ctx) + [default]
  → validate + dedup
  → loop with shared budget:
       for c in candidates:
           result = run_search(c, remaining)
           if solved: return ExecuteCZ(...)
           remaining -= result.nodes_expanded
  → return AtomState.bottom() if none solved
```

## Validation

Each candidate is validated before the search runs. All failures raise
`ValueError`:

1. **Missing qid**: every key in the current `placement` must also appear
   in the candidate.
2. **Unknown location**: every value must be a `LocationAddress` that the
   `arch_spec` recognizes (via existing `arch_spec` validation).
3. **Invalid CZ pair**: for every `(control_qid, target_qid)` in
   `zip(controls, targets)`,
   `arch_spec.get_cz_partner(candidate[target_qid]) == candidate[control_qid]`
   must hold. A candidate whose pair is not CZ-blockade-partnered cannot
   execute the CZ and indicates a plugin bug.

Rationale: malformed candidates indicate a plugin contract violation.
Silent skipping would hide bugs. The "default as fallback" mechanism is
for *search* failure, not for input malformedness.

## Error handling

| Case | Behavior |
|---|---|
| Plugin returns `[]` | Default is still appended; behavior identical to `target_generator=None`. |
| Plugin returns duplicate of default | Deduped; default runs once. |
| Plugin returns malformed candidate | `ValueError` with the offending candidate + reason. |
| Budget exhausted before default | Returns `AtomState.bottom()`. The user configured the budget. |
| `max_expansions=None` | All candidates run unbounded. |
| Plugin raises | Exception propagates. |
| Candidate equals current placement | Search returns immediately with zero expansions; normal. |

## Type safety

The design is pyright-clean:

- Public `target_generator` field is a `TargetGeneratorABC | Callable[...] | None`
  union; `__post_init__` narrows to `TargetGeneratorABC | None` internally.
- `TargetContext.state: ConcreteState` (subclass-specific), not
  `AtomState`. Callers already narrow via `isinstance` before building
  the context.
- `run_one` has two implementations, one per traversal path; each lives
  behind a structural `isinstance(self.traversal, ...)` check so the
  result type is determined.
- `_rust_nodes_expanded_total: int` accumulation uses `int(...)` cast at
  point of incrementing (same pattern as today).

## Testing

### Unit (new `python/tests/heuristics/test_target_generator.py`)

- `DefaultTargetGenerator.generate` reproduces current
  `_target_from_stage_controls_only` output on representative inputs.
- `TargetContext.placement` derives correctly from `state.layout`.
- Callable-wrapping: a plain function returns the same candidates as an
  ABC subclass with equivalent logic.
- Validation raises on each of: missing qid, unknown location, CZ pair
  not blockade-partnered.

### Integration (extend `python/tests/heuristics/test_physical_placement.py`)

- `target_generator=None` produces byte-for-byte identical output to
  today (regression guard across all existing traversals).
- Plugin returning `[]` behaves identically to `None`.
- Plugin returning a cheaper candidate → search finds it before trying
  the default; fewer total expansions than default-only.
- Plugin returning multiple candidates with shared budget →
  `sum(nodes_expanded) ≤ configured max_expansions` (budget-respect
  invariant).
- Duplicate dedup: plugin returns the default verbatim → only one search
  runs.
- Rust path: plugin applies identically and `_rust_nodes_expanded_total`
  accumulates across candidates.
- Plugin raising → exception propagates.

### Out of scope

- Specific new heuristics (each ships with its own tests when added).
- Performance/benchmark tests (separate benchmarks suite already
  exercises `PhysicalPlacementStrategy`).

## Migration & compatibility

- Strictly additive. The default `target_generator=None` path retains
  current behavior bit-for-bit, so no existing tests need to change
  beyond the regression guards listed above.
- No Rust API changes.
- No change to `PlacementTraversalABC`, `RustPlacementTraversal`, or the
  public `PhysicalPlacementStrategy` construction surface except the new
  optional field.
- `DefaultTargetGenerator` is exposed; future PRs can deprecate
  `_target_from_stage_controls_only` once callers migrate.
