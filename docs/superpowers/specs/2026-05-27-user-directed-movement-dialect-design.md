# User-Directed Atom Movement Dialect — Design

**Status:** Draft
**Date:** 2026-05-27
**Tracking issue:** [#669](https://github.com/QuEraComputing/bloqade-lanes/issues/669)

## Goal

Let users express explicit intermediate atom-placement requests inside a Squin
kernel via a dedicated `movement` Kirin dialect. The dialect is **not** part of
the base Squin dialect group; users opt in by compiling with a composed
kernel decorator that unions Squin + `movement`.

User-directed moves are a **hard constraint**: if the requested placement is
not physically realizable (unreachable destination, occupancy conflict, AOD
lane violation), compilation fails with a diagnostic. A soft-hint variant
(`prefer_move_to`) is out of scope for v1.

## Non-goals

- **Adding movement statements to the base Squin dialect.** Squin remains
  circuit-level; physical movement intent stays in its own dialect.
- **Exposing low-level `move.Move` IR construction as the user API.** Users
  speak in terms of `(qubit, LocationAddress)` pairs; the move synthesizer
  emits AOD lanes.
- **A `prefer_move_to` soft-hint variant.** Follow-up once usage patterns are
  understood.
- **`persistent=True` flag.** v1 picks one palindrome semantic (see §3); a
  per-call override is a follow-up.
- **`require_parallel=True` flag.** v1 packs into AOD shots best-effort and
  emits a diagnostic when a `move_to` call spans multiple shots; forcing a
  single shot is a follow-up.
- **Modifications to the move synthesizer itself.** v1 reuses the existing
  Rust-backed `MoveSolver` unchanged. The only synthesis-level change is what
  layout it sees as input (post-user-move instead of pre-user-move).

## Package split

Mirrors the explicit-qubit-allocation spec ([2026-04-27](./2026-04-27-explicit-qubit-allocation-design.md)):

- **`bloqade.lanes`** — owns the IR statement (`movement.MoveTo`), the new
  placement-analysis lattice element, and the `PalindromePlacementStrategy`
  update. The placement analysis must recognize `MoveTo` directly, so the
  statement lives next to `place` and `move` to avoid a `lanes → gemini`
  import cycle.
- **`bloqade.gemini`** — owns the user-facing composed kernel decorator,
  eager validation, and any Squin-side glue. Already depends on
  `bloqade.lanes`.

## Architecture

```text
user kernel:        @movement_kernel
                    def k():
                        q = qubit.qalloc(4)
                        movement.move_to([q[0], q[1]], [loc_a, loc_b])
                        squin.cz(q[0], q[2])

  ┌─ eager validation (bloqade.gemini) ─────────────────┐
  │  • per-stmt method-table impl (lanes validator key): │
  │      - const-foldability of LocationAddress args     │
  │      - len(qubits) == len(locations)                 │
  │      - destinations in-range vs ArchSpec             │
  │      - no duplicate destinations within one call     │
  └──────────────────────────────────────────────────────┘
       ↓
existing circuit→place rewrites (movement.MoveTo passes through unchanged —
 it already lives at the place-compatible layer of the IR)
       ↓
PlacementAnalysis (extended):
   • new interp method for movement.MoveTo updates layout
   • emits a new UserMoved lattice element accumulating user-move layers
     since the last CZ
       ↓
PalindromePlacementStrategy (extended):
   • on ExecuteCZ, drains accumulated user_move_layers from UserMoved
   • appends compiler-synthesized pairing moves
   • produces ExecuteCZReturn whose move_layers covers both, and whose
     return_move_layers palindromes the full segment
       ↓
InsertMoves (place→move, mostly unchanged):
   • walks states; emits forward + return Move IR from the (now-bigger)
     move_layers / return_move_layers
       ↓
existing post-compile lanes validator (unchanged) catches semantic illegality
```

## §1 — IR statements & attributes

### `bloqade.lanes.dialects.movement.MoveTo` (new statement)

```python
@statement(dialect=movement)
class MoveTo(ir.Statement):
    name = "move_to"
    traits = frozenset({lowering.FromPythonCall()})
    qubits: tuple[ir.SSAValue, ...] = info.argument(type=QubitType)
    locations: tuple[ir.SSAValue, ...] = info.argument(type=LocationAddressType)
```

- `qubits` and `locations` are parallel SSA-value tuples. Arity is implicit
  in operand count and equal between the two; `len(qubits) == len(locations)`
  is enforced by eager validation (§4).
- The statement produces **no SSA result**. It is a placement directive, not
  a value-producing op. State threading happens at the move-dialect level
  (the placement analysis updates the layout in its lattice element).
- `QubitType` and `LocationAddressType` are reused from existing dialects
  (`bloqade.types.QubitType`, `bloqade.lanes.bytecode.encoding.LocationAddressType`).

Python lowering stub (consumed by `lowering.FromPythonCall`):

```python
@wraps(MoveTo)
def move_to(qubits: IList[Qubit, N], locations: IList[LocationAddress, N]) -> None:
    ...
```

### `bloqade.lanes.dialects.movement` (new dialect)

New Kirin dialect. v1 contains exactly one statement (`MoveTo`).

The dialect is intentionally **not** included in the default Squin or Lanes
prelude. Users opt in via the composed kernel decorator (§4).

## §2 — Placement-analysis extensions

The placement analysis (`python/bloqade/lanes/analysis/placement/`) is the
seam where user moves become first-class. Two changes:

### 2.1 — New lattice element `UserMoved`

In `python/bloqade/lanes/analysis/placement/lattice.py`, alongside
`ExecuteCZ` / `ExecuteCZReturn`:

```python
@dataclass
class UserMoved(ConcreteState):
    """A state representing the result of one or more user-directed
    `movement.MoveTo` statements, executed in sequence since the last CZ.

    `user_move_layers` accumulates the AOD lane layers for each user
    `move_to` call. Each call contributes one layer (best-effort packing
    into a single AOD shot) or multiple layers (if the requested moves
    cannot be parallelized — see §6).

    `pre_user_layout` records the atom layout *before* the first user move
    in this segment, so PalindromePlacementStrategy can compute the full
    palindrome back to the home position when the next CZ arrives.
    """

    user_move_layers: tuple[tuple[LaneAddress, ...], ...]
    """Sequence of AOD lane layers for user moves since the last CZ."""

    pre_user_layout: tuple[LocationAddress, ...]
    """Layout before the first user move in this segment."""

    def get_move_layers(self) -> tuple[tuple[LaneAddress, ...], ...]:
        return self.user_move_layers
```

- `UserMoved` extends `ConcreteState`. Its `layout` reflects positions
  **after** the user moves (so subsequent statements see updated positions).
- `get_move_layers()` returns the accumulated user-move layers so
  `InsertMoves` can emit forward Move IR for them.
- Joining two `UserMoved` states with different `user_move_layers` joins to
  `AnyState` (top) — divergent user-move histories are not meaningful.

### 2.2 — Interpreter method for `movement.MoveTo`

Register an abstract-interpretation method (method-table impl) for the
placement analysis on `movement.MoveTo`. The handler:

1. Reads the const-folded `LocationAddress` values for the `locations`
   operands. (Const-foldability is enforced by eager validation in §4.)
2. Reads the qubit-id-to-current-location mapping from the input state.
3. Calls the move synthesizer to compute the AOD lane layer(s) that
   transport each qubit from its current location to its requested
   location. (Best-effort packing — see §6.)
4. Produces an output `UserMoved` state whose:
   - `layout` is updated with the new positions for the moved qubits
   - `user_move_layers` is the input state's layers (if input was already
     `UserMoved`, otherwise empty) extended with the new layer(s)
   - `pre_user_layout` is copied from input (if `UserMoved`) or set to
     the input state's `layout` (if any other `ConcreteState`)
5. If synthesis fails (no legal AOD lane assignment), the handler returns
   `AnyState` (top) and emits a validation diagnostic. Compilation halts at
   the validation stage; the analysis does not raise from inside the
   interpreter (matches the existing convention).

### 2.3 — Interaction with existing CZ analysis

When the analysis reaches a `place.CZ` statement and the input state is
`UserMoved`:

- The inner strategy (`PhysicalPlacementStrategy` or any
  `PlacementStrategyABC`) is invoked as usual, using the current
  `UserMoved.layout` as the starting layout.
- If the current layout already satisfies the CZ pairing (i.e.
  `ExecuteCZ.verify()` returns True with no further moves), no compiler
  moves are needed; the output state's `move_layers` is exactly
  `UserMoved.user_move_layers`.
- Otherwise the inner strategy synthesizes additional pairing moves and
  appends them after the user-move layers.

When the input state is any other `ConcreteState` (no pending user moves),
the existing behavior is preserved bit-for-bit.

## §3 — Palindrome interaction

### 3.1 — Stance

User-directed moves participate in the palindrome whenever the
`PalindromePlacementStrategy` is active. This means:

- **Without palindrome (default placement strategy)**: user moves persist.
  Atoms end the CZ at the position resulting from `user_moves +
  compiler_pairing_moves`, with no return moves.
- **With palindrome**: the strategy palindromes the **full inter-CZ
  segment** including user moves. Atoms return to `pre_user_layout` after
  the CZ pulse.

This keeps the palindrome strategy's contract intact ("undo the inter-CZ
segment") while making user moves first-class participants. A per-call
`persistent=True` override is a v2 follow-up.

### 3.2 — `PalindromePlacementStrategy` changes

In `python/bloqade/lanes/analysis/placement/strategy.py`,
`PalindromePlacementStrategy.handle_cz` (or equivalent — see existing impl
at lines 150-214) gains awareness of `UserMoved`:

- If the input state to a CZ is `UserMoved`:
  - Run the inner strategy on `UserMoved.layout` (the post-user-move
    position) to obtain compiler-synthesized pairing moves.
  - Build `ExecuteCZReturn` with:
    - `move_layers = user_move_layers + compiler_pairing_layers`
    - `initial_layout = UserMoved.pre_user_layout`
    - `return_move_layers` is auto-computed in `ExecuteCZReturn.__post_init__`
      from the combined `move_layers` (palindrome of the whole segment).
- If the input state is not `UserMoved`, existing behavior is preserved
  bit-for-bit.

### 3.3 — `ExecuteCZReturn` — no shape changes

The existing `ExecuteCZReturn` class already carries `move_layers`,
`initial_layout`, and computes `return_move_layers`. The only behavioral
change is that, when `PalindromePlacementStrategy` produces it from a
`UserMoved` input, `move_layers` is **longer** than before (covers both
user and compiler segments) and `initial_layout` reaches further back.

No new fields, no new methods, no new emission machinery in
`InsertMoves` — it already consumes `get_move_layers()` and
`get_reverse_moves()` uniformly.

## §4 — Validation

Failure modes surface eagerly at the gemini level (matching the
`new_at` pattern). All eager errors are Kirin diagnostics pointing at the
user's `movement.move_to(...)` source line.

| # | Failure | Where | When |
|---|---|---|---|
| 1 | `len(qubits) != len(locations)` | gemini per-stmt method-table impl (registered against lanes validator key) | eager |
| 2 | `LocationAddress` SSA arg isn't const-foldable | gemini per-stmt method-table impl | eager |
| 3 | Destination out of ArchSpec range | gemini per-stmt method-table impl | eager |
| 4 | Duplicate destinations within one `move_to` call | gemini per-stmt method-table impl | eager |
| 5 | Duplicate destination vs an unmoved occupied location | placement analysis (raises diagnostic) | mid-compile |
| 6 | Destination unreachable under AOD constraints | placement analysis (raises diagnostic) | mid-compile |
| 7 | Qubit already at requested location | no-op (silent) | n/a |

### 4.1 — Eager per-statement validation (failures 1-4)

A method-table impl for `movement.MoveTo` registered against the existing
lanes validation interpreter key (using the same pattern as
`bloqade.gemini.common.validation.new_at`). The impl runs against the
gemini IR before any lowering to place.

### 4.2 — Mid-compile validation (failures 5-6)

These are caught by the placement-analysis interpreter method for
`movement.MoveTo` (§2.2). When the move synthesizer reports infeasibility,
the analysis produces `AnyState` and queues a diagnostic via the existing
`ValidationSuite` plumbing. Compilation halts before move-IR emission.

### 4.3 — Already-at-destination (failure 7)

If a qubit's current location equals its requested destination, the
synthesizer simply emits no AOD lane for it. The analysis records the
no-op in `user_move_layers` as an empty layer (or merges it into the next
non-empty layer — implementation detail). User-visible behavior: identical
to omitting that pair from the call.

## §5 — Composed kernel decorator

In `python/bloqade/gemini/logical/movement.py` (sibling to `group.py`),
define a new composed kernel decorator that unions Squin with the
movement dialect:

```python
from bloqade.lanes.dialects import movement as movement_dialect
# ... other imports matching gemini/logical/group.py ...

@ir.dialect_group(
    structural_no_opt.union(
        [gate, qubit, operations, annotate,
         gemini_common.dialects.qubit, movement_dialect]
    )
)
def kernel(self):
    """Compile a function to a Gemini logical kernel with user-directed
    atom-movement support."""
    # run_pass body matches gemini/logical/group.py with the addition of
    # eager movement validation (§4.1)
    ...
```

User import path:

```python
from bloqade.gemini.logical.movement import kernel as movement_kernel

@movement_kernel
def k():
    q = qubit.qalloc(4)
    movement.move_to([q[0], q[1]], [loc_a, loc_b])
    squin.cz(q[0], q[2])
```

The exact final name (`movement_kernel`, `physical_squin`, etc.) is
flagged as an open question; the spec assumes `movement_kernel` for
concreteness.

The plain `gemini.logical.kernel` decorator (from `group.py`) is unchanged
— users who don't import `movement` get the existing surface bit-for-bit.

## §6 — Place→Move emission

`InsertMoves` (`python/bloqade/lanes/rewrite/place2move.py:14-61`) is
**not changed**. It already iterates over the analysis result and emits
forward + return Move IR via `state_after.get_move_layers()` and
`state.get_reverse_moves()`. Because:

- `UserMoved.get_move_layers()` returns the accumulated user-move layers,
- `ExecuteCZReturn` (built from a `UserMoved` input) carries the combined
  forward layers and the full-segment palindrome,

…the existing emission code emits correct Move IR with no modifications.
The only edit needed is wherever `InsertMoves` matches on the lattice
class set — `UserMoved` must be included in the recognized-state set.

### 6.1 — AOD packing for a single `move_to` call

The placement-analysis interpreter method (§2.2) calls
`bloqade.lanes.heuristics.move_synthesis.compute_move_layers` to produce
the AOD lane layer(s) for the user's requested transport. The synthesizer
packs compatible moves into the same layer; incompatible moves split
across multiple layers.

When a single `move_to` call results in **more than one** AOD layer, the
analysis emits a non-fatal diagnostic indicating that the requested
transport cannot be performed in a single AOD shot. This is informational
only — compilation continues. A `require_parallel=True` flag (v2) would
turn this into a hard error.

### 6.2 — Inter-call coalescing

Two adjacent `movement.MoveTo` statements with no intervening
non-movement op are **not** coalesced into one AOD shot in v1, even if
their requested transports are mutually compatible. Each call produces
its own layer(s). Coalescing is a future optimization.

## §7 — Testing strategy

### 7.1 — Eager validation (§4.1)

For each failure mode, write a kernel that exhibits it and assert the
diagnostic surfaces at the right source line:

- Mismatched `qubits` / `locations` lengths → length error.
- Non-const `LocationAddress` arg → const-foldability error.
- Out-of-range zone/word/site → range error.
- Two identical destinations in one call → duplicate-destination error,
  points at the duplicate and references the first.
- Negative cases: valid single-qubit move, valid multi-qubit move,
  already-at-destination no-op.

### 7.2 — Placement-analysis interpreter (§2.2)

- Single `move_to` → `UserMoved` lattice element with one layer and
  updated `layout`.
- Sequence of two `move_to` calls (no CZ between) → `UserMoved` with
  two layers, `pre_user_layout` is the layout before the first call,
  final `layout` reflects both calls.
- `move_to` followed by a non-CZ op (e.g. `R`) → `UserMoved` propagated
  unchanged through the non-movement op.
- Infeasible `move_to` (occupied destination, unreachable site) →
  `AnyState` + diagnostic.

### 7.3 — Palindrome interaction (§3)

- `move_to` + CZ under default placement strategy → `ExecuteCZ` whose
  `move_layers` includes the user moves; no return moves emitted.
- `move_to` + CZ under `PalindromePlacementStrategy` → `ExecuteCZReturn`
  whose `move_layers` covers user + compiler segments and whose
  `return_move_layers` palindromes the full segment.
- `move_to` placing qubits into CZ-compatible positions → compiler
  synthesizes **zero** additional pairing moves; under palindrome the
  return reverses only the user moves.
- Sequence: `move_to` + CZ + `move_to` + CZ — verifies the lattice resets
  between CZs (the second `UserMoved` does not see the first
  segment's history).

### 7.4 — Place→Move emission (§6)

- Each scenario from §7.3 lowered through `InsertMoves` → resulting
  `move.Move` statements in the expected order with correct AOD lanes.
- Single-call multi-layer split → multiple `move.Move` statements emitted
  for one `MoveTo` source statement.

### 7.5 — Composed kernel decorator (§5)

- Compile a kernel that uses both `squin.*` and `movement.move_to` →
  succeeds; resulting move IR contains both circuit-derived and
  user-directed transport.
- Compile a kernel that uses `movement.move_to` under the plain
  `gemini.logical.kernel` decorator → fails at dialect-group validation
  (movement is not in the group).

### 7.6 — Regression

- Existing demo kernels under `demo/` (zero `movement.move_to` usage)
  compile to byte-identical move IR before vs after this change. This is
  the primary canary that the analysis/strategy changes preserve
  semantics for non-movement kernels.

## §8 — File layout summary

```
python/bloqade/lanes/
├── dialects/
│   └── movement.py                       # NEW — dialect + MoveTo statement
├── analysis/placement/
│   ├── lattice.py                        # EDIT — add UserMoved lattice element
│   ├── strategy.py                       # EDIT — PalindromePlacementStrategy
│   │                                     #        handles UserMoved input
│   └── impl.py                           # EDIT — interp method for movement.MoveTo
└── rewrite/place2move.py                 # EDIT (minimal) — recognize UserMoved
                                          #        in lattice-class match set

python/bloqade/gemini/
├── logical/movement.py                   # NEW — composed kernel decorator
│                                         #        (movement_kernel)
└── common/validation/move_to.py          # NEW — eager validation impl
```

## Open questions

- **Decorator name.** `movement_kernel` (assumed in this spec) vs
  `physical_squin` vs simply re-exporting from `bloqade.lanes` as
  `bloqade.lanes.movement.kernel`. Final choice should match whatever
  convention emerges for other Squin-extending kernel groups.
- **Layer granularity in `UserMoved`.** Each `move_to` call contributes
  one or more layers. Two adjacent calls produce two separate layer
  groups even when their transports could pack into one AOD shot. v1
  defers coalescing; the open question is whether the lattice element
  should pre-emptively flatten to a single tuple or preserve per-call
  grouping (e.g. for diagnostics that map back to the source call).
- **Diagnostic richness for multi-layer splits (§6.1).** Should the
  diagnostic name the specific qubit/destination pairs that forced the
  split, or just report "your call spans N AOD shots"? Affects how much
  state the synthesizer surfaces back.
- **`movement.MoveTo` interaction with `place.NewLogicalQubit`.** Can a
  user `move_to` a qubit that was placed via explicit `new_at(z, w, s)`
  on the same kernel? The semantics are well-defined (the qubit moves
  from its pinned position to the new destination), but worth a regression
  test ensuring the explicit-allocation and movement-intent paths compose
  cleanly.

## Risk register

- **Highest risk:** the new `UserMoved` lattice element joining
  semantics. If two control-flow paths produce different
  `user_move_layers`, the join goes to `AnyState`, halting analysis.
  v1 only supports straight-line kernels (matching today's Squin), so
  this is theoretical — but the spec should call it out so future
  branching support knows it has work to do here.
- **Medium risk:** `PalindromePlacementStrategy` regression. The
  change is narrow (handle a new input state type) but the strategy is
  load-bearing for every palindrome-using kernel today. Byte-identical
  regression on the existing demo corpus is the gate.
- **Low risk:** `InsertMoves` accidentally double-emitting user-move
  layers (once as part of `UserMoved.get_move_layers()`, once as part of
  `ExecuteCZReturn.get_move_layers()` after the strategy folds them in).
  Mitigation: `UserMoved` is consumed by the strategy at the CZ
  boundary; the strategy's output replaces it. As long as the analysis
  doesn't emit both states for the same program point this is fine —
  but worth an explicit assertion in tests.
