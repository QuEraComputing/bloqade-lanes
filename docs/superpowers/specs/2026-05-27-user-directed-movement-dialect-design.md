# User-Directed Atom Movement Dialect — Design

**Status:** Draft
**Date:** 2026-05-27
**Tracking issue:** [#669](https://github.com/QuEraComputing/bloqade-lanes/issues/669)

## Goal

Let users express explicit intermediate atom-placement requests inside a
Gemini-logical kernel via a dedicated `movement` Kirin dialect. The dialect
lives in `bloqade.lanes` (alongside `place` and `move`) and is unioned into
the existing `gemini.logical.kernel` decorator's dialect group — Gemini
logical kernels gain `movement.move_to(...)` automatically without opting
into a separate decorator. Bare `bloqade.lanes` preludes are unchanged.

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
  emits a `UserWarning` via `warnings.warn` when a `move_to` call spans
  multiple shots (see §6.1); forcing a single shot — by short-circuiting
  to `bottom` and routing through the post-compile validator — is a
  follow-up.
- **Modifications to the move synthesizer itself.** v1 reuses the existing
  Rust-backed `MoveSolver` unchanged. The only synthesis-level change is what
  layout it sees as input (post-user-move instead of pre-user-move).

## Package split

Mirrors the explicit-qubit-allocation spec ([2026-04-27](./2026-04-27-explicit-qubit-allocation-design.md)):

- **`bloqade.lanes`** — owns the IR statements (`movement.MoveTo` and
  `place.MoveTo`), the new placement-analysis lattice element
  (`UserMoved`), the new strategy method
  (`PlacementStrategyABC.move_to_placements`), the
  `PalindromePlacementStrategy.cz_placements` splice, and the
  `ExecuteCZReturn` shape extension. The placement analysis must
  recognize `place.MoveTo` directly, so both statements live next to
  `place` and `move` to avoid a `lanes → gemini` import cycle.
- **`bloqade.gemini`** — owns the eager validation pass
  (`MoveToValidation`), and extends the existing `gemini.logical.kernel`
  decorator (`bloqade/gemini/logical/group.py`) to include the movement
  dialect in its union and the new validator in its `ValidationSuite`.
  Already depends on `bloqade.lanes`.

## Architecture

```text
user kernel:        @gemini.logical.kernel
                    def k():
                        q = qubit.qalloc(4)
                        movement.move_to([q[0], q[1]], [loc_a, loc_b])
                        squin.cz(q[0], q[2])

  ┌─ eager validation (MoveToValidation, bloqade.gemini) ─┐
  │  • per-stmt method-table impl (lanes validator key):  │
  │      - len(qubits) == len(locations)                  │
  │      - locations IList const-foldable                 │
  │      - destinations in-range vs ArchSpec              │
  │      - no duplicate destinations within one call      │
  │      - no duplicate qubit SSA values within one call  │
  └───────────────────────────────────────────────────────┘
       ↓
circuit→place rewrites + new RewritePlaceOperations.rewrite_MoveTo:
  movement.MoveTo (IList SSA operands) → place.MoveTo (QuantumStmt;
  qubits=int indices, locations=const-folded LocationAddress tuple)
       ↓
PlacementAnalysis (extended):
   • new interp method for place.MoveTo delegates to strategy
   • strategy.move_to_placements returns a UserMoved lattice element
     carrying:
       - move_layers          (this MoveTo's AOD layers, for InsertMoves)
       - accumulated_move_layers (full inter-CZ history, for palindrome)
       - pre_user_layout      (home position before the segment started)
   • analysis failures (occupancy conflict, AOD infeasibility) → bottom
       ↓
PalindromePlacementStrategy.cz_placements (extended):
   • on UserMoved input: inner.cz_placements computes compiler pairing
     from UserMoved.layout (post-user-move position)
   • wraps into ExecuteCZReturn(
       move_layers = compiler_pairing,                ← forward at CZ
       user_move_layers = UserMoved.accumulated_move_layers,
       initial_layout = UserMoved.pre_user_layout)
   • ExecuteCZReturn.__post_init__ computes
       return_move_layers = compiler_reverse + user_reverse
       ↓
InsertMoves (place→move, unchanged):
   • polymorphic on the lattice element: emits forward from
     state.get_move_layers() and reverse from state.get_reverse_moves()
   • at place.MoveTo:  UserMoved.move_layers   → forward Move IR
   • at place.CZ:      ExecuteCZ(Return).move_layers → forward,
                        ExecuteCZReturn.return_move_layers → reverse
       ↓
existing post-compile lanes validator (unchanged) catches semantic
illegality (bottom states → user-facing diagnostic)
```

## §1 — IR statements & attributes

The dialect introduces two statements at two layers of the IR — the
user-facing `movement.MoveTo` and the lower-level `place.MoveTo` — connected
by a single rewrite. This matches the existing pattern for gate ops
(`squin.gate.CZ` → `place.CZ`, etc. — see `circuit2place.py:265-291`).

### `bloqade.lanes.dialects.movement.MoveTo` (new, user-facing)

```python
@statement(dialect=movement)
class MoveTo(ir.Statement):
    name = "move_to"
    traits = frozenset({lowering.FromPythonCall()})
    qubits: ir.SSAValue = info.argument(
        type=ilist.IListType[bloqade_types.QubitType, types.Any]
    )
    locations: ir.SSAValue = info.argument(
        type=ilist.IListType[LocationAddressType, types.Any]
    )
```

- Two single SSA operands, each holding an `IList`. Matches the user-facing
  Python API verbatim (`move_to(qubits: IList[Qubit, N], locations: IList[LocationAddress, N])`).
- No SSA result. The statement is a placement directive consumed by the
  movement→place rewrite (below) and the placement analysis (§2).
- Length equality (`len(qubits) == len(locations)`) and const-foldability of
  the `locations` IList are eager-validation concerns (§4), not statement
  shape concerns.

Python lowering stub (consumed by `lowering.FromPythonCall`):

```python
@wraps(MoveTo)
def move_to(qubits: IList[Qubit, N], locations: IList[LocationAddress, N]) -> None:
    ...
```

### `bloqade.lanes.dialects.movement` (new dialect)

New Kirin dialect. v1 contains exactly one statement (`MoveTo`).

The dialect is intentionally **not** included in the default Squin or Lanes
prelude. Users opt in via the composed kernel decorator (§5).

### `bloqade.lanes.dialects.place.MoveTo` (new, place-level)

```python
@statement(dialect=dialect)  # the existing place dialect
class MoveTo(QuantumStmt):
    qubits: tuple[int, ...] = info.attribute()
    locations: tuple[LocationAddress, ...] = info.attribute()
```

- Inherits `QuantumStmt` (`place.py:85-99`) — threads `state_before: ir.SSAValue`
  / `state_after: ir.ResultValue` of `StateType`, matching the existing
  pattern for `place.CZ`, `place.R`, `place.Rz`.
- `qubits` is a tuple of integer indices into the surrounding
  `place.StaticPlacement` block's qubit SSA list — the wrapping
  `StaticPlacement` carries the actual SSA `Qubit` values.
- `locations` is a tuple of const `LocationAddress` attributes (one per
  qubit index), populated by the rewrite below from the const-prop hint on
  `movement.MoveTo.locations`.
- Invariant: `len(qubits) == len(locations)`. Established by the rewrite,
  asserted by the statement constructor.

### Rewrite — `movement.MoveTo` → `place.MoveTo`

Lives on `RewritePlaceOperations` (`python/bloqade/lanes/rewrite/circuit2place.py`),
alongside the existing per-gate rewrite methods. Mirrors `rewrite_CZ`
(lines 265-291) structurally:

```python
def rewrite_MoveTo(self, node: movement.MoveTo) -> abc.RewriteResult:
    # qubits side: chase ilist.New to recover the SSA Qubit values that
    # the StaticPlacement wrapper needs to thread.
    if not isinstance(qubits_list := node.qubits.owner, ilist.New):
        return abc.RewriteResult()

    # locations side: read the const-prop hint. Eager validation (§4)
    # requires every LocationAddress to be const-foldable, so the hint
    # resolves to a const.Value whose .data is an IList of LocationAddress
    # (IList implements the Sequence interface, so tuple(...) extracts).
    locations_hint = node.locations.hints.get("const")
    if not isinstance(locations_hint, const.Value):
        return abc.RewriteResult()

    inputs = qubits_list.values
    location_attrs = tuple(locations_hint.data)

    body, block, entry_state = self.prep_region()
    move_stmt = place.MoveTo(
        entry_state,
        qubits=tuple(range(len(inputs))),
        locations=location_attrs,
    )
    node.replace_by(
        self.construct_execute(move_stmt, qubits=inputs, body=body, block=block)
    )
    return abc.RewriteResult(has_done_something=True)
```

The rewrite gives up silently (returns `RewriteResult()` with no
modification) when:
- The `qubits` operand isn't produced by an `ilist.New` (the user passed in
  a runtime-computed IList rather than a literal).
- The `locations` const-prop hint isn't a `const.Value` (the locations IList
  wasn't const-folded — likely a validation gap or a kernel that hasn't been
  unrolled yet).

In both cases the validation pass (§4) will surface a diagnostic; the
rewrite stays defensive rather than raising.

## §2 — Placement-analysis extensions

The placement analysis (`python/bloqade/lanes/analysis/placement/`) is the
seam where user moves become first-class. Three changes, in three files:
the new lattice element, a new method on the placement strategy ABC
(implemented by each concrete strategy), and an interpreter method on the
analysis that delegates to the strategy.

### 2.1 — New lattice element `UserMoved`

In `python/bloqade/lanes/analysis/placement/lattice.py`, alongside
`ExecuteCZ` / `ExecuteCZReturn`:

```python
@final
@dataclass
class UserMoved(ConcreteState):
    """A state representing the result of one or more user-directed
    `place.MoveTo` statements, executed in sequence since the last CZ.

    Carries two layer tuples for two distinct consumers:

    - `move_layers` is the AOD lane layer(s) for *this* MoveTo only.
      ``InsertMoves`` reads this via ``get_move_layers()`` and emits
      the forward Move IR immediately before the corresponding
      ``place.MoveTo`` statement. Each MoveTo is responsible for its
      own forward emission.

    - `accumulated_move_layers` is every user-move layer in the
      inter-CZ segment so far (i.e. ``prev.accumulated_move_layers +
      move_layers`` if the previous state was already ``UserMoved``,
      else just ``move_layers``). ``PalindromePlacementStrategy``
      reads this at the next CZ to build the return-side palindrome
      that undoes the full user-move history.

    `pre_user_layout` records the atom layout *before* the first user
    move in this segment — the home position
    ``PalindromePlacementStrategy`` returns to after the CZ pulse.
    """

    move_layers: tuple[tuple[LaneAddress, ...], ...] = field(kw_only=True)
    """AOD lane layers for *this* MoveTo only (emitted at the MoveTo site)."""

    accumulated_move_layers: tuple[tuple[LaneAddress, ...], ...] = field(kw_only=True)
    """All user-move layers in the inter-CZ segment so far (for palindrome)."""

    pre_user_layout: tuple[LocationAddress, ...] = field(kw_only=True)
    """Layout before the first user move in this segment."""

    def get_move_layers(self) -> tuple[tuple[LaneAddress, ...], ...]:
        return self.move_layers

    def is_subseteq(self, other: AtomState) -> bool:
        return (
            super().is_subseteq(other)
            and isinstance(other, UserMoved)
            and self.move_layers == other.move_layers
            and self.accumulated_move_layers == other.accumulated_move_layers
            and self.pre_user_layout == other.pre_user_layout
        )

    @classmethod
    def from_concrete_state(
        cls,
        state: ConcreteState,
        move_layers: tuple[tuple[LaneAddress, ...], ...],
        accumulated_move_layers: tuple[tuple[LaneAddress, ...], ...],
        pre_user_layout: tuple[LocationAddress, ...],
    ) -> "UserMoved":
        return cls(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            move_layers=move_layers,
            accumulated_move_layers=accumulated_move_layers,
            pre_user_layout=pre_user_layout,
        )
```

- Extends `ConcreteState`; inherits `occupied`, `layout`, `move_count`. The
  inherited `layout` reflects positions **after** the user moves (so
  subsequent statements see updated positions).
- `get_move_layers()` returns `move_layers` (just this MoveTo's portion)
  so `InsertMoves` emits at the MoveTo site without double-emitting on
  subsequent statements.
- `is_subseteq` matches the precision used by `ExecuteCZ` / `ExecuteCZReturn`.
- `from_concrete_state` mirrors the lifting helpers on `ExecuteCZ` and
  `ExecuteMeasure` so the analysis can promote a `ConcreteState` into
  `UserMoved` when it sees the first `place.MoveTo` in a segment.
- **Join semantics**: divergent `move_layers`, `accumulated_move_layers`,
  or `pre_user_layout` across control-flow paths join to `AnyState` (top)
  via the inherited `SimpleJoinMixin`. v1 is straight-line-only, so this
  is theoretical — branching support is a future extension (see Risk
  register).

### 2.2 — New strategy method `move_to_placements`

In `python/bloqade/lanes/analysis/placement/strategy.py`,
`PlacementStrategyABC` gains an abstract method matching the existing
naming convention (`cz_placements`, `sq_placements`, `measure_placements`):

```python
class PlacementStrategyABC(abc.ABC):
    # … existing methods …

    @abc.abstractmethod
    def move_to_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
        locations: tuple[LocationAddress, ...],
    ) -> AtomState:
        """Apply a user-directed move to ``state``.

        Returns a ``UserMoved`` state on success, or
        ``AtomState.bottom()`` if the requested move is infeasible —
        occupancy conflict (§4 #6) or AOD lane assignment failure
        (§4 #7). The post-compile validator picks up the ``bottom``
        state and surfaces the user-facing diagnostic; the strategy
        never emits diagnostics itself."""
```

#### `SingleZonePlacementStrategyABC.move_to_placements`

Concrete implementation that synthesizes the user move:

1. Short-circuit: if `state == AtomState.bottom()` propagate `bottom`; if
   `state` is not a `ConcreteState`, return `AtomState.top()`. Matches
   the polarity of `cz_placements` (`strategy.py:100-104`).
2. **Occupancy precondition.** Let `moved_qubits = set(qubits)` and
   `requested_destinations = set(locations)`. For each location in
   `requested_destinations`, check whether `state.layout` currently maps
   any qubit *outside* `moved_qubits` to that location. If yes — a user
   move would displace an unmoved qubit — return `AtomState.bottom()`.
   This is how the "swap into an unmoved qubit's slot" case becomes an
   analysis failure (see §4).
3. For each `(qubit_index, destination)` pair, compute the source
   `LocationAddress` from `state.layout[qubit_index]`.
4. Build a target layout: copy `state.layout`, overwrite the moved
   indices with their requested destinations.
5. Call `self.compute_moves(state, synthetic_target_state)` (the existing
   strategy method that delegates to the Rust-backed
   `compute_move_layers` in
   `python/bloqade/lanes/heuristics/move_synthesis.py:20-53`) — returns
   AOD lane layers that effect the transport. If the synthesizer
   reports infeasibility (AOD lane assignment fails), return
   `AtomState.bottom()`.
6. On success, return `UserMoved.from_concrete_state(...)` with:
   - `layout` = target layout
   - `move_layers` = the layers from step 5 (just this MoveTo's portion)
   - `accumulated_move_layers` = `state.accumulated_move_layers + new_layers`
     if `state` is already `UserMoved`, else just `new_layers`
   - `pre_user_layout` = `state.pre_user_layout` if `state` is already
     `UserMoved`, else `state.layout` (the home position for this segment)

Diagnostic surfacing follows the existing convention: the strategy does
**not** emit Kirin diagnostics directly. A `bottom` state at any
`place.QuantumStmt`'s `state_after` is the contract for "this program
point is infeasible," and the existing post-compile lanes validator
(referenced in CLAUDE.md and `validation/address.py`) catches it and
emits the user-facing error.

#### `PalindromePlacementStrategy.move_to_placements`

```python
def move_to_placements(self, state, qubits, locations):
    return self.inner.move_to_placements(self._unwrap(state), qubits, locations)
```

`_unwrap` already converts `ExecuteCZReturn` → home `ConcreteState` (so a
MoveTo immediately after a palindrome CZ starts from the home position).
`_unwrap` leaves `UserMoved` and other states alone, so a MoveTo after
another MoveTo sees the previous `UserMoved` state directly — that's what
lets `SingleZonePlacementStrategyABC.move_to_placements` (step 5) extend
the accumulator. Pure delegation otherwise; the palindrome twist happens
at the next `place.CZ` (see §3.2).

### 2.3 — `sq_placements` / `measure_placements` propagate `UserMoved`

The existing `SingleZonePlacementStrategyABC.sq_placements` (and
`measure_placements`) strips any `ConcreteState` subclass down to a bare
`ConcreteState` (`strategy.py:122-130`) — the intent is "don't let
CZ-specific metadata leak into downstream rewrites for non-CZ ops". For
`UserMoved`, that strip is wrong: it discards `accumulated_move_layers`
and `pre_user_layout`, which the palindrome strategy still needs at the
next CZ.

Both methods grow a UserMoved-aware branch:

```python
def sq_placements(self, state, qubits):
    if isinstance(state, UserMoved):
        return UserMoved(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            move_layers=(),  # already emitted at the corresponding MoveTo
            accumulated_move_layers=state.accumulated_move_layers,
            pre_user_layout=state.pre_user_layout,
        )
    if isinstance(state, ConcreteState):
        return ConcreteState(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
        )
    return state
```

Empty `move_layers=()` on the propagated `UserMoved` ensures
`InsertMoves` emits nothing extra for the SQ/measure op; the accumulator
fields stay intact so the next `cz_placements` can splice them.

The same shape applies to `measure_placements`. `ExecuteMeasure`
(`lattice.py:147-180`) extends `ConcreteState` similarly to `UserMoved`,
so when the input is `UserMoved` the propagation either returns a
plain `UserMoved` with `move_layers=()` (preserving the accumulator for
any post-measure CZ) or a hybrid that carries both `ExecuteMeasure`
fields and `UserMoved` fields. v1 picks the former — a `UserMoved`
output for `place.MoveTo`-then-measure sequences — which keeps the
combinatorial complexity of the lattice classes flat. Measurement
followed by user moves and then a CZ is rare in practice; if it turns
out to be common, the hybrid is a localized follow-up.

### 2.4 — Interpreter method for `place.MoveTo`

In `python/bloqade/lanes/analysis/placement/impl.py`, register an
abstract-interpretation method for `place.MoveTo`:

```python
@dialect.register(key=PlacementAnalysis.keys)
class PlaceMoveToImpl(interp.MethodTable):

    @interp.impl(place.MoveTo)
    def move_to(
        self,
        interp_: PlacementAnalysis,
        frame: ForwardFrame[AtomState],
        stmt: place.MoveTo,
    ) -> tuple[AtomState, ...]:
        state = frame.get(stmt.state_before)
        if not isinstance(state, ConcreteState):
            return (state,)  # NotState / AnyState propagate
        new_state = interp_.strategy.move_to_placements(
            state, stmt.qubits, stmt.locations,
        )
        return (new_state,)
```

The method reads `stmt.qubits` (indices) and `stmt.locations`
(const-folded `LocationAddress` attributes) directly off the statement —
both are populated by `RewritePlaceOperations.rewrite_MoveTo` (§1). The
analysis does no const-folding of its own here.

### 2.5 — Interaction with existing CZ analysis

When the analysis reaches a `place.CZ` statement and the input state is
`UserMoved`:

- The strategy's existing `cz_placements` (`strategy.py:92-120` for the
  single-zone abstract base) is invoked as today, using `UserMoved.layout`
  (the post-user-move position) as the starting layout. The inner
  strategy does not need to know about `UserMoved` — it sees it as a
  `ConcreteState` and produces `ExecuteCZ.move_layers` = just the
  compiler-pairing moves needed from the post-user-move layout.
- If the post-user-move layout already satisfies the CZ pairing
  (`ExecuteCZ.verify(...)` returns True with no further moves), the
  inner strategy returns `ExecuteCZ` with `move_layers = ()` — no
  additional compiler moves needed.
- Otherwise the inner strategy synthesizes the pairing moves; user
  moves are *not* included in `ExecuteCZ.move_layers` (they were
  already emitted at the MoveTo sites).

Whether the user-move history is folded into the **return** side depends
on the strategy: `PalindromePlacementStrategy` reads
`UserMoved.accumulated_move_layers` and `UserMoved.pre_user_layout` to
build an `ExecuteCZReturn` that undoes the full segment — detailed in §3.2.

When the input state is any other `ConcreteState` (no pending user moves),
existing behavior is preserved bit-for-bit.

**Inner-strategy isinstance updates.** Any concrete strategy that
branches on `isinstance(state, X)` for a specific lattice subtype
(`ExecuteCZ`, `ExecuteCZReturn`, etc.) should be audited for whether it
also needs to recognize `UserMoved`. The single-zone abstract base in the
tree today only checks `isinstance(state, ConcreteState)`, which
naturally accepts `UserMoved`, so no change is needed there — but custom
downstream strategies may.

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

### 3.2 — `PalindromePlacementStrategy.cz_placements` splice

In `python/bloqade/lanes/analysis/placement/strategy.py`,
`PalindromePlacementStrategy.cz_placements` (`strategy.py:187-205`) gains
a branch for `UserMoved` input:

```python
def cz_placements(self, state, controls, targets, lookahead_cz_layers=()):
    home = self._unwrap(state)  # ExecuteCZReturn → ConcreteState; UserMoved passes through
    result = self.inner.cz_placements(home, controls, targets, lookahead_cz_layers)
    if not isinstance(result, ExecuteCZ) or not isinstance(home, ConcreteState):
        return result

    if isinstance(home, UserMoved):
        return ExecuteCZReturn(
            occupied=result.occupied,
            layout=result.layout,
            move_count=result.move_count,
            active_cz_zones=result.active_cz_zones,
            move_layers=result.move_layers,          # compiler pairing only
            user_move_layers=home.accumulated_move_layers,  # for the return
            initial_layout=home.pre_user_layout,     # true home
        )

    return ExecuteCZReturn(                          # existing path
        occupied=result.occupied,
        layout=result.layout,
        move_count=result.move_count,
        active_cz_zones=result.active_cz_zones,
        move_layers=result.move_layers,
        initial_layout=home.layout,
    )
```

Key points:

- `_unwrap` is unchanged — it only converts `ExecuteCZReturn` back to a
  home `ConcreteState`. `UserMoved` passes through, so `home` is the
  `UserMoved` itself when the previous segment included user moves.
- The inner strategy receives the `UserMoved` (typed as `ConcreteState`)
  and computes its `ExecuteCZ.move_layers` from `UserMoved.layout` (the
  post-user-move position). It does not know about user moves.
- The palindrome wrap-up reads `UserMoved.accumulated_move_layers` and
  `UserMoved.pre_user_layout` to populate the new `user_move_layers`
  field and the correct `initial_layout` on the resulting
  `ExecuteCZReturn`.

### 3.3 — `ExecuteCZReturn` shape change

`ExecuteCZReturn` (`lattice.py:183-224`) gains a new field
`user_move_layers` and an extended `__post_init__` that palindromes both
segments:

```python
@final
@dataclass
class ExecuteCZReturn(ExecuteCZ):
    initial_layout: tuple[LocationAddress, ...] = field(kw_only=True)

    user_move_layers: tuple[tuple[LaneAddress, ...], ...] = field(
        kw_only=True, default=()
    )
    """User-move layers already emitted forward at the MoveTo sites.
    Included in `return_move_layers` so the palindrome undoes the full
    inter-CZ segment, not just the compiler-pairing portion."""

    return_move_layers: tuple[tuple[LaneAddress, ...], ...] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        compiler_reverse = tuple(
            tuple(lane.reverse() for lane in layer)
            for layer in reversed(self.move_layers)
        )
        user_reverse = tuple(
            tuple(lane.reverse() for lane in layer)
            for layer in reversed(self.user_move_layers)
        )
        # CZ unwinds first (compiler moves last → first in reverse),
        # then user moves unwind back to the home position.
        self.return_move_layers = compiler_reverse + user_reverse
```

- `user_move_layers` defaults to `()` — pre-existing call sites that
  construct `ExecuteCZReturn` without user moves keep their current
  behavior bit-for-bit.
- `is_subseteq` extends to compare `user_move_layers` (matches the
  precision of the other fields).
- `InsertMoves` (`place2move.py:14-61`) is unchanged: it consumes
  `get_move_layers()` (forward — compiler pairing only) and
  `get_reverse_moves()` (full-segment palindrome) uniformly.

## §4 — Validation

Failure modes split into two tiers: **eager** (gemini-side method-table
impl on `movement.MoveTo`, before lowering to place) and **analysis**
(placement strategy returns `AtomState.bottom()` at the `place.MoveTo`'s
`state_after`; downstream post-compile validator catches the bottom and
emits a diagnostic — no new diagnostic plumbing in the strategy).

Eager errors are Kirin diagnostics pointing at the user's
`movement.move_to(...)` source line; analysis errors surface from the
existing post-compile lanes validator with kernel-level context.

| # | Failure | Tier | Mechanism |
|---|---|---|---|
| 1 | `len(qubits) != len(locations)` | eager | gemini per-stmt method-table impl (registered against the lanes validator key, matching `bloqade.gemini.common.validation.new_at`) |
| 2 | `locations` IList isn't const-foldable to a tuple of `LocationAddress` | eager | gemini per-stmt method-table impl |
| 3 | Any `LocationAddress` out of ArchSpec range | eager | gemini per-stmt method-table impl |
| 4 | Duplicate destinations within one `move_to` call | eager | gemini per-stmt method-table impl |
| 5 | Duplicate qubit references within one `move_to` call (same `Qubit` SSA appears more than once in the `qubits` IList) | eager | gemini per-stmt method-table impl |
| 6 | Destination is currently held by a qubit *not* in this `move_to` call ("swap into unmoved slot") | analysis | `move_to_placements` returns `bottom` (§2.2 step 2); post-compile validator surfaces |
| 7 | AOD lane assignment infeasible (synthesizer failure) | analysis | `move_to_placements` returns `bottom` (§2.2 step 5); post-compile validator surfaces |
| 8 | Qubit already at requested location | n/a | silent no-op — empty layer emitted; user-visible behavior identical to omitting that `(qubit, location)` pair |

### 4.1 — Eager per-statement validation (failures 1-5)

A method-table impl for `movement.MoveTo` registered against the existing
lanes validation interpreter key (same pattern as
`bloqade.gemini.common.validation.new_at`). The impl runs against the
gemini IR before lowering to place.

For failures 4 and 5 — duplicate-destination and duplicate-qubit — the
check reads the corresponding const-folded IList values (locations and
qubits respectively). For qubits the check is on SSA-value identity
rather than const data: two slots in the `qubits` IList referencing the
same `Qubit` SSA value is a duplicate.

### 4.2 — Analysis failures (failures 6-7)

The placement analysis interpreter method for `place.MoveTo` (§2.4)
delegates to `strategy.move_to_placements`. When that returns
`AtomState.bottom()`, the analysis frame records `bottom` for the
statement's `state_after`. The existing post-compile lanes validator
notices the `bottom` and emits a kernel-level diagnostic naming the
infeasible MoveTo. No new diagnostic-emission code in the strategy
itself; the contract is "`bottom` ⇒ infeasible point ⇒ validator
complains."

### 4.3 — Already-at-destination (failure 8)

If a qubit's current location equals its requested destination, the
synthesizer simply emits no AOD lane for that pair. The resulting
`UserMoved.move_layers` may be empty (or shorter than the number of
input pairs); `accumulated_move_layers` extends only by the non-empty
layers. User-visible behavior: identical to omitting that
`(qubit, location)` pair from the call.

## §5 — Extending `gemini.logical.kernel`

The user-facing surface is folded into the existing
`gemini.logical.kernel` decorator (`bloqade/gemini/logical/group.py`) so
that every Gemini logical kernel can call `movement.move_to(...)` without
opting into a separate decorator. The movement dialect joins the
existing union and the movement-specific validator joins the existing
`ValidationSuite`.

### 5.1 — Dialect union

In `python/bloqade/gemini/logical/group.py`:

```python
from bloqade.lanes.dialects import movement as movement_dialect

@ir.dialect_group(
    structural_no_opt.union(
        [gate, qubit, operations, annotate,
         gemini_common.dialects.qubit, movement_dialect]   # ← new
    )
)
def kernel(self):
    """Compile a function to a Gemini logical kernel."""
    ...
```

No other Gemini-side import or wrapper changes are needed — the dialect
is purely additive, and kernels that never call `movement.move_to`
produce IR with no `movement.MoveTo` statements (the movement-side
rewrites and analysis paths are simply no-ops).

### 5.2 — Validation registration

The existing `ValidationSuite` constructed inside `run_pass`
(`group.py:102-109`) gains the new movement validator:

```python
validator = ValidationSuite([
    GeminiLogicalValidation,
    GeminiTerminalMeasurementValidation,
    FlatKernelNoCloningValidation,
    DuplicateAddressValidation,
    MoveToValidation,          # ← new — implements failures 1-5 from §4
])
```

`MoveToValidation` lives in `python/bloqade/gemini/common/validation/move_to.py`
and registers the per-statement method-table impl described in §4.1. It
follows the structural shape of `bloqade.gemini.common.validation.new_at`
— a `ValidationPass` subclass whose interpreter recognizes the
movement-dialect statement.

### 5.3 — User-facing example

```python
from bloqade.gemini.logical import kernel
from bloqade.lanes.dialects import movement

@kernel
def k():
    q = qubit.qalloc(4)
    movement.move_to([q[0], q[1]], [loc_a, loc_b])
    squin.cz(q[0], q[2])
```

`movement` lives in `bloqade.lanes.dialects` (per §1) but is unioned into
the gemini logical kernel group; users import the statement from lanes
and use the kernel decorator from gemini. The base
`bloqade.lanes._prelude` / `bloqade.lanes.prelude` dialect groups
(`_prelude.py:10`, `prelude.py:10`) are **not** extended — lanes-only
kernels still don't see `movement.MoveTo`. The dialect is a
gemini-logical-level surface.

### 5.4 — Backward compatibility

Existing kernels that never reference `movement.move_to` produce
byte-identical IR before vs after this change:

- The dialect being in the group's union but unused costs only a small
  amount of dialect-registration overhead at kernel construction;
- `MoveToValidation` is a per-statement impl that runs only against
  `movement.MoveTo` instances — kernels without those statements emit
  no diagnostics and no extra work;
- The placement-analysis interpreter for `place.MoveTo` likewise only
  runs against that statement type; the post-rewrite IR for a non-
  movement kernel contains zero `place.MoveTo` statements, so the
  analysis is unaffected.

The regression test in §7.9 (existing demo corpus → byte-identical move
IR) is the canary for this property.

## §6 — Place→Move emission

`InsertMoves` (`python/bloqade/lanes/rewrite/place2move.py:14-61`) is
**not changed**. It iterates over `place.QuantumStmt` nodes and emits
forward + return Move IR via `state_after.get_move_layers()` and
`state_after.get_reverse_moves()` — both polymorphic on the lattice
element. With the §2/§3 lattice changes:

- `UserMoved.get_move_layers()` returns `move_layers` (just this
  MoveTo's portion), so each `place.MoveTo` emits exactly its own
  forward Move IR.
- `UserMoved.get_reverse_moves()` returns `()` (inherited from
  `AtomState`), so no return moves emit at the MoveTo site.
- `ExecuteCZ` (compiler pairing without palindrome) is unchanged.
- `ExecuteCZReturn` carries `move_layers` (compiler pairing only) for
  the forward emit, and `return_move_layers = compiler_reverse +
  user_reverse` for the return emit (computed in `__post_init__`).

No changes to `InsertMoves` itself — the polymorphic dispatch already
covers the new state class.

### 6.1 — Multi-shot warning channel

The placement strategy synthesizes user-move layers via `compute_moves`
(§2.2 step 5). When `compute_moves` returns more than one layer for a
single `move_to` call, the requested transport cannot be performed in a
single AOD shot. v1 still accepts the call (best-effort packing — see
Non-goals on `require_parallel`), but the strategy records a non-fatal
warning so the user knows.

#### Warning storage

`SingleZonePlacementStrategyABC` (and any future strategy) gains a
mutable warning accumulator:

```python
@dataclass
class MultiShotWarning:
    qubits: tuple[int, ...]
    locations: tuple[LocationAddress, ...]
    layer_count: int


@dataclass
class SingleZonePlacementStrategyABC(PlacementStrategyABC):
    # … existing fields …
    multi_shot_warnings: list[MultiShotWarning] = field(
        default_factory=list, repr=False, compare=False
    )

    def move_to_placements(self, state, qubits, locations):
        ...
        new_layers = self.compute_moves(state, target_state)
        if len(new_layers) > 1:
            self.multi_shot_warnings.append(
                MultiShotWarning(qubits=qubits, locations=locations,
                                 layer_count=len(new_layers))
            )
        ...
```

`repr=False` / `compare=False` keep the accumulator out of the dataclass
identity — two strategy instances are still equal iff their constructor
fields match, so existing equality-based tests are unaffected.

For `PalindromePlacementStrategy`, no extra accumulator is needed — its
`move_to_placements` (§2.2) delegates to `self.inner`, so warnings land
on the inner strategy and are reachable as
`palindrome_strategy.inner.multi_shot_warnings`.

#### Surfacing warnings

The `run_pass` in `bloqade/gemini/logical/group.py` reads
`strategy.multi_shot_warnings` after the placement analysis completes
and emits one `UserWarning` per entry via Python's `warnings.warn`:

```python
import warnings
...
for w in strategy.multi_shot_warnings:
    warnings.warn(
        f"movement.move_to(qubits={w.qubits}, locations={w.locations}) "
        f"could not be packed into a single AOD shot "
        f"(split across {w.layer_count} layers)",
        UserWarning,
    )
```

Reasons for `warnings.warn` over an analysis-frame channel:
- Python idiom — composes cleanly with `pytest.warns` and `-W error`.
- No new diagnostic infrastructure inside `ValidationSuite`, which is
  error-tier today.
- v2's `require_parallel=True` flag (Non-goals) would short-circuit
  *before* the accumulator: return `bottom()` from
  `move_to_placements` on `len(new_layers) > 1`, surfacing via the
  existing post-compile validator.

### 6.2 — Inter-call coalescing

Two adjacent `place.MoveTo` statements with no intervening non-movement
op are **not** coalesced into one AOD shot in v1, even if their
requested transports are mutually compatible. Each statement is a
separate `move_to_placements` call and produces its own
`UserMoved.move_layers` (which `InsertMoves` emits as its own
`move.Move` group). Coalescing is a future optimization.

## §7 — Testing strategy

### 7.1 — Eager validation (§4.1)

For each failure mode in §4 rows 1-5, write a kernel that exhibits it
and assert the Kirin diagnostic surfaces at the right source line:

- Mismatched `qubits` / `locations` IList lengths → length error.
- `locations` IList contains a non-const `LocationAddress` element
  (e.g., one produced by a runtime call) → const-foldability error
  pointing at the offending element.
- Out-of-range `(zone_id, word_id, site_id)` → range error.
- Two identical `LocationAddress` values in one `locations` IList →
  duplicate-destination error, points at the duplicate and references
  the first.
- Same `Qubit` SSA value appearing twice in one `qubits` IList →
  duplicate-qubit error (failure #5).
- Negative cases: valid single-qubit move, valid multi-qubit move,
  already-at-destination no-op, permutation among the moved qubits
  (qubit_a → loc_of_b, qubit_b → loc_of_a) → eager validation passes.

### 7.2 — `movement.MoveTo` → `place.MoveTo` rewrite (§1)

- Literal-IList kernel
  (`movement.move_to([q[0], q[1]], [loc_a, loc_b])`) post-ConstantFold →
  rewrite produces a `place.MoveTo` with `qubits=(0, 1)` (or whatever
  range maps onto the enclosing `StaticPlacement` qubits) and
  `locations=(loc_a, loc_b)` as a static attribute tuple.
- Const hint absent on `locations` (e.g., ConstantFold hasn't run) →
  rewrite returns `RewriteResult()` with no modification (defers,
  doesn't raise).
- `qubits` operand not produced by `ilist.New` → rewrite returns
  `RewriteResult()`.

### 7.3 — Placement-analysis interpreter and strategy (§2)

- Single `move_to` from a fresh `ConcreteState` → strategy returns
  `UserMoved` with `move_layers` containing this call's layers,
  `accumulated_move_layers == move_layers`, `pre_user_layout ==
  input_state.layout`, and `layout` reflecting the moved positions.
- Sequence of two `move_to` calls (no intervening CZ) → second
  `UserMoved` has `accumulated_move_layers ==
  (first.move_layers + second.move_layers)`,
  `pre_user_layout == first.pre_user_layout` (unchanged), and `layout`
  reflecting both moves. The per-call `move_layers` field on the second
  state contains only the second call's layers.
- `move_to` followed by a non-CZ op (`place.R` / `place.Rz`) →
  `sq_placements` returns a `UserMoved` with `move_layers=()` but
  `accumulated_move_layers` and `pre_user_layout` preserved. Confirms
  the §2.3 propagation rule.
- Same setup followed by `place.EndMeasure` → equivalent assertion
  on `measure_placements`.
- "Swap into unmoved slot" — `move_to([q0], [loc_currently_held_by_q1])`
  while q1 stays put → strategy returns `AtomState.bottom()`
  (occupancy precondition, §2.2 step 2).
- "Permutation among moved qubits" — `move_to([q0, q1], [loc_of_q1,
  loc_of_q0])` → strategy returns `UserMoved` (the precondition
  excludes locations held by qubits *in* the call, so a clean
  permutation is allowed).
- Synthesizer infeasibility — set up a layout where the requested
  transport has no legal AOD assignment → strategy returns
  `AtomState.bottom()` (§2.2 step 5).
- Interpreter method (§2.4): non-`ConcreteState` input (top or bottom
  state) propagates unchanged without calling the strategy.

### 7.4 — Palindrome interaction (§3)

- `move_to` + CZ under the default (non-palindrome) placement strategy →
  `ExecuteCZ` whose `move_layers` is **only** the compiler-pairing
  moves (or empty when user moves already land on CZ-compatible
  positions). User moves were already accounted for in the `UserMoved`
  state that fed `cz_placements`.
- `move_to` + CZ under `PalindromePlacementStrategy` →
  `ExecuteCZReturn` whose:
  - `move_layers` is the compiler-pairing portion only,
  - `user_move_layers` is exactly `UserMoved.accumulated_move_layers`,
  - `initial_layout == UserMoved.pre_user_layout`,
  - `return_move_layers == compiler_reverse + user_reverse` (verified
    by reconstructing the expected palindrome).
- `move_to` placing qubits into CZ-compatible positions → compiler
  synthesizes **zero** additional pairing moves
  (`ExecuteCZ.move_layers == ()`); under palindrome the return
  reverses only the user moves (`return_move_layers ==
  reversed(user_move_layers_palindromed)`, `move_layers == ()`).
- Sequence: `move_to` + CZ + `move_to` + CZ — verifies the lattice
  resets between CZs. After the first CZ (palindrome), the next
  `move_to` runs against the `_unwrap`'d home `ConcreteState`, and its
  `UserMoved.pre_user_layout` is the home position — not the previous
  segment's `pre_user_layout`.

### 7.5 — Place→Move emission (§6)

- Each scenario from §7.4 lowered through `InsertMoves` → resulting
  `move.Move` statements in the expected order:
  - At each `place.MoveTo`: one `Load` / one or more `Move` / one
    `Store` triple emitting `UserMoved.move_layers`.
  - At each `place.CZ` (default strategy): forward triple emitting
    `ExecuteCZ.move_layers` (empty when user moves already paired the
    qubits → no triple emitted).
  - At each `place.CZ` (palindrome): forward triple for
    `ExecuteCZReturn.move_layers` (compiler pairing only) before the
    CZ; return triple for `return_move_layers` (compiler reverse +
    user reverse) after the CZ.
- Single-call multi-layer split → multiple consecutive `move.Move`
  statements emit for one `place.MoveTo` source statement.

### 7.6 — `MultiShotWarning` channel (§6.1)

- A `move_to` call whose transport packs into one AOD shot →
  `strategy.multi_shot_warnings` is empty after compilation; no
  `UserWarning` emitted.
- A `move_to` call whose transport requires N > 1 AOD shots →
  `strategy.multi_shot_warnings` contains one `MultiShotWarning(...)`
  with the right `qubits`, `locations`, and `layer_count == N`.
- Compile the same kernel via `gemini.logical.kernel` and use
  `pytest.warns(UserWarning)` to confirm the warning surfaces to the
  user. Run with `-W error::UserWarning` to confirm the warning can be
  upgraded to a failure (the v2 `require_parallel=True` short-circuit
  would behave similarly).
- Palindrome wrapper: warnings raised by `inner.move_to_placements`
  remain accessible as `palindrome.inner.multi_shot_warnings`. No
  duplicate accumulator on the wrapper.

### 7.7 — `gemini.logical.kernel` extension (§5)

- Compile a kernel that uses both `squin.*` and `movement.move_to`
  under the existing `gemini.logical.kernel` decorator → succeeds;
  resulting move IR contains both circuit-derived and user-directed
  transport.
- Compile a kernel that uses `movement.move_to` under a bare
  `bloqade.lanes.prelude` / `_prelude` group → fails at dialect-group
  validation (the movement dialect is not in the lanes prelude unions
  — see `prelude.py:10`, `_prelude.py:10`). Confirms the dialect
  remains gemini-logical-scoped.
- Sanity test that `MoveToValidation` is in the `ValidationSuite` list
  constructed by `gemini.logical.kernel.run_pass` — a small kernel with
  a length-mismatched `move_to` call must trip the validator.

### 7.8 — Composition with `new_at` (Open Question)

- `move_to` a qubit that was placed via explicit `new_at(z, w, s)` on
  the same kernel → succeeds; the qubit moves from its pinned position
  to the requested destination. Confirms the explicit-allocation and
  movement-intent paths compose cleanly (Open Question item in this
  spec).

### 7.9 — Regression

- Existing demo kernels under `demo/` (zero `movement.move_to` usage)
  compile to byte-identical move IR before vs after this change. This
  is the primary canary that:
  - the new `movement` dialect being in the gemini logical kernel
    group's union does not perturb output for non-movement kernels,
  - `ExecuteCZReturn`'s new `user_move_layers` field defaults to `()`
    and leaves the existing palindrome computation byte-identical,
  - `sq_placements` / `measure_placements` continue to strip plain
    `ConcreteState` subclasses to bare `ConcreteState` when no
    `UserMoved` is involved.

## §8 — File layout summary

```
python/bloqade/lanes/
├── dialects/
│   ├── movement.py                       # NEW — movement dialect + MoveTo
│   │                                     #        statement (user-facing,
│   │                                     #        IList SSA operands)
│   └── place.py                          # EDIT — add place.MoveTo
│                                         #        (QuantumStmt; attribute
│                                         #        qubits/locations)
├── analysis/placement/
│   ├── lattice.py                        # EDIT — add UserMoved lattice
│   │                                     #        element; extend
│   │                                     #        ExecuteCZReturn with
│   │                                     #        user_move_layers field
│   ├── strategy.py                       # EDIT — add move_to_placements
│   │                                     #        to PlacementStrategyABC;
│   │                                     #        sq/measure preserve
│   │                                     #        UserMoved; palindrome
│   │                                     #        cz_placements splice
│   └── impl.py                           # EDIT — interp method for
│                                         #        place.MoveTo
├── rewrite/circuit2place.py              # EDIT — RewritePlaceOperations
│                                         #        gains rewrite_MoveTo
└── rewrite/place2move.py                 # (no edits — InsertMoves
                                          #  already uniform)

python/bloqade/gemini/
├── logical/group.py                      # EDIT — union movement dialect
│                                         #        into kernel; register
│                                         #        MoveToValidation
└── common/validation/move_to.py          # NEW — eager validation impl
                                          #        (failures 1-5 from §4)
```

## Open questions

- **Layer granularity in `UserMoved`.** Each `move_to` call contributes
  one or more layers. Two adjacent calls produce two separate layer
  groups even when their transports could pack into one AOD shot. v1
  defers coalescing; the open question is whether the lattice element
  should pre-emptively flatten to a single tuple or preserve per-call
  grouping (e.g. for diagnostics that map back to the source call).
- **`MultiShotWarning` granularity (§6.1).** The current spec records
  the full `(qubits, locations, layer_count)` for each warning. If the
  synthesizer can identify *which subset* of pairs forced the split,
  that information would be more actionable — but it requires the
  Rust-backed `compute_move_layers` to surface per-layer membership,
  which it does not today. Tracked here so the synthesizer side knows
  there's a consumer for richer information if it ever wants to expose
  it.
- **`movement.MoveTo` interaction with `place.NewLogicalQubit`.** Can a
  user `move_to` a qubit that was placed via explicit `new_at(z, w, s)`
  on the same kernel? The semantics are well-defined (the qubit moves
  from its pinned position to the new destination), but worth a regression
  test ensuring the explicit-allocation and movement-intent paths compose
  cleanly.

## Risk register

- **Highest risk:** the new `UserMoved` lattice element joining
  semantics. If two control-flow paths produce different `move_layers`,
  `accumulated_move_layers`, or `pre_user_layout`, the join goes to
  `AnyState` (top), halting analysis. v1 only supports straight-line
  kernels (matching today's Squin), so this is theoretical — but the
  spec calls it out so future branching support knows it has work to
  do here.
- **Medium risk:** `PalindromePlacementStrategy.cz_placements`
  regression. The change is narrow (handle a `UserMoved` input branch
  + populate the new `user_move_layers` field on `ExecuteCZReturn`)
  but the strategy is load-bearing for every palindrome-using kernel
  today. Byte-identical regression on the existing demo corpus
  (§7.9) is the gate.
- **Medium risk:** `ExecuteCZReturn.__post_init__` change. The
  `return_move_layers` formula moves from `reverse(move_layers)` to
  `compiler_reverse + user_reverse`. When `user_move_layers` defaults
  to `()` the result is bit-for-bit identical — but any direct
  construction of `ExecuteCZReturn` from outside
  `PalindromePlacementStrategy` (e.g. in tests) must not break. The
  field default and §7.9 regression gate this.
- **Low risk:** `sq_placements` / `measure_placements` strip-vs-preserve
  branching. The new code path adds an `isinstance(state, UserMoved)`
  check before the existing strip. A misordering (e.g. checking
  `ConcreteState` first) would silently drop the accumulator. Test:
  §7.3's "move_to followed by R" / "followed by EndMeasure" cases
  catch this directly.
