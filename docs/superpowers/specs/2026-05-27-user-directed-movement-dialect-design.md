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
  (`PlacementStrategyABC.move_to_placements`), and the
  `PalindromePlacementStrategy.cz_placements` splice. The placement analysis
  must recognize `place.MoveTo` directly, so both statements live next to
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
  qubits=int indices, locations=const-folded LocationAddress tuple,
  multi_move_warning=bool)
       ↓
PlacementAnalysis (extended):
   • new interp method for place.MoveTo in PlacementMethods (place.py)
     delegates to strategy.move_to_placements
   • strategy.move_to_placements returns a UserMoved lattice element
     carrying:
       - move_layers          (this MoveTo's AOD layers, for InsertMoves)
       - accumulated_move_layers (full inter-CZ history, for palindrome)
       - pre_user_layout      (home position before the segment started)
   • analysis failures (occupancy conflict, AOD infeasibility) → bottom
   • sq_placements(UserMoved) → bottom  (move_to before SQ gate: invalid)
   • measure_placements(UserMoved) → bottom  (move_to before measure: invalid)
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
InsertMoves (place→move, minimal edit):
   • polymorphic on the lattice element: emits forward from
     state.get_move_layers() and reverse from state.get_reverse_moves()
   • at place.MoveTo:  UserMoved.move_layers   → forward Move IR
                       emits UserWarning if multi_move_warning=True
                       and len(move_layers) > 1
   • at place.CZ:      ExecuteCZ(Return).move_layers → forward,
                        ExecuteCZReturn.return_move_layers → reverse
       ↓
RewriteGates (place→move, new handler):
   • place.MoveTo handler deletes the node after InsertMoves has
     emitted its layers, leaving the enclosing StaticPlacement with
     only a Yield (eligible for RemoveNoOpStaticPlacements)
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
    multi_move_warning: bool = info.attribute(default=True)
```

- Two SSA operands (`qubits`, `locations`) plus one compile-time boolean
  attribute (`multi_move_warning`, default `True`).
- No SSA result. The statement is a placement directive consumed by the
  movement→place rewrite (below) and the placement analysis (§2).
- Length equality (`len(qubits) == len(locations)`) and const-foldability of
  the `locations` IList are eager-validation concerns (§4), not statement
  shape concerns.

Python lowering stub (consumed by `lowering.FromPythonCall`):

```python
@wraps(MoveTo)
def move_to(
    qubits: IList[Qubit, N],
    locations: IList[LocationAddress, N],
    multi_move_warning: bool = True,
) -> None:
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
    multi_move_warning: bool = info.attribute(default=True)
```

- Inherits `QuantumStmt` (`place.py:85-99`) — threads `state_before: ir.SSAValue`
  / `state_after: ir.ResultValue` of `StateType`, matching the existing
  pattern for `place.CZ`, `place.R`, `place.Rz`.
- `qubits` is a tuple of integer indices into the surrounding
  `place.StaticPlacement` block's qubit SSA list.
- `locations` is a tuple of const `LocationAddress` attributes (one per
  qubit index), populated by the rewrite below from the const-prop hint on
  `movement.MoveTo.locations`.
- `multi_move_warning` is copied verbatim from the source `movement.MoveTo`.
- Invariant: `len(qubits) == len(locations)`. Established by the rewrite,
  asserted by the statement constructor.

### Rewrite — `movement.MoveTo` → `place.MoveTo`

Lives on `RewritePlaceOperations` (`python/bloqade/lanes/rewrite/circuit2place.py`),
alongside the existing per-gate rewrite methods. Mirrors `rewrite_CZ`
(lines 265-291) structurally:

```python
def rewrite_MoveTo(self, node: movement.MoveTo) -> abc.RewriteResult:
    # qubits side: after full unrolling and CSE (which run before this
    # rewrite), the qubits IList is always produced by ilist.New.
    # The guard here is defensive — in practice it cannot trigger on a
    # well-formed kernel. Qubit correctness is validated by AddressAnalysis.
    if not isinstance(qubits_list := node.qubits.owner, ilist.New):
        return abc.RewriteResult()

    # locations side: read the const-prop hint. Eager validation (§4)
    # requires every LocationAddress to be const-foldable, so the hint
    # resolves to a const.Value whose .data is an IList of LocationAddress.
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
        multi_move_warning=node.multi_move_warning,
    )
    node.replace_by(
        self.construct_execute(move_stmt, qubits=inputs, body=body, block=block)
    )
    return abc.RewriteResult(has_done_something=True)
```

The rewrite gives up silently (returns `RewriteResult()` with no
modification) when:
- The `qubits` operand isn't produced by an `ilist.New` — defensive only;
  see note above.
- The `locations` const-prop hint isn't a `const.Value` — a validation gap
  or a kernel that hasn't been unrolled yet.

In both cases the validation pass (§4) will surface a diagnostic; the
rewrite stays defensive rather than raising.

## §2 — Placement-analysis extensions

The placement analysis (`python/bloqade/lanes/analysis/placement/`) is the
seam where user moves become first-class. Three changes: the new lattice
element, a new method on the placement strategy ABC (implemented by each
concrete strategy), and an interpreter method on the analysis that delegates
to the strategy.

### 2.1 — New lattice element `UserMoved`

In `python/bloqade/lanes/analysis/placement/lattice.py`, alongside
`ExecuteCZ` / `ExecuteCZReturn`:

```python
@final
@dataclass
class UserMoved(ConcreteState):
    """State produced by one or more consecutive user-directed
    `place.MoveTo` statements, with no intervening non-MoveTo ops.

    Carries two layer tuples for two distinct consumers:

    - `move_layers` is the AOD lane layer(s) for *this* MoveTo only.
      ``InsertMoves`` reads this via ``get_move_layers()`` and emits
      the forward Move IR immediately before the corresponding
      ``place.MoveTo`` statement.

    - `accumulated_move_layers` is every user-move layer in the
      inter-CZ segment so far (i.e. ``prev.accumulated_move_layers +
      move_layers`` if the previous state was already ``UserMoved``,
      else just ``move_layers``). ``PalindromePlacementStrategy``
      reads this at the next CZ to build the return-side palindrome
      that undoes the full user-move history.

    `pre_user_layout` records the atom layout *before* the first user
    move in this segment — the home position
    ``PalindromePlacementStrategy`` returns to after the CZ pulse.

    Outside of ``cz_placements``, ``UserMoved`` must be treated as a
    plain ``ConcreteState``. ``sq_placements`` and
    ``measure_placements`` receiving a ``UserMoved`` state signals that
    the user inserted a move before a non-CZ operation, which is
    invalid — both methods return ``AtomState.bottom()`` in that case
    (see §2.3).
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
- `get_reverse_moves()` returns `()` — inherited from `AtomState`. Reverse
  moves at a MoveTo site are never emitted; the palindrome return is handled
  entirely at the subsequent `place.CZ` site via `ExecuteCZReturn`.
- `is_subseteq` matches the precision used by `ExecuteCZ` / `ExecuteCZReturn`.
- `from_concrete_state` mirrors the lifting helpers on `ExecuteCZ` and
  `ExecuteMeasure` so the analysis can promote a `ConcreteState` into
  `UserMoved` when it sees the first `place.MoveTo` in a segment. Works
  correctly whether `state` is a bare `ConcreteState` or a `UserMoved`
  (both expose `occupied`, `layout`, `move_count`).
- **Join semantics**: divergent `move_layers`, `accumulated_move_layers`,
  or `pre_user_layout` across control-flow paths join to `AnyState` (top)
  via the inherited `SimpleJoinMixin`. v1 only supports straight-line
  kernels (matching today's Squin), so this is theoretical — branching
  support is a future extension (see Risk register).

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
catches it and emits the user-facing error.

#### `PalindromePlacementStrategy.move_to_placements`

```python
def move_to_placements(self, state, qubits, locations):
    return self.inner.move_to_placements(self._unwrap(state), qubits, locations)
```

`_unwrap` converts `ExecuteCZReturn` → home `ConcreteState` (so a MoveTo
immediately after a palindrome CZ starts from the home position). `_unwrap`
leaves `UserMoved` and other states alone, so a MoveTo after another MoveTo
sees the previous `UserMoved` state directly — allowing
`SingleZonePlacementStrategyABC.move_to_placements` (step 6) to extend the
accumulator across consecutive MoveTo calls.

### 2.3 — `sq_placements` and `measure_placements` with `UserMoved` input

`movement.move_to` is only valid immediately before a CZ gate (or another
`move_to`). If the placement analysis reaches a `sq_placements` or
`measure_placements` call with a `UserMoved` input state, it means a user
move was followed by a non-CZ operation — an invalid program. Both methods
must signal this via `AtomState.bottom()`:

```python
def sq_placements(self, state, qubits):
    if isinstance(state, UserMoved):
        return AtomState.bottom()     # move_to before SQ gate — invalid
    if isinstance(state, ConcreteState):
        return ConcreteState(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
        )
    return state
```

The same pattern applies to `measure_placements` — return
`AtomState.bottom()` when `isinstance(state, UserMoved)`.

**isinstance ordering**: the `UserMoved` branch MUST appear before the
`ConcreteState` branch. Since `UserMoved IS ConcreteState`, checking
`ConcreteState` first would silently accept an invalid state rather than
returning `bottom`. See Risk register.

The `bottom` state propagates to the `state_after` of the subsequent
`place.QuantumStmt`, where the existing post-compile lanes validator catches
it and surfaces a kernel-level diagnostic. The structural constraint
("move_to must precede a CZ") cannot be checked eagerly on the current IR —
pipeline-level enforcement via `bottom` is the correct mechanism.

**Terminal `UserMoved`**: if `move_to` is the last statement in a kernel
(no subsequent gate or measurement), the analysis leaves the state as
`UserMoved` without triggering the bottom path. `InsertMoves` emits the
forward moves at the `place.MoveTo` site; atoms remain at the user-moved
positions. This is valid and intentional.

### 2.4 — Interpreter method for `place.MoveTo`

In `python/bloqade/lanes/dialects/place.py`, add to the existing
`PlacementMethods` class (key `"runtime.placement"`, registered at line 247):

```python
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

- The strategy's existing `cz_placements` is invoked as today, using
  `UserMoved.layout` (the post-user-move position) as the starting layout.
  The inner strategy does not need to know about `UserMoved` — it sees it
  as a `ConcreteState` and produces `ExecuteCZ.move_layers` = just the
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
  behavior bit-for-bit. When `user_move_layers = ()`, `user_reverse = ()`,
  so `return_move_layers = compiler_reverse` — identical to the original
  formula.
- `is_subseteq` extends to compare `user_move_layers` (matches the
  precision of the other fields).
- `InsertMoves` (`place2move.py`) is unchanged in its consumption of
  `get_move_layers()` (forward — compiler pairing only) and
  `get_reverse_moves()` (full-segment palindrome) — both polymorphic.

## §4 — Validation

Failure modes split into two tiers: **eager** (gemini-side method-table
impl on `movement.MoveTo`, before lowering to place) and **analysis**
(placement strategy returns `AtomState.bottom()` at the relevant
`state_after`; downstream post-compile validator catches the bottom and
emits a diagnostic — no new diagnostic plumbing in the strategy).

Eager errors are Kirin diagnostics pointing at the user's
`movement.move_to(...)` source line; analysis errors surface from the
existing post-compile lanes validator with kernel-level context.

| # | Failure | Tier | Mechanism |
|---|---|---|---|
| 1 | `len(qubits) != len(locations)` | eager | gemini per-stmt method-table impl |
| 2 | `locations` IList isn't const-foldable to a tuple of `LocationAddress` | eager | gemini per-stmt method-table impl |
| 3 | Any `LocationAddress` out of ArchSpec range | eager | gemini per-stmt method-table impl (uses `arch_spec` injected via `run_pass`, see §5) |
| 4 | Duplicate destinations within one `move_to` call | eager | gemini per-stmt method-table impl |
| 5 | Duplicate qubit references within one `move_to` call (same `Qubit` SSA appears more than once in the `qubits` IList) | eager | gemini per-stmt method-table impl |
| 6 | `move_to` followed by a non-CZ, non-MoveTo operation (SQ gate or measurement) | analysis | `sq_placements(UserMoved)` / `measure_placements(UserMoved)` return `bottom` (§2.3); post-compile validator surfaces |
| 7 | Destination is currently held by a qubit *not* in this `move_to` call ("swap into unmoved slot") | analysis | `move_to_placements` returns `bottom` (§2.2 step 2); post-compile validator surfaces |
| 8 | AOD lane assignment infeasible (synthesizer failure) | analysis | `move_to_placements` returns `bottom` (§2.2 step 5); post-compile validator surfaces |
| 9 | Qubit already at requested location | n/a | silent no-op — empty layer emitted; user-visible behavior identical to omitting that `(qubit, location)` pair |

Note on failure #6: the structural constraint "move_to must precede a CZ"
cannot be checked eagerly on the current IR. Pipeline-level enforcement via
`bottom` is the correct mechanism for v1.

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

### 4.2 — Analysis failures (failures 6-8)

The placement analysis returns `AtomState.bottom()` for each failure:
- **#6**: `sq_placements(UserMoved)` or `measure_placements(UserMoved)` (§2.3)
- **#7**: `move_to_placements` occupancy check (§2.2 step 2)
- **#8**: `move_to_placements` synthesizer failure (§2.2 step 5)

In all cases the existing post-compile lanes validator notices the `bottom`
at the statement's `state_after` and emits a kernel-level diagnostic. No
new diagnostic-emission code in the strategy itself.

### 4.3 — Already-at-destination (failure 9)

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

### 5.2 — `arch_spec` parameter and validation registration

`run_pass` gains an `arch_spec` parameter so `MoveToValidation` can
range-check `LocationAddress` values against the architecture:

```python
from bloqade.lanes.arch.gemini.logical import get_arch_spec

def run_pass(
    self,
    ...,
    arch_spec: ArchSpec | None = None,
    ...
) -> ...:
    if arch_spec is None:
        arch_spec = get_arch_spec()
    ...
```

The existing `ValidationSuite` gains `MoveToValidation` as an
**instance** (not a class — it requires the `arch_spec` at construction):

```python
validator = ValidationSuite([
    GeminiLogicalValidation(),
    GeminiTerminalMeasurementValidation(),
    FlatKernelNoCloningValidation(),
    DuplicateAddressValidation(),
    MoveToValidation(arch_spec=arch_spec),   # ← new — instance with arch_spec
])
```

`MoveToValidation` lives in `python/bloqade/gemini/common/validation/move_to.py`
and follows the structural shape of `bloqade.lanes.validation.Validation`
— a `ValidationPass` subclass with `arch_spec: ArchSpec = field(kw_only=True)`
whose internal `_ValidationAnalysis` receives the `arch_spec` at run time.
It registers the per-statement method-table impl described in §4.1.

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
`bloqade.lanes._prelude` / `bloqade.lanes.prelude` dialect groups are
**not** extended — lanes-only kernels still don't see `movement.MoveTo`.

### 5.4 — Backward compatibility

Existing kernels that never reference `movement.move_to` produce
byte-identical IR before vs after this change:

- The dialect being in the group's union but unused costs only dialect-
  registration overhead at kernel construction;
- `MoveToValidation` is a per-statement impl that runs only against
  `movement.MoveTo` instances — kernels without those statements emit
  no diagnostics and no extra work;
- The placement-analysis interpreter for `place.MoveTo` likewise only
  runs against that statement type; the post-rewrite IR for a non-
  movement kernel contains zero `place.MoveTo` statements.

The regression test in §7.9 (existing demo corpus → byte-identical move
IR) is the canary for this property.

## §6 — Place→Move emission

### 6.1 — `InsertMoves` (minimal edit)

`InsertMoves` (`python/bloqade/lanes/rewrite/place2move.py`) requires a
small addition to support the `multi_move_warning` attribute. Its core
polymorphic dispatch — emitting forward moves via
`state_after.get_move_layers()` and reverse moves via
`state_after.get_reverse_moves()` — is unchanged. With the §2/§3 lattice
changes:

- `UserMoved.get_move_layers()` returns `move_layers` (just this
  MoveTo's portion), so each `place.MoveTo` emits exactly its own
  forward Move IR.
- `UserMoved.get_reverse_moves()` returns `()` (inherited from
  `AtomState`), so no return moves emit at the MoveTo site.
- `ExecuteCZ` (compiler pairing without palindrome) is unchanged.
- `ExecuteCZReturn` carries `move_layers` (compiler pairing only) for
  the forward emit, and `return_move_layers = compiler_reverse +
  user_reverse` for the return emit (computed in `__post_init__`).

The one new code path in `InsertMoves.rewrite_Statement`: after emitting
move IR at a `place.MoveTo` site, check whether the transport required
more than one AOD shot and warn if the statement requests it:

```python
if isinstance(stmt, place.MoveTo) and stmt.multi_move_warning:
    layers = state_after.get_move_layers()
    if len(layers) > 1:
        warnings.warn(
            f"movement.move_to(qubits={stmt.qubits}, "
            f"locations={stmt.locations}) could not be packed into a "
            f"single AOD shot (split across {len(layers)} layers)",
            UserWarning,
        )
```

### 6.2 — `RewriteGates` (new handler for `place.MoveTo`)

After `InsertMoves` emits the forward moves at a `place.MoveTo` site,
the `place.MoveTo` node remains in the IR inside its `StaticPlacement`
region. `RewriteGates` gains a handler that deletes it:

```python
@stmts_to_insert.register(place.MoveTo)
def _(node: place.MoveTo) -> list[ir.Statement]:
    node.erase()   # or equivalent removal API
    return []
```

After deletion, the `StaticPlacement` contains only its `Yield` statement
and is eligible for removal by the existing `RemoveNoOpStaticPlacements`
pass — no additional cleanup needed.

### 6.3 — Inter-call coalescing

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
- `locations` IList contains a non-const `LocationAddress` element → error.
- Out-of-range `(zone_id, word_id, site_id)` → range error.
- Two identical `LocationAddress` values in one `locations` IList →
  duplicate-destination error.
- Same `Qubit` SSA value appearing twice in one `qubits` IList →
  duplicate-qubit error.
- Negative cases: valid single-qubit move, valid multi-qubit move,
  already-at-destination no-op, permutation among moved qubits → eager
  validation passes.

### 7.2 — `movement.MoveTo` → `place.MoveTo` rewrite (§1)

- Literal-IList kernel post-ConstantFold → rewrite produces a
  `place.MoveTo` with correct `qubits`, `locations`, and
  `multi_move_warning` attribute copied from the source.
- Const hint absent on `locations` → rewrite returns `RewriteResult()`
  (defers, doesn't raise).

### 7.3 — Placement-analysis interpreter and strategy (§2)

- Single `move_to` from a fresh `ConcreteState` → strategy returns
  `UserMoved` with correct `move_layers`, `accumulated_move_layers ==
  move_layers`, `pre_user_layout == input_state.layout`.
- Sequence of two `move_to` calls (no intervening op) → second
  `UserMoved` has `accumulated_move_layers == first.move_layers +
  second.move_layers`, `pre_user_layout == first.pre_user_layout`
  (unchanged), per-call `move_layers` contains only second call's layers.
- `move_to` followed by a non-CZ op (`place.R` / `place.Rz`) →
  `sq_placements(UserMoved)` returns `AtomState.bottom()`. Confirms
  the §2.3 invalid-program path.
- `move_to` followed by `place.EndMeasure` → `measure_placements(UserMoved)`
  returns `AtomState.bottom()`.
- "Swap into unmoved slot" → strategy returns `AtomState.bottom()`
  (occupancy precondition, §2.2 step 2).
- "Permutation among moved qubits" → strategy returns `UserMoved`
  (destinations held by qubits *in* the call are allowed).
- Synthesizer infeasibility → strategy returns `AtomState.bottom()`.
- Non-`ConcreteState` input (top or bottom) propagates unchanged.

### 7.4 — Palindrome interaction (§3)

- `move_to` + CZ under the default (non-palindrome) placement strategy →
  `ExecuteCZ` whose `move_layers` is only the compiler-pairing moves.
- `move_to` + CZ under `PalindromePlacementStrategy` →
  `ExecuteCZReturn` with `user_move_layers ==
  UserMoved.accumulated_move_layers`, `initial_layout ==
  UserMoved.pre_user_layout`, `return_move_layers == compiler_reverse +
  user_reverse`.
- `move_to` placing qubits into CZ-compatible positions → compiler
  synthesizes zero additional pairing moves; palindrome return reverses
  only the user moves.
- Sequence: `move_to` + CZ + `move_to` + CZ — verifies the lattice
  resets between CZs (second segment's `pre_user_layout` is the home
  position after the first palindrome, not the first segment's
  `pre_user_layout`).

### 7.5 — Place→Move emission (§6)

- Each scenario from §7.4 lowered through `InsertMoves` → resulting
  `move.Move` statements in the expected order.
- At each `place.MoveTo`: forward triple for `UserMoved.move_layers`.
- At each `place.CZ` (default): forward triple for `ExecuteCZ.move_layers`.
- At each `place.CZ` (palindrome): forward triple for
  `ExecuteCZReturn.move_layers` before; return triple for
  `return_move_layers` after.
- `place.MoveTo` node is absent from the IR after `RewriteGates` runs
  (verify the enclosing `StaticPlacement` is removed by
  `RemoveNoOpStaticPlacements`).

### 7.6 — `multi_move_warning` attribute (§6.1)

- A `move_to` call that packs into one AOD shot → no `UserWarning` emitted.
- A `move_to` call requiring N > 1 AOD shots with `multi_move_warning=True`
  (default) → one `UserWarning` emitted via `pytest.warns(UserWarning)`.
- Same call with `multi_move_warning=False` → no `UserWarning` emitted.
- Confirm the warning can be upgraded to failure with `-W error::UserWarning`.

### 7.7 — `gemini.logical.kernel` extension (§5)

- Compile a kernel that uses both `squin.*` and `movement.move_to`
  under the existing `gemini.logical.kernel` decorator → succeeds.
- Compile a kernel that uses `movement.move_to` under a bare
  `bloqade.lanes.prelude` group → fails at dialect-group validation.
- A kernel with a length-mismatched `move_to` call trips `MoveToValidation`.

### 7.8 — Composition with `new_at`

- `move_to` a qubit placed via explicit `new_at(z, w, s)` → succeeds;
  qubit moves from its pinned position to the requested destination.
  Confirms explicit-allocation and movement-intent paths compose cleanly.

### 7.9 — Regression

- Existing demo kernels under `demo/` (zero `movement.move_to` usage)
  compile to byte-identical move IR before vs after this change. This
  is the primary canary that:
  - the new `movement` dialect in the kernel group's union does not
    perturb output for non-movement kernels,
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
│   │                                     #        statement (user-facing;
│   │                                     #        qubits/locations IList SSA
│   │                                     #        operands + multi_move_warning
│   │                                     #        bool attribute)
│   └── place.py                          # EDIT — (1) add place.MoveTo
│                                         #        (QuantumStmt; attribute
│                                         #        qubits/locations/
│                                         #        multi_move_warning)
│                                         #        (2) add @interp.impl(place.MoveTo)
│                                         #        to PlacementMethods class
├── analysis/placement/
│   ├── lattice.py                        # EDIT — add UserMoved lattice
│   │                                     #        element; extend
│   │                                     #        ExecuteCZReturn with
│   │                                     #        user_move_layers field
│   └── strategy.py                       # EDIT — add move_to_placements
│                                         #        to PlacementStrategyABC;
│                                         #        sq/measure return bottom
│                                         #        for UserMoved input;
│                                         #        palindrome cz_placements
│                                         #        splice
├── rewrite/circuit2place.py              # EDIT — RewritePlaceOperations
│                                         #        gains rewrite_MoveTo
└── rewrite/place2move.py                 # EDIT — InsertMoves gains
                                          #        multi_move_warning check;
                                          #        RewriteGates gains
                                          #        place.MoveTo delete handler

python/bloqade/gemini/
├── logical/group.py                      # EDIT — (1) union movement dialect
│                                         #        into kernel
│                                         #        (2) add arch_spec param
│                                         #        to run_pass (default
│                                         #        get_arch_spec())
│                                         #        (3) register
│                                         #        MoveToValidation instance
└── common/validation/move_to.py          # NEW — eager validation impl
                                          #        (failures 1-5 from §4;
                                          #        arch_spec: ArchSpec field)
```

## Open questions

- **Layer granularity in `UserMoved`.** Each `move_to` call contributes
  one or more layers. Two adjacent calls produce two separate layer
  groups even when their transports could pack into one AOD shot. v1
  defers coalescing; the open question is whether the lattice element
  should pre-emptively flatten to a single tuple or preserve per-call
  grouping (e.g. for diagnostics that map back to the source call).
- **`MultiShotWarning` granularity (§6.1).** If the Rust-backed
  `compute_move_layers` could identify *which subset* of pairs forced
  the split, the `UserWarning` message would be more actionable — but
  this requires the synthesizer to surface per-layer membership. Tracked
  here so the synthesizer side knows there's a consumer if it ever
  exposes it.
- **`movement.MoveTo` interaction with `place.NewLogicalQubit`.** Can a
  user `move_to` a qubit placed via explicit `new_at(z, w, s)`? The
  semantics are well-defined (qubit moves from its pinned position to the
  new destination); §7.8 is the regression that confirms the paths
  compose cleanly.

## Risk register

- **Highest risk:** the new `UserMoved` lattice element join semantics.
  If two control-flow paths produce different `move_layers`,
  `accumulated_move_layers`, or `pre_user_layout`, the join goes to
  `AnyState` (top), halting analysis. v1 only supports straight-line
  kernels (matching today's Squin), so this is theoretical — but future
  branching support has work to do here.
- **High risk:** `sq_placements` / `measure_placements` isinstance
  ordering. The `UserMoved` branch MUST appear before the `ConcreteState`
  branch. A misordering silently strips `UserMoved` to `ConcreteState`
  instead of returning `bottom`, letting invalid programs through. The
  §7.3 "move_to followed by R/EndMeasure" tests catch this directly.
- **Medium risk:** `PalindromePlacementStrategy.cz_placements`
  regression. The change is narrow (handle a `UserMoved` input branch
  + populate `user_move_layers` on `ExecuteCZReturn`) but the strategy
  is load-bearing for every palindrome-using kernel today. The §7.9
  byte-identical regression is the gate.
- **Medium risk:** `ExecuteCZReturn.__post_init__` change. When
  `user_move_layers` defaults to `()` the result is bit-for-bit
  identical to the original — but any direct construction of
  `ExecuteCZReturn` outside `PalindromePlacementStrategy` (e.g. in
  tests) must not break. The field default and §7.9 regression gate this.
- **Low risk:** `RewriteGates` handler for `place.MoveTo`. The handler
  simply deletes the node; if accidentally omitted, `place.MoveTo`
  nodes remain in the IR and later passes will encounter unexpected
  statement types. The §7.5 test (verify `place.MoveTo` absent after
  `RewriteGates`) catches this.
