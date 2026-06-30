# Permute qubit locations — design

**Date:** 2026-06-23
**Status:** Approved (ready for implementation plan)

## Goal

Add a user-directed `permute` statement to the Gemini movement dialect that
moves a set of qubits into a **permutation of their own current locations** —
a simultaneous swap/cycle. Unlike `move_to`, the user does not name target
`LocationAddress`es; they give an index permutation and the targets are derived
from where the moving qubits currently are.

```python
@physical.kernel
def k():
    q = squin.qalloc(3)
    permute([q[0], q[1], q[2]], [1, 2, 0])   # q0→q1's spot, q1→q2's spot, q2→q0's spot
    squin.cz(q[0], q[1])
```

Semantics: `qubits[i]` moves to the location currently occupied by
`qubits[perm[i]]`.

## Background

The movement dialect already provides user-directed movement via `move_to`:

- `movement.MoveTo` statement (`qubits`, `locations` as SSA ilists) →
  `movement.move_to` interface.
- `circuit2place.RewritePlaceOperations.rewrite_MoveTo` lowers it to
  `place.MoveTo` (`qubits: tuple[int,...]`, `locations: tuple[LocationAddress,...]`
  as resolved attributes).
- The placement interpreter (`dialects/place.py` `impl_move_to`) gates on
  `MoveToPlacementStrategyABC` and calls `strategy.move_to_placements(state,
  qubits, locations)`, which routes via `compute_moves`, accumulates a
  `UserMoved` state, and is palindrome-returned by `PalindromePlacementStrategy`.
- `MoveToValidation` (eager, at decoration time) checks `move_to` inputs.

Two facts make `permute` a thin addition on top of this:

1. **Targets need the current layout.** The permutation's destination
   `LocationAddress`es are the moving qubits' own current positions, which are
   only known inside the placement analysis (the threaded `AtomState`). So
   `permute` cannot be statically desugared to `move_to` at a rewrite stage; the
   targets must be resolved during placement interpretation.
2. **`move_to_placements` already accepts a permutation.** Its occupancy guard
   rejects a destination only if held by an *unmoved* qubit
   (`idx not in moved_set`). In a permutation every destination is held by a
   qubit in the moving set, so the guard passes — the simultaneous swap is
   already legal once the targets are resolved.

Confirmed design decisions (from brainstorming):

- API: index permutation `permute(qubits, perm)`, `qubits[i] → loc(qubits[perm[i]])`.
- `perm` is a **compile-time-constant bijection**, validated eagerly.
- `permute` is the **same lifecycle family as `move_to`**: it extends the same
  `UserMoved` state, composes with `move_to`/SQ gates in an inter-CZ segment, and
  is palindrome-returned.
- Implementation: **Approach A** — distinct user-facing and place-layer
  statements, but the placement interpreter resolves the permutation to target
  locations and **delegates to the existing `move_to_placements`** (no new
  strategy method).

## Design

### 1. Movement-dialect statement + interface

`python/bloqade/gemini/common/dialects/movement/stmts.py`:

```python
@statement(dialect=dialect)
class Permute(ir.Statement):
    """User-facing permute directive: move qubits into a permutation of their
    own current locations. qubits[i] moves to the current location of
    qubits[perm[i]]."""

    name = "permute"
    traits = frozenset({lowering.FromPythonCall()})
    qubits: ir.SSAValue = info.argument(type=ilist.IListType[QubitType, Len])
    perm: ir.SSAValue = info.argument(type=ilist.IListType[types.Int, Len])
```

`python/bloqade/gemini/common/dialects/movement/_interface.py` — add a
`@lowering.wraps(Permute)` `permute` with typed overloads mirroring `move_to`
(`IList[Qubit, Len]` / `IList[int, Len]`, and the `list[...]` form), exported
from the movement package `__init__`.

`perm` is a Python list lowered to an ilist; it is const-folded during
`circuit2place` (same mechanism as `move_to`'s `locations`).

### 2. Place-layer statement + lowering

`python/bloqade/lanes/dialects/place.py`:

```python
@statement(dialect=dialect)
class Permute(QuantumStmt):
    """Place-layer permute directive. Produced by
    RewritePlaceOperations.rewrite_Permute from movement.Permute; consumed by
    placement analysis (delegates to move_to_placements) and deleted by
    RewriteGates after InsertMoves emits the forward Move IR."""

    qubits: tuple[int, ...] = info.attribute()
    perm: tuple[int, ...] = info.attribute()
```

`python/bloqade/lanes/rewrite/circuit2place.py`:

- Add `rewrite_Permute(node: movement.stmts.Permute)` mirroring `rewrite_MoveTo`:
  resolve the SSA `qubits` → integer indices via the existing address-analysis
  path, const-fold the `perm` ilist → `tuple[int,...]`, and emit `place.Permute`.
- Add `place.Permute` to **every** place where `place.MoveTo` is special-cased
  (the MoveTo-only-block detection/merge logic and the iterables listing the
  user-movement place statements), so a `permute` block is treated as
  user-directed movement that precedes a gate block — exactly like a `move_to`
  block.
- `RewriteGates` / `InsertMoves` delete `place.Permute` after the forward Move IR
  is emitted, the same as `place.MoveTo`.

### 3. Placement interpretation (the only new behavior)

`python/bloqade/lanes/dialects/place.py` — add `impl_permute`:

```python
@interp.impl(Permute)
def impl_permute(self, _interp, frame, stmt: Permute):
    strategy = _interp.placement_strategy
    if not isinstance(strategy, MoveToPlacementStrategyABC):
        return (AtomState.bottom(),)
    state = frame.get(stmt.state_before)
    if not isinstance(state, ConcreteState):
        return (state,)
    layout = state.layout
    locations = tuple(
        layout[stmt.qubits[stmt.perm[i]]] for i in range(len(stmt.qubits))
    )
    new_state = strategy.move_to_placements(state, stmt.qubits, locations)
    return (new_state,)
```

No new strategy method: `move_to_placements` provides `compute_moves` routing,
`UserMoved` accumulation, segment composition, and palindrome return.

### 4. Validation (eager, `PermuteValidation`)

`python/bloqade/gemini/common/validation/permute.py` — mirror
`MoveToValidation`. It does **not** need `arch_spec` (targets are existing,
in-range locations by construction), so it is a plain `ValidationSuite` member.
Failure classes:

- **P1**: `perm` is not a compile-time constant.
- **P2**: `len(perm) != len(qubits)`.
- **P3**: `perm` is not a bijection of `{0..n-1}` (an index is out of range or
  appears more than once).
- **P4**: the same qubit SSA value appears twice in `qubits`.

Wire `PermuteValidation` into the movement kernel's validation alongside
`MoveToValidation` (in `bloqade.gemini.physical.group.kernel`'s verify suite,
and any other kernel that includes the movement dialect + validation).

## Files

| File | Change |
|------|--------|
| `python/bloqade/gemini/common/dialects/movement/stmts.py` | Add `Permute` statement |
| `python/bloqade/gemini/common/dialects/movement/_interface.py` | Add `permute` wrapper + overloads |
| `python/bloqade/gemini/common/dialects/movement/__init__.py` | Export `permute` |
| `python/bloqade/lanes/dialects/place.py` | Add `place.Permute` statement + `impl_permute` interpreter |
| `python/bloqade/lanes/rewrite/circuit2place.py` | Add `rewrite_Permute`; treat `place.Permute` like `place.MoveTo` in block-merge / gate-rewrite paths |
| `python/bloqade/gemini/common/validation/permute.py` | New: `PermuteValidation` (P1–P4) |
| `python/bloqade/gemini/physical/group.py` (and any other movement kernel) | Wire `PermuteValidation` into the verify suite |
| `python/tests/...` | Tests (below) |

No new placement-strategy method; `impl_permute` reuses `move_to_placements`.

## Testing

- **Statement shape**: `movement.Permute` and `place.Permute` field/trait checks.
- **Lowering**: `circuit2place` turns `movement.Permute` into `place.Permute`
  with `perm` const-folded to `tuple[int,...]` and `qubits` resolved to indices.
- **Placement interpretation**:
  - `permute` produces the expected `UserMoved` target layout (each qubit at the
    permuted current location).
  - Identity `perm` (`[0,1,...,n-1]`) is a no-op (no net relocation).
  - `permute` composes with a following `move_to` in the same inter-CZ segment
    (`UserMoved` accumulates both).
  - Under `PalindromePlacementStrategy`, a `permute` segment is palindrome-returned.
- **Validation**: P1 (non-const perm), P2 (length mismatch), P3 (non-bijection:
  out-of-range and duplicate index), P4 (duplicate qubit) each rejected eagerly;
  a valid permutation passes.
- **End-to-end** (`@physical.kernel`): `permute → cz` compiles; a 2-cycle swap
  routes through the mover (or fails gracefully to `bottom` if the mover cannot
  stage it — see limitation).

## Out of scope / limitations

- **Cycle/swap routing depends on the mover.** Exchanging two atoms' positions
  requires `compute_moves` (the Rust solver) to stage atoms through an
  intermediate location. If it cannot, that solve returns infeasible (`bottom`)
  — the same failure mode as an impossible `move_to`. `permute` does not add new
  routing capability; it only derives targets and reuses the existing router.
- No runtime (non-constant) `perm`.
- No partial permutations — `perm` must cover exactly the given `qubits`
  (fixed points / identity entries are allowed, since identity is a permutation).
- No standalone `swap(a, b)` sugar (a 2-element `permute` covers it); can be
  added later if desired.
