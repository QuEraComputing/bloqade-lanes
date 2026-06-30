# Permute qubit locations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a user-directed `permute(qubits, perm)` statement to the Gemini movement dialect that moves qubits into a permutation of their own current locations (`qubits[i]` → current location of `qubits[perm[i]]`).

**Architecture:** Approach A — distinct `movement.Permute` and `place.Permute` statements, but the place-layer interpreter resolves the permutation against the current layout into target `LocationAddress`es and delegates to the existing `move_to_placements`, inheriting `UserMoved` accumulation, `compute_moves` routing, and palindrome return. The only new logic is the permutation→locations resolution (extracted into a small pure helper) plus eager validation.

**Tech Stack:** Python, Kirin IR (dialects/statements/interpreter method tables, rewrites, validation), `uv` + `pytest`. Spec: `docs/superpowers/specs/2026-06-23-permute-qubit-locations-design.md`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `python/bloqade/gemini/common/dialects/movement/stmts.py` | **Modify** — add `Permute` statement |
| `python/bloqade/gemini/common/dialects/movement/_interface.py` | **Modify** — add `permute` wrapper + overloads |
| `python/bloqade/gemini/common/dialects/movement/__init__.py` | **Modify** — export `permute` |
| `python/bloqade/lanes/dialects/place.py` | **Modify** — add `place.Permute` statement, `_resolve_permute_locations` helper, `impl_permute` interpreter |
| `python/bloqade/lanes/rewrite/circuit2place.py` | **Modify** — `rewrite_Permute` + treat `place.Permute` like `place.MoveTo` in the dispatch filter, block-merge type tuples, clone/attribute-remap regions, and `_is_move_to_only_block` |
| `python/bloqade/gemini/common/validation/move_to.py` | **Modify** — add `@interp.impl(Permute)` validation (P1–P4) to the existing `move.address.validation` table |
| `python/tests/gemini/test_permute.py` | **Create** — all permute tests |

**Validation note (refinement of spec §4):** the spec proposed a separate `permute.py` `PermuteValidation` ValidationSuite member. The codebase's actual per-statement validation mechanism is the `move.address.validation` interpreter key, run by `_ValidationAnalysis` (invoked by the `MoveToValidation` pass the movement kernel already calls in `bloqade.gemini.physical.group.kernel`). So we register the `Permute` check in that same table — it is then validated by the existing pass with **no `group.py` change** and without needing `arch_spec` (the permute checks don't use it). This satisfies the spec's intent (eager P1–P4 at decoration time).

Lint after each task: `uv run black python && uv run isort python && uv run ruff check <changed files> && uv run pyright <changed files>`. Ignore unrelated pre-existing `cirq` test failures elsewhere in the suite.

---

## Task 1: `movement.Permute` statement + `permute` interface + export

**Files:**
- Modify: `python/bloqade/gemini/common/dialects/movement/stmts.py`
- Modify: `python/bloqade/gemini/common/dialects/movement/_interface.py`
- Modify: `python/bloqade/gemini/common/dialects/movement/__init__.py`
- Test (create): `python/tests/gemini/test_permute.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/gemini/test_permute.py`:

```python
"""Tests for the movement-dialect permute(qubits, perm) feature."""

from kirin import ir, lowering
from kirin.dialects import ilist

from bloqade.gemini.common.dialects import movement
from bloqade.gemini.common.dialects.movement.stmts import Permute


def test_permute_statement_shape():
    assert issubclass(Permute, ir.Statement)
    assert Permute.name == "permute"
    # Lowered from a Python call; NOT pure (it mutates placement state).
    assert any(isinstance(t, lowering.FromPythonCall) for t in Permute.traits)
    assert not any(isinstance(t, ir.Pure) for t in Permute.traits)


def test_permute_interface_exported():
    # `permute` is exported from the movement package and wraps Permute.
    assert movement.permute is not None
    assert callable(movement.permute)
```

- [ ] **Step 2: Run it, verify it FAILS**

Run: `uv run pytest python/tests/gemini/test_permute.py -v`
Expected: FAIL — `ImportError: cannot import name 'Permute'`.

- [ ] **Step 3: Add the `Permute` statement**

In `python/bloqade/gemini/common/dialects/movement/stmts.py`, append after the `MoveTo` class (the file already imports `ir, lowering, types`, `QubitType`, `ilist`, `info`, `statement`, and defines `Len`):

```python
@statement(dialect=dialect)
class Permute(ir.Statement):
    """User-facing permute directive: move qubits into a permutation of their
    own current locations. qubits[i] moves to the current location of
    qubits[perm[i]].

    Lowered from a Python call by FromPythonCall. RewritePlaceOperations rewrites
    this to a StaticPlacement(place.Permute) after const-folding the perm ilist.
    """

    name = "permute"
    traits = frozenset({lowering.FromPythonCall()})
    qubits: ir.SSAValue = info.argument(type=ilist.IListType[QubitType, Len])
    perm: ir.SSAValue = info.argument(type=ilist.IListType[types.Int, Len])
```

- [ ] **Step 4: Add the `permute` interface wrapper**

In `python/bloqade/gemini/common/dialects/movement/_interface.py`, change the stmts import to include `Permute` and append the wrapper after `move_to`:

Change:
```python
from .stmts import Loc, MoveTo
```
to:
```python
from .stmts import Loc, MoveTo, Permute
```

Append (after the `move_to` definition):
```python
@overload
def permute(
    qubits: ilist.IList[Qubit, Len],
    perm: ilist.IList[int, Len],
): ...


@overload
def permute(
    qubits: list[Qubit],
    perm: list[int],
): ...


@lowering.wraps(Permute)
def permute(
    qubits,
    perm,
) -> None:
    """Move qubits into a permutation of their own current locations.

    ``qubits[i]`` moves to the location currently held by ``qubits[perm[i]]``.
    ``perm`` must be a compile-time-constant permutation of
    ``range(len(qubits))``. Like ``move_to``, this is user-directed movement
    within an inter-CZ segment.
    """
    ...
```

- [ ] **Step 5: Export `permute`**

In `python/bloqade/gemini/common/dialects/movement/__init__.py`, change:
```python
from ._interface import loc as loc, move_to as move_to
```
to:
```python
from ._interface import loc as loc, move_to as move_to, permute as permute
```

- [ ] **Step 6: Run the test, verify it PASSES**

Run: `uv run pytest python/tests/gemini/test_permute.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Lint**

Run: `uv run black python && uv run isort python && uv run ruff check python/bloqade/gemini/common/dialects/movement/ python/tests/gemini/test_permute.py && uv run pyright python/bloqade/gemini/common/dialects/movement/`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add python/bloqade/gemini/common/dialects/movement/ python/tests/gemini/test_permute.py
git commit -m "feat(gemini): add movement.permute(qubits, perm) statement + interface"
```

---

## Task 2: `place.Permute` statement + resolution helper + `impl_permute`

**Files:**
- Modify: `python/bloqade/lanes/dialects/place.py`
- Test: `python/tests/gemini/test_permute.py`

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/gemini/test_permute.py`. First extend the imports at the top of the file so they include:

```python
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import place
from bloqade.lanes.dialects.place import _resolve_permute_locations
```

Then append:

```python
def _loc(w, s):
    return LocationAddress(zone_id=0, word_id=w, site_id=s)


def test_place_permute_statement_shape():
    stmt = place.Permute(ir.TestValue(), qubits=(0, 1, 2), perm=(1, 2, 0))
    assert stmt.qubits == (0, 1, 2)
    assert stmt.perm == (1, 2, 0)


def test_resolve_permute_locations_cycle():
    # layout: qubit 0@(0,0), 1@(1,0), 2@(2,0); perm [1,2,0]
    # qubit 0 -> loc of qubit 1, qubit 1 -> loc of qubit 2, qubit 2 -> loc of 0
    layout = (_loc(0, 0), _loc(1, 0), _loc(2, 0))
    locations = _resolve_permute_locations(layout, qubits=(0, 1, 2), perm=(1, 2, 0))
    assert locations == (_loc(1, 0), _loc(2, 0), _loc(0, 0))


def test_resolve_permute_locations_identity():
    layout = (_loc(0, 0), _loc(1, 0))
    locations = _resolve_permute_locations(layout, qubits=(0, 1), perm=(0, 1))
    assert locations == (_loc(0, 0), _loc(1, 0))


def test_resolve_permute_locations_remapped_indices():
    # qubits are global layout indices (StaticPlacement merging remaps them);
    # perm indexes positions within the qubits tuple.
    layout = (_loc(9, 0), _loc(0, 0), _loc(1, 0), _loc(2, 0))
    locations = _resolve_permute_locations(layout, qubits=(1, 2, 3), perm=(2, 0, 1))
    # position0 -> loc of qubits[2]=index3=(2,0); pos1 -> qubits[0]=index1=(0,0);
    # pos2 -> qubits[1]=index2=(1,0)
    assert locations == (_loc(2, 0), _loc(0, 0), _loc(1, 0))
```

- [ ] **Step 2: Run, verify it FAILS**

Run: `uv run pytest python/tests/gemini/test_permute.py -v -k "place_permute or resolve_permute"`
Expected: FAIL — `cannot import name 'Permute'`/`_resolve_permute_locations` from `place`.

- [ ] **Step 3: Add the `place.Permute` statement**

In `python/bloqade/lanes/dialects/place.py`, append after the `MoveTo` statement (around line 163):

```python
@statement(dialect=dialect)
class Permute(QuantumStmt):
    """Place-layer permute directive.

    Produced by RewritePlaceOperations.rewrite_Permute from movement.Permute.
    Consumed by placement analysis (the interpreter resolves the permutation
    against the current layout and delegates to move_to_placements, producing
    UserMoved) and deleted by RewriteGates after InsertMoves emits the forward
    Move IR. ``perm`` is a permutation over the positions of ``qubits``.
    """

    qubits: tuple[int, ...] = info.attribute()
    perm: tuple[int, ...] = info.attribute()
```

- [ ] **Step 4: Add the resolution helper**

In `python/bloqade/lanes/dialects/place.py`, add a module-level pure helper (place it near the top after the imports / `dialect` definition, before the statement classes — `LocationAddress` is already imported):

```python
def _resolve_permute_locations(
    layout: tuple[LocationAddress, ...],
    qubits: tuple[int, ...],
    perm: tuple[int, ...],
) -> tuple[LocationAddress, ...]:
    """Resolve a permutation to target locations against the current layout.

    ``qubits`` are layout indices; ``perm`` permutes the *positions* of
    ``qubits``. The target for position ``i`` is the current location of the
    qubit at position ``perm[i]`` (i.e. ``layout[qubits[perm[i]]]``).
    """
    return tuple(layout[qubits[perm[i]]] for i in range(len(qubits)))
```

- [ ] **Step 5: Add the `impl_permute` interpreter**

In `python/bloqade/lanes/dialects/place.py`, add immediately after the `impl_move_to` method in the placement interpreter method table (it already imports `MoveToPlacementStrategyABC`, `ConcreteState`, `AtomState`, `PlacementAnalysis`, `ForwardFrame`, and `interp`):

```python
    @interp.impl(Permute)
    def impl_permute(
        self,
        _interp: PlacementAnalysis,
        frame: ForwardFrame[AtomState],
        stmt: Permute,
    ):
        strategy = _interp.placement_strategy
        if not isinstance(strategy, MoveToPlacementStrategyABC):
            # Strategy does not support user-directed movement.
            return (AtomState.bottom(),)
        state = frame.get(stmt.state_before)
        if not isinstance(state, ConcreteState):
            return (state,)
        locations = _resolve_permute_locations(state.layout, stmt.qubits, stmt.perm)
        new_state = strategy.move_to_placements(state, stmt.qubits, locations)
        return (new_state,)
```

- [ ] **Step 6: Run the tests, verify PASS**

Run: `uv run pytest python/tests/gemini/test_permute.py -v -k "place_permute or resolve_permute"`
Expected: PASS (4 tests).

- [ ] **Step 7: Lint**

Run: `uv run black python && uv run isort python && uv run ruff check python/bloqade/lanes/dialects/place.py python/tests/gemini/test_permute.py && uv run pyright python/bloqade/lanes/dialects/place.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add python/bloqade/lanes/dialects/place.py python/tests/gemini/test_permute.py
git commit -m "feat(lanes): place.Permute statement + interpreter delegating to move_to_placements"
```

---

## Task 3: `circuit2place` lowering (`rewrite_Permute` + special-cases)

**Files:**
- Modify: `python/bloqade/lanes/rewrite/circuit2place.py`
- Test: `python/tests/gemini/test_permute.py`

- [ ] **Step 1: Write the failing test**

Append to `python/tests/gemini/test_permute.py`. Add these imports at the top:

```python
from kirin import rewrite
from kirin.analysis import const
from kirin.dialects import py
from bloqade.lanes.rewrite.circuit2place import RewritePlaceOperations
```

Then append:

```python
def test_rewrite_permute_produces_place_permute():
    q0, q1, q2 = ir.TestValue(), ir.TestValue(), ir.TestValue()
    qubits_new = ilist.New(values=(q0, q1, q2))

    # perm [1, 2, 0] as a const-hinted ilist of ints
    p0, p1, p2 = py.Constant(1), py.Constant(2), py.Constant(0)
    perm_new = ilist.New(values=(p0.result, p1.result, p2.result))
    perm_new.result.hints["const"] = const.Value((1, 2, 0))

    stmt = movement.stmts.Permute(qubits_new.result, perm_new.result)

    # Only statements go in the block; q0/q1/q2 are SSA operands (TestValues).
    block = ir.Block([qubits_new, p0, p1, p2, perm_new, stmt])
    region = ir.Region([block])

    rewrite.Walk(RewritePlaceOperations()).rewrite(region)

    permutes = [s for s in region.walk() if isinstance(s, place.Permute)]
    assert len(permutes) == 1
    assert permutes[0].qubits == (0, 1, 2)
    assert permutes[0].perm == (1, 2, 0)
    # original movement.Permute is gone
    assert not any(isinstance(s, movement.stmts.Permute) for s in region.walk())
```

(If `ir.TestValue()` instances cannot be added directly to a block in this codebase's harness, mirror the exact block-construction the existing `test_rewrite_move_to_produces_static_placement` in `python/tests/rewrite/test_movement_rewrite.py` uses — read it and follow the same pattern for assembling `q*`, `ilist.New`, `py.Constant`, and the statement.)

- [ ] **Step 2: Run, verify it FAILS**

Run: `uv run pytest python/tests/gemini/test_permute.py -v -k rewrite_permute`
Expected: FAIL — no `place.Permute` produced (and/or `AttributeError: 'RewritePlaceOperations' object has no attribute 'rewrite_Permute'` once `Permute` is added to the dispatch filter).

- [ ] **Step 3: Add `movement.Permute` to the dispatch filter**

In `python/bloqade/lanes/rewrite/circuit2place.py`, in `RewritePlaceOperations.rewrite_Statement` (around line 186), add `movement_dialect.stmts.Permute` to the isinstance tuple so it dispatches to `rewrite_Permute`:

```python
    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not isinstance(
            node,
            (
                gemini_stmts.TerminalLogicalMeasurement,
                gemini_stmts.Initialize,
                gemini_stmts.StarRz,
                gate.CZ,
                gate.R,
                gate.Rz,
                movement_dialect.stmts.MoveTo,
                movement_dialect.stmts.Permute,
            ),
        ):
            return abc.RewriteResult()
        rewrite_method_name = f"rewrite_{type(node).__name__}"
        rewrite_method = getattr(self, rewrite_method_name)
        return rewrite_method(node)
```

- [ ] **Step 4: Add `rewrite_Permute`**

In the same class, add immediately after `rewrite_MoveTo` (around line 377):

```python
    def rewrite_Permute(self, node: movement_dialect.stmts.Permute) -> abc.RewriteResult:
        # qubits: after unrolling, always an ilist.New
        if not isinstance(qubits_list := node.qubits.owner, ilist.New):
            return abc.RewriteResult()

        # perm: must be const-foldable
        perm_hint = node.perm.hints.get("const")
        if not isinstance(perm_hint, const.Value):
            return abc.RewriteResult()

        inputs = qubits_list.values
        perm_attr = tuple(int(p) for p in perm_hint.data)

        body, block, entry_state = self.prep_region()
        permute_stmt = place.Permute(
            entry_state,
            qubits=tuple(range(len(inputs))),
            perm=perm_attr,
        )
        node.replace_by(
            self.construct_execute(permute_stmt, qubits=inputs, body=body, block=block)
        )
        return abc.RewriteResult(has_done_something=True)
```

- [ ] **Step 5: Treat `place.Permute` like `place.MoveTo` in the merge logic**

Make the following edits in `python/bloqade/lanes/rewrite/circuit2place.py`:

(a) `_GATE_STMT_TYPES` tuple (around line 603) — add `place.Permute`:
```python
_GATE_STMT_TYPES = (
    place.R,
    place.Rz,
    place.StarRz,
    place.CZ,
    place.Yield,
    place.MoveTo,
    place.Permute,
)
```

(b) `_is_move_to_only_block` (around line 632) — include `place.Permute`:
```python
def _is_move_to_only_block(sp: place.StaticPlacement) -> bool:
    return all(
        isinstance(stmt, (place.MoveTo, place.Permute, place.Yield))
        for stmt in sp.body.blocks[0].stmts
    )
```

(c) **Both** clone/attribute-remap regions (around lines 540–562 and 695–717) — add `place.Permute` to the isinstance tuple and copy its `perm` attribute. In each region, change the isinstance tuple to include `place.Permute`:
```python
            if isinstance(
                stmt,
                (
                    place.R,
                    place.Rz,
                    place.StarRz,
                    place.CZ,
                    place.EndMeasure,
                    place.Initialize,
                    place.MoveTo,
                    place.Permute,
                ),
            ):
```
and add a branch alongside the existing `place.MoveTo` branch (in both regions):
```python
                if isinstance(stmt, place.Permute):
                    attributes["perm"] = ir.PyAttr(stmt.perm)
```
(The generic `attributes["qubits"]` remap already covers `place.Permute.qubits`; `perm` indexes positions within `qubits`, which the remap preserves, so `perm` is copied verbatim.)

- [ ] **Step 6: Run the test, verify PASS**

Run: `uv run pytest python/tests/gemini/test_permute.py -v -k rewrite_permute`
Expected: PASS.

- [ ] **Step 7: Lint**

Run: `uv run black python && uv run isort python && uv run ruff check python/bloqade/lanes/rewrite/circuit2place.py python/tests/gemini/test_permute.py && uv run pyright python/bloqade/lanes/rewrite/circuit2place.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add python/bloqade/lanes/rewrite/circuit2place.py python/tests/gemini/test_permute.py
git commit -m "feat(lanes): lower movement.Permute to place.Permute in circuit2place"
```

---

## Task 4: Eager validation (P1–P4)

**Files:**
- Modify: `python/bloqade/gemini/common/validation/move_to.py`
- Test: `python/tests/gemini/test_permute.py`

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/gemini/test_permute.py`. Add these imports at the top:

```python
from kirin import types
from kirin.dialects import func
from kirin.prelude import structural_no_opt
from bloqade.gemini.common.dialects.movement import dialect as movement_dialect
from bloqade.gemini.common.validation.move_to import MoveToValidation
from bloqade.lanes.arch.gemini.logical import get_arch_spec
```

Then append the helpers + tests (mirrors `python/tests/gemini/validation/test_move_to_validation.py`):

```python
_VAL_DIALECTS = structural_no_opt.union([movement_dialect])


def _build_method(stmts):
    block = ir.Block(list(stmts) + [func.Return()])
    block.args.append_from(types.PyClass(object), name="self")
    region = ir.Region([block])
    fn = func.Function(
        sym_name="test_fn",
        signature=func.Signature(inputs=(), output=types.NoneType),
        body=region,
    )
    return ir.Method(
        dialects=_VAL_DIALECTS, sym_name="test_fn", arg_names=["self"], code=fn
    )


def _permute_stmts(qubit_vals, perm_data, *, stamp_const=True):
    qubits_new = ilist.New(values=tuple(qubit_vals))
    perm_consts = [py.Constant(p) for p in perm_data]
    perm_new = ilist.New(values=tuple(c.result for c in perm_consts))
    if stamp_const:
        perm_new.result.hints["const"] = const.Value(tuple(perm_data))
    stmt = movement.stmts.Permute(qubits_new.result, perm_new.result)
    # Only statements; qubit_vals are SSA operands (TestValues), not statements.
    return [qubits_new, *perm_consts, perm_new, stmt]


def _errors(stmts):
    _, errs = MoveToValidation(arch_spec=get_arch_spec()).run(_build_method(stmts))
    return errs


def test_permute_valid_passes():
    q0, q1, q2 = ir.TestValue(), ir.TestValue(), ir.TestValue()
    assert _errors(_permute_stmts([q0, q1, q2], [1, 2, 0])) == []


def test_permute_p1_non_const_perm_rejected():
    q0, q1 = ir.TestValue(), ir.TestValue()
    assert len(_errors(_permute_stmts([q0, q1], [1, 0], stamp_const=False))) >= 1


def test_permute_p2_length_mismatch_rejected():
    q0, q1, q2 = ir.TestValue(), ir.TestValue(), ir.TestValue()
    assert len(_errors(_permute_stmts([q0, q1, q2], [1, 0]))) >= 1


def test_permute_p3_not_a_bijection_rejected():
    q0, q1, q2 = ir.TestValue(), ir.TestValue(), ir.TestValue()
    # duplicate index / out of range
    assert len(_errors(_permute_stmts([q0, q1, q2], [0, 0, 1]))) >= 1
    assert len(_errors(_permute_stmts([q0, q1, q2], [0, 1, 3]))) >= 1


def test_permute_p4_duplicate_qubit_rejected():
    q0 = ir.TestValue()
    assert len(_errors(_permute_stmts([q0, q0], [0, 1]))) >= 1
```

- [ ] **Step 2: Run, verify it FAILS**

Run: `uv run pytest python/tests/gemini/test_permute.py -v -k "permute_p1 or permute_p2 or permute_p3 or permute_p4 or permute_valid"`
Expected: FAIL — no `Permute` validation registered, so invalid cases report no errors (or the valid case errors unexpectedly), i.e. assertions about error counts fail.

- [ ] **Step 3: Add the `Permute` validation impl**

In `python/bloqade/gemini/common/validation/move_to.py`, change the stmts import to include `Permute`:
```python
from bloqade.gemini.common.dialects.movement.stmts import Loc, MoveTo, Permute
```

Add this method to the `_MoveToValidationMethods` table (after `check_move_to`):

```python
    @interp.impl(Permute)
    def check_permute(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: Permute,
    ):
        from kirin.analysis import const
        from kirin.ir import ResultValue

        if not isinstance(node.qubits, ResultValue) or not isinstance(
            qubits_owner := node.qubits.owner, ilist.New
        ):
            _interp.add_validation_error(
                node,
                ir.ValidationError(node, "permute: qubits must be a literal list"),
            )
            return (EmptyLattice.bottom(),)

        qubit_values = qubits_owner.values

        # P1: perm must be compile-time constant
        perm_hint = node.perm.hints.get("const")
        if not isinstance(perm_hint, const.Value):
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    "permute: perm must be compile-time constants "
                    "(pass a literal list of ints)",
                ),
            )
            return (EmptyLattice.bottom(),)

        perm_values = tuple(int(p) for p in perm_hint.data)

        # P2: length mismatch
        if len(qubit_values) != len(perm_values):
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"permute: len(qubits)={len(qubit_values)} != "
                    f"len(perm)={len(perm_values)}",
                ),
            )
            return (EmptyLattice.bottom(),)

        # P3: perm must be a bijection of range(n)
        if sorted(perm_values) != list(range(len(perm_values))):
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    "permute: perm must be a permutation of range(len(qubits)) "
                    "(each index 0..n-1 exactly once)",
                ),
            )

        # P4: duplicate qubit SSA values
        seen_ids: set[int] = set()
        for qv in qubit_values:
            if id(qv) in seen_ids:
                _interp.add_validation_error(
                    node,
                    ir.ValidationError(
                        node,
                        "permute: same Qubit SSA value appears more than once "
                        "in qubits list",
                    ),
                )
                break
            seen_ids.add(id(qv))

        return (EmptyLattice.bottom(),)
```

- [ ] **Step 4: Run the tests, verify PASS**

Run: `uv run pytest python/tests/gemini/test_permute.py -v -k "permute_p1 or permute_p2 or permute_p3 or permute_p4 or permute_valid"`
Expected: PASS (5 tests).

- [ ] **Step 5: Lint**

Run: `uv run black python && uv run isort python && uv run ruff check python/bloqade/gemini/common/validation/move_to.py python/tests/gemini/test_permute.py && uv run pyright python/bloqade/gemini/common/validation/move_to.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add python/bloqade/gemini/common/validation/move_to.py python/tests/gemini/test_permute.py
git commit -m "feat(gemini): eager permute validation (const bijection, lengths, distinct qubits)"
```

---

## Task 5: End-to-end integration

**Files:**
- Test: `python/tests/gemini/test_permute.py`

- [ ] **Step 1: Write the failing test**

Append to `python/tests/gemini/test_permute.py`. Add these imports at the top:

```python
import bloqade.squin as squin
from bloqade.gemini import physical
```

Then append:

```python
def test_permute_identity_compiles():
    """An identity permutation is a valid no-op user-movement before a CZ."""

    @physical.kernel(aggressive_unroll=True, verify=False)
    def k():
        q = squin.qalloc(2)
        movement.permute([q[0], q[1]], [0, 1])
        squin.cz(q[0], q[1])

    assert k is not None


def test_permute_then_cz_compiles():
    """permute followed by a CZ lowers without error (routing handled by the
    placement strategy)."""

    @physical.kernel(aggressive_unroll=True, verify=False)
    def k():
        q = squin.qalloc(3)
        movement.permute([q[0], q[1], q[2]], [1, 2, 0])
        squin.cz(q[0], q[1])

    assert k is not None


def test_permute_rejected_on_plain_logical_kernel():
    """A plain logical kernel (no movement dialect) cannot lower permute."""
    import pytest

    from bloqade.gemini import logical

    with pytest.raises(Exception):

        @logical.kernel(aggressive_unroll=True)
        def k():
            q = squin.qalloc(2)
            movement.permute([q[0], q[1]], [1, 0])
            squin.cz(q[0], q[1])
```

- [ ] **Step 2: Run, verify it FAILS or PASSES appropriately**

Run: `uv run pytest python/tests/gemini/test_permute.py -v -k "compiles or plain_logical"`
Expected: with Tasks 1–4 done these should already PASS (the chain is wired). If `test_permute_then_cz_compiles` fails because the mover cannot route a 3-cycle, change its assertion to tolerate the documented limitation: replace the body's assertion with a compile-or-graceful check (compile the kernel inside a `try/except` and assert it either returns a kernel or raises a placement/infeasibility error — not an unexpected crash). Document the limitation in a comment referencing the spec's "Out of scope / limitations" section.

- [ ] **Step 3: Run the full permute test file**

Run: `uv run pytest python/tests/gemini/test_permute.py -v`
Expected: all PASS.

- [ ] **Step 4: Regression — existing movement + placement suites still pass**

Run:
```bash
uv run pytest python/tests/gemini/test_movement_kernel.py \
  python/tests/analysis/placement/test_user_moved.py \
  python/tests/rewrite/test_movement_rewrite.py \
  python/tests/gemini/validation/test_move_to_validation.py \
  python/tests/dialects/test_movement_dialect.py -q
```
Expected: all PASS (permute additions did not disturb move_to).

- [ ] **Step 5: Lint**

Run: `uv run black python && uv run isort python && uv run ruff check python/tests/gemini/test_permute.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add python/tests/gemini/test_permute.py
git commit -m "test(gemini): end-to-end permute kernel + plain-kernel rejection"
```

---

## Final Verification

- [ ] Run the full permute suite + movement regression:
  ```bash
  uv run pytest python/tests/gemini/test_permute.py python/tests/gemini/test_movement_kernel.py python/tests/analysis/placement python/tests/rewrite/test_movement_rewrite.py python/tests/gemini/validation -q
  ```
- [ ] Lint the whole tree: `uv run black --check python && uv run isort --check python && uv run ruff check python && uv run pyright python` (ignore unrelated `cirq` import failures).
- [ ] Confirm `bloqade.gemini.common.dialects.movement.permute` is importable and a `permute → cz` kernel compiles.

## Notes

- **Reuse, not duplication:** `impl_permute` resolves the permutation (via `_resolve_permute_locations`) and delegates to the existing `move_to_placements`; no new placement-strategy method. This is why permute inherits `UserMoved` accumulation, `compute_moves` routing, and palindrome return.
- **Validation mechanism (spec §4 refinement):** the `Permute` check lives in the existing `move.address.validation` table and is run by the `MoveToValidation` pass the movement kernel already invokes — no separate `PermuteValidation` pass and no `group.py` change. Permute errors currently surface under the "MoveToValidation failed" message; renaming that grouping is optional follow-up.
- **Known limitation:** routing a pure cycle/swap needs the mover to stage atoms through an intermediate location; if it cannot, the solve returns `bottom` (infeasible) — same as an impossible `move_to`. Permute adds no new routing capability.
- The untracked `demo/move_demo.py` is unrelated; leave it alone.
