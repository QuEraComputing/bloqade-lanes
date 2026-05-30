# `loc` Statement Design

**Goal:** Add a `loc(zone_id, word_id, site_id)` statement to the `lanes.movement` dialect that constructs a `LocationAddress` from integer arguments inside a kernel body, producing a `LocationAddressType` SSA result that const-folds when all args are compile-time constants.

**Architecture:** New `Loc` statement in `movement.py` alongside `MoveTo`. A const-fold interpreter impl registered on `movement_dialect` stamps `const.Value(LocationAddress(...))` on the result when all args have `const.Value(int)` hints. All downstream validation and rewrite logic (`MoveToValidation`, `rewrite_MoveTo`) works without changes — they already operate on the `const.Value` hint of the `ilist.New` result, which Kirin const-folds automatically when all elements have const hints.

**Tech Stack:** Kirin IR framework (`@statement`, `info.argument`, `info.result`, `interp.MethodTable`, `@interp.impl`, `kirin.analysis.const`), `lowering.FromPythonCall`, `types.Int`, `types.PyClass(LocationAddress)`.

---

## File Layout

| File | Status | Responsibility |
|------|--------|----------------|
| `python/bloqade/lanes/dialects/movement.py` | **EDIT** | Add `Loc` statement + `loc` stub + `"main"` interpreter impl |
| `python/bloqade/gemini/logical/__init__.py` | **EDIT** | Re-export `loc` alongside `movement_kernel` |
| `python/tests/dialects/test_movement_dialect.py` | **EDIT** | Add `Loc` statement shape tests |
| `python/tests/gemini/test_movement_kernel.py` | **EDIT** | Add end-to-end `loc(...)` inside `move_to` test |

---

## Statement Definition

```python
@statement(dialect=dialect)
class Loc(ir.Statement):
    name = "loc"
    traits = frozenset({lowering.FromPythonCall(), ir.Pure()})
    zone_id: ir.SSAValue = info.argument(types.Int)
    word_id: ir.SSAValue = info.argument(types.Int)
    site_id: ir.SSAValue = info.argument(types.Int)
    location: ir.ResultValue = info.result(LocationAddressType)

@wraps(Loc)
def loc(zone_id: int, word_id: int, site_id: int) -> LocationAddress:
    """Construct a LocationAddress from compile-time constant integers.

    All three arguments must be compile-time constants. Use inside
    movement_kernel bodies instead of capturing LocationAddress from
    outer Python scope.
    """
    ...
```

## Const-Fold Interpreter

`Loc` carries the `ir.Pure` trait, which tells Kirin's `HintConst` pass (part of `Default`) to attempt constant evaluation whenever all argument SSA values have `const.Value` hints. The pass uses `interp.Interpreter` (key `"main"`) for the actual evaluation, so a `"main"` method table must be registered on `movement_dialect`:

```python
@dialect.register(key="main")
class _MovementMainMethods(interp.MethodTable):
    @interp.impl(Loc)
    def loc(self, interp_, frame, stmt):
        z = frame.get(stmt.zone_id)
        w = frame.get(stmt.word_id)
        s = frame.get(stmt.site_id)
        return (LocationAddress(zone_id=z, word_id=w, site_id=s),)
```

When `loc(0, 1, 0)` appears in a kernel body, `Default` const-props `0`, `1`, `0` to `const.Value(0)` etc., then evaluates `Loc` via the `"main"` impl and stamps `const.Value(LocationAddress(0, 1, 0))` on the result SSA. If any arg is non-const, the result gets `Unknown` — no `const.Value` hint is stamped, and `MoveToValidation` F2 fires naturally.

**Note on `ir.Pure` trait instantiation:** Check whether `ir.Pure` is used as a class (e.g., `ir.Pure` directly in `frozenset`) or must be instantiated (`ir.Pure()`). Follow the same pattern used by other statements in the codebase (e.g., `place.MoveTo` or `place.ExecuteGate`).

## Downstream Integration — No Changes Required

| Pipeline step | Current behaviour | With `loc` |
|---|---|---|
| `ilist.New([loc(...).result])` | Folds when all elements have `const.Value` | Same — `Loc` result gets `const.Value` hint when args are const |
| `MoveToValidation` F2 | Fires when ilist lacks `const.Value` hint | Fires naturally if `loc` args are non-const |
| `MoveToValidation` F3 | Fires on out-of-range `LocationAddress` | Fires naturally on `loc(99, 0, 0)` |
| `rewrite_MoveTo` | Reads `const.Value` from ilist | Unchanged |

## Public API (after this change)

```python
from bloqade.gemini.logical import movement_kernel
from bloqade.lanes.dialects.movement import move_to, loc

@movement_kernel(verify=False)
def k():
    q = squin.qalloc(2)
    move_to([q[0], q[1]], [loc(0, 0, 0), loc(0, 1, 0)])
    squin.cz(q[0], q[1])
```

`loc` is also exported from `bloqade.gemini.logical` (add to `__init__.py`) so users have one import location for the full movement API.

## Error Behaviour

| User writes | What happens |
|---|---|
| `loc(0, 0, 0)` inside `move_to` | Const-folds → `LocationAddress(0, 0, 0)` → valid |
| `loc(99, 0, 0)` | Const-folds → F3 validation error "Invalid location address" |
| `loc(fn_arg, 0, 0)` (non-const) | No const hint → F2 "compile-time constant" error |
| `loc(0, 0, 0)` used outside `move_to` | Produces a `LocationAddressType` SSA value; behaviour depends on consumer |

## Testing

### `test_movement_dialect.py` additions

```python
def test_loc_in_dialect():
    assert Loc in movement.dialect.stmts

def test_loc_attributes():
    assert hasattr(Loc, "zone_id")
    assert hasattr(Loc, "word_id")
    assert hasattr(Loc, "site_id")
    assert hasattr(Loc, "location")

def test_loc_from_python_call_trait():
    assert any(isinstance(t, lowering.FromPythonCall) for t in Loc.traits)

def test_loc_produces_location_address_type():
    from bloqade.lanes.dialects.movement import LocationAddressType
    assert Loc.location.type == LocationAddressType  # or equivalent type check
```

### `test_movement_kernel.py` addition

```python
def test_movement_kernel_loc_inside_move_to():
    """loc(zone, word, site) inside move_to works without outer-scope capture."""
    @movement_kernel(verify=False)
    def k():
        q = squin.qalloc(2)
        move_to([q[0], q[1]], [loc(0, 0, 0), loc(0, 1, 0)])
        squin.cz(q[0], q[1])

    assert k is not None
```

### `test_move_to_validation.py` additions

```python
def test_loc_non_const_arg_triggers_f2():
    """loc(fn_arg, 0, 0) with non-const zone_id triggers F2."""
    # (test using movement_kernel with a kernel argument)

def test_loc_out_of_range_triggers_f3():
    """loc(99, 0, 0) triggers F3 (invalid location address)."""
    # (test using movement_kernel)
```
