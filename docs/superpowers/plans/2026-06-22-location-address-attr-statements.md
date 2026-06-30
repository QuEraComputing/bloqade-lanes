# LocationAddress attribute statements + GetAttr rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `WordId`/`SiteId`/`ZoneId` statements to the Gemini movement dialect and a post-inference rewrite so `addr.word_id` / `addr.site_id` / `addr.zone_id` inside a kernel lower to those statements (folding to constant ints when the address is constant).

**Architecture:** Three pure movement-dialect statements (one `LocationAddress` arg → `int`), each with a concrete interpreter impl. A `RewriteRule` converts `py.GetAttr` on a `LocationAddressType` value (attr `word_id`/`site_id`/`zone_id`) into the matching statement; it is registered — wrapped in `Walk` — on the movement dialect's post-inference rules, so it runs automatically after type inference in any kernel that includes the dialect. No kernel-pipeline edit; the subsequent fold step collapses constant addresses (the concrete interp impl is what makes them foldable).

**Tech Stack:** Python, the Kirin IR framework (`kirin-toolchain`), `uv` + `pytest`. Spec: `docs/superpowers/specs/2026-06-22-location-address-attr-statements-design.md`. Related upstream issue: QuEraComputing/kirin#676 (un-walked post-inference rules — why the `Walk` wrapper is needed).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `python/bloqade/gemini/common/dialects/movement/stmts.py` | **Modify** — add `WordId`, `SiteId`, `ZoneId` statements |
| `python/bloqade/gemini/common/dialects/movement/rewrite.py` | **Create** — `RewriteLocationAttr` rule + post-inference registration |
| `python/bloqade/gemini/common/dialects/movement/__init__.py` | **Modify** — import `rewrite` so registration runs at dialect load |
| `python/bloqade/gemini/common/impl/movement.py` | **Modify** — interpreter impls for the three statements |
| `python/tests/gemini/test_location_attr.py` | **Create** — all tests |

Lint/format after each task: `uv run black python && uv run isort python && uv run ruff check python && uv run pyright python`.

---

## Task 1: Add the three statements

**Files:**
- Modify: `python/bloqade/gemini/common/dialects/movement/stmts.py`
- Test: `python/tests/gemini/test_location_attr.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/gemini/test_location_attr.py`:

```python
"""Tests for LocationAddress attribute statements + GetAttr rewrite."""

from kirin import ir, lowering

from bloqade.gemini.common.dialects.movement.stmts import SiteId, WordId, ZoneId


def test_statements_are_pure_and_named():
    for stmt_cls, name in [
        (WordId, "word_id"),
        (SiteId, "site_id"),
        (ZoneId, "zone_id"),
    ]:
        assert issubclass(stmt_cls, ir.Statement)
        assert stmt_cls.name == name
        # Pure so the fold pass can evaluate them.
        assert any(isinstance(t, ir.Pure) for t in stmt_cls.traits)
        # Not lowered from a Python call (constructed only by the rewrite).
        assert not any(
            isinstance(t, lowering.FromPythonCall) for t in stmt_cls.traits
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest python/tests/gemini/test_location_attr.py::test_statements_are_pure_and_named -v`
Expected: FAIL — `ImportError: cannot import name 'WordId'` (statements don't exist yet).

- [ ] **Step 3: Add the statements**

In `python/bloqade/gemini/common/dialects/movement/stmts.py`, append after the existing `MoveTo` class (the file already imports `ir`, `lowering`, `types`, `info`, `statement`, and defines `LocationAddressType`):

```python
@statement(dialect=dialect)
class WordId(ir.Statement):
    """Read the word_id of a LocationAddress (produced by the GetAttr rewrite)."""

    name = "word_id"
    traits = frozenset({ir.Pure()})
    address: ir.SSAValue = info.argument(LocationAddressType)
    result: ir.ResultValue = info.result(types.Int)


@statement(dialect=dialect)
class SiteId(ir.Statement):
    """Read the site_id of a LocationAddress (produced by the GetAttr rewrite)."""

    name = "site_id"
    traits = frozenset({ir.Pure()})
    address: ir.SSAValue = info.argument(LocationAddressType)
    result: ir.ResultValue = info.result(types.Int)


@statement(dialect=dialect)
class ZoneId(ir.Statement):
    """Read the zone_id of a LocationAddress (produced by the GetAttr rewrite)."""

    name = "zone_id"
    traits = frozenset({ir.Pure()})
    address: ir.SSAValue = info.argument(LocationAddressType)
    result: ir.ResultValue = info.result(types.Int)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest python/tests/gemini/test_location_attr.py::test_statements_are_pure_and_named -v`
Expected: PASS.

- [ ] **Step 5: Lint**

Run: `uv run black python && uv run isort python && uv run ruff check python/bloqade/gemini/common/dialects/movement/stmts.py python/tests/gemini/test_location_attr.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add python/bloqade/gemini/common/dialects/movement/stmts.py python/tests/gemini/test_location_attr.py
git commit -m "feat(gemini): add WordId/SiteId/ZoneId movement statements"
```

---

## Task 2: Add the rewrite rule, register it post-inference, and wire it in

**Files:**
- Create: `python/bloqade/gemini/common/dialects/movement/rewrite.py`
- Modify: `python/bloqade/gemini/common/dialects/movement/__init__.py`
- Test: `python/tests/gemini/test_location_attr.py`

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/gemini/test_location_attr.py`. First extend the imports at the top of the file so the block reads:

```python
"""Tests for LocationAddress attribute statements + GetAttr rewrite."""

from kirin import ir, lowering, types
from kirin.dialects import func, py

from bloqade.gemini.common.dialects.movement.rewrite import RewriteLocationAttr
from bloqade.gemini.common.dialects.movement.stmts import (
    LocationAddressType,
    SiteId,
    WordId,
    ZoneId,
)
from bloqade.gemini.logical import loc
from bloqade.gemini.physical import kernel as movement_kernel
from bloqade.lanes.bytecode.encoding import LocationAddress
```

Then append these tests:

```python
# -- rule unit tests: the guard cases must NOT rewrite --


def test_rewrite_skips_unsupported_attrname():
    addr = ir.TestValue(type=LocationAddressType)
    node = py.GetAttr(obj=addr, attrname="nonexistent")
    result = RewriteLocationAttr().rewrite_Statement(node)
    assert not result.has_done_something


def test_rewrite_skips_non_location_type():
    obj = ir.TestValue(type=types.Int)
    node = py.GetAttr(obj=obj, attrname="word_id")
    result = RewriteLocationAttr().rewrite_Statement(node)
    assert not result.has_done_something


def test_rewrite_skips_bottom_type():
    # Bottom is a subtype of every type, so without the explicit guard this
    # would wrongly match. It must be skipped.
    obj = ir.TestValue(type=types.Bottom)
    node = py.GetAttr(obj=obj, attrname="word_id")
    result = RewriteLocationAttr().rewrite_Statement(node)
    assert not result.has_done_something


# -- integration: a symbolic (non-constant) address rewrites to the statement --


def test_param_word_id_rewrites_to_statement():
    @movement_kernel(verify=False)
    def k(a: LocationAddress):
        return a.word_id

    stmts = list(k.callable_region.walk())
    assert not any(isinstance(s, py.GetAttr) for s in stmts)
    ret = next(s for s in stmts if isinstance(s, func.Return))
    assert isinstance(ret.value.owner, WordId)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest python/tests/gemini/test_location_attr.py -v -k "rewrite_skips or param_word_id"`
Expected: FAIL — `ModuleNotFoundError: No module named 'bloqade.gemini.common.dialects.movement.rewrite'`.

- [ ] **Step 3: Create the rewrite rule + registration**

Create `python/bloqade/gemini/common/dialects/movement/rewrite.py`:

```python
from dataclasses import dataclass

from kirin import ir, types
from kirin.dialects import py
from kirin.rewrite import Walk
from kirin.rewrite.abc import RewriteResult, RewriteRule

from ._dialect import dialect
from .stmts import LocationAddressType, SiteId, WordId, ZoneId

_ATTR_TO_STMT = {"word_id": WordId, "site_id": SiteId, "zone_id": ZoneId}


@dataclass
class RewriteLocationAttr(RewriteRule):
    """Lower ``py.GetAttr`` on a ``LocationAddress`` (``word_id`` / ``site_id`` /
    ``zone_id``) into the corresponding movement-dialect statement."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, py.GetAttr):
            return RewriteResult()
        stmt_cls = _ATTR_TO_STMT.get(node.attrname)
        if stmt_cls is None:
            return RewriteResult()
        obj_type = node.obj.type
        # Bottom is a subtype of every type; exclude it explicitly so that
        # never-typed values are not rewritten.
        if obj_type is types.Bottom or not obj_type.is_subseteq(LocationAddressType):
            return RewriteResult()
        node.replace_by(stmt_cls(node.obj))
        return RewriteResult(has_done_something=True)


# Registered wrapped in Walk: PostInference applies registered rules as
# `rule.rewrite(mt.code)` on the top function node only (no recursion), so a
# bare rule would never reach nested GetAttr statements. See QuEraComputing/
# kirin#676.
dialect.rules.inference.append(Walk(RewriteLocationAttr()))
```

- [ ] **Step 4: Wire the module so registration runs at dialect load**

In `python/bloqade/gemini/common/dialects/movement/__init__.py`, add the `rewrite` import. The file becomes:

```python
from . import rewrite as rewrite
from . import stmts as stmts
from ._dialect import dialect as dialect
from ._interface import loc as loc, move_to as move_to
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest python/tests/gemini/test_location_attr.py -v -k "rewrite_skips or param_word_id"`
Expected: PASS (4 tests).

- [ ] **Step 6: Lint**

Run: `uv run black python && uv run isort python && uv run ruff check python/bloqade/gemini/common/dialects/movement/ python/tests/gemini/test_location_attr.py && uv run pyright python/bloqade/gemini/common/dialects/movement/`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add python/bloqade/gemini/common/dialects/movement/rewrite.py python/bloqade/gemini/common/dialects/movement/__init__.py python/tests/gemini/test_location_attr.py
git commit -m "feat(gemini): rewrite LocationAddress GetAttr to WordId/SiteId/ZoneId

Registered as Walk(RewriteLocationAttr()) on the movement dialect's
post-inference rules so it runs after type inference. See
QuEraComputing/kirin#676 for the un-walked-rule caveat."
```

---

## Task 3: Interpreter impls so constant addresses fold

**Files:**
- Modify: `python/bloqade/gemini/common/impl/movement.py`
- Test: `python/tests/gemini/test_location_attr.py`

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/gemini/test_location_attr.py`:

```python
# -- integration: a constant address folds the attribute read to a constant --


def test_constant_word_id_folds():
    @movement_kernel(verify=False)
    def k():
        return loc(zone_id=2, word_id=3, site_id=5).word_id

    stmts = list(k.callable_region.walk())
    assert not any(isinstance(s, py.GetAttr) for s in stmts)
    ret = next(s for s in stmts if isinstance(s, func.Return))
    assert isinstance(ret.value.owner, py.Constant)
    assert ret.value.owner.value.unwrap() == 3


def test_constant_site_id_folds():
    @movement_kernel(verify=False)
    def k():
        return loc(zone_id=2, word_id=3, site_id=5).site_id

    stmts = list(k.callable_region.walk())
    assert not any(isinstance(s, py.GetAttr) for s in stmts)
    ret = next(s for s in stmts if isinstance(s, func.Return))
    assert isinstance(ret.value.owner, py.Constant)
    assert ret.value.owner.value.unwrap() == 5


def test_constant_zone_id_folds():
    @movement_kernel(verify=False)
    def k():
        return loc(zone_id=2, word_id=3, site_id=5).zone_id

    stmts = list(k.callable_region.walk())
    assert not any(isinstance(s, py.GetAttr) for s in stmts)
    ret = next(s for s in stmts if isinstance(s, func.Return))
    assert isinstance(ret.value.owner, py.Constant)
    assert ret.value.owner.value.unwrap() == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest python/tests/gemini/test_location_attr.py -v -k "folds"`
Expected: FAIL — the attribute read rewrites to a `WordId`/`SiteId`/`ZoneId` statement but, with no interpreter impl, it is *not* folded, so `ret.value.owner` is the statement, not a `py.Constant`.

- [ ] **Step 3: Add the interpreter impls**

In `python/bloqade/gemini/common/impl/movement.py`, extend the `stmts` import and add three impls to the existing `_LocInterpreter` method table. The file becomes:

```python
from kirin import interp
from kirin.interp import Frame

from bloqade.gemini.common.dialects.movement import dialect as movement_dialect
from bloqade.gemini.common.dialects.movement.stmts import Loc, SiteId, WordId, ZoneId


@movement_dialect.register
class _LocInterpreter(interp.MethodTable):
    @interp.impl(Loc)
    def loc(self, interp_: interp.Interpreter, frame: Frame, stmt: Loc):
        from bloqade.lanes.bytecode.encoding import LocationAddress

        z = frame.get(stmt.zone_id)
        w = frame.get(stmt.word_id)
        s = frame.get(stmt.site_id)
        return (LocationAddress(zone_id=z, word_id=w, site_id=s),)

    @interp.impl(WordId)
    def word_id(self, interp_: interp.Interpreter, frame: Frame, stmt: WordId):
        return (frame.get(stmt.address).word_id,)

    @interp.impl(SiteId)
    def site_id(self, interp_: interp.Interpreter, frame: Frame, stmt: SiteId):
        return (frame.get(stmt.address).site_id,)

    @interp.impl(ZoneId)
    def zone_id(self, interp_: interp.Interpreter, frame: Frame, stmt: ZoneId):
        return (frame.get(stmt.address).zone_id,)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest python/tests/gemini/test_location_attr.py -v -k "folds"`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the whole test file + the existing movement-dialect suite**

Run: `uv run pytest python/tests/gemini/test_location_attr.py python/tests/gemini/test_movement_kernel.py -v`
Expected: PASS (all tests, including the existing movement-kernel tests — confirms the new post-inference rule didn't disturb them).

- [ ] **Step 6: Lint**

Run: `uv run black python && uv run isort python && uv run ruff check python/bloqade/gemini/common/impl/movement.py python/tests/gemini/test_location_attr.py && uv run pyright python/bloqade/gemini/common/impl/movement.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add python/bloqade/gemini/common/impl/movement.py python/tests/gemini/test_location_attr.py
git commit -m "feat(gemini): interpret WordId/SiteId/ZoneId so constant addresses fold"
```

---

## Final verification

- [ ] Run the full Gemini test suite: `uv run pytest python/tests/gemini -q` — Expected: all pass.
- [ ] Lint the whole tree: `uv run black --check python && uv run isort --check python && uv run ruff check python && uv run pyright python` — Expected: clean.

## Notes / follow-up

- The `Walk(RewriteLocationAttr())` direct registration is the approach that works today. If QuEraComputing/kirin#676 lands a `walk=` option on `@dialect.post_inference`, simplify the registration to the decorator form.
- The uncommitted `_interface.py` change in the working tree is unrelated to this plan; leave it untouched.
