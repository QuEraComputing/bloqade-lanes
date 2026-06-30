# LocationAddress attribute statements + GetAttr rewrite — design

**Date:** 2026-06-22
**Status:** Approved (ready for implementation plan)

## Goal

Let kernel authors read the fields of a `LocationAddress` (`addr.word_id`,
`addr.site_id`, `addr.zone_id`) inside a Gemini kernel and have those reads
compile to first-class movement-dialect statements that are pure and
interpretable (so constant addresses fold to constant ints).

Concretely:

1. Add three statements to the movement dialect:
   - `WordId(address: LocationAddressType) -> int`
   - `SiteId(address: LocationAddressType) -> int`
   - `ZoneId(address: LocationAddressType) -> int`
2. Add a rewrite rule that converts `py.GetAttr` on a value of type
   `LocationAddressType` with attribute `word_id` / `site_id` / `zone_id`
   into the corresponding statement.
3. Register that rewrite to run automatically after type inference.

## Background

- The movement dialect lives at
  `python/bloqade/gemini/common/dialects/movement/` — `stmts.py` (`Loc`,
  `MoveTo`), `_interface.py` (Python-facing `@lowering.wraps`), `_dialect.py`
  (`dialect = ir.Dialect("gemini.common.movement")`), and interpreter impls in
  `python/bloqade/gemini/common/impl/movement.py`.
- `LocationAddressType = types.PyClass(LocationAddress)` is already defined in
  `stmts.py`. `LocationAddress` (in `bloqade.lanes.bytecode.encoding`) exposes
  `word_id`, `site_id`, and `zone_id` as `int` properties.
- `Loc` is the model to follow: `ir.Pure()` + `lowering.FromPythonCall()`, SSA
  arguments, a single result, plus an interpreter impl.
- The movement dialect is included by the Gemini physical kernel
  (`bloqade.gemini.physical.group.kernel`, which unions the logical kernel with
  the movement dialect). Registering behavior on the dialect makes it active in
  any kernel that includes the movement dialect.

## Design

### 1. Statements (`movement/stmts.py`)

Three statements, each pure (so the fold pass can evaluate them), taking one
`LocationAddressType` argument and producing an `Int`:

```python
@statement(dialect=dialect)
class WordId(ir.Statement):
    name = "word_id"
    traits = frozenset({ir.Pure()})
    address: ir.SSAValue = info.argument(LocationAddressType)
    result: ir.ResultValue = info.result(types.Int)
```

…and identical `SiteId` (`name = "site_id"`) and `ZoneId`
(`name = "zone_id"`).

No `_interface.py` (`@lowering.wraps`) wrappers are added: the only intended
entry point is `addr.word_id` via the rewrite below, not a standalone callable.
Consequently the statements **omit the `lowering.FromPythonCall()` trait** that
`Loc` carries — that trait only supports lowering from a Python call, and these
statements are never lowered from one (the rewrite constructs them directly).
They keep `ir.Pure()`, which is what lets the fold pass evaluate them.

### 2. Interpreter impls (`gemini/common/impl/movement.py`)

Three impls registered on the movement dialect, mirroring `_LocInterpreter`:

```python
@interp.impl(WordId)
def word_id(self, interp_, frame, stmt: WordId):
    return (frame.get(stmt.address).word_id,)
```

…and the same for `SiteId` / `ZoneId`. These make a constant `LocationAddress`
(e.g. produced by `loc(...)`) fold to a constant int.

### 3. Rewrite rule (`movement/rewrite.py`)

A `RewriteRule` that replaces a matching `py.GetAttr` with the corresponding
statement:

```python
from kirin import types
from kirin.dialects.py.attr import GetAttr
from kirin.rewrite.abc import RewriteRule, RewriteResult

from .stmts import LocationAddressType, SiteId, WordId, ZoneId

_ATTR_TO_STMT = {"word_id": WordId, "site_id": SiteId, "zone_id": ZoneId}


@dataclass
class RewriteLocationAttr(RewriteRule):
    def rewrite_Statement(self, node) -> RewriteResult:
        if not isinstance(node, GetAttr):
            return RewriteResult()
        stmt_cls = _ATTR_TO_STMT.get(node.attrname)
        if stmt_cls is None:
            return RewriteResult()
        obj_type = node.obj.type
        if obj_type is types.Bottom or not obj_type.is_subseteq(LocationAddressType):
            return RewriteResult()
        node.replace_by(stmt_cls(node.obj))
        return RewriteResult(has_done_something=True)
```

**`Bottom` guard (important):** in the kirin type lattice `Bottom` is a subtype
of every type, so `Bottom.is_subseteq(LocationAddressType)` is `True`. A
`Bottom`-typed `obj` (from unreachable / never-typed code) must be excluded
explicitly via `obj_type is types.Bottom`, otherwise it would be wrongly
rewritten. Values of unknown type (`Any`) are *not* a subtype of
`LocationAddressType`, so they are correctly skipped without a special case.

### 4. Registration: post-inference, walked (`movement/rewrite.py` + `__init__.py`)

Register the rule on the dialect's post-inference rules, **wrapped in `Walk`**:

```python
from kirin.rewrite import Walk
from ._dialect import dialect

dialect.rules.inference.append(Walk(RewriteLocationAttr()))
```

`movement/__init__.py` imports `rewrite` so the registration executes when the
dialect loads.

**Why `Walk` is required (not optional).** `TypeInfer.unsafe_run` runs
`PostInference.fixpoint(mt)` after inference (`kirin/passes/typeinfer.py:44`),
and `PostInference` applies each registered rule as `rule.rewrite(mt.code)`
(`kirin/passes/post_inference.py`). `RewriteRule.rewrite(node)`
(`kirin/rewrite/abc.py`) dispatches **only on the node passed** — no recursion —
and `mt.code` is the top-level `func.Function`. A bare `RewriteLocationAttr`
would therefore only ever see the function node and never the nested `GetAttr`
statements, making it a silent no-op. Wrapping in `Walk` recursively descends
region → block → statements and applies the rule to each `GetAttr`.

`@dialect.post_inference` cannot express this because it instantiates the rule
with no arguments (`rule()`); we therefore append `Walk(RewriteLocationAttr())`
directly to `dialect.rules.inference`.

**Why this needs no pipeline (`group.py`) edit.** Both compilation branches run
type inference, and `TypeInfer` runs the post-inference rules:
- `aggressive_unroll=False` → `Default` pass calls the `TypeInfer` pass.
- `aggressive_unroll=True` → bloqade's `AggressiveUnroll` calls
  `self.typeinfer.unsafe_run(mt)`.

In both, type inference runs first (so `obj.type` is known), the rewrite fires,
and the fold step that follows in each branch collapses constant addresses to
constant ints. Symbolic addresses keep the interpretable statement. The
post-inference fixpoint converges: after the first pass the `GetAttr` is gone,
so the second pass reports no change.

### Upstream caveat / follow-up

The need to register `Walk(RewriteLocationAttr())` rather than a bare rule stems
from kirin applying `canonicalize` / `post_inference` rules un-walked. This is
tracked upstream in **QuEraComputing/kirin#676** (proposes a `walk=True` option
on the decorators). When that is resolved we can simplify the registration to
the decorator form; until then the manual `Walk` registration is the working
approach.

## Files

| File | Change |
|------|--------|
| `python/bloqade/gemini/common/dialects/movement/stmts.py` | Add `WordId`, `SiteId`, `ZoneId` statements |
| `python/bloqade/gemini/common/impl/movement.py` | Add interpreter impls for the three statements |
| `python/bloqade/gemini/common/dialects/movement/rewrite.py` | New: `RewriteLocationAttr` + `dialect.rules.inference.append(Walk(...))` |
| `python/bloqade/gemini/common/dialects/movement/__init__.py` | Import `rewrite` so registration runs |
| `python/tests/gemini/test_location_attr.py` | New: tests (below) |

No change to `python/bloqade/gemini/physical/group.py`.

## Testing

- **Constant fold:** a kernel doing `loc(z, w, s).word_id` (and `site_id` /
  `zone_id`) compiles so the `GetAttr` is gone and the value folds to the
  expected constant int.
- **Symbolic address:** an address that is not a compile-time constant (e.g. a
  kernel parameter typed `LocationAddress`) keeps a `WordId` / `SiteId` /
  `ZoneId` statement (no `GetAttr`), and interprets to the correct int.
- **Non-matches left alone:** `GetAttr` on a non-`LocationAddress` object, and
  an unsupported attribute name, are untouched.
- **Rule unit test:** apply `RewriteLocationAttr` (wrapped in `Walk`) to a small
  IR and assert the replacement; include a `Bottom`-typed `obj` case to confirm
  it is *not* rewritten.

## Out of scope

- No standalone `word_id()` / `site_id()` / `zone_id()` callable wrappers.
- No change to the kernel pass pipeline; behavior rides on the dialect's
  post-inference registration.
- The upstream kirin decorator enhancement (#676) is a separate follow-up.
