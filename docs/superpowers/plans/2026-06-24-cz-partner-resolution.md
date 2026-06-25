# `cz_partner` Resolution Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `movement.cz_partner(loc)` so its result is materialized through Kirin's standard const-prop / fold instead of a custom `ResolveCzPartner` walk, and move address validation into the pipeline so kernels that use `cz_partner` no longer need `verify=False`.

**Architecture:** `CzPartner` becomes `ir.Pure` with an `arch_spec: ArchSpec | None` attribute. A new const-prop method table on the movement dialect returns `const.Value(partner)` when both attribute and address operand are constant, otherwise `const.Result.top()`. A new `BindCzPartnerArchSpec` rewrite, run as a `CallGraphPass` before `SquinToNative` in both `pipeline/base.py` and `upstream.py`, populates the attribute from the pipeline's arch spec. The old `ResolveCzPartner` walk + trailing `Fold` block and the duplicate `get_validation(arch_spec)` invocation in `@physical.kernel`'s `run_pass` are removed.

**Tech Stack:** Kirin IR (`@statement`, `info.attribute`, `ir.Pure`, `interp.MethodTable`, `@interp.impl`, `kirin.analysis.const`), `bloqade.rewrite.passes.callgraph.CallGraphPass`, `bloqade.lanes.arch.spec.ArchSpec`, pytest.

**Spec:** `docs/superpowers/specs/2026-06-24-cz-partner-resolution-design.md`.

---

## Resume Status (2026-06-25)

Execution paused after Task 6. Currently positioned to start **Task 7** (drop `get_validation(arch_spec)` from `python/bloqade/gemini/physical/group.py`'s `ValidationSuite` and update the `arch_spec` parameter `Doc`).

Completed commits on branch `feat-user-movement-dialect`:

| Task | Commit | Note |
|------|--------|------|
| T1 | `da977d22` | CzPartner: `ir.Pure()` + `arch_spec: ArchSpec \| None` attribute |
| T2 | `f7226433` | `CzPartnerConstProp` method table |
| T2 (polish) | `6c2db20a` | Annotated `frame: Frame`; `isinstance(table, CzPartnerConstProp)` |
| T3 | `7447667b` | `BindCzPartnerArchSpec` rewrite + tests |
| T4 | `a0490595` | Wire bind into `pipeline/base.py` — e2e tests green again |
| T5 | `d2c364ff` | Wire bind into `upstream.py` |
| T6 | `fa2d471a` | Delete `ResolveCzPartner` |
| T6 (extras) | `f4c5cfb7` | `ArchSpec.__hash__` + concrete-interpreter `@interp.impl(CzPartner)` on `_LocInterpreter` |

**Plan-text correction noticed during execution:** the plan's Step 2 code blocks for T4 and T5 show `CallGraphPass(out.dialects, BindCzPartnerArchSpec(self.arch_spec))(out)`, but the actual implementation needs `CallGraphPass(out.dialects, rewrite.Walk(BindCzPartnerArchSpec(self.arch_spec)))(out)` — `CallGraphPass` calls `rule.rewrite(region)`, and a bare `RewriteRule.rewrite_Region` is a no-op. Both T4 and T5 commits use the `rewrite.Walk(...)`-wrapped form. The plan text for T4/T5 is stale on this point but the code is correct.

**Extra commit beyond the original plan (T6 extras, `f4c5cfb7`):** required to make the refactor function end-to-end:

- `ArchSpec.__hash__` (identity-based) — Kirin's IR attribute machinery needs `ArchSpec` to be hashable now that it's an `info.attribute` value on `CzPartner`.
- Concrete-interpreter `@interp.impl(CzPartner)` on `_LocInterpreter` (key `"main"`) — fallback used by const-prop's `try_eval_const_pure` when unrolling pure lambdas. Raises `NotImplementedError` when arch_spec is unbound / address is non-const / no partner exists, so kernel-decoration-time `run_no_raise` skips folding gracefully.

**Remaining tasks:** T7 (this), T8 (drop `verify=False` in `test_cz_partner.py` + add no-partner regression), T9 (full test suite + lint sweep).

To resume: pick up at "Task 7" below, on HEAD `f4c5cfb7` (or wherever the branch tip is after this status commit pushes).

---

## File Layout

| File | Status | Responsibility |
|------|--------|----------------|
| `python/bloqade/gemini/common/dialects/movement/stmts.py` | **EDIT** | Add `ir.Pure()` trait; add `arch_spec` attribute on `CzPartner`; rewrite the docstring/comment |
| `python/bloqade/gemini/common/dialects/movement/constprop.py` | **NEW** | Const-prop method table for the movement dialect (currently: `CzPartner`) |
| `python/bloqade/gemini/common/dialects/movement/__init__.py` | **EDIT** | Import `constprop` so its registrations run |
| `python/bloqade/gemini/common/dialects/movement/rewrite.py` | **EDIT** | Delete `ResolveCzPartner`; add `BindCzPartnerArchSpec` |
| `python/bloqade/lanes/pipeline/base.py` | **EDIT** | Replace `ResolveCzPartner` walk + `Fold` with `CallGraphPass(BindCzPartnerArchSpec(...))` before `SquinToNative` |
| `python/bloqade/lanes/upstream.py` | **EDIT** | Same wiring change for `NativeToPlace.emit` |
| `python/bloqade/gemini/physical/group.py` | **EDIT** | Remove `get_validation(arch_spec)` from `run_pass`'s `ValidationSuite` |
| `python/tests/gemini/test_cz_partner.py` | **EDIT** | Drop `ResolveCzPartner` unit tests; update statement-shape test; drop `verify=False` from kernels; add no-partner const-prop regression test |
| `python/tests/dialects/test_movement_constprop.py` | **NEW** | Unit tests for the `CzPartner` const-prop impl |

---

## Task 1: Update `CzPartner` statement shape

**Files:**
- Modify: `python/bloqade/gemini/common/dialects/movement/stmts.py:59-87`
- Test: `python/tests/gemini/test_cz_partner.py:28-37`

- [ ] **Step 1: Update the failing statement-shape test**

The current test asserts `CzPartner` is NOT `ir.Pure`. After this task it must be `ir.Pure` and carry an `arch_spec` attribute that defaults to `None`. Replace lines 28-37 of `python/tests/gemini/test_cz_partner.py` with:

```python
def test_cz_partner_statement_shape():
    assert issubclass(CzPartner, ir.Statement)
    assert CzPartner.name == "cz_partner"
    assert any(isinstance(t, lowering.FromPythonCall) for t in CzPartner.traits)
    # CzPartner is now Pure: materialization rides on the standard
    # const-prop / fold machinery once arch_spec is bound by the pipeline.
    assert any(isinstance(t, ir.Pure) for t in CzPartner.traits)
    assert CzPartner in movement.dialect.stmts

    # arch_spec is an attribute (default None) — populated by the pipeline's
    # BindCzPartnerArchSpec pass before unrolling.
    addr = ir.TestValue()
    stmt = CzPartner(addr)
    assert stmt.arch_spec is None
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `uv run pytest python/tests/gemini/test_cz_partner.py::test_cz_partner_statement_shape -x`
Expected: FAIL — the `Pure` assertion fails (current `CzPartner` is not `Pure`) and `stmt.arch_spec` doesn't exist.

- [ ] **Step 3: Edit `CzPartner` in `stmts.py`**

Replace lines 59-87 of `python/bloqade/gemini/common/dialects/movement/stmts.py` (the `CzPartner` class and its preceding comment) with:

```python
@statement(dialect=dialect)
class CzPartner(ir.Statement):
    """Resolve the CZ blockade-partner LocationAddress of a location.

    ``movement.cz_partner(loc)`` returns the location an atom must occupy to
    be CZ-entangled with an atom at ``loc``. The result is materialized by
    standard const-prop: once ``arch_spec`` is bound (by the pipeline's
    ``BindCzPartnerArchSpec`` pass) and the ``address`` operand is constant,
    the registered constprop impl returns a ``const.Value(LocationAddress)``
    so the rest of the fold pipeline propagates it (e.g. into a ``move_to``
    locations list).

    Any ``CzPartner`` that survives resolution (because ``arch_spec`` was not
    bound, ``address`` did not const-fold, or the architecture has no partner
    for that location) keeps the downstream ``move_to`` non-const, which the
    existing move_to validation reports.
    """

    name = "cz_partner"
    traits = frozenset({ir.Pure(), lowering.FromPythonCall()})
    address: ir.SSAValue = info.argument(LocationAddressType)
    arch_spec: ArchSpec | None = info.attribute(default=None)
    result: ir.ResultValue = info.result(LocationAddressType)
```

Add the missing import at the top of the file (immediately below the existing `from bloqade.lanes.bytecode.encoding import LocationAddress` line on line 6):

```python
from bloqade.lanes.arch.spec import ArchSpec
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `uv run pytest python/tests/gemini/test_cz_partner.py::test_cz_partner_statement_shape -x`
Expected: PASS.

- [ ] **Step 5: Run the full module to catch regressions**

Run: `uv run pytest python/tests/gemini/test_cz_partner.py -x`
Expected: several end-to-end tests still pass because `ResolveCzPartner` is still wired in the pipeline. If any unrelated test fails, fix the regression before moving on.

- [ ] **Step 6: Commit**

```bash
git add python/bloqade/gemini/common/dialects/movement/stmts.py python/tests/gemini/test_cz_partner.py
git commit -m "$(cat <<'EOF'
refactor(gemini-movement): make CzPartner Pure with arch_spec attribute

Lays the groundwork for resolving cz_partner via standard const-prop
instead of a bespoke rewrite. ResolveCzPartner is still wired in the
pipeline; const-prop impl and pipeline rewiring follow.
EOF
)"
```

---

## Task 2: Add `CzPartner` const-prop impl

**Files:**
- Create: `python/bloqade/gemini/common/dialects/movement/constprop.py`
- Modify: `python/bloqade/gemini/common/dialects/movement/__init__.py`
- Test: `python/tests/dialects/test_movement_constprop.py` (NEW)

- [ ] **Step 1: Write the failing registration test**

Create `python/tests/dialects/test_movement_constprop.py`. The full logic of the impl (resolve / unbound / no-partner / non-const) is exercised end-to-end by the pipeline tests in Task 4 (success path) and Task 8 (no-partner path); this file just confirms the const-prop method table is registered on the dialect.

```python
"""Const-prop registration for the movement dialect."""

from kirin import interp

from bloqade.gemini.common.dialects.movement import dialect as movement_dialect


def test_movement_dialect_registers_constprop_method_table():
    # ``Dialect.interps`` is the dict of registered method tables keyed by
    # interpreter key. The const-prop analysis looks under "constprop".
    table = movement_dialect.interps.get("constprop")
    assert table is not None, "movement dialect has no 'constprop' registration"
    assert isinstance(table, interp.MethodTable)
    assert type(table).__name__ == "CzPartnerConstProp"
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `uv run pytest python/tests/dialects/test_movement_constprop.py -x`
Expected: FAIL — no `constprop` registration for `CzPartner` (the method lookup returns `None`).

- [ ] **Step 3: Create the const-prop module**

Create `python/bloqade/gemini/common/dialects/movement/constprop.py`:

```python
from kirin import interp
from kirin.analysis import const

from bloqade.lanes.bytecode.encoding import LocationAddress

from ._dialect import dialect
from .stmts import CzPartner


@dialect.register(key="constprop")
class CzPartnerConstProp(interp.MethodTable):
    """Resolve ``movement.cz_partner(loc)`` during const propagation.

    Returns the partner location as ``const.Value`` once ``arch_spec`` is
    bound (by ``BindCzPartnerArchSpec`` in the pipeline) and the ``address``
    operand is itself a constant ``LocationAddress``. Returns ``top()`` for
    every other case so downstream consumers stay non-const — that's how
    unresolved ``CzPartner`` surfaces as a compilation error today (via the
    existing move_to "locations must be compile-time constants" check).
    """

    @interp.impl(CzPartner)
    def cz_partner(self, _, frame, stmt: CzPartner):
        if stmt.arch_spec is None:
            return (const.Result.top(),)
        addr = frame.get(stmt.address)
        if not isinstance(addr, const.Value) or not isinstance(
            addr.data, LocationAddress
        ):
            return (const.Result.top(),)
        partner = stmt.arch_spec.get_cz_partner(addr.data)
        if partner is None:
            return (const.Result.top(),)
        return (const.Value(partner),)
```

- [ ] **Step 4: Wire the module into the dialect package**

Edit `python/bloqade/gemini/common/dialects/movement/__init__.py` (currently line 1):

```python
from . import constprop as constprop, rewrite as rewrite, stmts as stmts
```

- [ ] **Step 5: Run the test to confirm it passes**

Run: `uv run pytest python/tests/dialects/test_movement_constprop.py -x`
Expected: all four tests PASS.

- [ ] **Step 6: Run the cz_partner module to confirm nothing else regressed**

Run: `uv run pytest python/tests/gemini/test_cz_partner.py -x`
Expected: PASS — the pipeline still resolves via `ResolveCzPartner`, the new impl is registered but no `CzPartner` has its `arch_spec` set yet, so const-prop returns `top()` and the existing walk handles resolution.

- [ ] **Step 7: Commit**

```bash
git add python/bloqade/gemini/common/dialects/movement/constprop.py \
        python/bloqade/gemini/common/dialects/movement/__init__.py \
        python/tests/dialects/test_movement_constprop.py
git commit -m "$(cat <<'EOF'
feat(gemini-movement): const-prop impl for CzPartner

Resolves cz_partner(loc) to const.Value(partner) when arch_spec is bound
on the statement and the address operand is constant. Not yet exercised:
the pipeline still uses ResolveCzPartner; the bind pass and rewiring
follow.
EOF
)"
```

---

## Task 3: Add `BindCzPartnerArchSpec` rewrite

**Files:**
- Modify: `python/bloqade/gemini/common/dialects/movement/rewrite.py:45-90`
- Test: `python/tests/gemini/test_cz_partner.py` (new test alongside the existing rewrite tests)

- [ ] **Step 1: Write the failing test**

Append to `python/tests/gemini/test_cz_partner.py` (before the `_build_kernel` helper):

```python
def test_bind_arch_spec_sets_attribute_on_unbound_statement():
    from bloqade.gemini.common.dialects.movement.rewrite import (
        BindCzPartnerArchSpec,
    )

    arch = get_physical_layout_arch_spec()
    addr = ir.TestValue()
    stmt = CzPartner(addr)
    assert stmt.arch_spec is None
    region = ir.Region([ir.Block([stmt])])

    rewrite.Walk(BindCzPartnerArchSpec(arch)).rewrite(region)

    assert stmt.arch_spec is arch


def test_bind_arch_spec_leaves_already_bound_statement_alone():
    from bloqade.gemini.common.dialects.movement.rewrite import (
        BindCzPartnerArchSpec,
    )

    arch_a = get_physical_layout_arch_spec()
    arch_b = get_physical_layout_arch_spec()
    assert arch_a is not arch_b

    addr = ir.TestValue()
    stmt = CzPartner(addr, arch_spec=arch_a)
    region = ir.Region([ir.Block([stmt])])

    rewrite.Walk(BindCzPartnerArchSpec(arch_b)).rewrite(region)

    assert stmt.arch_spec is arch_a
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run pytest python/tests/gemini/test_cz_partner.py::test_bind_arch_spec_sets_attribute_on_unbound_statement python/tests/gemini/test_cz_partner.py::test_bind_arch_spec_leaves_already_bound_statement_alone -x`
Expected: FAIL — `BindCzPartnerArchSpec` doesn't exist yet.

- [ ] **Step 3: Add the rewrite**

Append to `python/bloqade/gemini/common/dialects/movement/rewrite.py` (do NOT delete `ResolveCzPartner` yet — Task 6 does that):

```python
@dataclass
class BindCzPartnerArchSpec(RewriteRule):
    """Populate ``CzPartner.arch_spec`` from the pipeline's arch spec.

    Run once as a ``CallGraphPass`` before any folding so the const-prop impl
    can resolve every ``CzPartner`` during ``AggressiveUnroll``. Only binds
    statements whose attribute is still ``None`` — a statement that was
    constructed with an explicit arch spec is left alone.
    """

    arch_spec: ArchSpec

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, CzPartner) or node.arch_spec is not None:
            return RewriteResult()
        node.arch_spec = self.arch_spec
        return RewriteResult(has_done_something=True)
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run pytest python/tests/gemini/test_cz_partner.py::test_bind_arch_spec_sets_attribute_on_unbound_statement python/tests/gemini/test_cz_partner.py::test_bind_arch_spec_leaves_already_bound_statement_alone -x`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/gemini/common/dialects/movement/rewrite.py \
        python/tests/gemini/test_cz_partner.py
git commit -m "$(cat <<'EOF'
feat(gemini-movement): BindCzPartnerArchSpec rewrite

Populates CzPartner.arch_spec from the pipeline's arch spec, only when
the attribute is still None. ResolveCzPartner is still in place; the
pipeline rewires to use the new rule next.
EOF
)"
```

---

## Task 4: Wire the bind pass into `pipeline/base.py`

**Files:**
- Modify: `python/bloqade/lanes/pipeline/base.py:1-115`

- [ ] **Step 1: Update the imports**

Edit `python/bloqade/lanes/pipeline/base.py`:

- Remove line 10: `from bloqade.rewrite.passes.aggressive_unroll import Fold`
- Remove line 18: `from bloqade.gemini.common.dialects.movement.rewrite import ResolveCzPartner`
- After line 8 (`from bloqade.rewrite.passes import AggressiveUnroll`), insert:
  ```python
  from bloqade.rewrite.passes.callgraph import CallGraphPass
  ```
- After the (now-deleted) line 18, insert:
  ```python
  from bloqade.gemini.common.dialects.movement.rewrite import BindCzPartnerArchSpec
  ```

- [ ] **Step 2: Replace the resolve block in `emit`**

Edit `_NativeToPlaceBase.emit` (currently lines 70-83). Replace:

```python
def emit(self, mt: Method, no_raise: bool = True) -> Method:
    out = mt.similar(mt.dialects.add(place))
    out = self._pre_native_rewrites(mt, out, no_raise)

    out = SquinToNative().emit(out, no_raise=no_raise)
    AggressiveUnroll(out.dialects, no_raise=no_raise).fixpoint(out)

    if self.arch_spec is not None:
        # Resolve movement.cz_partner against the arch spec, then re-fold so
        # the resulting constant locations propagate into move_to lists
        # before they are validated / lowered.
        rewrite.Walk(ResolveCzPartner(self.arch_spec)).rewrite(out.code)
        Fold(out.dialects, no_raise=no_raise)(out)
```

With:

```python
def emit(self, mt: Method, no_raise: bool = True) -> Method:
    out = mt.similar(mt.dialects.add(place))
    out = self._pre_native_rewrites(mt, out, no_raise)

    if self.arch_spec is not None:
        # Bind arch_spec on every CzPartner reachable through the call graph
        # so const-prop resolves them during AggressiveUnroll.
        CallGraphPass(
            out.dialects, BindCzPartnerArchSpec(self.arch_spec)
        )(out)

    out = SquinToNative().emit(out, no_raise=no_raise)
    AggressiveUnroll(out.dialects, no_raise=no_raise).fixpoint(out)
```

The rest of the method (`_post_unroll_validation`, `ScfToCfRule`, `HoistConstants`, the validation suite, `_lower_qubits`, etc.) is unchanged.

- [ ] **Step 3: Run the end-to-end test**

Run: `uv run pytest python/tests/gemini/test_cz_partner.py::test_cz_partner_end_to_end_compiles_and_no_residual_nodes -x`
Expected: PASS — `BindCzPartnerArchSpec` runs as `CallGraphPass`, `AggressiveUnroll`'s fold resolves each `CzPartner` via const-prop, no residual `CzPartner` statements survive.

- [ ] **Step 4: Run the broader pipeline test suite**

Run: `uv run pytest python/tests/gemini/test_cz_partner.py python/tests/gemini/test_movement_kernel.py python/tests/test_validation_squin_kernels.py -x`
Expected: all pass. If anything fails, the most likely cause is that `CallGraphPass` placement is wrong (must be before `SquinToNative`) or that an arch-spec-less pipeline path silently lost validation — debug before continuing.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/pipeline/base.py
git commit -m "$(cat <<'EOF'
refactor(lanes): resolve cz_partner via CallGraphPass + const-prop

Replaces the ResolveCzPartner walk + trailing Fold in _NativeToPlaceBase
with a single CallGraphPass(BindCzPartnerArchSpec(arch_spec)) before
SquinToNative. AggressiveUnroll's built-in folding now handles
materialization through the const-prop impl added in the previous commit.
EOF
)"
```

---

## Task 5: Mirror the wiring in `upstream.py`

**Files:**
- Modify: `python/bloqade/lanes/upstream.py:1-108`

- [ ] **Step 1: Update the imports**

Edit `python/bloqade/lanes/upstream.py`:

- After line 20 (`from bloqade.lanes.rewrite import ...`), insert:
  ```python
  from bloqade.gemini.common.dialects.movement.rewrite import BindCzPartnerArchSpec
  ```
- Remove the two deferred imports inside `emit` (currently lines 48-52):
  ```python
  from bloqade.rewrite.passes.aggressive_unroll import Fold

  from bloqade.gemini.common.dialects.movement.rewrite import (
      ResolveCzPartner,
  )
  ```

- [ ] **Step 2: Replace the resolve block in `NativeToPlace.emit`**

Edit `NativeToPlace.emit` (currently lines 29-55). Replace:

```python
def emit(self, mt: Method, no_raise: bool = True):
    out = mt.similar(mt.dialects.add(place))
    if self.logical_initialize:
        rule = rewrite.Chain(
            rewrite.Walk(
                RewriteNonCliffordToU3(),
            ),
            rewrite.Walk(
                _RewriteU3ToInitialize(),
            ),
        )
        CallGraphPass(mt.dialects, rule)(out)

    out = SquinToNative().emit(out, no_raise=no_raise)
    AggressiveUnroll(out.dialects, no_raise=no_raise).fixpoint(out)

    if self.arch_spec is not None:
        # Resolve movement.cz_partner against the arch spec, then re-fold so
        # the resulting constant locations propagate into move_to lists.
        from bloqade.rewrite.passes.aggressive_unroll import Fold

        from bloqade.gemini.common.dialects.movement.rewrite import (
            ResolveCzPartner,
        )

        rewrite.Walk(ResolveCzPartner(self.arch_spec)).rewrite(out.code)
        Fold(out.dialects, no_raise=no_raise)(out)
```

With:

```python
def emit(self, mt: Method, no_raise: bool = True):
    out = mt.similar(mt.dialects.add(place))
    if self.logical_initialize:
        rule = rewrite.Chain(
            rewrite.Walk(
                RewriteNonCliffordToU3(),
            ),
            rewrite.Walk(
                _RewriteU3ToInitialize(),
            ),
        )
        CallGraphPass(mt.dialects, rule)(out)

    if self.arch_spec is not None:
        # Bind arch_spec on every CzPartner reachable through the call graph
        # so const-prop resolves them during AggressiveUnroll.
        CallGraphPass(
            out.dialects, BindCzPartnerArchSpec(self.arch_spec)
        )(out)

    out = SquinToNative().emit(out, no_raise=no_raise)
    AggressiveUnroll(out.dialects, no_raise=no_raise).fixpoint(out)
```

The rest of the method (`ScfToCfRule`, `HoistConstants`, the validation suite, the logical-initialize block, `RewritePlaceOperations`, etc.) is unchanged.

- [ ] **Step 3: Run the upstream-driven tests**

Run: `uv run pytest python/tests/test_validation_squin_kernels.py python/tests/gemini/test_movement_kernel.py -x`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add python/bloqade/lanes/upstream.py
git commit -m "$(cat <<'EOF'
refactor(lanes): wire BindCzPartnerArchSpec into NativeToPlace

Mirrors the pipeline/base.py change in upstream.py: replaces the
ResolveCzPartner walk + Fold with a single CallGraphPass before
SquinToNative so const-prop resolves cz_partner during AggressiveUnroll.
EOF
)"
```

---

## Task 6: Delete `ResolveCzPartner`

**Files:**
- Modify: `python/bloqade/gemini/common/dialects/movement/rewrite.py:45-90`
- Modify: `python/tests/gemini/test_cz_partner.py:11-13, 44-89`

- [ ] **Step 1: Remove `ResolveCzPartner` and its tests**

Edit `python/bloqade/gemini/common/dialects/movement/rewrite.py`:

- Delete the entire `@dataclass class ResolveCzPartner(...)` definition (currently lines 45-90).
- The remaining file should keep `RewriteLocationAttr`, its `dialect.rules.inference` registration, and the new `BindCzPartnerArchSpec` from Task 3.

Edit `python/tests/gemini/test_cz_partner.py`:

- In the imports block (lines 5-13), remove:
  ```python
  from bloqade.gemini.common.dialects.movement.rewrite import ResolveCzPartner
  ```
  and any unused imports it leaves behind (e.g. `py` from `kirin.dialects` if no longer referenced — check by running ruff).
- Delete the helper and two tests that exercise `ResolveCzPartner` (currently lines 44-89: `_const_loc_value`, `test_resolve_replaces_const_input_with_partner_loc`, `test_resolve_leaves_non_const_input_untouched`, plus the section comment block).

- [ ] **Step 2: Run the cz_partner tests**

Run: `uv run pytest python/tests/gemini/test_cz_partner.py -x`
Expected: PASS (the deleted tests are gone; remaining tests still pass).

- [ ] **Step 3: Run ruff to catch leftover unused imports**

Run: `uv run ruff check python/bloqade/gemini/common/dialects/movement/rewrite.py python/tests/gemini/test_cz_partner.py`
Expected: clean. Remove anything ruff flags as unused.

- [ ] **Step 4: Commit**

```bash
git add python/bloqade/gemini/common/dialects/movement/rewrite.py \
        python/tests/gemini/test_cz_partner.py
git commit -m "$(cat <<'EOF'
refactor(gemini-movement): drop ResolveCzPartner

The pipeline now resolves CzPartner through const-prop driven by
BindCzPartnerArchSpec — ResolveCzPartner is unreachable.
EOF
)"
```

---

## Task 7: Move address validation out of `@physical.kernel`

**Files:**
- Modify: `python/bloqade/gemini/physical/group.py:52-118`

- [ ] **Step 1: Strip `get_validation(arch_spec)` from the kernel decorator**

Edit `python/bloqade/gemini/physical/group.py`:

- Remove the import on (currently) line 106: `from bloqade.lanes.validation.address import get_validation`.
- Remove the explanatory comment on lines 108-110 (`# ``get_validation(arch_spec)`` is the move/movement address ...`).
- Remove the `get_validation(arch_spec),` entry from the `ValidationSuite` list (currently line 117).
- Update the `Doc(...)` on the `arch_spec` parameter (currently line 53) to reflect that the address check is now the pipeline's responsibility. Replace:
  ```python
  arch_spec: Annotated[
      ArchSpec | None, Doc("architecture spec for MoveToValidation")
  ] = None,
  ```
  with:
  ```python
  arch_spec: Annotated[
      ArchSpec | None,
      Doc(
          "architecture spec; reserved for future kernel-level checks. "
          "Address validation now runs inside PhysicalPipeline.emit."
      ),
  ] = None,
  ```
- The block guarded by `if arch_spec is None:` (currently lines 59-62) that falls back to `get_arch_spec()` becomes dead code for the validation path but may still be read by callers; leave it alone — `arch_spec` is still part of the kernel's public signature.

The resulting `ValidationSuite` should be:

```python
validator = ValidationSuite(
    [
        GeminiLogicalValidation,
        GeminiTerminalMeasurementValidation,
        FlatKernelNoCloningValidation,
        DuplicateAddressValidation,
    ]
)
```

- [ ] **Step 2: Run the gemini physical test set**

Run: `uv run pytest python/tests/gemini/ python/tests/test_validation_squin_kernels.py -x`
Expected: PASS. Failures here probably mean a structural validation regressed or that a test was relying on the address check firing at definition time — investigate before continuing.

- [ ] **Step 3: Commit**

```bash
git add python/bloqade/gemini/physical/group.py
git commit -m "$(cat <<'EOF'
refactor(gemini-physical): move address validation into the pipeline

@physical.kernel no longer runs get_validation(arch_spec) in its
ValidationSuite. PhysicalPipeline.emit already runs the full address
validation after BindCzPartnerArchSpec resolves cz_partner, so the
duplicate at kernel-definition time was both redundant and the reason
kernels using cz_partner had to pass verify=False.
EOF
)"
```

---

## Task 8: Drop `verify=False` from `cz_partner` kernels and add the no-partner regression

**Files:**
- Modify: `python/tests/gemini/test_cz_partner.py`

- [ ] **Step 1: Remove `verify=False` from the helper kernels**

In `python/tests/gemini/test_cz_partner.py`, replace every `@krn(verify=False)` and `@krn(aggressive_unroll=True, verify=False)` in `_build_kernel`, `with_partner`, and `hardcoded` with their `verify`-default forms:

- `@krn(verify=False)` → `@krn()`
- `@krn(aggressive_unroll=True, verify=False)` → `@krn(aggressive_unroll=True)`

There are six such decorators total (search for `verify=False` to confirm).

- [ ] **Step 2: Run the file to confirm the kernels still build with verify on**

Run: `uv run pytest python/tests/gemini/test_cz_partner.py -x`
Expected: PASS. The combination of "address validation moved out of the kernel decorator" + "CzPartner resolved via const-prop inside the pipeline" means `verify=True` no longer trips on `cz_partner`. If anything fails, the most likely cause is that some other validation in the decorator (e.g. `GeminiLogicalValidation`) doesn't tolerate the `cz_partner` shape — investigate.

- [ ] **Step 3: Add the no-partner end-to-end regression test**

Append to `python/tests/gemini/test_cz_partner.py`:

```python
def test_cz_partner_on_partnerless_location_fails_pipeline_emit():
    """A kernel that calls cz_partner on a location with no CZ partner in the
    arch spec must fail compilation (today: via the MoveTo non-const check;
    a future dedicated validator will give a more specific error)."""
    import pytest

    arch = get_physical_layout_arch_spec()
    partnerless = None
    for word_id in range(len(arch.words)):
        candidate = LocationAddress(zone_id=0, word_id=word_id, site_id=0)
        if arch.get_cz_partner(candidate) is None:
            partnerless = (candidate.word_id, candidate.site_id)
            break
    assert partnerless is not None, "no partnerless location in physical arch"
    partnerless_word, partnerless_site = partnerless

    krn = physical.kernel

    @krn(aggressive_unroll=True)
    def main():
        bad = movement.loc(0, partnerless_word, partnerless_site)
        q = qubit.new_at(0, partnerless_word, partnerless_site)
        movement.move_to([q], [movement.cz_partner(bad)])

    strat = make_physical_placement_strategy(return_moves=False)
    pipeline = PhysicalPipeline(placement_strategy=strat)
    with pytest.raises(Exception):
        pipeline.emit(main, no_raise=False)
```

- [ ] **Step 4: Run the new test**

Run: `uv run pytest python/tests/gemini/test_cz_partner.py::test_cz_partner_on_partnerless_location_fails_pipeline_emit -x`
Expected: PASS. If the test passes spuriously (no exception raised), the const-prop impl is silently producing a value for the partnerless location — debug; the impl in `constprop.py` must return `top()` when `arch_spec.get_cz_partner(addr) is None`.

- [ ] **Step 5: Commit**

```bash
git add python/tests/gemini/test_cz_partner.py
git commit -m "$(cat <<'EOF'
test(gemini): cz_partner kernels run with verify=True

Drop verify=False from every cz_partner test kernel — the refactor made
this unnecessary — and add a regression test that pipeline.emit fails
when cz_partner targets a location with no CZ partner.
EOF
)"
```

---

## Task 9: Full test suite + lint sweep

- [ ] **Step 1: Run the full Python test suite**

Run: `just test-python` (or `uv run coverage run -m pytest python/tests` if `just` is unavailable).
Expected: all tests pass.

- [ ] **Step 2: Run lint and formatting**

Run in parallel:
- `uv run isort python`
- `uv run black python`
- `uv run ruff check python`
- `uv run pyright python`

Expected: all clean. Auto-formatters may rewrite imports — review the diff and commit any cosmetic changes.

- [ ] **Step 3: Commit any formatting changes**

```bash
git status
# If anything is modified by isort/black:
git add -u python/
git commit -m "style: isort/black after cz_partner refactor"
```

(Skip the commit if `git status` is clean.)

- [ ] **Step 4: Optional sanity check**

Run: `uv run pytest python/tests/gemini/test_cz_partner.py python/tests/dialects/test_movement_constprop.py -v`
Expected: a green list of every test added or modified by this plan.

---

## Notes for the Implementer

- **`ir.TestValue` and helpers**: `ir.TestValue` is a fixture-friendly SSA stand-in; `kirin.analysis.const.Value` / `const.Result.top()` are the lattice constructors. Existing tests in this repo use both.
- **`dialect.register(key=...)`**: the dialect's `registry` indexes method tables by key. The `constprop` key is consumed by `kirin.analysis.const.prop.Propagate`. The `interp.impl(StmtClass)` decorator attaches a function to a method table for a specific statement.
- **`CallGraphPass`**: `bloqade.rewrite.passes.callgraph.CallGraphPass(dialects, rule)(method)` walks the call graph reachable from `method` and applies `rule` to each method in topological order. The pre-existing usage in `upstream.py:40` is a good reference.
- **`info.attribute(default=None)`**: Kirin wraps the Python value into a `PyAttr` automatically. `stmt.arch_spec` returns the bare Python `ArchSpec` (or `None`); setting it back rewraps. No special handling needed.
- **Order matters for `BindCzPartnerArchSpec`**: it must run before any fold pass. Placing it before `SquinToNative` keeps the lifecycle continuous — the attribute is set as soon as `CzPartner` exists in the IR.
- **`Fold` import**: present in `pipeline/base.py` before this plan. Removing it after Task 4 leaves no other callers in that file.
