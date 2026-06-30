# `cz_partner` Resolution via Pure Statement + Const-Prop Design

**Goal:** Refactor how `movement.cz_partner(loc)` materializes its result so the resolution rides on Kirin's standard const-prop / fold machinery instead of a bespoke walk-based rewrite. This makes the materialization correct under future call-graph traversal of `ilist.map` bodies ([bloqade-circuit#830](https://github.com/QuEraComputing/bloqade-circuit/issues/830)) and lays the groundwork for a follow-up "no CZ partner" validation pass.

**Architecture:** `CzPartner` becomes `ir.Pure` with an `arch_spec` statement attribute (default `None`). A const-prop impl registered on the movement dialect resolves the statement when both the attribute and the address operand are constant. A new `BindCzPartnerArchSpec` rewrite — run as a `CallGraphPass` before `SquinToNative` in both `pipeline/base.py` and `upstream.py` — populates the attribute from the pipeline's arch spec. The existing `ResolveCzPartner` rewrite, its companion `Fold` re-pass, and the address validation currently invoked by `@physical.kernel`'s `run_pass` are removed; the pipeline becomes the single owner of address validation.

**Tech Stack:** Kirin IR framework (`@statement`, `info.attribute`, `ir.Pure`, `interp.MethodTable`, `@interp.impl`, `kirin.analysis.const`), `bloqade.rewrite.passes.callgraph.CallGraphPass`, `bloqade.lanes.arch.spec.ArchSpec`.

---

## File Layout

| File | Status | Responsibility |
|------|--------|----------------|
| `python/bloqade/gemini/common/dialects/movement/stmts.py` | **EDIT** | Add `ir.Pure()` to `CzPartner`; add `arch_spec: ArchSpec \| None` attribute; rewrite the existing comment to describe the new resolution model |
| `python/bloqade/gemini/common/dialects/movement/rewrite.py` | **EDIT** | Delete `ResolveCzPartner`; add `BindCzPartnerArchSpec` |
| `python/bloqade/gemini/common/dialects/movement/__init__.py` | **EDIT** | Register a `key="constprop"` `MethodTable` with a `CzPartner` impl (new file `constprop.py` imported by the package init) |
| `python/bloqade/gemini/common/dialects/movement/constprop.py` | **NEW** | Const-prop method table for the movement dialect (currently just `CzPartner`) |
| `python/bloqade/lanes/pipeline/base.py` | **EDIT** | Replace the `ResolveCzPartner` walk + `Fold` block with a `CallGraphPass(BindCzPartnerArchSpec(...))` run before `SquinToNative` |
| `python/bloqade/lanes/upstream.py` | **EDIT** | Same wiring change for `NativeToPlace.emit` |
| `python/bloqade/gemini/physical/group.py` | **EDIT** | Drop `get_validation(arch_spec)` from the `ValidationSuite` in `run_pass`; the pipeline owns address validation |
| `python/tests/gemini/test_cz_partner.py` | **EDIT** | Drop `verify=False` on kernels that only used it because of `cz_partner` |
| `python/tests/...` | **EDIT** | New const-prop unit test for `CzPartner` (resolved / unbound / no-partner cases); regression test that the residual-CzPartner check still passes |

---

## Statement Definition

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
    locations ilist).
    """

    name = "cz_partner"
    traits = frozenset({ir.Pure(), lowering.FromPythonCall()})
    address: ir.SSAValue = info.argument(LocationAddressType)
    arch_spec: ArchSpec | None = info.attribute(default=None)
    result: ir.ResultValue = info.result(LocationAddressType)
```

`arch_spec` defaults to `None` so kernels lower without needing arch-spec context. The attribute is stored as a Kirin `PyAttr` (same wrapping `LocationAddress` / `LaneAddress` get in `lanes.dialects.stack_move` / `lanes.dialects.move`).

The existing block-comment about purity is replaced — the previous reasoning (Pure would hoist the `ilist.map` closure beyond the walk's reach) no longer applies because materialization no longer needs a walk: it rides on the standard const-prop / fold machinery, which evaluates the closure body when the map is unrolled / inlined.

---

## Const-Prop Impl

```python
# python/bloqade/gemini/common/dialects/movement/constprop.py
from kirin import interp
from kirin.analysis import const

from bloqade.lanes.bytecode.encoding import LocationAddress

from ._dialect import dialect
from .stmts import CzPartner


@dialect.register(key="constprop")
class CzPartnerConstProp(interp.MethodTable):
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

The three "unresolved" branches return `const.Result.top()` (Unknown), keeping downstream `move_to.locations` non-const. The MoveTo validation already reports non-const locations, so until the dedicated "no CZ partner" validator lands (issue 1), unresolved `CzPartner` still surfaces as a compilation error — just via the existing MoveTo path.

---

## Bind Rewrite

```python
# python/bloqade/gemini/common/dialects/movement/rewrite.py
@dataclass
class BindCzPartnerArchSpec(RewriteRule):
    """Populate ``CzPartner.arch_spec`` from the pipeline's arch spec.

    Run once as a ``CallGraphPass`` before any folding so the const-prop impl
    can resolve every ``CzPartner`` during ``AggressiveUnroll``. Only binds
    statements whose attribute is still ``None`` — a kernel that explicitly
    constructed a ``CzPartner`` with a different arch spec is left alone.
    """

    arch_spec: ArchSpec

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, CzPartner) or node.arch_spec is not None:
            return RewriteResult()
        node.arch_spec = self.arch_spec
        return RewriteResult(has_done_something=True)
```

`ResolveCzPartner` is deleted (along with its import).

---

## Pipeline Wiring

### `python/bloqade/lanes/pipeline/base.py` (`_NativeToPlaceBase.emit`)

Before (lines 70-83):

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

After:

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

The standalone `Fold` re-pass and the unused `Fold` import go away. `AggressiveUnroll`'s built-in folding now also resolves `CzPartner`. The post-unroll `ValidationSuite([DuplicateAddressValidation, get_validation(self.arch_spec)])` (line 91) is unchanged.

### `python/bloqade/lanes/upstream.py` (`NativeToPlace.emit`)

The same shape replaces lines 30-55: `CallGraphPass(BindCzPartnerArchSpec)` moves up to run before `SquinToNative`, the trailing `ResolveCzPartner` walk and `Fold` block are deleted, and the deferred imports inside the block go with them.

---

## Validation Ownership

Address validation (`get_validation(arch_spec)`) is removed from `gemini/physical/group.py:run_pass` (currently lines 106-118). The kernel decorator keeps the validations that don't depend on `CzPartner` being resolved:

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

The `arch_spec` parameter on `run_pass` becomes effectively unused inside the validation block; it stays for backward compatibility of the decorator signature (`arch_spec: ArchSpec | None = None`) until a separate cleanup, but the `Doc("architecture spec for MoveToValidation")` comment is updated to reflect that the address check is now the pipeline's responsibility.

`PhysicalPipeline.emit` already runs the full `ValidationSuite([DuplicateAddressValidation, get_validation(self.arch_spec)])` after the unroll/fold stage (post-resolution), so the address validation still happens — just at the right point in the pipeline.

**Trade-off:** physical kernels that are compiled with the `@physical.kernel` decorator but never go through `PhysicalPipeline.emit` lose address validation entirely. This was always a partial coverage gap (it didn't cover the move dialect; only the movement dialect). Callers who want pre-pipeline address validation can invoke the suite directly. This trade-off is acceptable for now and is the explicit motivation for the move.

---

## Pipeline Order, in One Place

After this change, the relevant prefix of both `pipeline/base.py` and `upstream.py` looks like:

1. `out = mt.similar(mt.dialects.add(place))`
2. `_pre_native_rewrites` hook (logical / physical specific)
3. **`CallGraphPass(BindCzPartnerArchSpec(arch_spec))(out)`** — bind on every reachable `CzPartner`
4. `SquinToNative().emit(out)`
5. `AggressiveUnroll(...).fixpoint(out)` — const-prop / fold now resolves `CzPartner`
6. `_post_unroll_validation` hook
7. `ScfToCfRule` → `HoistConstants`
8. `ValidationSuite([DuplicateAddressValidation, get_validation(arch_spec)])` — runs against resolved IR
9. `_lower_qubits` hook → `RewritePlaceOperations` → DCE/CSE → discard squin/qubit dialects → TypeInfer

`SquinToNative` does not introduce `CzPartner` statements, so binding before it is correct; placing the bind earliest keeps the lifecycle of "arch_spec attribute is set" continuous through every later pass.

---

## Tests

Unit tests for the new const-prop impl belong with the movement dialect tests:

- `arch_spec=None` → result is `const.Result.top()`.
- `arch_spec` set, `address` is `const.Value(LocationAddress)` with a partner → result is `const.Value(partner)`.
- `arch_spec` set, address has a partner of `None` → result is `const.Result.top()`.
- `arch_spec` set, `address` is unknown → result is `const.Result.top()`.

End-to-end tests in `python/tests/gemini/test_cz_partner.py` drop `verify=False` from the kernels that only used it because of `cz_partner` (e.g. `test_cz_partner_end_to_end_compiles_and_no_residual_nodes`, `test_cz_partner_matches_hardcoded_partner_words`). The existing assertion that no residual `CzPartner` survives compilation continues to hold.

A targeted regression test confirms that a kernel using `cz_partner` on a location **with no partner** raises during pipeline emit (today: via the MoveTo non-const error; after issue 1 lands: via the new "no CZ partner" validator).

---

## Follow-Ups (Out of Scope for This Spec)

- **Issue 1 — dedicated "no CZ partner" validation pass.** Walks the post-bind IR and reports every `CzPartner` whose `arch_spec.get_cz_partner(addr)` returns `None`, producing a specific error instead of the generic MoveTo "locations must be compile-time constants" failure. Becomes straightforward once this refactor lands.
- **bloqade-circuit#830.** Once call-graph rewrites descend into `ilist.map` function bodies, the `CallGraphPass(BindCzPartnerArchSpec(...))` here automatically reaches `CzPartner` statements inside `ilist.map` closures. No code change required in this repo when #830 lands.
