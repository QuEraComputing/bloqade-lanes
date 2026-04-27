# Explicit Qubit Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a user-facing `bloqade.gemini.operations.new_at(zone, word, site)` statement that pins a logical qubit to an explicit physical address, while making `place.NewLogicalQubit.location_address` the canonical source of truth for where each qubit lives.

**Architecture:** The new gemini statement carries runtime SSA int args; eager validation enforces const-foldability, range, and uniqueness. Augmented circuit→place rewrites stamp constants onto a new optional `location_address` attribute on `place.NewLogicalQubit`. After `LayoutAnalysis` runs, a new `ResolvePinnedAddresses` rewrite fills in the heuristic's picks for un-pinned qubits — establishing the post-resolve invariant that every `NewLogicalQubit` has a concrete address. Downstream `InsertInitialize` / `InsertFill` rewrites read attributes directly instead of consulting analysis frames.

**Tech Stack:** Python 3.10+, Kirin IR framework, Maturin/PyO3 (Rust ↔ Python), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-04-27-explicit-qubit-allocation-design.md`

---

## Human-Review Checkpoints — two tiers

This plan has **two tiers of human review**:

1. **Per-task reviews (implicit):** Every task ends with a `git commit`. After each commit, the executor pauses and the human inspects the diff before the next task begins. If using subagent-driven-development, this is automatic — a fresh subagent runs each task and the human reviews between subagents. If using inline execution, treat each commit as a natural pause point.

2. **Per-phase checkpoints (explicit):** At the end of each phase, the plan contains a 🔍 **Human Review Checkpoint** with a focused list of what to verify (architectural correctness, regression gates, API decisions). The executor MUST stop at these and wait for explicit approval before starting the next phase. These are stronger gates than per-task reviews.

The per-task reviews catch implementation slips; the per-phase checkpoints catch design drift before it compounds across phases.

| Phase | Theme | Files touched | Review focus |
|---|---|---|---|
| A | Add `location_address` attribute | `dialects/place.py`, tests | IR shape, default behavior |
| B | Heuristic API + `LayoutAnalysis` + `ResolvePinnedAddresses` | `analysis/layout/`, `heuristics/`, new rewrite | API change correctness, no regressions |
| C | Place→Move consumer refactor | `rewrite/place2move.py`, `upstream.py` | Byte-identical move IR for un-pinned kernels |
| D | Gemini IR + lowering | `bloqade/gemini/...`, `rewrite/circuit2place.py` | New statement plumbing; rewrites stamp address |
| E | Eager validation | `bloqade/gemini/...` | Errors at correct source line |
| F | Integration + demo | E2E tests, `demo/` | Full-pipeline correctness |

---

## File Structure

### Files to create

| Path | Responsibility |
|---|---|
| `python/bloqade/gemini/logical/dialects/operations/_new_at_lowering.py` | Python `@lowering.wraps` interface for `NewAt` (so users can call `operations.new_at(z, w, s)` from a kernel) |
| `python/bloqade/gemini/logical/validation/__init__.py` | Package init |
| `python/bloqade/gemini/logical/validation/new_at.py` | Per-stmt method-table impl (const + range checks) registered against `"move.address.validation"` |
| `python/bloqade/gemini/logical/validation/duplicates.py` | Cross-statement duplicate-address validation pass |
| `python/bloqade/lanes/rewrite/resolve_pinned.py` | The `ResolvePinnedAddresses` rewrite |
| `python/tests/dialects/place/__init__.py` | Test package init |
| `python/tests/dialects/place/test_new_logical_qubit.py` | Round-trip the new attribute |
| `python/tests/rewrite/test_resolve_pinned.py` | Tests for `ResolvePinnedAddresses` |
| `python/tests/gemini/dialects/__init__.py` | Test package init |
| `python/tests/gemini/dialects/test_new_at.py` | Tests for the new statement |
| `python/tests/gemini/validation/__init__.py` | Test package init |
| `python/tests/gemini/validation/test_new_at_validation.py` | Tests for per-stmt + duplicate-address validation |
| `python/tests/integration/__init__.py` | Test package init |
| `python/tests/integration/test_explicit_allocation.py` | E2E mixed-pinning + regression tests |
| `demo/explicit_allocation.py` | Demo script |

### Files to modify

| Path | Reason |
|---|---|
| `python/bloqade/lanes/dialects/place.py` | Add `location_address` attribute to `NewLogicalQubit` |
| `python/bloqade/lanes/analysis/layout/analysis.py` | Heuristic `pinned` parameter; collect from IR; hard-fail on no-legal-layout |
| `python/bloqade/lanes/heuristics/simple_layout.py` | Accept `pinned` parameter |
| `python/bloqade/lanes/heuristics/physical/layout.py` | Accept `pinned` parameter |
| `python/bloqade/lanes/heuristics/logical/layout.py` | Accept `pinned` parameter (two impls) |
| `python/bloqade/lanes/rewrite/circuit2place.py` | Augment `RewriteLogicalInitializeToNewLogical` and `InitializeNewQubits` to handle `NewAt` |
| `python/bloqade/lanes/rewrite/place2move.py` | Refactor `InsertInitialize` and `InsertFill` to read attributes directly |
| `python/bloqade/lanes/upstream.py` | Wire `ResolvePinnedAddresses`; remove now-dead `address_entries`/`initial_layout` plumbing; invoke eager validation |
| `python/bloqade/gemini/logical/dialects/operations/stmts.py` | Add `NewAt` statement |
| `python/bloqade/gemini/logical/dialects/operations/__init__.py` | Re-export `new_at` |
| `python/bloqade/gemini/__init__.py` (or appropriate subpackage init) | Wire validation registration |

---

## Conventions

- **Run tests** with `uv run pytest <path> -v`. Run the full suite with `just test-python`.
- **Pre-commit hooks** run black, isort, ruff, pyright. Don't bypass with `-n` unless explicitly told.
- **Commit style** — Conventional Commits: `feat(scope): ...`, `refactor(scope): ...`, `test(scope): ...`. Scope from the spec: `gemini`, `lanes`, `place`, `move`, `validation`.
- **Branch** — work happens on `worktree-feat-explicit-qubit-allocation`.

---

## Phase A — Lanes IR foundation

The smallest possible change: add an optional attribute. No behavior change yet.

### Task A1: Add `location_address` attribute to `place.NewLogicalQubit`

**Files:**
- Modify: `python/bloqade/lanes/dialects/place.py:36-51`
- Create: `python/tests/dialects/place/__init__.py` (empty)
- Create: `python/tests/dialects/place/test_new_logical_qubit.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/dialects/place/__init__.py` (empty file).

Create `python/tests/dialects/place/test_new_logical_qubit.py`:

```python
from kirin import ir, types
from kirin.dialects import py

from bloqade.lanes.dialects import place
from bloqade.lanes.layout.encoding import LocationAddress


def _make_zero() -> ir.SSAValue:
    return py.Constant(0.0).result


def test_new_logical_qubit_default_location_address_is_none():
    stmt = place.NewLogicalQubit(_make_zero(), _make_zero(), _make_zero())
    assert stmt.location_address is None


def test_new_logical_qubit_accepts_location_address():
    addr = LocationAddress(0, 1, 2)
    stmt = place.NewLogicalQubit(
        _make_zero(),
        _make_zero(),
        _make_zero(),
        location_address=addr,
    )
    assert stmt.location_address == addr
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest python/tests/dialects/place/test_new_logical_qubit.py -v
```

Expected: FAIL — `NewLogicalQubit.__init__()` does not accept `location_address`.

- [ ] **Step 3: Add the attribute**

Modify `python/bloqade/lanes/dialects/place.py:36-51`:

```python
@statement(dialect=dialect)
class NewLogicalQubit(ir.Statement):
    """Allocate new logical qubits with initial state u3(theta, phi, lam)|0>.

    Args:
        theta (float): Angle for rotation around the Y axis
        phi (float): angle for rotation around the Z axis
        lam (float): angle for rotation around the Z axis
        location_address (LocationAddress | None): Pinned physical address; None means
            the layout heuristic chooses. After ResolvePinnedAddresses runs, this is
            never None in well-formed IR.
    """

    traits = frozenset()
    theta: ir.SSAValue = info.argument(types.Float)
    phi: ir.SSAValue = info.argument(types.Float)
    lam: ir.SSAValue = info.argument(types.Float)
    location_address: LocationAddress | None = info.attribute(default=None)
    result: ir.ResultValue = info.result(bloqade_types.QubitType)
```

Add the import at the top of the file (next to existing imports):

```python
from bloqade.lanes.layout.encoding import LocationAddress
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest python/tests/dialects/place/test_new_logical_qubit.py -v
```

Expected: PASS, both test cases.

- [ ] **Step 5: Run the existing test corpus to confirm no regressions**

```
just test-python
```

Expected: all existing tests still pass. The default value of `None` means existing call sites are unaffected.

- [ ] **Step 6: Commit**

```bash
git add python/bloqade/lanes/dialects/place.py python/tests/dialects/place/
git commit -m "$(cat <<'EOF'
feat(place): add optional location_address attribute to NewLogicalQubit

Defaults to None (current behavior). Pinned-address support builds on top
of this in subsequent phases.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### 🔍 Human Review Checkpoint A

**Stop here.** Hand the diff back to the human and wait for approval.

Things to check:
- Does `info.attribute(default=None)` in this codebase support `LocationAddress | None`? (It should — `LocationAddress` is a Kirin-friendly attribute already.)
- Does the import order match isort rules?
- Does pyright pass on the modified file?

**Resume only after the human gives explicit approval.**

---

## Phase B — Heuristic API + LayoutAnalysis driver + ResolvePinnedAddresses

This phase makes pinned addresses *internally* meaningful: the heuristic respects them, the analysis collects them from IR, and a new rewrite stamps heuristic picks back into attributes after layout. No user-facing surface yet.

### Task B1: Update `LayoutHeuristicABC.compute_layout` signature

**Files:**
- Modify: `python/bloqade/lanes/analysis/layout/analysis.py:14-37`

- [ ] **Step 1: Update the abstract signature**

Modify `python/bloqade/lanes/analysis/layout/analysis.py`:

```python
@dataclass
class LayoutHeuristicABC(abc.ABC):

    @abc.abstractmethod
    def compute_layout(
        self,
        all_qubits: tuple[int, ...],
        stages: list[tuple[tuple[int, int], ...]],
        pinned: dict[int, LocationAddress] | None = None,
    ) -> tuple[LocationAddress, ...]:
        """
        Compute the initial qubit layout from circuit stages.

        Args:
            all_qubits: Tuple of logical qubit indices to be mapped.
            stages: List of circuit stages, where each stage is a tuple of
                (control, target) qubit pairs representing two-qubit gates.
            pinned: Map from logical qubit ID to pre-pinned LocationAddress.
                Implementations MUST place each pinned qubit at its requested
                address and MUST NOT use any address in pinned.values() for
                un-pinned qubits. None or empty preserves previous behavior.

        Returns:
            A tuple of LocationAddress objects mapping logical qubit indices
            to physical locations. Pinned IDs return their pinned address;
            un-pinned IDs return the heuristic's choice. Raises if no legal
            layout exists.
        """
        ...  # pragma: no cover
```

- [ ] **Step 2: Run the existing test corpus to identify what breaks**

```
just test-python
```

Expected: tests for existing heuristics fail with "abstract method not implemented" — the existing impls don't yet accept `pinned`. We'll fix them in B2.

- [ ] **Step 3: Commit**

```bash
git add python/bloqade/lanes/analysis/layout/analysis.py
git commit -m "$(cat <<'EOF'
refactor(lanes): add optional pinned param to LayoutHeuristicABC.compute_layout

Concrete implementations updated in subsequent commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B2: Update existing `LayoutHeuristicABC` implementations

There are four implementations to update. Each gets the same treatment: accept `pinned`, pre-stamp pinned IDs into the output, restrict the heuristic's search to addresses not in `pinned.values()`, and raise if no legal layout exists for un-pinned qubits.

**Files:**
- Modify: `python/bloqade/lanes/heuristics/simple_layout.py:30`
- Modify: `python/bloqade/lanes/heuristics/physical/layout.py:434`
- Modify: `python/bloqade/lanes/heuristics/logical/layout.py:78,108` (two impls)

#### B2a: `simple_layout.py`

- [ ] **Step 1: Read the existing impl to understand its strategy**

Read `python/bloqade/lanes/heuristics/simple_layout.py` end-to-end so you know what it does (it's small).

- [ ] **Step 2: Add the parameter and respect it**

The pattern is:

1. Accept `pinned: dict[int, LocationAddress] | None = None`.
2. Normalize `pinned = pinned or {}`.
3. Compute the existing-style layout for `all_qubits`, but exclude `pinned.values()` from the candidate addresses the heuristic can pick.
4. After computation, build the final tuple: for each `i in all_qubits`, return `pinned[i]` if `i in pinned`, else the heuristic's choice.
5. If no valid layout exists for the un-pinned set under exclusion, raise `ValueError(f"layout heuristic cannot place {len(unpinned)} un-pinned qubits given {len(pinned)} pinned addresses; no legal positions remain")`.

Apply this pattern to the existing impl. The exact code depends on the heuristic's internal structure — read the existing code, then make the minimal changes.

- [ ] **Step 3: Add a unit test for the new parameter**

Find or create the test file for this heuristic (look under `python/tests/heuristics/`). Add tests for:
- Empty `pinned` produces byte-identical output to previous behavior (regression).
- Non-empty `pinned` returns pinned addresses at pinned indices and never returns those addresses for un-pinned indices.
- Over-constraining `pinned` raises with a message containing "no legal positions remain".

- [ ] **Step 4: Run tests**

```
uv run pytest python/tests/heuristics/<test_file>.py -v
```

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/heuristics/simple_layout.py python/tests/heuristics/
git commit -m "refactor(heuristics): SimpleLayout respects pinned in compute_layout"
```

#### B2b: `physical/layout.py`

Same pattern as B2a applied to the physical heuristic. Look at lines around 434 and `_compute_layout_from_cz_layers` (line 417) — the exclusion-set logic likely lives in the helper.

- [ ] **Step 1-5:** as in B2a, applied to `python/bloqade/lanes/heuristics/physical/layout.py`.

- [ ] **Step 6: Commit**

```bash
git commit -m "refactor(heuristics): PhysicalLayout respects pinned in compute_layout"
```

#### B2c: `logical/layout.py` (both impls)

Same pattern, applied to both `compute_layout` methods (lines 78 and 108) and the helper `_compute_layout_from_weighted_edges` (line 46).

- [ ] **Step 1-5:** as in B2a.

- [ ] **Step 6: Commit**

```bash
git commit -m "refactor(heuristics): LogicalLayout impls respect pinned in compute_layout"
```

---

### Task B3: `LayoutAnalysis` driver — collect pinned from IR

**Files:**
- Modify: `python/bloqade/lanes/analysis/layout/analysis.py:39-79`

- [ ] **Step 1: Write the failing test**

Create `python/tests/analysis/layout/__init__.py` (empty) if it doesn't exist.

Create `python/tests/analysis/layout/test_pinned_collection.py`:

```python
from kirin import ir
from kirin.dialects import py

from bloqade.analysis import address
from bloqade.lanes.analysis.layout import LayoutAnalysis
from bloqade.lanes.dialects import place
from bloqade.lanes.heuristics.simple_layout import SimpleLayout  # adjust import to actual class
from bloqade.lanes.layout.encoding import LocationAddress


# This is a unit test of LayoutAnalysis._collect_pinned only.
# It does not require running the full forward analysis.

def test_collect_pinned_extracts_addresses_from_attribute():
    # Build three NewLogicalQubit stmts: one pinned, one unpinned, one pinned.
    addr_a = LocationAddress(0, 1, 2)
    addr_c = LocationAddress(0, 3, 4)
    z = py.Constant(0.0).result

    stmt_a = place.NewLogicalQubit(z, z, z, location_address=addr_a)
    stmt_b = place.NewLogicalQubit(z, z, z)  # unpinned
    stmt_c = place.NewLogicalQubit(z, z, z, location_address=addr_c)

    address_entries = {
        stmt_a.result: address.AddressQubit(0),
        stmt_b.result: address.AddressQubit(1),
        stmt_c.result: address.AddressQubit(2),
    }

    analysis = LayoutAnalysis(
        dialects=None,  # not used by _collect_pinned
        heuristic=None,
        address_entries=address_entries,
        all_qubits=(0, 1, 2),
    )

    method = ...  # build a kirin Method whose body contains stmt_a, stmt_b, stmt_c
    pinned = analysis._collect_pinned(method)
    assert pinned == {0: addr_a, 2: addr_c}
```

(Filling in the `method = ...` construction requires picking up the patterns used in existing `tests/analysis/` files — copy a minimal block construction.)

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest python/tests/analysis/layout/test_pinned_collection.py -v
```

Expected: FAIL — `_collect_pinned` doesn't exist yet.

- [ ] **Step 3: Implement `_collect_pinned` and use it**

In `python/bloqade/lanes/analysis/layout/analysis.py`, add:

```python
def _collect_pinned(self, method: ir.Method) -> dict[int, LocationAddress]:
    """Walk the method's IR and collect pinned addresses from
    place.NewLogicalQubit.location_address attributes."""
    from bloqade.lanes.dialects import place  # local import to avoid cycle
    pinned: dict[int, LocationAddress] = {}
    for stmt in method.callable_region.walk():
        if not isinstance(stmt, place.NewLogicalQubit):
            continue
        if stmt.location_address is None:
            continue
        addr_entry = self.address_entries.get(stmt.result)
        if not isinstance(addr_entry, address.AddressQubit):
            continue
        pinned[addr_entry.data] = stmt.location_address
    return pinned
```

Update `process_results` (currently line 67) to take `method` and pass `pinned`:

```python
def process_results(self, method: ir.Method):
    pinned = self._collect_pinned(method)
    layout = self.heuristic.compute_layout(
        self.all_qubits, self.stages, pinned=pinned
    )
    return layout

def get_layout_no_raise(self, method: ir.Method):
    self.run_no_raise(method)
    return self.process_results(method)

def get_layout(self, method: ir.Method):
    self.run(method)
    return self.process_results(method)
```

- [ ] **Step 4: Run the test to verify it passes**

```
uv run pytest python/tests/analysis/layout/test_pinned_collection.py -v
```

- [ ] **Step 5: Run the full corpus**

```
just test-python
```

Expected: all pass — no kernel currently sets `location_address`, so `pinned` is always empty, and `compute_layout` is called with `pinned={}`, which (per B2) is byte-identical to the prior behavior.

- [ ] **Step 6: Commit**

```bash
git add python/bloqade/lanes/analysis/layout/analysis.py python/tests/analysis/layout/
git commit -m "feat(lanes): LayoutAnalysis collects pinned addresses from NewLogicalQubit"
```

---

### Task B4: Hard-failure surface in `LayoutAnalysis`

**Files:**
- Modify: `python/bloqade/lanes/analysis/layout/analysis.py` (the `process_results` method just edited)

- [ ] **Step 1: Write the failing test**

Append to `python/tests/analysis/layout/test_pinned_collection.py`:

```python
import pytest


def test_overconstraining_pins_raises_at_layout_analysis_level():
    # Construct an arch and pins such that the heuristic cannot find a layout.
    # Use a small ArchSpec where total capacity == number of pins, leaving zero
    # for un-pinned qubits.
    ...  # see existing tests for ArchSpec construction
    method = ...
    with pytest.raises(Exception) as exc_info:
        analysis.get_layout(method)
    assert "no legal positions remain" in str(exc_info.value).lower() or \
           "cannot place" in str(exc_info.value).lower()
```

- [ ] **Step 2: Run to verify it fails (or already passes if B2 raises propagate)**

```
uv run pytest python/tests/analysis/layout/test_pinned_collection.py::test_overconstraining_pins_raises_at_layout_analysis_level -v
```

If the heuristic already raises and the analysis lets it propagate, this test may pass without code changes. If not, wrap the heuristic call:

```python
def process_results(self, method: ir.Method):
    pinned = self._collect_pinned(method)
    try:
        layout = self.heuristic.compute_layout(
            self.all_qubits, self.stages, pinned=pinned
        )
    except Exception as e:
        raise type(e)(
            f"layout heuristic cannot satisfy pinned addresses for this kernel: {e}"
        ) from e
    return layout
```

- [ ] **Step 3: Re-run the test**

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add python/bloqade/lanes/analysis/layout/analysis.py python/tests/analysis/layout/
git commit -m "feat(lanes): hard-fail LayoutAnalysis when pins admit no legal layout"
```

---

### Task B5: `ResolvePinnedAddresses` rewrite

**Files:**
- Create: `python/bloqade/lanes/rewrite/resolve_pinned.py`
- Create: `python/tests/rewrite/test_resolve_pinned.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/rewrite/test_resolve_pinned.py`:

```python
from kirin import ir
from kirin.dialects import py

from bloqade.analysis import address
from bloqade.lanes.dialects import place
from bloqade.lanes.layout.encoding import LocationAddress
from bloqade.lanes.rewrite.resolve_pinned import ResolvePinnedAddresses


def _zero():
    return py.Constant(0.0).result


def test_unpinned_qubits_get_attribute_filled_from_layout():
    # Build a block with three NewLogicalQubit stmts: A pinned, B unpinned, C pinned.
    addr_a = LocationAddress(0, 1, 2)
    addr_b_chosen_by_heuristic = LocationAddress(0, 0, 0)
    addr_c = LocationAddress(0, 3, 4)

    stmt_a = place.NewLogicalQubit(_zero(), _zero(), _zero(), location_address=addr_a)
    stmt_b = place.NewLogicalQubit(_zero(), _zero(), _zero())
    stmt_c = place.NewLogicalQubit(_zero(), _zero(), _zero(), location_address=addr_c)

    # Build a small block + method around them (use existing test patterns).
    method = ...

    address_entries = {
        stmt_a.result: address.AddressQubit(0),
        stmt_b.result: address.AddressQubit(1),
        stmt_c.result: address.AddressQubit(2),
    }
    initial_layout = (addr_a, addr_b_chosen_by_heuristic, addr_c)

    rule = ResolvePinnedAddresses(
        address_entries=address_entries,
        initial_layout=initial_layout,
    )
    rewrite.Walk(rule).rewrite(method.code)

    assert stmt_a.location_address == addr_a
    assert stmt_b.location_address == addr_b_chosen_by_heuristic
    assert stmt_c.location_address == addr_c


def test_postcondition_no_none_remains():
    # As above: assert that for every NewLogicalQubit in the method,
    # location_address is non-None after the rewrite runs.
    ...


def test_no_op_when_all_pinned():
    # Build a block with all NewLogicalQubits already pinned.
    # After the rewrite, the attributes are unchanged.
    ...
```

- [ ] **Step 2: Run to verify it fails**

```
uv run pytest python/tests/rewrite/test_resolve_pinned.py -v
```

Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Create the rewrite**

Create `python/bloqade/lanes/rewrite/resolve_pinned.py`:

```python
from dataclasses import dataclass

from bloqade.analysis import address
from kirin import ir
from kirin.rewrite import abc

from bloqade.lanes.dialects import place
from bloqade.lanes.layout.encoding import LocationAddress


@dataclass
class ResolvePinnedAddresses(abc.RewriteRule):
    """Stamp each NewLogicalQubit's location_address from the analysis frame.

    For NewLogicalQubits that already have a non-None location_address (i.e.
    user-pinned), the attribute is left alone — the heuristic respected it
    and the layout entry should match.

    For NewLogicalQubits with location_address=None, the heuristic's choice is
    looked up via address_entries[stmt.result] -> AddressQubit.data, which
    indexes into initial_layout.

    Post-condition: every NewLogicalQubit has a non-None location_address.
    """

    address_entries: dict[ir.SSAValue, address.Address]
    initial_layout: tuple[LocationAddress, ...]

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not isinstance(node, place.NewLogicalQubit):
            return abc.RewriteResult()
        if node.location_address is not None:
            return abc.RewriteResult()
        addr_entry = self.address_entries.get(node.result)
        if not isinstance(addr_entry, address.AddressQubit):
            return abc.RewriteResult()
        if addr_entry.data >= len(self.initial_layout):
            return abc.RewriteResult()
        node.location_address = self.initial_layout[addr_entry.data]
        return abc.RewriteResult(has_done_something=True)
```

- [ ] **Step 4: Run tests**

```
uv run pytest python/tests/rewrite/test_resolve_pinned.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/resolve_pinned.py python/tests/rewrite/test_resolve_pinned.py
git commit -m "feat(lanes): add ResolvePinnedAddresses rewrite to stamp heuristic picks"
```

---

### Task B6: Wire `ResolvePinnedAddresses` into `PlaceToMove`

**Files:**
- Modify: `python/bloqade/lanes/upstream.py:113-145` (the `PlaceToMove.emit` method)

- [ ] **Step 1: Insert the rewrite call**

In `PlaceToMove.emit`, after the line that computes `initial_layout` (currently `initial_layout = layout.LayoutAnalysis(...).get_layout_no_raise(out)` on line 119-121), and before any rewrites consume `initial_layout`, add:

```python
from bloqade.lanes.rewrite.resolve_pinned import ResolvePinnedAddresses

rewrite.Walk(
    ResolvePinnedAddresses(
        address_entries=address_frame.entries,
        initial_layout=initial_layout,
    )
).rewrite(out.code)
```

(Add the import at the top of the file with the other `from bloqade.lanes.rewrite ...` imports.)

This must run for **both** the `no_raise=True` branch and the `no_raise=False` branch — i.e., after each of the two `LayoutAnalysis(...).get_layout(...)` blocks.

- [ ] **Step 2: Run the full test corpus**

```
just test-python
```

Expected: all pass. The rewrite stamps attributes that no consumer reads yet (Phase C wires that up), so this should be observationally a no-op.

- [ ] **Step 3: Commit**

```bash
git add python/bloqade/lanes/upstream.py
git commit -m "feat(lanes): run ResolvePinnedAddresses after LayoutAnalysis in PlaceToMove"
```

---

### 🔍 Human Review Checkpoint B

**Stop here.** Hand the diff back to the human and wait for approval.

Things to check:
- All four `LayoutHeuristicABC` impls correctly refuse to assign pinned addresses to un-pinned qubits, and exclude pinned addresses from candidate sets.
- Empty-pin / no-pin behavior is byte-identical to before this PR (the `just test-python` regression suite is the gate).
- The post-resolve invariant ("every `NewLogicalQubit` has a non-None `location_address` after `ResolvePinnedAddresses`") holds in practice — the human can spot-check with a debugger or print statement on a real kernel.
- The `_collect_pinned` walk uses the right Kirin walking API (`method.callable_region.walk()` vs `method.code.walk()`); confirm with the human if uncertain.
- No race between `ResolvePinnedAddresses` and `PlacementAnalysis` — `PlacementAnalysis` runs after `ResolvePinnedAddresses` in this design? (Re-inspect line 123-141 of `upstream.py` and confirm with the human; we may need to adjust ordering.)

**Resume only after the human gives explicit approval.**

---

## Phase C — Place→Move consumer refactor

Now that the post-resolve invariant holds, refactor `InsertInitialize` and `InsertFill` to read attributes directly. Drop `address_entries` / `initial_layout` parameters.

### Task C1: Refactor `InsertFill`

**Files:**
- Modify: `python/bloqade/lanes/rewrite/place2move.py:355-375` (the `InsertFill` class)

`InsertFill` is simpler than `InsertInitialize`, so do it first.

- [ ] **Step 1: Write the failing test**

Locate or create `python/tests/rewrite/test_place2move.py`. Add:

```python
def test_insert_fill_reads_from_attributes():
    # Build a function whose first stmts are NewLogicalQubits with
    # location_address attributes set. Run InsertFill (without initial_layout
    # parameter) and assert the emitted move.Fill has the right
    # location_addresses tuple.
    ...
```

- [ ] **Step 2: Run to verify it fails**

```
uv run pytest python/tests/rewrite/test_place2move.py::test_insert_fill_reads_from_attributes -v
```

Expected: FAIL (or compile error — `InsertFill(initial_layout=...)` is the current signature).

- [ ] **Step 3: Refactor `InsertFill`**

Replace the existing class:

```python
@dataclass
class InsertFill(RewriteRule):
    """Emit move.Fill at function entry, with location_addresses collected
    from place.NewLogicalQubit.location_address in allocation order.

    Pre-condition: every NewLogicalQubit has a non-None location_address
    (established by ResolvePinnedAddresses).
    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, func.Function):
            return RewriteResult()

        first_stmt = node.body.blocks[0].first_stmt
        if first_stmt is None or isinstance(first_stmt, move.Fill):
            return RewriteResult()

        location_addresses: list[LocationAddress] = []
        for stmt in node.body.walk():
            if not isinstance(stmt, place.NewLogicalQubit):
                continue
            assert stmt.location_address is not None, (
                "InsertFill expects post-ResolvePinnedAddresses IR, "
                "but found NewLogicalQubit with location_address=None"
            )
            location_addresses.append(stmt.location_address)

        if not location_addresses:
            return RewriteResult()

        (current_state := move.Load()).insert_before(first_stmt)
        (
            current_state := move.Fill(
                current_state.result,
                location_addresses=tuple(location_addresses),
            )
        ).insert_before(first_stmt)
        move.Store(current_state.result).insert_before(first_stmt)
        return RewriteResult(has_done_something=True)
```

- [ ] **Step 4: Update the call site in `upstream.py`**

In `PlaceToMove.emit`, change `place2move.InsertFill(initial_layout)` to `place2move.InsertFill()`.

- [ ] **Step 5: Run tests**

```
uv run pytest python/tests/rewrite/ -v
just test-python
```

Expected: all pass. The Fill output should be byte-identical to before (same addresses in same order).

- [ ] **Step 6: Commit**

```bash
git commit -m "refactor(place2move): InsertFill reads location_address from attributes"
```

---

### Task C2: Refactor `InsertInitialize`

**Files:**
- Modify: `python/bloqade/lanes/rewrite/place2move.py:303-352` (the `InsertInitialize` class)

- [ ] **Step 1: Write the failing test**

Add to `python/tests/rewrite/test_place2move.py`:

```python
def test_insert_initialize_reads_from_attributes():
    # Build a block with NewLogicalQubits whose location_address attributes
    # are set. Run InsertInitialize (without address_entries or initial_layout
    # parameters) and assert the emitted move.LogicalInitialize has the right
    # location_addresses, theta, phi, lam.
    ...
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL (signature mismatch).

- [ ] **Step 3: Refactor `InsertInitialize`**

Replace the existing class:

```python
@dataclass
class InsertInitialize(RewriteRule):
    """Emit move.LogicalInitialize for the run of NewLogicalQubits in a block,
    with location_addresses, thetas, phis, lams collected directly from each
    statement.

    Pre-condition: every NewLogicalQubit has a non-None location_address
    (established by ResolvePinnedAddresses).
    """

    def rewrite_Block(self, node: ir.Block) -> RewriteResult:
        stmt = node.first_stmt
        thetas: list[ir.SSAValue] = []
        phis: list[ir.SSAValue] = []
        lams: list[ir.SSAValue] = []
        location_addresses: list[LocationAddress] = []

        while stmt is not None:
            if not isinstance(stmt, place.NewLogicalQubit):
                stmt = stmt.next_stmt
                continue
            assert stmt.location_address is not None, (
                "InsertInitialize expects post-ResolvePinnedAddresses IR"
            )
            location_addresses.append(stmt.location_address)
            thetas.append(stmt.theta)
            phis.append(stmt.phi)
            lams.append(stmt.lam)
            stmt = stmt.next_stmt

        if stmt is None or len(location_addresses) == 0:
            return RewriteResult()

        (current_state := move.Load()).insert_before(stmt)
        (
            current_state := move.LogicalInitialize(
                current_state.result,
                tuple(thetas),
                tuple(phis),
                tuple(lams),
                location_addresses=tuple(location_addresses),
            )
        ).insert_before(stmt)
        (move.Store(current_state.result)).insert_before(stmt)

        return RewriteResult(has_done_something=True)
```

(Note: this matches the existing logic of stopping at the first non-`NewLogicalQubit` statement to find the insertion point. Verify against the original `rewrite_Block` semantics — particularly the loop that limits to `len(initial_layout)`. With attributes, the natural stopping condition is "no more `NewLogicalQubit`s in this block", which is what the rewritten version uses.)

- [ ] **Step 4: Update the call site in `upstream.py`**

Change `place2move.InsertInitialize(address_frame.entries, initial_layout)` to `place2move.InsertInitialize()`.

- [ ] **Step 5: Run tests**

```
just test-python
```

- [ ] **Step 6: Commit**

```bash
git commit -m "refactor(place2move): InsertInitialize reads location_address from attributes"
```

---

### Task C3: Drop `initial_layout` from `LayoutAnalysis` output

**Files:**
- Modify: `python/bloqade/lanes/analysis/layout/analysis.py`
- Modify: `python/bloqade/lanes/upstream.py` (consumers of `initial_layout`)

- [ ] **Step 1: Identify remaining consumers**

```
grep -rn "initial_layout" python/bloqade/lanes/ python/tests/
```

After C1 and C2, the only places that should still use `initial_layout` as a name are:
- `LayoutAnalysis.process_results` (returns it)
- `PlaceToMove.emit` (reads it)
- `placement.PlacementAnalysis(..., initial_layout, ...)` (consumes it!)

`PlacementAnalysis` still needs the layout. So **`initial_layout` is not dead** — `PlacementAnalysis` consumes it. Keep it.

- [ ] **Step 2: Update the spec**

The spec said "drop initial_layout from analysis output" but this is wrong: `PlacementAnalysis` still reads it. Update the spec to acknowledge that `LayoutAnalysis.process_results` still returns the layout for `PlacementAnalysis`'s consumption, but that the `Place→Move` rewrites no longer take it as a parameter.

Edit `docs/superpowers/specs/2026-04-27-explicit-qubit-allocation-design.md` §4.3 to say:

```markdown
### 4.3 — `LayoutAnalysis.initial_layout` output

`LayoutAnalysis.process_results` continues to return the layout tuple,
because `PlacementAnalysis` consumes it. What changes: `InsertInitialize`
and `InsertFill` no longer take it as a parameter; they read attributes
directly. `PlacementAnalysis`'s consumption is unchanged.
```

- [ ] **Step 3: Commit the spec update**

```bash
git add docs/superpowers/specs/2026-04-27-explicit-qubit-allocation-design.md
git commit -m "docs(specs): clarify initial_layout still feeds PlacementAnalysis"
```

---

### 🔍 Human Review Checkpoint C

**Stop here.** Hand the diff back to the human and wait for approval.

Things to check:
- **Most important:** the existing test corpus produces byte-identical move IR for kernels with zero `new_at` usage. This is the regression gate.
- The `assert stmt.location_address is not None` in `InsertInitialize` and `InsertFill` is load-bearing: if `ResolvePinnedAddresses` didn't run, these crash. Confirm `ResolvePinnedAddresses` runs unconditionally in `PlaceToMove.emit`.
- `PlacementAnalysis` continues to receive `initial_layout` — the spec wording was over-optimistic; review the corrected wording.
- Any usage of `address_entries` in `InsertInitialize` / `InsertFill` that we missed.

**Resume only after the human gives explicit approval.**

---

## Phase D — Gemini IR + lowering

Add the user-facing statement and teach the existing rewrite chain to handle it.

### Task D1: Define `gemini.operations.NewAt` statement

**Files:**
- Modify: `python/bloqade/gemini/logical/dialects/operations/stmts.py`
- Modify: `python/bloqade/gemini/logical/dialects/operations/__init__.py`
- Create: `python/bloqade/gemini/logical/dialects/operations/_new_at_lowering.py`
- Create: `python/tests/gemini/dialects/__init__.py` (empty)
- Create: `python/tests/gemini/dialects/test_new_at.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/gemini/dialects/test_new_at.py`:

```python
from kirin import ir, types
from kirin.dialects import py

from bloqade.gemini.logical.dialects.operations import stmts


def test_new_at_takes_three_int_args_and_produces_qubit():
    z = py.Constant(0).result
    w = py.Constant(1).result
    s = py.Constant(2).result
    stmt = stmts.NewAt(zone_id=z, word_id=w, site_id=s)
    assert stmt.zone_id is z
    assert stmt.word_id is w
    assert stmt.site_id is s
    assert stmt.qubit.type.is_subseteq(types.Any)  # produces a qubit-typed result
```

- [ ] **Step 2: Run to verify it fails**

```
uv run pytest python/tests/gemini/dialects/test_new_at.py -v
```

Expected: FAIL — `stmts.NewAt` does not exist.

- [ ] **Step 3: Add the statement**

Append to `python/bloqade/gemini/logical/dialects/operations/stmts.py`:

```python
@statement(dialect=dialect)
class NewAt(ir.Statement):
    """Allocate a new logical qubit pinned to the given physical address.

    The three int args MUST be compile-time constants (enforced by validation).
    The constant values are stamped into place.NewLogicalQubit.location_address
    by the circuit→place rewrite chain.
    """

    traits = frozenset({lowering.FromPythonCall()})
    zone_id: ir.SSAValue = info.argument(types.Int)
    word_id: ir.SSAValue = info.argument(types.Int)
    site_id: ir.SSAValue = info.argument(types.Int)
    qubit: ir.ResultValue = info.result(QubitType)
```

- [ ] **Step 4: Add the Python lowering interface**

Create `python/bloqade/gemini/logical/dialects/operations/_new_at_lowering.py`:

```python
from bloqade.types import Qubit
from kirin import lowering

from .stmts import NewAt


@lowering.wraps(NewAt)
def new_at(zone_id: int, word_id: int, site_id: int) -> Qubit:
    """Allocate a logical qubit pinned to (zone_id, word_id, site_id).

    All three arguments must be compile-time constants. Use of non-constant
    values raises a validation error before lowering.
    """
    ...
```

- [ ] **Step 5: Re-export from the package init**

Modify `python/bloqade/gemini/logical/dialects/operations/__init__.py`:

```python
from . import _typeinfer as _typeinfer, stmts as stmts
from ._dialect import dialect as dialect
from ._interface import terminal_measure as terminal_measure
from ._new_at_lowering import new_at as new_at
```

- [ ] **Step 6: Run tests to verify it passes**

```
uv run pytest python/tests/gemini/dialects/test_new_at.py -v
```

Expected: PASS.

- [ ] **Step 7: Run the full corpus**

```
just test-python
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add python/bloqade/gemini/logical/dialects/operations/ python/tests/gemini/dialects/
git commit -m "feat(gemini): add operations.new_at statement for pinned-qubit allocation"
```

---

### Task D2: Augment `RewriteLogicalInitializeToNewLogical` to handle `NewAt`

**Files:**
- Modify: `python/bloqade/lanes/rewrite/circuit2place.py:36-53`
- Modify: `python/tests/rewrite/test_circuit2place.py`

- [ ] **Step 1: Write the failing test**

Add to `python/tests/rewrite/test_circuit2place.py`:

```python
def test_rewrite_logical_initialize_handles_new_at_with_const_args():
    # Build a kernel that uses gemini.operations.new_at(c1, c2, c3) and
    # passes the qubit to gemini.logical.Initialize. Run const-prop, then
    # RewriteLogicalInitializeToNewLogical. Assert that the resulting
    # place.NewLogicalQubit has location_address=LocationAddress(c1, c2, c3).
    ...

def test_rewrite_logical_initialize_handles_qubit_new_unchanged():
    # Pure qubit.stmts.New kernel: byte-identical output to before this PR.
    ...

def test_rewrite_logical_initialize_handles_mixed():
    # Both qubit.stmts.New and gemini.operations.new_at in the same kernel.
    # Both become NewLogicalQubit; only the latter has location_address set.
    ...
```

- [ ] **Step 2: Run to verify it fails**

```
uv run pytest python/tests/rewrite/test_circuit2place.py -v -k new_at
```

Expected: FAIL (the rewrite filters to `qubit.stmts.New` only).

- [ ] **Step 3: Augment the rewrite**

Modify `python/bloqade/lanes/rewrite/circuit2place.py:36-53`:

```python
class RewriteLogicalInitializeToNewLogical(abc.RewriteRule):
    """Rewrite qubit references in place.LogicalInitialize statements to
    place.NewLogicalQubit allocations.

    Handles both qubit.stmts.New (un-pinned, no location_address) and
    gemini.operations.new_at (pinned, with location_address built from
    constant-folded args).
    """

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not isinstance(node, place.LogicalInitialize):
            return abc.RewriteResult()

        def is_alloc(owner: ir.Statement | ir.Block) -> TypeGuard[
            qubit.stmts.New | gemini_stmts.NewAt
        ]:
            return isinstance(owner, (qubit.stmts.New, gemini_stmts.NewAt))

        alloc_stmts = tuple(
            filter(is_alloc, (q.owner for q in node.qubits))
        )

        any_replaced = False
        for alloc_stmt in alloc_stmts:
            if isinstance(alloc_stmt, gemini_stmts.NewAt):
                addr = _resolve_location_from_new_at(alloc_stmt)
                replacement = place.NewLogicalQubit(
                    node.theta, node.phi, node.lam, location_address=addr
                )
            else:
                replacement = place.NewLogicalQubit(node.theta, node.phi, node.lam)
            alloc_stmt.replace_by(replacement)
            any_replaced = True

        return abc.RewriteResult(has_done_something=any_replaced)


def _resolve_location_from_new_at(node: gemini_stmts.NewAt) -> LocationAddress:
    """Read const-prop hints to build a LocationAddress from a NewAt's args.

    By the time this runs, eager validation has already proved the args are
    constants and in range, so failures here indicate a pipeline bug.
    """
    z = _expect_const_int(node.zone_id, "zone_id")
    w = _expect_const_int(node.word_id, "word_id")
    s = _expect_const_int(node.site_id, "site_id")
    return LocationAddress(z, w, s)


def _expect_const_int(value: ir.SSAValue, name: str) -> int:
    # Use Kirin's idiomatic const-hint API. AbstractInterpreter.maybe_const /
    # expect_const are the right entry points; in raw rewrite context, the
    # const hint lives at value.hints["const"] as a const.Result.
    hint = value.hints.get("const")
    if hint is None or not hasattr(hint, "data"):
        raise AssertionError(
            f"NewAt.{name} not const-folded; eager validation should have caught this"
        )
    if not isinstance(hint.data, int):
        raise AssertionError(
            f"NewAt.{name} const value is {type(hint.data).__name__}, expected int"
        )
    return hint.data
```

Add the import at the top of the file:

```python
from bloqade.lanes.layout.encoding import LocationAddress
```

- [ ] **Step 4: Run tests**

```
uv run pytest python/tests/rewrite/test_circuit2place.py -v
just test-python
```

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/circuit2place.py python/tests/rewrite/test_circuit2place.py
git commit -m "feat(circuit2place): RewriteLogicalInitializeToNewLogical handles new_at"
```

---

### Task D3: Augment `InitializeNewQubits` to handle bare `NewAt`

**Files:**
- Modify: `python/bloqade/lanes/rewrite/circuit2place.py:71-86`

- [ ] **Step 1: Write the failing test**

Add to `python/tests/rewrite/test_circuit2place.py`:

```python
def test_initialize_new_qubits_handles_bare_new_at():
    # Build a kernel where new_at is NOT referenced by any Initialize.
    # After InitializeNewQubits, it becomes
    # place.NewLogicalQubit(0, 0, 0, location_address=...).
    ...
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL.

- [ ] **Step 3: Augment the rewrite**

Modify `python/bloqade/lanes/rewrite/circuit2place.py:71-86`:

```python
class InitializeNewQubits(abc.RewriteRule):
    """Rewrite bare allocation statements (qubit.stmts.New or
    gemini.operations.new_at) to place.NewLogicalQubit with default angles.
    """

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if isinstance(node, qubit.stmts.New):
            (zero := py.Constant(0.0)).insert_before(node)
            node.replace_by(
                place.NewLogicalQubit(
                    theta=zero.result,
                    phi=zero.result,
                    lam=zero.result,
                )
            )
            return abc.RewriteResult(has_done_something=True)

        if isinstance(node, gemini_stmts.NewAt):
            (zero := py.Constant(0.0)).insert_before(node)
            addr = _resolve_location_from_new_at(node)
            node.replace_by(
                place.NewLogicalQubit(
                    theta=zero.result,
                    phi=zero.result,
                    lam=zero.result,
                    location_address=addr,
                )
            )
            return abc.RewriteResult(has_done_something=True)

        return abc.RewriteResult()
```

- [ ] **Step 4: Run tests**

```
just test-python
```

- [ ] **Step 5: Commit**

```bash
git add python/bloqade/lanes/rewrite/circuit2place.py python/tests/rewrite/test_circuit2place.py
git commit -m "feat(circuit2place): InitializeNewQubits handles bare new_at"
```

---

### 🔍 Human Review Checkpoint D

**Stop here.** Hand the diff back to the human and wait for approval.

Things to check:
- **Const-prop hint access:** the plan reads `value.hints.get("const")` directly; the human noted earlier that `AbstractInterpreter.maybe_const` / `expect_const` are the idiomatic entry points. Confirm with the human whether the rewrite-layer code should use the AbstractInterpreter API instead, and adjust if so.
- The `_resolve_location_from_new_at` helper is shared by D2 and D3 — confirm it's defined exactly once.
- A pure-`qubit.stmts.New` kernel produces byte-identical output (regression).
- A pure-`new_at` kernel compiles, even though we haven't wired up validation yet — at this point a non-constant arg would `AssertionError` in the rewrite. That's OK because Phase E adds the eager validator that turns this into a clean diagnostic, but the human should know.

**Resume only after the human gives explicit approval.**

---

## Phase E — Eager validation

Two pieces: the per-statement method-table impl (registered against the lanes validation key) and the cross-statement duplicate-address pass. Both run on gemini IR before the lowering chain.

### Task E1: Per-statement validation method-table impl

**Files:**
- Create: `python/bloqade/gemini/logical/validation/__init__.py` (empty)
- Create: `python/bloqade/gemini/logical/validation/new_at.py`
- Create: `python/tests/gemini/validation/__init__.py` (empty)
- Create: `python/tests/gemini/validation/test_new_at_validation.py`

- [ ] **Step 1: Write the failing tests**

Create `python/tests/gemini/validation/test_new_at_validation.py`:

```python
import pytest
from kirin import ir

from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.validation.address import Validation


def test_const_foldability_failure(some_arch_spec):
    # Build a kernel: q = operations.new_at(some_var, 1, 2) where some_var
    # is genuinely non-constant (e.g. a function argument).
    # Run Validation. Assert at least one ValidationError points at the
    # new_at statement and mentions "constant".
    ...


def test_range_failure(some_arch_spec):
    # Build a kernel: q = operations.new_at(99, 0, 0) where zone 99 is
    # out of the arch's range. Run Validation. Assert ValidationError
    # mentions the address and "not valid for this architecture".
    ...


def test_valid_new_at_no_diagnostics(some_arch_spec):
    # operations.new_at(0, 1, 2) for a valid arch produces no errors.
    ...
```

(`some_arch_spec` is a pytest fixture you'll need to either use from existing tests or build inline. Look at `python/tests/validation/` for examples.)

- [ ] **Step 2: Run to verify they fail**

```
uv run pytest python/tests/gemini/validation/ -v
```

Expected: FAIL — no validation impl for `NewAt`.

- [ ] **Step 3: Create the validation impl**

Create `python/bloqade/gemini/logical/validation/__init__.py` (empty for now; we'll wire it up in E3).

Create `python/bloqade/gemini/logical/validation/new_at.py`:

```python
from kirin import interp, ir
from kirin.analysis.forward import ForwardFrame
from kirin.lattice.empty import EmptyLattice

from bloqade.gemini.logical.dialects.operations import dialect, stmts
from bloqade.lanes.layout.encoding import LocationAddress
from bloqade.lanes.validation.address import _ValidationAnalysis


@dialect.register(key="move.address.validation")
class _NewAtValidation(interp.MethodTable):
    @interp.impl(stmts.NewAt)
    def check_new_at(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: stmts.NewAt,
    ):
        # 1. Const-foldability check.
        try:
            z = _expect_const_int(node.zone_id, "zone_id", node, _interp)
            w = _expect_const_int(node.word_id, "word_id", node, _interp)
            s = _expect_const_int(node.site_id, "site_id", node, _interp)
        except _ConstError:
            return (EmptyLattice.bottom(),)

        # 2. Range check via ArchSpec.
        candidate = LocationAddress(z, w, s)
        _interp.report_location_errors(node, (candidate,))

        return (EmptyLattice.bottom(),)


class _ConstError(Exception):
    pass


def _expect_const_int(
    value: ir.SSAValue,
    arg_name: str,
    node: ir.Statement,
    interpreter: _ValidationAnalysis,
) -> int:
    hint = value.hints.get("const")
    if hint is None or not hasattr(hint, "data") or not isinstance(hint.data, int):
        interpreter.add_validation_error(
            node,
            ir.ValidationError(
                node,
                f"address argument '{arg_name}' is not a compile-time constant; "
                f"explicit allocation requires constant zone/word/site",
            ),
        )
        raise _ConstError()
    return hint.data
```

Note: `_ValidationAnalysis._collect_pinned`-style helpers may already be available — verify by reading `bloqade/lanes/validation/address.py`. The `report_location_errors` method exists at line 28 and is used by other registered impls.

- [ ] **Step 4: Wire registration**

The `@dialect.register(key="...")` decorator should auto-register, but the module needs to be **imported** at the right time so the registration runs. Add an import statement to ensure the module is loaded:

In `python/bloqade/gemini/logical/__init__.py` (read it first to find the right place):

```python
from .validation import new_at as _new_at_validation  # noqa: F401  - registers method table
```

(Adjust the path based on actual `__init__.py` structure.)

- [ ] **Step 5: Run tests**

```
uv run pytest python/tests/gemini/validation/ -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/bloqade/gemini/logical/validation/ python/tests/gemini/validation/ python/bloqade/gemini/logical/__init__.py
git commit -m "feat(gemini): per-stmt validation for new_at (const + range checks)"
```

---

### Task E2: Cross-statement duplicate-address validation pass

**Files:**
- Create: `python/bloqade/gemini/logical/validation/duplicates.py`
- Add to: `python/tests/gemini/validation/test_new_at_validation.py`

- [ ] **Step 1: Write the failing test**

Add to `python/tests/gemini/validation/test_new_at_validation.py`:

```python
def test_duplicate_addresses_reported(some_arch_spec):
    # Two new_at calls with identical constants. Run the duplicate-address
    # pass. Assert one ValidationError is reported on the second occurrence
    # mentioning the first.
    ...


def test_distinct_addresses_no_diagnostics(some_arch_spec):
    # Two new_at calls with distinct constants. No errors.
    ...
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — pass doesn't exist.

- [ ] **Step 3: Implement the pass**

Create `python/bloqade/gemini/logical/validation/duplicates.py`:

```python
from dataclasses import dataclass, field
from typing import Any

from kirin import ir
from kirin.validation import ValidationPass

from bloqade.gemini.logical.dialects.operations import stmts
from bloqade.lanes.layout.encoding import LocationAddress


@dataclass
class DuplicateAddressValidation(ValidationPass):
    """Report any pair of gemini.operations.new_at statements that pin the
    same physical address.

    Pre-condition: per-statement validation (const-foldability + range) has
    already run, so every new_at's args have const hints populated.
    """

    def name(self) -> str:
        return "gemini.new_at.duplicates"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        seen: dict[LocationAddress, stmts.NewAt] = {}
        errors: list[ir.ValidationError] = []

        for stmt in method.callable_region.walk():
            if not isinstance(stmt, stmts.NewAt):
                continue
            addr = _addr_from_new_at(stmt)
            if addr is None:
                continue  # per-stmt validation will have already errored
            existing = seen.get(addr)
            if existing is not None:
                errors.append(
                    ir.ValidationError(
                        stmt,
                        f"address ({addr.zone_id}, {addr.word_id}, {addr.site_id}) "
                        f"is pinned by two operations.new_at calls; first defined at "
                        f"{existing.print_str() if hasattr(existing, 'print_str') else existing}",
                    )
                )
            else:
                seen[addr] = stmt

        return None, errors


def _addr_from_new_at(node: stmts.NewAt) -> LocationAddress | None:
    z = node.zone_id.hints.get("const")
    w = node.word_id.hints.get("const")
    s = node.site_id.hints.get("const")
    if not all(h is not None and hasattr(h, "data") and isinstance(h.data, int)
               for h in (z, w, s)):
        return None
    return LocationAddress(z.data, w.data, s.data)
```

- [ ] **Step 4: Run tests**

```
uv run pytest python/tests/gemini/validation/ -v
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(gemini): DuplicateAddressValidation pass for new_at"
```

---

### Task E3: Wire eager validation into the pipeline

**Files:**
- Modify: `python/bloqade/lanes/upstream.py` (in `NativeToPlace.emit`, before lowering)

- [ ] **Step 1: Pick the invocation point**

The validation must run on the gemini IR (before `circuit2place.RewriteLogicalInitializeToNewLogical` rewrites `new_at` away). The natural spot is at the start of `NativeToPlace.emit`, after const-prop has run but before the lowering rewrites.

Look at `NativeToPlace.emit` (line 40-99). The lowering rewrites start at line 60 (`RewriteInitializeToLogicalInitialize`). Const-prop is implicit in `AggressiveUnroll` (line 54). Insert validation between them.

- [ ] **Step 2: Write the failing test**

Add to `python/tests/gemini/validation/test_new_at_validation.py`:

```python
def test_pipeline_invokes_validation(some_arch_spec):
    # Build a kernel with a duplicate-address bug. Run squin_to_move with
    # the appropriate arch spec. Assert compilation fails with the
    # duplicate-address diagnostic.
    ...
```

- [ ] **Step 3: Run to verify it fails**

Expected: FAIL — no eager validation invoked yet, so the bug slips through to runtime or later validation.

- [ ] **Step 4: Wire validation into NativeToPlace**

Add a parameter `arch_spec: ArchSpec | None = None` to `NativeToPlace`, default `None`. When non-None, run validation after `AggressiveUnroll` and before `RewriteInitializeToLogicalInitialize`:

```python
if self.arch_spec is not None:
    from bloqade.lanes.validation.address import Validation
    from bloqade.gemini.logical.validation.duplicates import DuplicateAddressValidation

    per_stmt = Validation(arch_spec=self.arch_spec)
    _, per_stmt_errors = per_stmt.run(out)
    if per_stmt_errors:
        raise ir.ValidationError(out, f"validation failed: {per_stmt_errors}")

    duplicates = DuplicateAddressValidation()
    _, dup_errors = duplicates.run(out)
    if dup_errors:
        raise ir.ValidationError(out, f"duplicate addresses: {dup_errors}")
```

Update `squin_to_move` to thread `arch_spec` through to `NativeToPlace`.

- [ ] **Step 5: Run tests**

```
just test-python
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(lanes): invoke gemini validation eagerly before circuit→place lowering"
```

---

### 🔍 Human Review Checkpoint E

**Stop here.** Hand the diff back to the human and wait for approval.

Things to check:
- The validation invocation point (`NativeToPlace.emit`) is the right place — earlier (on raw squin?) might also be valid. Confirm with the human.
- The plan threads `arch_spec` from `squin_to_move` down to `NativeToPlace`. The human may prefer a different parameter shape (e.g. always-on validation, or moving the invocation to `compile_squin_to_move`). Confirm.
- The use of `ir.ValidationError` to halt compilation is consistent with how the existing post-compile validator surfaces errors. If not, switch to whatever halt mechanism matches.
- The plan reads `value.hints.get("const")` directly in the duplicate pass; if Checkpoint D updated this to use `AbstractInterpreter.maybe_const`, propagate that change here.

**Resume only after the human gives explicit approval.**

---

## Phase F — Integration tests + demo

End-to-end gates and a demo script.

### Task F1: End-to-end mixed-pinning happy path

**Files:**
- Create: `python/tests/integration/__init__.py` (empty)
- Create: `python/tests/integration/test_explicit_allocation.py`

- [ ] **Step 1: Write the test**

Create `python/tests/integration/test_explicit_allocation.py`:

```python
import pytest

# Use existing test patterns — copy compile setup from
# python/tests/rewrite/test_circuit2place.py or similar.

def test_e2e_mixed_pinning(small_arch_spec, layout_heuristic, placement_strategy):
    # squin kernel: half qubits via qubit.new, half via gemini.operations.new_at
    # at known addresses; one CZ between them; measurements.
    # Compile via squin_to_move(arch_spec=small_arch_spec, ...).
    # Assert:
    # - move.Fill location_addresses contains all pinned addresses.
    # - The post-compile lanes validator passes on the result.
    ...
```

- [ ] **Step 2: Run the test**

```
uv run pytest python/tests/integration/test_explicit_allocation.py::test_e2e_mixed_pinning -v
```

Expected: PASS (this is the smoke test).

- [ ] **Step 3: Commit**

```bash
git add python/tests/integration/
git commit -m "test(integration): e2e mixed-pinning happy path"
```

---

### Task F2: No-regressions gate

**Files:**
- Append to: `python/tests/integration/test_explicit_allocation.py`

- [ ] **Step 1: Write the test**

```python
def test_unannotated_kernel_unchanged():
    # Pick a small kernel that uses no new_at (e.g. one of the demo kernels).
    # Compile it both ways:
    #   - With this branch's compiler.
    #   - With the IR shape produced before this branch (snapshot, or a
    #     hand-built reference).
    # Assert byte-identical move IR (or, if too brittle, structural equality
    # of the relevant move ops).
    ...
```

A practical alternative: compile a fixed kernel and compare against a snapshot string committed to the repo. Use `pytest-snapshot` or hand-rolled comparison.

- [ ] **Step 2: Run**

Expected: PASS — un-pinned kernels produce identical output.

- [ ] **Step 3: Commit**

```bash
git commit -m "test(integration): no-regressions gate for kernels with zero new_at"
```

---

### Task F3: End-to-end failure modes

**Files:**
- Append to: `python/tests/integration/test_explicit_allocation.py`

- [ ] **Step 1: Write the tests**

Three failure cases:

```python
def test_e2e_const_prop_failure_surfaces_at_compile_time():
    # Kernel: q = operations.new_at(some_arg, 0, 0) where some_arg is a
    # function parameter. squin_to_move should raise.
    ...


def test_e2e_overconstraining_pins_fail():
    # Kernel with more pins than the arch can satisfy. squin_to_move raises
    # a LayoutAnalysis-level error.
    ...


def test_e2e_semantic_illegality_caught_by_post_compile_validator():
    # Kernel that pins a qubit to an address where a needed gate cannot run.
    # squin_to_move raises from the existing post-compile validator (failure
    # mode #4 in the spec).
    ...
```

- [ ] **Step 2: Run**

```
uv run pytest python/tests/integration/ -v
```

- [ ] **Step 3: Commit**

```bash
git commit -m "test(integration): e2e failure modes for explicit allocation"
```

---

### Task F4: Demo script

**Files:**
- Create: `demo/explicit_allocation.py`

- [ ] **Step 1: Write the demo**

Create `demo/explicit_allocation.py`:

```python
"""Demo: explicit qubit allocation.

Pins a small register of logical qubits to known-good physical addresses
and runs a basic circuit. Useful as both documentation and a regression
guard for the user-facing API.
"""
from bloqade import qubit
from bloqade.gemini.logical.dialects import operations
# ... (build squin kernel using both qubit.new and operations.new_at)
# ... (compile via squin_to_move)
# ... (print the resulting IR or a summary)

if __name__ == "__main__":
    main()
```

(Concrete kernel content depends on existing demos; copy the structure from `demo/move_lang.py` or similar.)

- [ ] **Step 2: Run the demo**

```
uv run python demo/explicit_allocation.py
```

Expected: completes without errors and prints sensible output.

- [ ] **Step 3: Add to `just demo`**

If `just demo` runs all demos by globbing or by an explicit list, ensure `demo/explicit_allocation.py` is included. Look at the `justfile`:

```
grep -A 5 "^demo" justfile
```

If it's an explicit list, add this script.

- [ ] **Step 4: Run all demos**

```
just demo
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add demo/explicit_allocation.py justfile
git commit -m "docs(demo): add explicit_allocation demo script"
```

---

### 🔍 Human Review Checkpoint F (Final)

**Stop here.** Hand the diff back to the human for the final review.

Things to check:
- Full test corpus passes: `just test-python`, `just test-rust`, `just test`.
- The integration test exercises all three failure modes (const-prop, overconstrain, semantic illegality).
- The no-regressions test gate is meaningful — i.e., it would actually catch a regression. (Manually break something temporarily, confirm the test fails, revert.)
- The demo script reads cleanly and serves as documentation.
- All commits use Conventional Commits style with the right scopes.
- The `breaking` label might apply here for the `LayoutHeuristicABC.compute_layout` signature change. Confirm with the human whether this should be tagged on the PR.
- Memory entries from `MEMORY.md` apply: `feedback_pr_breaking_changes.md` (categorize by Python/Rust/C surface), `feedback_alane_label.md` (A-Lane label on issues), `feedback_backport_labels.md` (S-backport + backport v0.7 for non-breaking).

**Once approved, the work is ready for `just lint` + `just test` + PR creation.**

---

## Final integration check before PR

After all phases ship, run as a sanity check:

```bash
just sync
just test
just lint
```

If everything passes, the work is ready for PR.
