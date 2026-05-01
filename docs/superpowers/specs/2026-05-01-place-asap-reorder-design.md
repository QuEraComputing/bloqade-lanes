# Place-Stage ASAP Reorder Design

**Date:** 2026-05-01
**Status:** Approved

## Overview

Two coordinated additions to the `place`-dialect compilation stage:

1. **`cz_disjoint_heuristic`** — a new `merge_heuristic` for `MergePlacementRegions` that prevents mixing CZ and non-CZ layers and only merges CZ layers whose qubit sets are disjoint.
2. **`ASAPReorder`** — a new rewrite rule that reorders statements within a `StaticPlacement` body block using ASAP (As Soon As Possible) scheduling, creating adjacency that `FuseAdjacentGates` can then exploit.

## Motivation

`MergePlacementRegions` currently uses a trivial heuristic (always merge). This collapses all placements into one block regardless of gate type, interleaving CZ and single-qubit operations in ways that obscure parallelism. Once merged into a single body, `FuseAdjacentGates` can only fuse gates that happen to be textually adjacent — it cannot fuse gates separated by an unrelated gate on a different qubit.

ASAP reordering moves each gate to the earliest position its qubit dependencies allow, grouping independent gates into contiguous runs. Combined with a CZ-aware merge heuristic, this produces well-structured placement bodies where `FuseAdjacentGates` achieves maximum broadcast fusion.

## Pass Ordering

```
RewritePlaceOperations                              (one StaticPlacement per gate)
       ↓
MergePlacementRegions(cz_disjoint_heuristic)        (new heuristic — CZ-aware)
       ↓
ASAPReorder                                         (new — reorders within body)
       ↓
FuseAdjacentGates                                   (unchanged)
```

Nothing downstream of `FuseAdjacentGates` changes.

## Component 1: `cz_disjoint_heuristic`

### Location

`python/bloqade/lanes/rewrite/circuit2place.py`, alongside `_default_merge_heuristic`.

### Helper

```python
def _cz_qubits(region: ir.Region) -> set[int]:
    """Return all qubit indices touched by CZ statements in the region's first block."""
    result: set[int] = set()
    for stmt in region.blocks[0].stmts:
        if isinstance(stmt, place.CZ):
            result.update(stmt.qubits)  # qubits = controls first, then targets
    return result
```

### Heuristic logic

```
is_cz_layer(r) = _cz_qubits(r) is non-empty

Rules (evaluated in order):
1. is_cz_layer(r1) != is_cz_layer(r2)  →  False   (never mix CZ and non-CZ layers)
2. not is_cz_layer(r1)                  →  True    (both non-CZ: merge freely)
3. both CZ layers                       →  _cz_qubits(r1).isdisjoint(_cz_qubits(r2))
```

The heuristic is passed to `MergePlacementRegions(merge_heuristic=cz_disjoint_heuristic)`. No changes to `MergePlacementRegions` itself.

## Component 2: `ASAPReorder`

### Location

New file: `python/bloqade/lanes/rewrite/asap_reorder.py`

### Interface

```python
@dataclass
class ASAPReorder(RewriteRule):
    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, place.StaticPlacement):
            return RewriteResult()
        ...
```

Applied as `rewrite.Walk(ASAPReorder())` in the pass pipeline.

### Algorithm

#### Step 1 — Segment extraction

Walk the body block's statement list (excluding the trailing `place.Yield`). Split at hard barriers: `place.Initialize` and `place.EndMeasure`. Each contiguous run of `place.R`, `place.Rz`, and `place.CZ` between barriers forms one segment. Barriers are not moved; they divide the body into independently reorderable segments.

#### Step 2 — Dependency DAG (per segment)

Use `rustworkx.PyDAG` (from the `rustworkx` package, already a declared project dependency).

For each segment:
- Add one DAG node per statement, carrying the statement as payload.
- Maintain `last_touch: dict[int, int]` mapping qubit index → DAG node index of the most recent statement that touched that qubit.
- For each new statement: for every qubit index `q` in `stmt.qubits`, add a directed edge from `last_touch[q]` → new node if `last_touch[q]` exists, then set `last_touch[q]` = new node index.

This builds the minimal necessary-dependency graph (only direct predecessor edges; no transitive redundancy).

#### Step 3 — ASAP layer assignment

```
layer[v] = 0                                   for nodes with no predecessors
layer[v] = max(layer[p] for p in pred(v)) + 1  for all others
```

Collect statements grouped by layer. Within a layer, preserve the original relative order as a stable tiebreaker.

#### Step 4 — State chain re-threading

Walk the new emission order. For each statement, reconstruct it with the updated `state_before` using the `from_stmt` pattern (matching `MergePlacementRegions`):

```python
new_stmt = stmt.from_stmt(stmt, args=(curr_state, *stmt.args[1:]), attributes=...)
old_stmt.state_after.replace_by(new_stmt.state_after)
curr_state = new_stmt.state_after
```

Barriers are similarly reconstructed in place with their updated `state_before`. The `place.Yield` receives the final `curr_state`.

Return `RewriteResult(has_done_something=True)` if any statement changed position.

## Testing

### `test_circuit2place.py` additions

- `test_cz_disjoint_heuristic_rejects_cz_plus_noncz` — one CZ layer + one R layer → False
- `test_cz_disjoint_heuristic_rejects_overlapping_cz_layers` — two CZ layers sharing a qubit → False
- `test_cz_disjoint_heuristic_accepts_disjoint_cz_layers` — two CZ layers with disjoint qubits → True
- `test_cz_disjoint_heuristic_accepts_noncz_layers` — two R-only layers → True

### `test_asap_reorder.py` (new file, hand-built IR, style of `test_fuse_gates.py`)

- `test_single_stmt_unchanged` — one statement, no reorder
- `test_two_independent_gates_same_type_reordered` — R(q0) then R(q1): already optimal, verify idempotence
- `test_dependent_gate_not_moved_before_predecessor` — R(q0), CZ(q0,q1), R(q1): CZ cannot move before R(q0)
- `test_independent_gate_moves_earlier` — R(q0), R(q1), CZ(q0,q1), R(q2): R(q2) is qubit-independent of everything before it; ASAP assigns it layer 0 alongside R(q0)/R(q1), moving it before CZ(q0,q1) which is layer 1
- `test_barrier_prevents_reorder_across` — R(q0), Initialize(q1), R(q0): second R stays after Initialize
- `test_multiple_layers_correct_ordering` — three gates spanning two layers, verify layer grouping
- `test_idempotence` — second application is a no-op
