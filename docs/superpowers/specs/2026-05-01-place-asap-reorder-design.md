# Place-Stage ASAP Reorder Design

**Date:** 2026-05-01
**Status:** Approved

## Overview

Three composable rewrite passes for the `place`-dialect compilation stage, each with a pluggable policy:

1. **`MergeStaticPlacement(merge_policy)`** — merges consecutive `StaticPlacement` nodes using a caller-supplied predicate. Replaces and generalises the existing `MergePlacementRegions`.
2. **`ReorderStaticPlacement(reorder_policy)`** — reorders the gate statements within a single `StaticPlacement` body using a caller-supplied ordering function.
3. **`SplitStaticPlacement(split_policy)`** — splits a single `StaticPlacement` body into multiple `StaticPlacement` nodes using a caller-supplied partitioning function.

Concrete policies supplied for this feature:
- `gate_only_merge` — merge predicate that only merges placements whose bodies contain exclusively `R`, `Rz`, `CZ`, and `Yield`; `Initialize` and `EndMeasure` placements are never merged.
- `asap_reorder_policy` — reorder policy implementing ASAP (As Soon As Possible) scheduling via a `rustworkx.PyDAG` dependency graph.
- `cz_layer_split_policy` — split policy that groups all single-qubit layers preceding each CZ layer together with that CZ layer into one `StaticPlacement`, following the CZ-anchored grouping rule (policy A).

## Motivation

After `RewritePlaceOperations` each gate occupies its own `StaticPlacement`. Naively merging everything produces a flat body where `FuseAdjacentGates` can only fuse gates that happen to be textually adjacent. ASAP scheduling groups independent gates into contiguous runs, maximising fusion opportunities, but it requires seeing the full gate sequence at once rather than working placement-by-placement. Splitting the scheduled result back into CZ-anchored layers preserves the `StaticPlacement` boundary semantics (palindrome move insertion sites) that downstream passes depend on.

## Pass Ordering

```
RewritePlaceOperations
       ↓
MergeStaticPlacement(gate_only_merge)        (collapse pure-gate layers; placements with classical outputs isolated)
       ↓
ReorderStaticPlacement(asap_reorder_policy)  (ASAP schedule within the body)
       ↓
SplitStaticPlacement(cz_layer_split_policy)  (re-split into CZ-anchored layers)
       ↓
FuseAdjacentGates                            (unchanged — fuse within each layer)
```

Nothing downstream of `FuseAdjacentGates` changes.

## Component 1: `MergeStaticPlacement`

### Location

`python/bloqade/lanes/rewrite/circuit2place.py`. Replaces `MergePlacementRegions`.

### Interface

```python
@dataclass
class MergeStaticPlacement(RewriteRule):
    merge_policy: Callable[[place.StaticPlacement, place.StaticPlacement], bool] = always_merge

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult: ...
```

### Policy signature change from `MergePlacementRegions`

The original `MergePlacementRegions.merge_heuristic` had signature `Callable[[ir.Region, ir.Region], bool]`. The new policy receives the full `StaticPlacement` statements:

```python
Callable[[place.StaticPlacement, place.StaticPlacement], bool]
```

This is necessary because `place.CZ.qubits` holds **local integer indices** into the outer `StaticPlacement.qubits` tuple; resolving those indices to identify physical qubits requires access to the placement, not just its body region.

### `gate_only_merge` policy (new default)

```python
_GATE_STMT_TYPES = (place.R, place.Rz, place.CZ, place.Yield)

def _is_pure_gate_block(sp: place.StaticPlacement) -> bool:
    return all(isinstance(stmt, _GATE_STMT_TYPES) for stmt in sp.body.blocks[0].stmts)

def gate_only_merge(sp1: place.StaticPlacement, sp2: place.StaticPlacement) -> bool:
    return _is_pure_gate_block(sp1) and _is_pure_gate_block(sp2)
```

Only merges placements whose bodies contain exclusively `R`, `Rz`, `CZ`, and a trailing `place.Yield` with no classical results. Two important non-gate statement types are excluded:

- **`EndMeasure`** — a `QuantumStmt` whose measurement results are threaded into the `Yield` as `classical_results`, giving the `StaticPlacement` non-empty results. Merging would require propagating those classical outputs incorrectly.
- **`place.Initialize`** — a `QuantumStmt` (with `theta`, `phi`, `lam`) that has no extra result types, so its `StaticPlacement` has empty `results`. A results-only check would incorrectly allow merging it with gate blocks. `Initialize` must remain isolated because `place2move` lowering treats it as a distinct hardware operation (lowered to `move.Fill` + `LogicalInitialize`) and expects it in its own `StaticPlacement`.

Checking body statement types directly (rather than `len(sp.results)`) handles both cases correctly.

This guarantee is load-bearing for `SplitStaticPlacement`: every body entering the split pass terminates in a plain `place.Yield` with no classical results, so the split policy can safely reuse that `Yield` in the last output block without special-casing.

## Component 2: `ReorderStaticPlacement`

### Location

New file: `python/bloqade/lanes/rewrite/reorder_static_placement.py`

### Interface

```python
@dataclass
class ReorderStaticPlacement(RewriteRule):
    reorder_policy: Callable[[list[place.QuantumStmt]], list[place.QuantumStmt]]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult: ...
```

The policy receives the gate statements of the body (excluding the trailing `place.Yield` and any hard barriers) and returns them in a new order. Hard barriers (`place.Initialize`, `place.EndMeasure`) divide the body into independent segments; the rewriter applies the policy to each segment separately and re-threads the state chain after reordering.

### `asap_reorder_policy`

Implements ASAP scheduling using `rustworkx.PyDAG` (declared project dependency `rustworkx>=0.17.1`).

#### Per-segment algorithm

1. **Build the dependency DAG** — add one node per statement. Maintain `last_touch: dict[int, int]` mapping qubit index → DAG node index of the most recent statement that touched that qubit. For each new statement, add a directed edge from `last_touch[q]` → new node for every qubit `q` in `stmt.qubits`, then update `last_touch[q]`. This produces the minimal necessary-dependency graph.

2. **ASAP layer assignment**:
   ```
   layer[v] = 0                                    for nodes with no predecessors
   layer[v] = max(layer[p] for p in pred(v)) + 1   for all others
   ```
   Statements are collected per layer; within a layer the original relative order is preserved as a stable tiebreaker.

3. **Re-thread the state chain** — walk the new emission order, reconstructing each statement via `stmt.from_stmt(stmt, args=(curr_state, *stmt.args[1:]), ...)` (the same pattern used in `MergePlacementRegions`). Propagate `old_result.replace_by(new_result)` for `state_after` outputs.

## Component 3: `SplitStaticPlacement`

### Location

New file: `python/bloqade/lanes/rewrite/split_static_placement.py`

### Interface

```python
@dataclass
class SplitStaticPlacement(RewriteRule):
    split_policy: Callable[[ir.Block], list[ir.Block]]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult: ...
```

The policy receives the body block (with its fully-threaded state chain) and returns a list of new blocks, each of which becomes one `StaticPlacement`. State threading is the policy's responsibility: because the number of statements changes across the output blocks, the policy must construct each output block with a correctly threaded state chain. Each output block must also terminate with a `place.Yield` carrying the block's final `state_after` value — this is required for the block to be a valid `StaticPlacement` body. Intermediate blocks must append a freshly constructed `place.Yield`; the **last block must reuse the original terminator** from the input block (not construct a new one), since that terminator may carry classical results or other outputs that downstream statements depend on. The rewriter wraps each returned block in a new `StaticPlacement` and replaces the original node.

The `reorder_policy` does not have this constraint — it is a pure permutation of statements with no change in count, so the rewriter can re-thread the state chain itself after reordering.

### `cz_layer_split_policy` (policy A)

After ASAP scheduling the body is a sequence of ASAP layers. The policy scans these layers sequentially:

- Accumulate single-qubit gate layers (`place.R`, `place.Rz` only).
- When a CZ layer is reached: flush `[accumulated SQ layers + CZ layer]` as one group.
- Continue until all layers are consumed.
- Any remaining SQ layers after the last CZ form the final group.

Example:
```
Layer 0: R(q2)              }
Layer 1: R(q0), R(q1)       }  group 1 → StaticPlacement 1
Layer 2: CZ(q0, q1)         }
Layer 3: R(q0), R(q3)       }  group 2 → StaticPlacement 2
Layer 4: CZ(q2, q3)         }
Layer 5: R(q1)                 group 3 → StaticPlacement 3
```

## Testing

### `MergeStaticPlacement`

Existing `MergePlacementRegions` tests migrated and updated for the renamed class and updated policy signature.

### `ReorderStaticPlacement` — `test_reorder_static_placement.py` (new file, hand-built IR)

- `test_single_stmt_unchanged` — one statement, no reorder
- `test_two_independent_gates_already_optimal` — R(q0) then R(q1): already layer 0, verify idempotence
- `test_dependent_gate_not_moved_before_predecessor` — R(q0), CZ(q0,q1), R(q1): CZ cannot move before R(q0)
- `test_independent_gate_moves_earlier` — R(q0), R(q1), CZ(q0,q1), R(q2): R(q2) is layer 0 alongside R(q0)/R(q1), moves before CZ(q0,q1) which is layer 1
- `test_barrier_prevents_reorder_across` — R(q0), Initialize(q1), R(q0): second R stays after Initialize
- `test_multiple_layers_correct_ordering` — three gates spanning two layers, verify layer grouping
- `test_idempotence` — second application is a no-op

### `SplitStaticPlacement` — `test_split_static_placement.py` (new file, hand-built IR)

- `test_no_cz_no_split` — body with only SQ gates: one group, no split
- `test_single_cz_groups_preceding_sq` — SQ layers before a CZ all go into the same group as the CZ
- `test_two_cz_layers_produce_two_groups` — SQ + CZ1, SQ + CZ2 → two StaticPlacements
- `test_trailing_sq_after_last_cz_forms_own_group` — confirms policy A trailing behaviour
- `test_policy_a_full_example` — the five-layer example from the design doc
