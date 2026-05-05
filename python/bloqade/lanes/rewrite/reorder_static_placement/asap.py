"""ASAP (As-Soon-As-Possible) scheduling policy."""

import rustworkx

from bloqade.lanes.rewrite.reorder_static_placement.types import (
    _BARRIERS,
    _build_dependency_dag,
    _group_within_layer,
    _SchedulableStmt,
)


def _asap_schedule(stmts: list[_SchedulableStmt]) -> list[_SchedulableStmt]:
    """Return stmts in ASAP layer order with fusable-equivalent gates adjacent within each layer."""
    if len(stmts) <= 1:
        return list(stmts)

    dag, node_for = _build_dependency_dag(stmts)

    layer: dict[int, int] = {}
    for node_idx in rustworkx.topological_sort(dag):
        preds = dag.predecessor_indices(node_idx)
        layer[node_idx] = 0 if not preds else max(layer[p] for p in preds) + 1

    # Bucket statements per layer in original order, then group within each layer
    # so fusable-equivalent gates (same type + params) are adjacent.
    layer_buckets: dict[int, list[_SchedulableStmt]] = {}
    for i, stmt in enumerate(stmts):
        layer_buckets.setdefault(layer[node_for[i]], []).append(stmt)

    result: list[_SchedulableStmt] = []
    for lyr in sorted(layer_buckets):
        result.extend(_group_within_layer(layer_buckets[lyr]))
    return result


def asap_reorder_policy(
    stmts: list[_SchedulableStmt],
) -> list[_SchedulableStmt]:
    """ASAP scheduling policy. Hard barriers (Initialize, EndMeasure) divide
    the body into independent segments; each segment is scheduled separately."""
    result: list[_SchedulableStmt] = []
    segment: list[_SchedulableStmt] = []

    for stmt in stmts:
        if isinstance(stmt, _BARRIERS):
            result.extend(_asap_schedule(segment))
            result.append(stmt)
            segment = []
        else:
            segment.append(stmt)
    result.extend(_asap_schedule(segment))
    return result
