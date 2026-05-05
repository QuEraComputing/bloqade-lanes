"""ALAP (As-Late-As-Possible) scheduling policy."""

import rustworkx

from bloqade.lanes.rewrite.reorder_static_placement.types import (
    _BARRIERS,
    _build_dependency_dag,
    _group_within_layer,
    _SchedulableStmt,
)


def _alap_schedule(stmts: list[_SchedulableStmt]) -> list[_SchedulableStmt]:
    """Return stmts in ALAP layer order with fusable-equivalent gates adjacent within each layer.

    Each gate is assigned to the *latest* layer that still respects all data
    dependencies (i.e. every successor is placed at a strictly later layer).
    Deferring single-qubit gates reduces the qubit footprint of early
    CZ-anchored StaticPlacement regions, lowering atom-move overhead.
    """
    if len(stmts) <= 1:
        return list(stmts)

    dag, node_for = _build_dependency_dag(stmts)
    topo_order = list(rustworkx.topological_sort(dag))

    # Forward pass to find max ASAP depth (= ALAP horizon for sink nodes).
    asap_layer: dict[int, int] = {}
    for node_idx in topo_order:
        preds = dag.predecessor_indices(node_idx)
        asap_layer[node_idx] = 0 if not preds else max(asap_layer[p] for p in preds) + 1

    max_layer = max(asap_layer.values()) if asap_layer else 0

    # Backward pass: ALAP[i] = min(ALAP[j] for j in successors(i)) - 1,
    # or max_layer if i has no successors.
    alap_layer: dict[int, int] = {}
    for node_idx in reversed(topo_order):
        succs = dag.successor_indices(node_idx)
        alap_layer[node_idx] = (
            max_layer if not succs else min(alap_layer[s] for s in succs) - 1
        )

    layer_buckets: dict[int, list[_SchedulableStmt]] = {}
    for i, stmt in enumerate(stmts):
        layer_buckets.setdefault(alap_layer[node_for[i]], []).append(stmt)

    result: list[_SchedulableStmt] = []
    for lyr in sorted(layer_buckets):
        result.extend(_group_within_layer(layer_buckets[lyr]))
    return result


def alap_reorder_policy(
    stmts: list[_SchedulableStmt],
) -> list[_SchedulableStmt]:
    """ALAP scheduling policy. Hard barriers (Initialize, EndMeasure) divide
    the body into independent segments; each segment is scheduled separately.

    Gates are assigned to their *latest* valid layer, deferring single-qubit
    gates as long as possible to minimise the qubit footprint of earlier
    CZ-anchored StaticPlacement regions.
    """
    result: list[_SchedulableStmt] = []
    segment: list[_SchedulableStmt] = []

    for stmt in stmts:
        if isinstance(stmt, _BARRIERS):
            result.extend(_alap_schedule(segment))
            result.append(stmt)
            segment = []
        else:
            segment.append(stmt)
    result.extend(_alap_schedule(segment))
    return result
