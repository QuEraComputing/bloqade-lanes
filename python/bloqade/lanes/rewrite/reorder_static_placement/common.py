"""Shared type aliases and layer-grouping utilities for scheduling policies."""

import rustworkx

from bloqade.lanes.dialects import place

_BARRIERS: tuple[type, ...] = (place.Initialize, place.EndMeasure)

# Union of all concrete QuantumStmt subclasses that carry a .qubits attribute.
_SchedulableStmt = (
    place.R | place.Rz | place.StarRz | place.CZ | place.Initialize | place.EndMeasure
)


def _group_key(stmt: _SchedulableStmt) -> tuple:
    """Hashable key identifying gates that FuseAdjacentGates can fuse together."""
    if isinstance(stmt, place.R):
        return (type(stmt), id(stmt.axis_angle), id(stmt.rotation_angle))
    if isinstance(stmt, place.Rz):
        return (type(stmt), id(stmt.rotation_angle))
    if isinstance(stmt, place.StarRz):
        return (type(stmt), id(stmt.rotation_angle), stmt.qubit_indices)
    return (type(stmt),)  # CZ has no non-qubit params


def _build_dependency_dag(
    stmts: list[_SchedulableStmt],
) -> tuple[rustworkx.PyDAG, list[int]]:
    """Build a qubit-dependency DAG for a flat list of schedulable statements.

    Returns ``(dag, node_for)`` where ``node_for[i]`` is the DAG node index
    corresponding to ``stmts[i]``.  An edge ``u → v`` means the gate at
    position ``v`` must execute strictly after the gate at position ``u``
    because they share at least one qubit.
    """
    dag: rustworkx.PyDAG = rustworkx.PyDAG()
    node_for: list[int] = [dag.add_node(i) for i in range(len(stmts))]
    last_touch: dict[int, int] = {}
    for i, stmt in enumerate(stmts):
        for q in stmt.qubits:
            if q in last_touch:
                dag.add_edge(last_touch[q], node_for[i], None)
            last_touch[q] = node_for[i]
    return dag, node_for


def _group_within_layer(layer_stmts: list[_SchedulableStmt]) -> list[_SchedulableStmt]:
    """Re-order layer_stmts so fusable-equivalent gates are adjacent.

    Uses Python dict insertion-order: the first time a (type, params) key is
    seen a new group is opened; subsequent matches append to that group.
    Iterating over groups in insertion order preserves first-seen ordering.
    """
    groups: dict[tuple, list[_SchedulableStmt]] = {}
    for stmt in layer_stmts:
        groups.setdefault(_group_key(stmt), []).append(stmt)
    return [stmt for group in groups.values() for stmt in group]
