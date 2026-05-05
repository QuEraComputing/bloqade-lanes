"""Shared type aliases and layer-grouping utilities for scheduling policies."""

from bloqade.lanes.dialects import place

_BARRIERS: tuple[type, ...] = (place.Initialize, place.EndMeasure)

# Union of all concrete QuantumStmt subclasses that carry a .qubits attribute.
_SchedulableStmt = place.R | place.Rz | place.CZ | place.Initialize | place.EndMeasure


def _group_key(stmt: _SchedulableStmt) -> tuple:
    """Hashable key identifying gates that FuseAdjacentGates can fuse together."""
    if isinstance(stmt, place.R):
        return (type(stmt), id(stmt.axis_angle), id(stmt.rotation_angle))
    if isinstance(stmt, place.Rz):
        return (type(stmt), id(stmt.rotation_angle))
    return (type(stmt),)  # CZ has no non-qubit params


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
