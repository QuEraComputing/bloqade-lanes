"""Common goal predicates for configuration search."""

from __future__ import annotations

from bloqade.lanes.layout import LocationAddress
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.search.configuration import ConfigurationNode
from bloqade.lanes.search.strategies import GoalPredicate


def placement_goal(target: dict[int, LocationAddress]) -> GoalPredicate:
    """Goal: all specified qubits are at their target locations.

    Every qubit in `target` must be at the exact location. Qubits not
    in `target` are ignored.

    Args:
        target: Mapping of qubit ID to desired location.

    Returns:
        A GoalPredicate that returns True when all targets are met.
    """

    def goal(node: ConfigurationNode) -> bool:
        return all(node.configuration.get(qid) == loc for qid, loc in target.items())

    return goal


def partial_placement_goal(
    target: dict[int, LocationAddress],
    min_placed: int | None = None,
) -> GoalPredicate:
    """Goal: at least some qubits are at their target locations.

    If `min_placed` is None, all qubits in `target` must be placed
    (same as `placement_goal`). Otherwise, at least `min_placed`
    qubits must be at their target.

    Args:
        target: Mapping of qubit ID to desired location.
        min_placed: Minimum number of qubits that must be at their
            target. None means all.

    Returns:
        A GoalPredicate.
    """
    required = min_placed if min_placed is not None else len(target)

    def goal(node: ConfigurationNode) -> bool:
        placed = sum(
            1 for qid, loc in target.items() if node.configuration.get(qid) == loc
        )
        return placed >= required

    return goal


def zone_goal(zone_id: int, arch_spec: ArchSpec) -> GoalPredicate:
    """Goal: all qubits are located in the specified zone.

    A qubit is "in the zone" if its location's word_id is in the
    zone's word list.

    Args:
        zone_id: The zone to target.
        arch_spec: Architecture specification (for zone → word mapping).

    Returns:
        A GoalPredicate that returns True when all qubits are in the zone.
    """
    zone_words = set(arch_spec.zones[zone_id])

    def goal(node: ConfigurationNode) -> bool:
        return all(loc.word_id in zone_words for loc in node.configuration.values())

    return goal
