"""Deadlock detection and diagnosis for configuration search."""

from __future__ import annotations

from dataclasses import dataclass, field

from bloqade.lanes.layout import LaneAddress, LocationAddress
from bloqade.lanes.search.configuration import ConfigurationNode
from bloqade.lanes.search.tree import ConfigurationTree


@dataclass(frozen=True)
class BlockedMove:
    """A single move that was considered but rejected."""

    qubit_id: int
    lane: LaneAddress
    src: LocationAddress
    dst: LocationAddress
    reason: str


@dataclass
class DeadlockDiagnosis:
    """Analysis of why a configuration has no valid moves.

    Produced by `diagnose_deadlock` when a node has no valid children.
    """

    blocked_moves: list[BlockedMove] = field(default_factory=list)
    """All individual moves that were considered and why each was rejected."""

    circular_dependencies: list[tuple[int, ...]] = field(default_factory=list)
    """Groups of qubit IDs that mutually block each other.

    E.g., (0, 1) means qubit 0's destination is occupied by qubit 1 and
    qubit 1's destination is occupied by qubit 0, and they cannot move
    simultaneously (different buses or AOD constraints).
    """

    isolated_qubits: list[int] = field(default_factory=list)
    """Qubit IDs that have no valid outgoing lanes at all."""

    @property
    def is_deadlocked(self) -> bool:
        """True if no moves are possible from this configuration."""
        return len(self.blocked_moves) > 0 or len(self.isolated_qubits) > 0


def diagnose_deadlock(
    node: ConfigurationNode,
    tree: ConfigurationTree,
) -> DeadlockDiagnosis:
    """Analyze why a configuration has no valid moves.

    For each atom, enumerates all possible lanes and records why each
    is blocked. Also detects circular dependencies where atoms mutually
    block each other.

    Args:
        node: The stuck configuration node.
        tree: The configuration tree (provides arch_spec and path_finder).

    Returns:
        A DeadlockDiagnosis with details on blocked moves, circular
        dependencies, and isolated atoms.
    """
    occupied = node.occupied_locations
    blocked: list[BlockedMove] = []
    isolated: list[int] = []

    # Track which qubits block which: blocker[qid] = set of qids it blocks
    blocked_by: dict[int, set[int]] = {}

    for qid, src_loc in node.configuration.items():
        src_idx = tree.path_finder.physical_address_map.get(src_loc)
        if src_idx is None:
            isolated.append(qid)
            continue

        has_any_edge = False
        for dst_idx in tree.path_finder.site_graph.successor_indices(src_idx):
            has_any_edge = True
            dst_loc = tree.path_finder.physical_addresses[dst_idx]
            lane = tree.path_finder.site_graph.get_edge_data(src_idx, dst_idx)
            if lane is None:
                continue

            if dst_loc in occupied:
                blocker_qid = node.get_qubit_at(dst_loc)
                reason = (
                    f"destination occupied by qubit {blocker_qid}"
                    if blocker_qid is not None
                    else "destination occupied"
                )
                blocked.append(BlockedMove(qid, lane, src_loc, dst_loc, reason))

                # Record blocking relationship
                if blocker_qid is not None:
                    blocked_by.setdefault(qid, set()).add(blocker_qid)
            else:
                # Destination is free — check lane validation
                errors = tree.arch_spec.validate_lane(lane)
                if errors:
                    reason = f"validation failed: {next(iter(errors))}"
                    blocked.append(BlockedMove(qid, lane, src_loc, dst_loc, reason))

        if not has_any_edge:
            isolated.append(qid)

    # Detect circular dependencies
    circular: list[tuple[int, ...]] = []
    visited: set[int] = set()

    for qid in blocked_by:
        if qid in visited:
            continue

        # Follow the blocking chain to find cycles
        chain: list[int] = []
        current = qid
        chain_set: set[int] = set()

        while current is not None and current not in chain_set:
            chain.append(current)
            chain_set.add(current)
            # Find if current blocks someone who blocks current back
            blockers = blocked_by.get(current, set())
            current = next(
                (
                    b
                    for b in blockers
                    if b in blocked_by and qid in blocked_by.get(b, set())
                ),
                None,
            )

        if current is not None and current in chain_set:
            # Found a cycle — extract it
            cycle_start = chain.index(current)
            cycle = tuple(sorted(chain[cycle_start:]))
            if cycle not in [tuple(sorted(c)) for c in circular]:
                circular.append(cycle)
                visited.update(cycle)

    return DeadlockDiagnosis(
        blocked_moves=blocked,
        circular_dependencies=circular,
        isolated_qubits=isolated,
    )
