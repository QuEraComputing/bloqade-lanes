"""Entropy-weighted heuristic move generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from bloqade.lanes.layout import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
)
from bloqade.lanes.search.generators.base import EntropyNode

if TYPE_CHECKING:
    from bloqade.lanes.search.configuration import ConfigurationNode
    from bloqade.lanes.search.scoring import CandidateScorer
    from bloqade.lanes.search.search_params import SearchParams
    from bloqade.lanes.search.tree import ConfigurationTree


@dataclass
class HeuristicMoveGenerator:
    """Generates ranked candidate movesets using entropy-weighted scoring.

    Implements the MoveGenerator protocol. The traversal sets
    ``search_nodes`` before the search loop begins so the generator
    can read per-node entropy.
    """

    scorer: CandidateScorer
    params: SearchParams
    search_nodes: dict[int, EntropyNode]
    """Mapping of id(ConfigurationNode) -> entropy metadata node."""

    def generate(
        self,
        node: ConfigurationNode,
        tree: ConfigurationTree,
    ) -> Iterator[frozenset[LaneAddress]]:
        """Yield candidate movesets ranked by moveset score (descending).

        Steps:
        1. Get entropy from traversal metadata node.
        2. Score all (qubit, move_type, bus_id, direction) pairs.
        3. Per-qubit: keep top C triples.
        4. Group by (move_type, bus_id, direction), collect positive-score qubits.
        5. Resolve conflicts within each group (greedy by score).
        6. Build moveset per group, score with score_moveset, sort descending.
        """
        search_node = self.search_nodes.get(id(node))
        entropy = search_node.entropy if search_node is not None else 1

        scores = self.scorer.score_all_qubit_bus_pairs(node, entropy, tree)
        if not scores:
            return

        # Per-qubit: keep top C (move_type, bus_id, direction) triples
        qubit_top: dict[int, list[tuple[MoveType, int, Direction, float]]] = {}
        for (qid, mt, bid, d), score in scores.items():
            qubit_top.setdefault(qid, []).append((mt, bid, d, score))

        for qid in qubit_top:
            qubit_top[qid].sort(key=lambda x: x[3], reverse=True)
            qubit_top[qid] = qubit_top[qid][: self.params.top_c]

        # Group by (move_type, bus_id, direction): collect positive-scoring qubits
        groups: dict[
            tuple[MoveType, int, Direction],
            list[tuple[int, float]],
        ] = {}
        for qid, top_triples in qubit_top.items():
            for mt, bid, d, score in top_triples:
                if score > 0:
                    groups.setdefault((mt, bid, d), []).append((qid, score))

        # Fallback: if no group has positive-scoring qubits, use best single entry
        if not groups:
            best_key = max(scores, key=lambda key: scores[key])
            qid, mt, bid, d = best_key
            groups[(mt, bid, d)] = [(qid, scores[best_key])]

        # Build one moveset per group with conflict resolution
        occupied = node.occupied_locations | tree.blocked_locations
        candidates: list[tuple[float, frozenset[LaneAddress]]] = []

        for (mt, bid, d), qubit_scores in groups.items():
            # Sort qubits by score descending for greedy conflict resolution
            qubit_scores.sort(key=lambda x: x[1], reverse=True)

            lanes: list[LaneAddress] = []
            used_dsts: set[LocationAddress] = set()

            for qid, _ in qubit_scores:
                loc = node.configuration[qid]
                lane = next(
                    (
                        la
                        for la in tree.outgoing_lanes(loc)
                        if la.move_type == mt and la.bus_id == bid and la.direction == d
                    ),
                    None,
                )
                if lane is None:
                    continue
                _, dst = tree.arch_spec.get_endpoints(lane)
                if dst in occupied:
                    continue
                # Destination conflict: another qubit in this group targets same dst
                if dst in used_dsts:
                    continue
                # Lane compatibility check with existing lanes in this group
                if lanes and not all(
                    tree.arch_spec.compatible_lanes(lane, existing)
                    for existing in lanes
                ):
                    continue
                lanes.append(lane)
                used_dsts.add(dst)

            if not lanes:
                continue

            moveset = frozenset(lanes)
            ms_score = self.scorer.score_moveset(moveset, node, tree)
            candidates.append((ms_score, moveset))

        # Sort by moveset score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        for _, moveset in candidates:
            yield moveset
