"""Candidate scoring for entropy-guided search.

Two-level scoring:
1. Per-qubit-bus: s(q, b, d; E) — entropy-weighted distance/mobility heuristic
2. Per-moveset: score[M] — alpha*distance + beta*arrived + gamma*mobility
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bloqade.lanes.layout import Direction, LaneAddress, LocationAddress, MoveType

if TYPE_CHECKING:
    from bloqade.lanes.search.configuration import ConfigurationNode
    from bloqade.lanes.search.search_params import SearchParams
    from bloqade.lanes.search.tree import ConfigurationTree


@dataclass
class CandidateScorer:
    """Scores qubit-bus pairs and movesets for entropy-guided search."""

    params: SearchParams
    target: dict[int, LocationAddress]

    def _distance_to_target(
        self,
        current: LocationAddress,
        target_loc: LocationAddress,
        tree: ConfigurationTree,
    ) -> float:
        """Weighted shortest path cost from current to target.

        Uses MoveMetricCalculator.get_lane_duration_us as edge weight.
        Does not pass an occupied set — paths are computed over the full
        graph, which is appropriate for a heuristic distance estimate.
        Returns float('inf') if no path.
        """
        result = tree.path_finder.find_path(
            current,
            target_loc,
            edge_weight=tree.path_finder.metrics.get_lane_duration_us,
        )
        if result is None:
            return float("inf")
        lanes, _ = result
        return sum(
            tree.path_finder.metrics.get_lane_duration_us(lane) for lane in lanes
        )

    def _mobility_at(
        self,
        position: LocationAddress,
        occupied: frozenset[LocationAddress],
        tree: ConfigurationTree,
    ) -> int:
        """Count distinct legal lane addresses from a position.

        A lane is legal if its destination is not occupied.
        """
        return sum(
            1
            for lane in tree.outgoing_lanes(position)
            if tree.arch_spec.get_endpoints(lane)[1] not in occupied
        )

    def score_all_qubit_bus_pairs(
        self,
        node: ConfigurationNode,
        entropy: int,
        tree: ConfigurationTree,
    ) -> dict[tuple[int, MoveType, int, Direction], float]:
        """Score all legal (qubit, move_type, bus_id, direction) tuples.

        Returns mapping of (qubit_id, move_type, bus_id, direction) -> score.
        Only includes legal tuples where the qubit is on the bus source and
        the destination is unoccupied.
        """
        occupied = node.occupied_locations | tree.blocked_locations
        e_eff = min(entropy, self.params.e_max)

        # Identify unresolved qubits
        unresolved = {
            qid: loc
            for qid, loc in node.configuration.items()
            if qid in self.target and loc != self.target[qid]
        }
        if not unresolved:
            return {}

        # Cache d_now and m_now per qubit (same across all buses)
        d_now: dict[int, float] = {}
        m_now: dict[int, int] = {}
        for qid, loc in unresolved.items():
            d_now[qid] = self._distance_to_target(loc, self.target[qid], tree)
            m_now[qid] = self._mobility_at(loc, occupied, tree)

        # Compute raw deltas for all legal (qubit, move_type, bus_id, direction)
        d_after_cache: dict[tuple[int, LocationAddress], float] = {}
        m_after_cache: dict[LocationAddress, int] = {}
        raw_scores: dict[tuple[int, MoveType, int, Direction], tuple[float, float]] = {}
        for qid, loc in unresolved.items():
            for lane in tree.outgoing_lanes(loc):
                mt = lane.move_type
                bus_id = lane.bus_id
                direction = lane.direction
                _, dst = tree.arch_spec.get_endpoints(lane)
                if dst in occupied:
                    continue
                d_key = (qid, dst)
                d_after = d_after_cache.get(d_key)
                if d_after is None:
                    d_after = self._distance_to_target(dst, self.target[qid], tree)
                    d_after_cache[d_key] = d_after
                m_after = m_after_cache.get(dst)
                if m_after is None:
                    m_after = self._mobility_at(dst, occupied, tree)
                    m_after_cache[dst] = m_after
                delta_d = d_now[qid] - d_after
                delta_m = m_after - m_now[qid]
                raw_scores[(qid, mt, bus_id, direction)] = (delta_d, delta_m)

        if not raw_scores:
            return {}

        # Normalize
        d_ref = max(1.0, max(abs(dd) for dd, _ in raw_scores.values()))
        m_ref = max(1.0, max(abs(dm) for _, dm in raw_scores.values()))

        # Apply entropy-weighted formula
        result: dict[tuple[int, MoveType, int, Direction], float] = {}
        for key, (delta_d, delta_m) in raw_scores.items():
            d_hat = delta_d / d_ref
            m_hat = delta_m / m_ref
            score = (self.params.w_d / e_eff) * d_hat + self.params.w_m * e_eff * m_hat
            result[key] = score

        return result

    def score_moveset(
        self,
        moveset: frozenset[LaneAddress],
        node: ConfigurationNode,
        tree: ConfigurationTree,
    ) -> float:
        """Score a candidate moveset.

        Returns alpha*distance_moved + beta*arrived_gain + gamma*mobility_gain.
        """
        occupied = node.occupied_locations | tree.blocked_locations
        distance_moved = 0.0
        arrived_gain = 0
        mobility_before = 0
        mobility_after = 0

        # Identify moved qubits and build post-move config
        new_config: dict[int, LocationAddress] = dict(node.configuration)
        moved_qubits: list[tuple[int, LocationAddress, LocationAddress]] = []

        for lane in moveset:
            src, dst = tree.arch_spec.get_endpoints(lane)
            qid = node.get_qubit_at(src)
            if qid is None:
                continue
            moved_qubits.append((qid, src, dst))
            new_config[qid] = dst

        new_occupied = frozenset(new_config.values()) | tree.blocked_locations

        for qid, src, dst in moved_qubits:
            if qid not in self.target:
                continue
            d_before = self._distance_to_target(src, self.target[qid], tree)
            d_after = self._distance_to_target(dst, self.target[qid], tree)
            distance_moved += max(0.0, d_before - d_after)
            if dst == self.target[qid]:
                arrived_gain += 1
            mobility_before += self._mobility_at(src, occupied, tree)
            mobility_after += self._mobility_at(dst, new_occupied, tree)

        mobility_gain = mobility_after - mobility_before

        return (
            self.params.alpha * distance_moved
            + self.params.beta * arrived_gain
            + self.params.gamma * mobility_gain
        )
