"""Candidate scoring for entropy-guided search.

Two-level scoring:
1. Per-qubit-lane context used to prepare legal rectangle candidates.
2. Per-moveset: score[M] — alpha*distance + beta*arrived + gamma*mobility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bloqade.lanes.layout import Direction, LaneAddress, LocationAddress, MoveType

if TYPE_CHECKING:
    from bloqade.lanes.search.configuration import ConfigurationNode
    from bloqade.lanes.search.search_params import SearchParams
    from bloqade.lanes.search.tree import ConfigurationTree


@dataclass(frozen=True)
class RectangleBusCandidates:
    """Rectangle candidate prep for one (move_type, bus_id, direction) group.

    Attributes:
        valid_entries: Positive-scoring unresolved movers for this bus context.
            Mapping is qid -> lane from that qubit's source.
        invalid_sources: Occupied sources that must invalidate any rectangle
            containing them (non-movers, collision-risk movers, or non-positive
            scoring movers for this bus context).
        qubit_scores: Score for each qubit in valid_entries.
    """

    valid_entries: dict[int, LaneAddress]
    invalid_sources: frozenset[LocationAddress]
    qubit_scores: dict[int, float]


@dataclass
class CandidateScorer:
    """Scores qubit-bus pairs and movesets for entropy-guided search."""

    params: SearchParams
    target: dict[int, LocationAddress]
    _fastest_lane_duration_cache: dict[int, float] = field(
        default_factory=dict, init=False, repr=False
    )

    def _fastest_lane_duration_us(self, tree: ConfigurationTree) -> float:
        """Return fastest lane duration for a tree (cached)."""
        tree_id = id(tree)
        cached = self._fastest_lane_duration_cache.get(tree_id)
        if cached is not None:
            return cached

        fastest = float("inf")
        for lanes in tree._lanes_by_triplet.values():
            for lane in lanes:
                duration = tree.path_finder.metrics.get_lane_duration_us(lane)
                if duration > 0.0:
                    fastest = min(fastest, duration)

        if fastest == float("inf"):
            fastest = 1.0
        self._fastest_lane_duration_cache[tree_id] = fastest
        return fastest

    def _distance_to_target(
        self,
        current: LocationAddress,
        target_loc: LocationAddress,
        tree: ConfigurationTree,
    ) -> float:
        """Blended shortest-path distance from current to target.

        Combines:
        - hop-count shortest path (uniform lane cost 1.0), and
        - approximate move-time shortest path (lane duration in microseconds),
          normalized into hop-like units by dividing by the fastest lane duration.

        Blend weight is controlled by SearchParams.w_t:
        - w_t = 0.0 => hop-count only
        - w_t = 1.0 => move-time only (normalized)

        Occupancy is intentionally ignored for this heuristic estimate.
        Returns float('inf') if no path.
        """
        hop_result = tree.path_finder.find_path(
            current,
            target_loc,
            edge_weight=lambda _lane: 1.0,
        )
        if hop_result is None:
            return float("inf")
        hop_lanes, _ = hop_result
        hop_distance = float(len(hop_lanes))

        if self.params.w_t <= 0.0:
            return hop_distance

        time_result = tree.path_finder.find_path(
            current,
            target_loc,
            edge_weight=tree.path_finder.metrics.get_lane_duration_us,
        )
        if time_result is None:
            return float("inf")
        time_lanes, _ = time_result
        time_us = sum(
            tree.path_finder.metrics.get_lane_duration_us(lane) for lane in time_lanes
        )
        fastest_lane_us = self._fastest_lane_duration_us(tree)
        time_distance = time_us / fastest_lane_us

        w_t = self.params.w_t
        return (1.0 - w_t) * hop_distance + w_t * time_distance

    def _mobility_at(
        self,
        position: LocationAddress,
        target_loc: LocationAddress,
        occupied: frozenset[LocationAddress],
        tree: ConfigurationTree,
    ) -> float:
        """Sum distance-weighted legal lane mobility from a position.

        A lane is legal if its destination is not occupied. Each legal lane
        contributes a weight inversely proportional to the post-lane distance
        to the qubit's target location, so closer destinations count more.
        """
        mobility = 0.0
        for lane in tree.outgoing_lanes(position):
            _, dst = tree.arch_spec.get_endpoints(lane)
            if dst in occupied:
                continue
            d_after = self._distance_to_target(dst, target_loc, tree)
            if d_after == float("inf"):
                continue
            mobility += 1.0 / (1.0 + d_after)
        return mobility

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
        m_now: dict[int, float] = {}
        for qid, loc in unresolved.items():
            d_now[qid] = self._distance_to_target(loc, self.target[qid], tree)
            m_now[qid] = self._mobility_at(loc, self.target[qid], occupied, tree)

        # Compute raw deltas for all legal (qubit, move_type, bus_id, direction)
        d_after_cache: dict[tuple[int, LocationAddress], float] = {}
        m_after_cache: dict[tuple[int, LocationAddress], float] = {}
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
                m_key = (qid, dst)
                m_after = m_after_cache.get(m_key)
                if m_after is None:
                    m_after = self._mobility_at(dst, self.target[qid], occupied, tree)
                    m_after_cache[m_key] = m_after
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

    def score_rectangle_bus_candidates(
        self,
        node: ConfigurationNode,
        entropy: int,
        tree: ConfigurationTree,
    ) -> dict[tuple[MoveType, int, Direction], RectangleBusCandidates]:
        """Prepare per-bus rectangle seeds and invalid source buckets.

        The moving set is unresolved target qubits only. For each bus context,
        any occupied source is placed into ``invalid_sources`` when:
        - the source qubit is not in the moving set,
        - the move collides with occupied/blocked destination, or
        - the qubit's score for this bus context is non-positive.
        """
        occupied = node.occupied_locations | tree.blocked_locations
        moving_qubits = {
            qid
            for qid, loc in node.configuration.items()
            if qid in self.target and loc != self.target[qid]
        }
        if not moving_qubits:
            return {}

        per_qubit_scores = self.score_all_qubit_bus_pairs(node, entropy, tree)
        result: dict[tuple[MoveType, int, Direction], RectangleBusCandidates] = {}

        for mt in (MoveType.SITE, MoveType.WORD):
            buses = (
                tree.arch_spec.site_buses
                if mt == MoveType.SITE
                else tree.arch_spec.word_buses
            )
            for bus_id in range(len(buses)):
                for direction in (Direction.FORWARD, Direction.BACKWARD):
                    valid_entries: dict[int, LaneAddress] = {}
                    valid_scores: dict[int, float] = {}
                    invalid_sources: set[LocationAddress] = set()

                    for lane in tree.lanes_for(mt, bus_id, direction):
                        src, dst = tree.arch_spec.get_endpoints(lane)
                        qid = node.get_qubit_at(src)
                        if qid is None:
                            continue

                        if qid not in moving_qubits:
                            invalid_sources.add(src)
                            continue

                        score_key = (qid, mt, bus_id, direction)
                        score = per_qubit_scores.get(score_key)
                        if score is None or dst in occupied or score <= 0.0:
                            invalid_sources.add(src)
                            continue

                        valid_entries[qid] = lane
                        valid_scores[qid] = score

                    if valid_entries or invalid_sources:
                        result[(mt, bus_id, direction)] = RectangleBusCandidates(
                            valid_entries=valid_entries,
                            invalid_sources=frozenset(invalid_sources),
                            qubit_scores=valid_scores,
                        )

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
            mobility_before += self._mobility_at(src, self.target[qid], occupied, tree)
            mobility_after += self._mobility_at(
                dst, self.target[qid], new_occupied, tree
            )

        mobility_gain = mobility_after - mobility_before

        return (
            self.params.alpha * distance_moved
            + self.params.beta * arrived_gain
            + self.params.gamma * mobility_gain
        )
