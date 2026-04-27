"""Entropy-weighted heuristic move generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from bloqade.lanes.layout import (
    LaneAddress,
    LocationAddress,
)
from bloqade.lanes.search.generators.aod_grouping import BusContext
from bloqade.lanes.search.generators.base import EntropyNode

if TYPE_CHECKING:
    from bloqade.lanes.search.configuration import ConfigurationNode
    from bloqade.lanes.search.scoring import CandidateScorer
    from bloqade.lanes.search.search_params import SearchParams
    from bloqade.lanes.search.tree import ConfigurationTree


@dataclass
class HeuristicMoveGenerator:
    """Generates globally-ranked rectangle candidates from entropy scoring.

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
        """Yield globally-ranked legal rectangle movesets.

        For each bus-direction context, positive-scoring unresolved movers
        seed candidate AOD grids. Any rectangle that includes a source in the
        context invalid bucket (non-mover, collision-risk, or non-positive
        score mover) is rejected.
        """
        search_node = self.search_nodes.get(id(node))
        entropy = search_node.entropy if search_node is not None else 1

        bus_candidates = self.scorer.score_rectangle_bus_candidates(node, entropy, tree)

        occupied = node.occupied_locations | tree.blocked_locations
        scored_candidates: dict[frozenset[LaneAddress], float] = {}

        for (mt, bus_id, direction), bucket in bus_candidates.items():
            if not bucket.valid_entries:
                continue

            context = BusContext.from_tree(
                tree=tree,
                occupied=occupied,
                move_type=mt,
                bus_id=bus_id,
                direction=direction,
            )
            for moveset in context.build_aod_grids(bucket.valid_entries):
                if not self._is_valid_rectangle_candidate(
                    moveset=moveset,
                    node=node,
                    tree=tree,
                    invalid_sources=bucket.invalid_sources,
                ):
                    continue
                ms_score = self.scorer.score_moveset(moveset, node, tree)
                best = scored_candidates.get(moveset)
                if best is None or ms_score > best:
                    scored_candidates[moveset] = ms_score

        for moveset, _ in sorted(
            scored_candidates.items(), key=lambda item: item[1], reverse=True
        ):
            yield moveset
        if not scored_candidates:
            fallback = self._best_singleton_fallback(node, tree, entropy)
            if fallback is not None:
                yield fallback

    def _is_valid_rectangle_candidate(
        self,
        moveset: frozenset[LaneAddress],
        node: ConfigurationNode,
        tree: ConfigurationTree,
        invalid_sources: frozenset[LocationAddress],
    ) -> bool:
        """Reject rectangles that include any invalid occupied source."""
        for lane in moveset:
            endpoints = tree.arch_spec.get_endpoints(lane)
            assert (
                endpoints is not None
            ), f"lane {lane!r} has no endpoints in this architecture"
            src, _ = endpoints
            if src in invalid_sources:
                return False
            qid = node.get_qubit_at(src)
            if qid is None:
                continue
            if qid not in self.scorer.target:
                return False
            if node.configuration[qid] == self.scorer.target[qid]:
                return False
        return True

    def _best_singleton_fallback(
        self,
        node: ConfigurationNode,
        tree: ConfigurationTree,
        entropy: int,
    ) -> frozenset[LaneAddress] | None:
        """Return best unresolved singleton lane when no rectangles survive."""
        scores = self.scorer.score_all_qubit_bus_pairs(node, entropy, tree)
        if not scores:
            return None

        qid, mt, bus_id, direction = max(scores, key=scores.__getitem__)
        current = node.configuration.get(qid)
        if current is None:
            return None
        if qid not in self.scorer.target or current == self.scorer.target[qid]:
            return None

        lane = tree.lane_for_source(mt, bus_id, direction, current)
        if lane is None:
            return None
        endpoints = tree.arch_spec.get_endpoints(lane)
        assert (
            endpoints is not None
        ), f"lane {lane!r} has no endpoints in this architecture"
        _, dst = endpoints
        occupied = node.occupied_locations | tree.blocked_locations
        if dst in occupied:
            return None
        return frozenset({lane})
