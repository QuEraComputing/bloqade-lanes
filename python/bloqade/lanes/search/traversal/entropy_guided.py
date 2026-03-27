"""Entropy-guided tree search traversal."""

from __future__ import annotations

__all__ = ["entropy_guided_search", "EntropyGuidedSearch"]

from dataclasses import dataclass, field
from typing import Callable

from bloqade.lanes.layout import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
)
from bloqade.lanes.search.configuration import ConfigurationNode
from bloqade.lanes.search.generators import EntropyNode, HeuristicMoveGenerator
from bloqade.lanes.search.scoring import CandidateScorer
from bloqade.lanes.search.search_params import SearchParams
from bloqade.lanes.search.traversal.goal import GoalPredicate, SearchResult
from bloqade.lanes.search.traversal.step_info import (
    DescendStepInfo,
    EntropyBumpStepInfo,
    FallbackStartStepInfo,
    FallbackStepInfo,
    GoalStepInfo,
    RevertStepInfo,
    StepInfo,
)
from bloqade.lanes.search.tree import ConfigurationTree, ExpansionStatus


@dataclass
class _SearchNode:
    """Per-node entropy-guided traversal state."""

    config_node: ConfigurationNode
    entropy: int = 1
    candidate_cache: list[frozenset[LaneAddress]] = field(default_factory=list)
    candidates_tried: int = 0
    last_entropy_bump_reason: str = "entropy"
    last_no_valid_moves_qubit: int | None = None
    last_state_seen_node_id: int | None = None


def _print_node_transition(
    prefix: str,
    node: ConfigurationNode,
    reason: str | None = None,
    seen_node_id: int | None = None,
    no_valid_moves_qubit: int | None = None,
) -> None:
    """Temporary debugging output for node traversal transitions."""
    parent_id = id(node.parent) if node.parent is not None else None
    child_ids = [id(child) for child in node.children.values()]
    reason_part = f" reason={reason}" if reason is not None else ""
    seen_part = f" seen_node_id={seen_node_id}" if seen_node_id is not None else ""
    qubit_part = (
        f" no_valid_moves_qubit={no_valid_moves_qubit}"
        if no_valid_moves_qubit is not None
        else ""
    )
    print(
        f"[{prefix}] depth={node.depth} node_id={id(node)} "
        f"parent_id={parent_id} child_ids={child_ids}"
        f"{reason_part}{seen_part}{qubit_part}"
    )


class EntropyGuidedSearch:
    """Entropy-guided depth-first search with reversion and sequential fallback.

    Encapsulates the internal state of the search so that helper methods
    can access shared context (tree, target, params, etc.) without
    requiring long parameter lists.
    """

    def __init__(
        self,
        tree: ConfigurationTree,
        target: dict[int, LocationAddress],
        goal: GoalPredicate,
        params: SearchParams = SearchParams(),
        max_depth: int | None = None,
        max_expansions: int | None = None,
        on_step: Callable[[str, ConfigurationNode, StepInfo], None] | None = None,
    ) -> None:
        self.tree = tree
        self.target = target
        self.goal = goal
        self.params = params
        self.max_depth = max_depth
        self.max_expansions = max_expansions
        self.on_step = on_step

        self.scorer = CandidateScorer(params=params)
        self.search_nodes: dict[int, _SearchNode] = {}
        # _SearchNode satisfies EntropyNode protocol; cast for dict invariance.
        entropy_view: dict[int, EntropyNode] = self.search_nodes  # type: ignore[assignment]
        self.generator = HeuristicMoveGenerator(
            scorer=self.scorer,
            params=params,
            target=target,
            search_nodes=entropy_view,
        )
        self.nodes_expanded = 0
        self.max_depth_reached = 0

    def _emit_step(self, event: str, node: ConfigurationNode, info: StepInfo) -> None:
        if self.on_step is not None:
            self.on_step(event, node, info)

    def _unresolved_count(self, node: ConfigurationNode) -> int:
        return sum(
            1 for qid, loc in self.target.items() if node.configuration.get(qid) != loc
        )

    def _get_or_create_search_node(self, config_node: ConfigurationNode) -> _SearchNode:
        node_id = id(config_node)
        sn = self.search_nodes.get(node_id)
        if sn is None:
            sn = _SearchNode(config_node=config_node)
            self.search_nodes[node_id] = sn
        return sn

    def _get_next_candidate(
        self, sn: _SearchNode, node: ConfigurationNode
    ) -> frozenset[LaneAddress] | None:
        """Get the next untried candidate, enforcing max_candidates (K) limit.

        After K candidates tried from current cache, forces regeneration.
        Returns None if all generated candidates are already node children
        (failed generation).
        """
        if sn.candidates_tried >= self.params.max_candidates:
            new_candidates = list(self.generator.generate(node, self.tree))
            sn.candidate_cache = new_candidates
            sn.candidates_tried = 0

            while sn.candidates_tried < len(sn.candidate_cache):
                candidate = sn.candidate_cache[sn.candidates_tried]
                if candidate not in node.children:
                    return candidate
                sn.candidates_tried += 1
            return None

        while sn.candidates_tried < len(sn.candidate_cache):
            candidate = sn.candidate_cache[sn.candidates_tried]
            if candidate not in node.children:
                return candidate
            sn.candidates_tried += 1

        new_candidates = list(self.generator.generate(node, self.tree))
        sn.candidate_cache = new_candidates
        sn.candidates_tried = 0

        while sn.candidates_tried < len(sn.candidate_cache):
            candidate = sn.candidate_cache[sn.candidates_tried]
            if candidate not in node.children:
                return candidate
            sn.candidates_tried += 1

        return None

    def _first_unresolved_qubit_without_valid_move(
        self, node: ConfigurationNode
    ) -> int | None:
        """Return one unresolved qubit that currently has no valid lane move."""
        occupied = node.occupied_locations
        for qid, target_loc in self.target.items():
            current_loc = node.configuration.get(qid)
            if current_loc is None or current_loc == target_loc:
                continue

            has_valid_lane = False
            for mt in (MoveType.SITE, MoveType.WORD):
                buses = (
                    range(len(self.tree.arch_spec.site_buses))
                    if mt == MoveType.SITE
                    else range(len(self.tree.arch_spec.word_buses))
                )
                for bus_id in buses:
                    for direction in (Direction.FORWARD, Direction.BACKWARD):
                        for lane in self.tree.lanes_for(mt, bus_id, direction):
                            src, dst = self.tree.arch_spec.get_endpoints(lane)
                            if src == current_loc and dst not in occupied:
                                has_valid_lane = True
                                break
                        if has_valid_lane:
                            break
                    if has_valid_lane:
                        break
                if has_valid_lane:
                    break

            if not has_valid_lane:
                return qid

        return None

    def _make_result(self, goal_node: ConfigurationNode | None = None) -> SearchResult:
        return SearchResult(
            goal_node=goal_node,
            nodes_expanded=self.nodes_expanded,
            max_depth_reached=self.max_depth_reached,
        )

    def _sequential_fallback(self, current_node: ConfigurationNode) -> SearchResult:
        """Move each unresolved qubit one at a time along its shortest path.

        Greedy: compute and execute one qubit's full path, then recompute
        for the next, accounting for new positions.
        """
        node = current_node

        unresolved = [
            qid
            for qid, loc in node.configuration.items()
            if qid in self.target and loc != self.target[qid]
        ]

        self._emit_step(
            "fallback_start",
            node,
            FallbackStartStepInfo(
                entropy=0,
                unresolved_count=self._unresolved_count(node),
                unresolved_qubits=unresolved,
            ),
        )

        for qid in unresolved:
            current_loc = node.configuration[qid]
            target_loc = self.target[qid]

            if current_loc == target_loc:
                continue

            occupied = frozenset(
                loc for q, loc in node.configuration.items() if q != qid
            )

            result = self.tree.path_finder.find_path(
                current_loc,
                target_loc,
                occupied=occupied,
                edge_weight=self.tree.path_finder.metrics.get_lane_duration_us,
            )

            if result is None:
                return self._make_result()

            lanes, _ = result

            for lane in lanes:
                moveset = frozenset({lane})
                child = self.tree.apply_move_set(node, moveset, strict=False)
                if child is None:
                    return self._make_result()
                self.nodes_expanded += 1
                self.max_depth_reached = max(self.max_depth_reached, child.depth)
                node = child
                _print_node_transition("fallback_step", node)

                self._emit_step(
                    "fallback_step",
                    node,
                    FallbackStepInfo(
                        entropy=0,
                        unresolved_count=self._unresolved_count(node),
                        qubit_id=qid,
                        moveset=moveset,
                    ),
                )

        if self.goal(node):
            return self._make_result(goal_node=node)

        return self._make_result()

    def run(self) -> SearchResult:
        """Execute the entropy-guided search and return the result."""
        if self.goal(self.tree.root):
            self._emit_step(
                "goal",
                self.tree.root,
                GoalStepInfo(entropy=0, unresolved_count=0, total_depth=0),
            )
            return SearchResult(
                goal_node=self.tree.root, nodes_expanded=0, max_depth_reached=0
            )

        current_node = self.tree.root
        _print_node_transition("start", current_node)

        while True:
            if (
                self.max_expansions is not None
                and self.nodes_expanded >= self.max_expansions
            ):
                break
            if self.max_depth is not None and current_node.depth >= self.max_depth:
                sn = self._get_or_create_search_node(current_node)
                sn.entropy = self.params.e_max
                sn.last_entropy_bump_reason = "entropy"
                sn.last_no_valid_moves_qubit = None
                sn.last_state_seen_node_id = None

            sn = self._get_or_create_search_node(current_node)

            if sn.entropy >= self.params.e_max:
                revert_reason = sn.last_entropy_bump_reason
                seen_node_id = sn.last_state_seen_node_id
                no_valid_qid = sn.last_no_valid_moves_qubit
                ancestor = current_node
                for _ in range(self.params.reversion_steps):
                    if ancestor.parent is None:
                        break
                    ancestor = ancestor.parent

                if ancestor.parent is None:
                    root_sn = self._get_or_create_search_node(ancestor)
                    if root_sn.entropy >= self.params.e_max:
                        return self._sequential_fallback(self.tree.root)

                ancestor_sn = self._get_or_create_search_node(ancestor)
                ancestor_sn.entropy += self.params.delta_e
                ancestor_sn.last_entropy_bump_reason = "entropy"
                ancestor_sn.last_no_valid_moves_qubit = None
                ancestor_sn.last_state_seen_node_id = None
                self._emit_step(
                    "revert",
                    ancestor,
                    RevertStepInfo(
                        entropy=ancestor_sn.entropy,
                        unresolved_count=self._unresolved_count(ancestor),
                        reversion_steps=self.params.reversion_steps,
                        ancestor_depth=ancestor.depth,
                        reason=revert_reason,
                        state_seen_node_id=seen_node_id,
                        no_valid_moves_qubit=no_valid_qid,
                    ),
                )
                current_node = ancestor
                _print_node_transition(
                    "revert",
                    current_node,
                    reason=revert_reason,
                    seen_node_id=seen_node_id,
                    no_valid_moves_qubit=no_valid_qid,
                )
                continue

            candidate = self._get_next_candidate(sn, current_node)

            if candidate is None:
                no_valid_qid = self._first_unresolved_qubit_without_valid_move(
                    current_node
                )
                sn.entropy += self.params.delta_e
                sn.last_entropy_bump_reason = "no-valid-moves"
                sn.last_no_valid_moves_qubit = no_valid_qid
                sn.last_state_seen_node_id = None
                self._emit_step(
                    "entropy_bump",
                    current_node,
                    EntropyBumpStepInfo(
                        entropy=sn.entropy,
                        unresolved_count=self._unresolved_count(current_node),
                        new_entropy=sn.entropy,
                        reason="no-valid-moves",
                        no_valid_moves_qubit=no_valid_qid,
                    ),
                )
                continue

            outcome = self.tree.try_move_set(current_node, candidate, strict=False)
            sn.candidates_tried += 1

            if outcome.status != ExpansionStatus.CREATED_CHILD:
                if outcome.status == ExpansionStatus.TRANSPOSITION_SEEN:
                    reason = "state-seen"
                    seen_node_id = (
                        id(outcome.existing_node)
                        if outcome.existing_node is not None
                        else None
                    )
                    no_valid_qid = None
                else:
                    reason = "no-valid-moves"
                    seen_node_id = None
                    no_valid_qid = self._first_unresolved_qubit_without_valid_move(
                        current_node
                    )

                sn.entropy += self.params.delta_e
                sn.last_entropy_bump_reason = reason
                sn.last_no_valid_moves_qubit = no_valid_qid
                sn.last_state_seen_node_id = seen_node_id
                self._emit_step(
                    "entropy_bump",
                    current_node,
                    EntropyBumpStepInfo(
                        entropy=sn.entropy,
                        unresolved_count=self._unresolved_count(current_node),
                        new_entropy=sn.entropy,
                        reason=reason,
                        state_seen_node_id=seen_node_id,
                        no_valid_moves_qubit=no_valid_qid,
                    ),
                )
                continue
            assert outcome.child is not None
            child = outcome.child

            self.nodes_expanded += 1
            self.max_depth_reached = max(self.max_depth_reached, child.depth)

            if self.on_step is not None:
                moveset_score = self.scorer.score_moveset(
                    candidate, current_node, self.target, self.tree
                )
                self._emit_step(
                    "descend",
                    child,
                    DescendStepInfo(
                        entropy=sn.entropy,
                        unresolved_count=self._unresolved_count(child),
                        moveset=candidate,
                        moveset_score=moveset_score,
                    ),
                )

            if self.goal(child):
                self._emit_step(
                    "goal",
                    child,
                    GoalStepInfo(
                        entropy=sn.entropy,
                        unresolved_count=0,
                        total_depth=child.depth,
                    ),
                )
                return self._make_result(goal_node=child)

            current_node = child
            _print_node_transition("descend", current_node)

        return self._make_result()


def entropy_guided_search(
    tree: ConfigurationTree,
    target: dict[int, LocationAddress],
    goal: GoalPredicate,
    params: SearchParams = SearchParams(),
    max_depth: int | None = None,
    max_expansions: int | None = None,
    on_step: Callable[[str, ConfigurationNode, StepInfo], None] | None = None,
) -> SearchResult:
    """Entropy-guided depth-first search with reversion and sequential fallback.

    Thin wrapper around ``EntropyGuidedSearch`` for backward compatibility.
    Also takes *target* directly (in addition to *goal*) because scoring needs
    the raw target mapping.
    """
    return EntropyGuidedSearch(
        tree=tree,
        target=target,
        goal=goal,
        params=params,
        max_depth=max_depth,
        max_expansions=max_expansions,
        on_step=on_step,
    ).run()
