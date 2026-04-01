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

    def force_entropy(self, value: int) -> None:
        """Set entropy to an absolute value and clear bump metadata."""
        self.entropy = value
        self.last_entropy_bump_reason = "entropy"
        self.last_no_valid_moves_qubit = None
        self.last_state_seen_node_id = None

    def bump_entropy(
        self,
        delta_e: int,
        reason: str = "entropy",
        no_valid_moves_qubit: int | None = None,
        state_seen_node_id: int | None = None,
    ) -> None:
        """Increment entropy and record the reason for the bump."""
        self.entropy += delta_e
        self.last_entropy_bump_reason = reason
        self.last_no_valid_moves_qubit = no_valid_moves_qubit
        self.last_state_seen_node_id = state_seen_node_id


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


class _DebugEmitter:
    """Emits debug/visualization events for entropy-guided search.

    Consolidates all ``on_step`` callback emissions and
    ``_print_node_transition`` calls so the main search class stays
    focused on algorithm logic.
    """

    def __init__(
        self,
        scorer: CandidateScorer,
        tree: ConfigurationTree,
        on_step: Callable[[str, ConfigurationNode, StepInfo], None] | None = None,
    ) -> None:
        self.scorer = scorer
        self.tree = tree
        self.on_step = on_step

    def _emit_step(self, event: str, node: ConfigurationNode, info: StepInfo) -> None:
        if self.on_step is not None:
            self.on_step(event, node, info)

    def _unresolved_count(self, node: ConfigurationNode) -> int:
        return sum(
            1
            for qid, loc in self.scorer.target.items()
            if node.configuration.get(qid) != loc
        )

    def goal(self, node: ConfigurationNode, entropy: int) -> None:
        self._emit_step(
            "goal",
            node,
            GoalStepInfo(entropy=entropy, unresolved_count=0, total_depth=node.depth),
        )

    def revert(
        self,
        ancestor: ConfigurationNode,
        ancestor_sn: _SearchNode,
        trigger_sn: _SearchNode,
    ) -> None:
        reason = trigger_sn.last_entropy_bump_reason
        seen_node_id = trigger_sn.last_state_seen_node_id
        no_valid_qid = trigger_sn.last_no_valid_moves_qubit
        self._emit_step(
            "revert",
            ancestor,
            RevertStepInfo(
                entropy=ancestor_sn.entropy,
                unresolved_count=self._unresolved_count(ancestor),
                reversion_steps=self.scorer.params.reversion_steps,
                ancestor_depth=ancestor.depth,
                reason=reason,
                state_seen_node_id=seen_node_id,
                no_valid_moves_qubit=no_valid_qid,
            ),
        )

    def entropy_bump(
        self,
        node: ConfigurationNode,
        sn: _SearchNode,
    ) -> None:
        self._emit_step(
            "entropy_bump",
            node,
            EntropyBumpStepInfo(
                entropy=sn.entropy,
                unresolved_count=self._unresolved_count(node),
                new_entropy=sn.entropy,
                reason=sn.last_entropy_bump_reason,
                state_seen_node_id=sn.last_state_seen_node_id,
                no_valid_moves_qubit=sn.last_no_valid_moves_qubit,
            ),
        )

    def descend(
        self,
        child: ConfigurationNode,
        sn: _SearchNode,
        candidate: frozenset[LaneAddress],
        parent: ConfigurationNode,
    ) -> None:
        if self.on_step is not None:
            moveset_score = self.scorer.score_moveset(candidate, parent, self.tree)
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

    def fallback_start(self, node: ConfigurationNode, unresolved: list[int]) -> None:
        self._emit_step(
            "fallback_start",
            node,
            FallbackStartStepInfo(
                entropy=0,
                unresolved_count=self._unresolved_count(node),
                unresolved_qubits=unresolved,
            ),
        )

    def fallback_step(
        self,
        node: ConfigurationNode,
        qid: int,
        moveset: frozenset[LaneAddress],
    ) -> None:
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

        self.scorer = CandidateScorer(params=params, target=target)
        self.search_nodes: dict[int, _SearchNode] = {}
        # _SearchNode satisfies EntropyNode protocol; cast for dict invariance.
        entropy_view: dict[int, EntropyNode] = self.search_nodes  # type: ignore[assignment]
        self.generator = HeuristicMoveGenerator(
            scorer=self.scorer,
            params=params,
            search_nodes=entropy_view,
        )
        self.debug = _DebugEmitter(scorer=self.scorer, tree=tree, on_step=on_step)
        self.nodes_expanded = 0
        self.max_depth_reached = 0

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
        occupied = node.occupied_locations | self.tree.blocked_locations
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

    @staticmethod
    def _goal_sort_key(
        node: ConfigurationNode,
    ) -> tuple[int, tuple[tuple[int, ...], ...]]:
        move_program = node.to_move_program()
        encoded_program = tuple(
            tuple(lane.encode() for lane in step) for step in move_program
        )
        return (node.depth, encoded_program)

    def _cutoff_ancestor(self, goal_node: ConfigurationNode) -> ConfigurationNode:
        """Return the first ancestor whose parent has multiple children.

        Walk upward from the goal branch and stop at the first node where its
        parent is an actual branch point (2+ children). If no branch point
        exists on the path, cut back to the root.
        """
        ancestor = goal_node
        while ancestor.parent is not None:
            if len(ancestor.parent.children) > 1:
                return ancestor
            ancestor = ancestor.parent
        return ancestor

    def _make_result(
        self,
        goal_nodes: tuple[ConfigurationNode, ...] = (),
    ) -> SearchResult:
        if goal_nodes:
            sorted_goal_nodes = tuple(sorted(goal_nodes, key=self._goal_sort_key))
        else:
            sorted_goal_nodes = ()
        return SearchResult(
            nodes_expanded=self.nodes_expanded,
            max_depth_reached=self.max_depth_reached,
            goal_nodes=sorted_goal_nodes,
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

        self.debug.fallback_start(node, unresolved)

        for qid in unresolved:
            current_loc = node.configuration[qid]
            target_loc = self.target[qid]

            if current_loc == target_loc:
                continue

            occupied = (
                frozenset(loc for q, loc in node.configuration.items() if q != qid)
                | self.tree.blocked_locations
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
                outcome = self.tree.try_move_set(node, moveset, strict=False)
                if (
                    outcome.status == ExpansionStatus.CREATED_CHILD
                    and outcome.child is not None
                ):
                    next_node = outcome.child
                elif (
                    outcome.status == ExpansionStatus.ALREADY_CHILD
                    and outcome.child is not None
                ):
                    next_node = outcome.child
                elif (
                    outcome.status == ExpansionStatus.TRANSPOSITION_SEEN
                    and outcome.existing_node is not None
                ):
                    next_node = outcome.existing_node
                else:
                    return self._make_result()
                self.nodes_expanded += 1
                self.max_depth_reached = max(self.max_depth_reached, next_node.depth)
                node = next_node
                self.debug.fallback_step(node, qid, moveset)

        if self.goal(node):
            return self._make_result(goal_nodes=(node,))

        return self._make_result()

    def run(self) -> SearchResult:
        """Execute the entropy-guided search and return the result."""
        found_goals: list[ConfigurationNode] = []
        if self.goal(self.tree.root):
            self.debug.goal(self.tree.root, entropy=0)
            return SearchResult(
                nodes_expanded=0,
                max_depth_reached=0,
                goal_nodes=(self.tree.root,),
            )

        current_node = self.tree.root

        while self.max_expansions is None or self.nodes_expanded < self.max_expansions:
            sn = self._get_or_create_search_node(current_node)
            if self.max_depth is not None and current_node.depth >= self.max_depth:
                sn.force_entropy(self.params.e_max)

            if sn.entropy >= self.params.e_max:
                ancestor = current_node
                for _ in range(self.params.reversion_steps):
                    if ancestor.parent is None:
                        break
                    ancestor = ancestor.parent

                ancestor_sn = self._get_or_create_search_node(ancestor)
                if ancestor.parent is None and ancestor_sn.entropy >= self.params.e_max:
                    fallback_result = self._sequential_fallback(self.tree.root)
                    combined_goals = found_goals + list(fallback_result.goal_nodes)
                    return self._make_result(goal_nodes=tuple(combined_goals))

                ancestor_sn.bump_entropy(self.params.delta_e)
                self.debug.revert(ancestor, ancestor_sn, sn)
                current_node = ancestor
                continue

            candidate = self._get_next_candidate(sn, current_node)

            if candidate is None:
                no_valid_qid = self._first_unresolved_qubit_without_valid_move(
                    current_node
                )
                sn.bump_entropy(
                    self.params.delta_e,
                    "no-valid-moves",
                    no_valid_moves_qubit=no_valid_qid,
                )
                self.debug.entropy_bump(current_node, sn)
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

                sn.bump_entropy(
                    self.params.delta_e,
                    reason,
                    no_valid_moves_qubit=no_valid_qid,
                    state_seen_node_id=seen_node_id,
                )
                self.debug.entropy_bump(current_node, sn)
                continue
            assert outcome.child is not None
            child = outcome.child

            self.nodes_expanded += 1
            self.max_depth_reached = max(self.max_depth_reached, child.depth)
            self.debug.descend(child, sn, candidate, current_node)

            if self.goal(child):
                self.debug.goal(child, entropy=sn.entropy)
                found_goals.append(child)
                if len(found_goals) >= self.params.max_goal_candidates:
                    return self._make_result(goal_nodes=tuple(found_goals))
                current_node = self._cutoff_ancestor(child)
                continue

            current_node = child

        return self._make_result(goal_nodes=tuple(found_goals))


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
