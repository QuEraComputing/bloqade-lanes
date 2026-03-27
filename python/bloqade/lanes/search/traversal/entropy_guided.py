"""Entropy-guided tree search traversal."""

from __future__ import annotations

__all__ = ["entropy_guided_search"]

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


def _unresolved_count(
    node: ConfigurationNode, target: dict[int, LocationAddress]
) -> int:
    return sum(1 for qid, loc in target.items() if node.configuration.get(qid) != loc)


def _get_or_create_search_node(
    config_node: ConfigurationNode,
    search_nodes: dict[int, _SearchNode],
) -> _SearchNode:
    node_id = id(config_node)
    sn = search_nodes.get(node_id)
    if sn is None:
        sn = _SearchNode(config_node=config_node)
        search_nodes[node_id] = sn
    return sn


def _get_next_candidate(
    sn: _SearchNode,
    node: ConfigurationNode,
    tree: ConfigurationTree,
    generator: HeuristicMoveGenerator,
    params: SearchParams,
) -> frozenset[LaneAddress] | None:
    """Get the next untried candidate, enforcing max_candidates (K) limit.

    After K candidates tried from current cache, forces regeneration.
    Returns None if all generated candidates are already node children
    (failed generation).
    """
    # If we've tried K candidates from the current cache, force regeneration
    if sn.candidates_tried >= params.max_candidates:
        new_candidates = list(generator.generate(node, tree))
        sn.candidate_cache = new_candidates
        sn.candidates_tried = 0

        while sn.candidates_tried < len(sn.candidate_cache):
            candidate = sn.candidate_cache[sn.candidates_tried]
            if candidate not in node.children:
                return candidate
            sn.candidates_tried += 1
        return None

    # Try cached candidates (within K limit)
    while sn.candidates_tried < len(sn.candidate_cache):
        candidate = sn.candidate_cache[sn.candidates_tried]
        if candidate not in node.children:
            return candidate
        sn.candidates_tried += 1

    # Cache empty or exhausted before hitting K -- initial generation
    new_candidates = list(generator.generate(node, tree))
    sn.candidate_cache = new_candidates
    sn.candidates_tried = 0

    while sn.candidates_tried < len(sn.candidate_cache):
        candidate = sn.candidate_cache[sn.candidates_tried]
        if candidate not in node.children:
            return candidate
        sn.candidates_tried += 1

    return None


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


def _first_unresolved_qubit_without_valid_move(
    node: ConfigurationNode,
    target: dict[int, LocationAddress],
    tree: ConfigurationTree,
) -> int | None:
    """Return one unresolved qubit that currently has no valid lane move."""
    occupied = node.occupied_locations
    for qid, target_loc in target.items():
        current_loc = node.configuration.get(qid)
        if current_loc is None or current_loc == target_loc:
            continue

        has_valid_lane = False
        for mt in (MoveType.SITE, MoveType.WORD):
            buses = (
                range(len(tree.arch_spec.site_buses))
                if mt == MoveType.SITE
                else range(len(tree.arch_spec.word_buses))
            )
            for bus_id in buses:
                for direction in (Direction.FORWARD, Direction.BACKWARD):
                    for lane in tree.lanes_for(mt, bus_id, direction):
                        src, dst = tree.arch_spec.get_endpoints(lane)
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


def _sequential_fallback(
    tree: ConfigurationTree,
    current_node: ConfigurationNode,
    target: dict[int, LocationAddress],
    goal: GoalPredicate,
    nodes_expanded: int,
    max_depth_reached: int,
    # NOTE: on_step callback is for debugging/visualization only.
    # Can be safely removed along with all on_step callsites in this function.
    on_step: Callable[[str, ConfigurationNode, StepInfo], None] | None = None,
) -> SearchResult:
    """Move each unresolved qubit one at a time along its shortest path.

    Greedy: compute and execute one qubit's full path, then recompute
    for the next, accounting for new positions.
    """
    node = current_node

    unresolved = [
        qid
        for qid, loc in node.configuration.items()
        if qid in target and loc != target[qid]
    ]

    if on_step is not None:
        on_step(
            "fallback_start",
            node,
            FallbackStartStepInfo(
                entropy=0,
                unresolved_count=_unresolved_count(node, target),
                unresolved_qubits=unresolved,
            ),
        )

    for qid in unresolved:
        current_loc = node.configuration[qid]
        target_loc = target[qid]

        if current_loc == target_loc:
            continue

        occupied = frozenset(loc for q, loc in node.configuration.items() if q != qid)

        result = tree.path_finder.find_path(
            current_loc,
            target_loc,
            occupied=occupied,
            edge_weight=tree.path_finder.metrics.get_lane_duration_us,
        )

        if result is None:
            return SearchResult(
                goal_node=None,
                nodes_expanded=nodes_expanded,
                max_depth_reached=max_depth_reached,
            )

        lanes, _ = result

        for lane in lanes:
            moveset = frozenset({lane})
            child = tree.apply_move_set(node, moveset, strict=False)
            if child is None:
                return SearchResult(
                    goal_node=None,
                    nodes_expanded=nodes_expanded,
                    max_depth_reached=max_depth_reached,
                )
            nodes_expanded += 1
            max_depth_reached = max(max_depth_reached, child.depth)
            node = child
            _print_node_transition("fallback_step", node)

            if on_step is not None:
                on_step(
                    "fallback_step",
                    node,
                    FallbackStepInfo(
                        entropy=0,
                        unresolved_count=_unresolved_count(node, target),
                        qubit_id=qid,
                        moveset=moveset,
                    ),
                )

    if goal(node):
        return SearchResult(
            goal_node=node,
            nodes_expanded=nodes_expanded,
            max_depth_reached=max_depth_reached,
        )

    return SearchResult(
        goal_node=None,
        nodes_expanded=nodes_expanded,
        max_depth_reached=max_depth_reached,
    )


def entropy_guided_search(
    tree: ConfigurationTree,
    target: dict[int, LocationAddress],
    goal: GoalPredicate,
    params: SearchParams = SearchParams(),
    max_depth: int | None = None,
    max_expansions: int | None = None,
    # NOTE: on_step callback is for debugging/visualization only.
    # Can be safely removed along with all on_step callsites in this function.
    on_step: Callable[[str, ConfigurationNode, StepInfo], None] | None = None,
) -> SearchResult:
    """Entropy-guided depth-first search with reversion and sequential fallback.

    Also takes target directly (in addition to goal) because scoring needs
    the raw target mapping.

    Does not use tree.expand_node() -- calls tree.apply_move_set() directly.
    """

    if goal(tree.root):
        if on_step is not None:
            on_step(
                "goal",
                tree.root,
                GoalStepInfo(
                    entropy=0,
                    unresolved_count=0,
                    total_depth=0,
                ),
            )
        return SearchResult(goal_node=tree.root, nodes_expanded=0, max_depth_reached=0)

    scorer = CandidateScorer(params=params)
    search_nodes: dict[int, _SearchNode] = {}
    # _SearchNode satisfies EntropyNode protocol; cast for dict invariance.
    entropy_view: dict[int, EntropyNode] = search_nodes  # type: ignore[assignment]
    generator = HeuristicMoveGenerator(
        scorer=scorer,
        params=params,
        target=target,
        search_nodes=entropy_view,
    )

    current_node = tree.root
    _print_node_transition("start", current_node)
    nodes_expanded = 0
    max_depth_reached = 0

    while True:
        # Limit checks
        if max_expansions is not None and nodes_expanded >= max_expansions:
            break
        if max_depth is not None and current_node.depth >= max_depth:
            sn = _get_or_create_search_node(current_node, search_nodes)
            sn.entropy = params.e_max  # Force reversion
            sn.last_entropy_bump_reason = "entropy"
            sn.last_no_valid_moves_qubit = None
            sn.last_state_seen_node_id = None

        sn = _get_or_create_search_node(current_node, search_nodes)

        # Entropy gate -- revert if entropy too high
        if sn.entropy >= params.e_max:
            revert_reason = sn.last_entropy_bump_reason
            seen_node_id = sn.last_state_seen_node_id
            no_valid_qid = sn.last_no_valid_moves_qubit
            ancestor = current_node
            for _ in range(params.reversion_steps):
                if ancestor.parent is None:
                    break
                ancestor = ancestor.parent

            if ancestor.parent is None:
                root_sn = _get_or_create_search_node(ancestor, search_nodes)
                if root_sn.entropy >= params.e_max:
                    return _sequential_fallback(
                        tree,
                        tree.root,
                        target,
                        goal,
                        nodes_expanded,
                        max_depth_reached,
                        on_step=on_step,
                    )

            ancestor_sn = _get_or_create_search_node(ancestor, search_nodes)
            ancestor_sn.entropy += params.delta_e
            ancestor_sn.last_entropy_bump_reason = "entropy"
            ancestor_sn.last_no_valid_moves_qubit = None
            ancestor_sn.last_state_seen_node_id = None
            if on_step is not None:
                on_step(
                    "revert",
                    ancestor,
                    RevertStepInfo(
                        entropy=ancestor_sn.entropy,
                        unresolved_count=_unresolved_count(ancestor, target),
                        reversion_steps=params.reversion_steps,
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

        # Get next candidate
        candidate = _get_next_candidate(sn, current_node, tree, generator, params)

        if candidate is None:
            # Failed generation -- bump entropy, retry
            no_valid_qid = _first_unresolved_qubit_without_valid_move(
                current_node, target, tree
            )
            sn.entropy += params.delta_e
            sn.last_entropy_bump_reason = "no-valid-moves"
            sn.last_no_valid_moves_qubit = no_valid_qid
            sn.last_state_seen_node_id = None
            if on_step is not None:
                on_step(
                    "entropy_bump",
                    current_node,
                    EntropyBumpStepInfo(
                        entropy=sn.entropy,
                        unresolved_count=_unresolved_count(current_node, target),
                        new_entropy=sn.entropy,
                        reason="no-valid-moves",
                        no_valid_moves_qubit=no_valid_qid,
                    ),
                )
            continue

        # Apply candidate with detailed outcome for reason-aware entropy bumps.
        outcome = tree.try_move_set(current_node, candidate, strict=False)
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
                no_valid_qid = _first_unresolved_qubit_without_valid_move(
                    current_node, target, tree
                )

            sn.entropy += params.delta_e
            sn.last_entropy_bump_reason = reason
            sn.last_no_valid_moves_qubit = no_valid_qid
            sn.last_state_seen_node_id = seen_node_id
            if on_step is not None:
                on_step(
                    "entropy_bump",
                    current_node,
                    EntropyBumpStepInfo(
                        entropy=sn.entropy,
                        unresolved_count=_unresolved_count(current_node, target),
                        new_entropy=sn.entropy,
                        reason=reason,
                        state_seen_node_id=seen_node_id,
                        no_valid_moves_qubit=no_valid_qid,
                    ),
                )
            continue
        assert outcome.child is not None
        child = outcome.child

        nodes_expanded += 1
        max_depth_reached = max(max_depth_reached, child.depth)

        if on_step is not None:
            moveset_score = scorer.score_moveset(candidate, current_node, target, tree)
            on_step(
                "descend",
                child,
                DescendStepInfo(
                    entropy=sn.entropy,
                    unresolved_count=_unresolved_count(child, target),
                    moveset=candidate,
                    moveset_score=moveset_score,
                ),
            )

        if goal(child):
            if on_step is not None:
                on_step(
                    "goal",
                    child,
                    GoalStepInfo(
                        entropy=sn.entropy,
                        unresolved_count=0,
                        total_depth=child.depth,
                    ),
                )
            return SearchResult(
                goal_node=child,
                nodes_expanded=nodes_expanded,
                max_depth_reached=max_depth_reached,
            )

        current_node = child
        _print_node_transition("descend", current_node)

    return SearchResult(
        goal_node=None,
        nodes_expanded=nodes_expanded,
        max_depth_reached=max_depth_reached,
    )
