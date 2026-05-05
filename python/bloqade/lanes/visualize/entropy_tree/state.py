"""Frame-by-frame state reducer for entropy tree visualization."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress
from bloqade.lanes.visualize.entropy_tree.tracer import TreeTraceStep

# Sentinel step_index for synthetic reducer-injected entries (e.g. the
# root candidate_expand seed that has no corresponding native trace step).
_SYNTHETIC_STEP_INDEX = -1


def candidate_color(generation: int, index: int) -> str:
    # Generate an unbounded sequence of distinct colors by combining
    # generation-level hue shifts with per-candidate spacing.
    golden_ratio_conj = 0.6180339887498948
    generation_shift = 0.7548776662466927
    hue = (generation * generation_shift + index * golden_ratio_conj) % 1.0
    sat_cycle = (0.72, 0.82, 0.68, 0.88)
    val_cycle = (0.90, 0.82, 0.95, 0.86)
    saturation = sat_cycle[index % len(sat_cycle)]
    value = val_cycle[(index // len(sat_cycle)) % len(val_cycle)]
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


@dataclass
class CandidateState:
    moveset: frozenset[LaneAddress]
    color: str
    attempted: bool = False
    selected: bool = False


@dataclass
class NodeState:
    node_id: int
    display_id: int
    parent_id: int | None
    depth: int
    entropy: int = 0
    show_entropy: bool = True
    fill_color: str = "white"
    outline_color: str = "black"
    outline_width: float = 1.0
    active_candidate_index: int | None = None
    candidates: list[CandidateState] = field(default_factory=list)
    candidate_generation: int = 0
    terminal_reason: str | None = None
    terminal_parent_entropy: int | None = None
    last_candidate_movesets: tuple[frozenset[LaneAddress], ...] = field(
        default_factory=tuple
    )
    configuration: dict[int, LocationAddress] = field(default_factory=dict)
    move_score: float | None = None
    is_goal: bool = False


@dataclass
class EdgeState:
    parent_id: int
    child_id: int
    color: str = "lightgray"
    width: float = 1.0
    dashed: bool = False


@dataclass
class SeenLinkState:
    source_id: int
    target_id: int
    color: str = "#F9A825"


@dataclass
class TreeFrameState:
    step_index: int
    event: str
    current_node_id: int
    current_node_display_id: int
    nodes: dict[int, NodeState]
    edges: list[EdgeState]
    event_reason: str | None = None
    event_state_seen_node_id: int | None = None
    event_state_seen_display_id: int | None = None
    event_no_valid_moves_qubit: int | None = None
    hardware_configuration: dict[int, LocationAddress] = field(default_factory=dict)
    hardware_parent_configuration: dict[int, LocationAddress] | None = None
    hardware_move_source_configuration: dict[int, LocationAddress] | None = None
    hardware_moveset: frozenset[LaneAddress] | None = None
    seen_link: SeenLinkState | None = None
    best_buffer_node_display_ids: tuple[int | None, ...] = ()
    best_goal_depth: int | None = None


class TreeStateReducer:
    """Converts trace events into renderable tree frames."""

    def __init__(
        self,
        steps: tuple[TreeTraceStep, ...] | list[TreeTraceStep],
        root_node_id: int,
        best_buffer_size: int = 2,
    ):
        self.root_node_id = root_node_id
        self.actions = self._expand_actions(list(steps))
        self._display_ids: dict[int, int] = {root_node_id: 0}
        self._next_display_id = 1
        self._root_configuration = self._infer_root_configuration(list(steps))
        self.best_buffer_size = max(0, best_buffer_size)

    @staticmethod
    def _infer_root_configuration(
        steps: list[TreeTraceStep],
    ) -> dict[int, LocationAddress]:
        for step in steps:
            if step.parent_configuration is not None:
                return dict(step.parent_configuration)
            if step.depth == 0:
                return dict(step.configuration)
        return {}

    def _get_or_assign_display_id(self, node_id: int) -> int:
        display_id = self._display_ids.get(node_id)
        if display_id is None:
            display_id = self._next_display_id
            self._next_display_id += 1
            self._display_ids[node_id] = display_id
        return display_id

    @property
    def frame_count(self) -> int:
        return len(self.actions) + 1

    def _expand_actions(self, steps: list[TreeTraceStep]) -> list[TreeTraceStep]:
        """Expand the native trace with synthetic root-candidate-seed entries.

        Inserts a ``candidate_expand`` TreeTraceStep with
        ``step_index=_SYNTHETIC_STEP_INDEX`` before the first root-child
        descend, so the root node's candidate boxes are visible from frame 0.
        Real steps preserve their original ``step_index``.
        """
        actions: list[TreeTraceStep] = []
        root_candidate_seeded = False
        for step in steps:
            if (
                step.event == "descend"
                and step.parent_node_id == self.root_node_id
                and step.candidate_movesets
                and not root_candidate_seeded
            ):
                actions.append(
                    TreeTraceStep(
                        step_index=_SYNTHETIC_STEP_INDEX,
                        event="candidate_expand",
                        node_id=self.root_node_id,
                        parent_node_id=None,
                        depth=0,
                        entropy=0,
                        unresolved_count=1,
                        moveset=None,
                        candidate_movesets=step.candidate_movesets,
                        candidate_index=None,
                        reason=None,
                        state_seen_node_id=None,
                        no_valid_moves_qubit=None,
                        trigger_node_id=None,
                        configuration=(
                            {}
                            if step.parent_configuration is None
                            else dict(step.parent_configuration)
                        ),
                        parent_configuration=None,
                        moveset_score=None,
                        best_buffer_node_ids=step.best_buffer_node_ids,
                    )
                )
                root_candidate_seeded = True
            actions.append(step)
        return actions

    @staticmethod
    def _register_candidates(
        node: NodeState, movesets: tuple[frozenset[LaneAddress], ...]
    ) -> None:
        if not movesets:
            return
        generation_changed = movesets != node.last_candidate_movesets
        if generation_changed:
            node.candidate_generation += 1
            node.last_candidate_movesets = movesets

        existing = {cand.moveset: cand for cand in node.candidates}
        merged: list[CandidateState] = []
        for idx, moveset in enumerate(movesets):
            previous = existing.get(moveset)
            if previous is None:
                merged.append(
                    CandidateState(
                        moveset=moveset,
                        color=candidate_color(node.candidate_generation, idx),
                    )
                )
                continue
            color = previous.color
            if generation_changed:
                color = candidate_color(node.candidate_generation, idx)
            merged.append(
                CandidateState(
                    moveset=moveset,
                    color=color,
                    attempted=previous.attempted,
                    selected=previous.selected,
                )
            )
        node.candidates = merged

    @staticmethod
    def _failure_color(reason: str | None) -> str:
        if reason == "state-seen":
            return "#EF5350"
        if reason == "no-valid-moves":
            return "#FFD54F"
        return "#BDBDBD"

    def _infer_future_candidate_movesets(
        self,
        *,
        node_id: int,
        start_idx: int,
    ) -> tuple[frozenset[LaneAddress], ...]:
        """Find the next candidate snapshot relevant to `node_id`.

        Candidate snapshots can appear either:
        - on events emitted at the node itself (e.g. entropy_bump), or
        - on a descend event to one of its children (snapshot captured on parent).
        """
        for future in self.actions[start_idx:]:
            if future.node_id == node_id and future.candidate_movesets:
                return future.candidate_movesets
            if (
                future.event == "descend"
                and future.parent_node_id == node_id
                and future.candidate_movesets
            ):
                return future.candidate_movesets
        return ()

    def frame_at(self, frame_index: int) -> TreeFrameState:
        nodes: dict[int, NodeState] = {
            self.root_node_id: NodeState(
                node_id=self.root_node_id,
                display_id=self._get_or_assign_display_id(self.root_node_id),
                parent_id=None,
                depth=0,
                configuration=dict(self._root_configuration),
            )
        }
        edge_map: dict[tuple[int, int], EdgeState] = {}
        current_node_id = self.root_node_id
        current_event = "initial"
        hardware_configuration = dict(nodes[self.root_node_id].configuration)
        hardware_parent_configuration: dict[int, LocationAddress] | None = None
        hardware_move_source_configuration: dict[int, LocationAddress] | None = None
        hardware_moveset: frozenset[LaneAddress] | None = None
        best_scores_by_node: dict[int, float] = {}
        best_goal_depth: int | None = None
        traced_best_buffer_ids: tuple[int, ...] | None = None

        if frame_index <= 0:
            return TreeFrameState(
                step_index=0,
                event="initial",
                current_node_id=self.root_node_id,
                current_node_display_id=self._get_or_assign_display_id(
                    self.root_node_id
                ),
                nodes=nodes,
                edges=[],
                event_reason=None,
                event_state_seen_node_id=None,
                event_state_seen_display_id=None,
                event_no_valid_moves_qubit=None,
                hardware_configuration=hardware_configuration,
                hardware_parent_configuration=hardware_parent_configuration,
                hardware_move_source_configuration=hardware_move_source_configuration,
                hardware_moveset=hardware_moveset,
                seen_link=None,
                best_buffer_node_display_ids=tuple(),
            )

        last_reason: str | None = None
        last_state_seen_node_id: int | None = None
        last_no_valid_moves_qubit: int | None = None
        seen_link: SeenLinkState | None = None
        seen_target_highlight_node_id: int | None = None
        active_candidate_node_id: int | None = None
        active_candidate_index: int | None = None
        for action_idx, action in enumerate(self.actions, start=1):
            if action_idx > frame_index:
                break

            current_node_id = action.node_id
            current_event = action.event
            last_reason = action.reason
            last_state_seen_node_id = action.state_seen_node_id
            last_no_valid_moves_qubit = action.no_valid_moves_qubit
            synthetic_goal_event = False
            if action.best_buffer_node_ids is not None:
                traced_best_buffer_ids = action.best_buffer_node_ids

            node = nodes.get(action.node_id)
            if node is None:
                node = NodeState(
                    node_id=action.node_id,
                    display_id=self._get_or_assign_display_id(action.node_id),
                    parent_id=action.parent_node_id,
                    depth=action.depth,
                )
                nodes[action.node_id] = node
            # Only descend events should alter tree topology. Other events
            # (goal/revert/entropy_bump) may reference a node but must not
            # re-parent it, or the layout will jump/overlap across frames.
            if action.event == "descend" and action.parent_node_id is not None:
                node.parent_id = action.parent_node_id
            node.depth = action.depth
            if action.event != "candidate_expand":
                node.entropy = action.entropy
            node.configuration = dict(action.configuration)
            if action.moveset_score is not None:
                node.move_score = action.moveset_score
            if action.event == "goal":
                if best_goal_depth is None:
                    best_goal_depth = action.depth
                else:
                    best_goal_depth = min(best_goal_depth, action.depth)
                if action.reason == "state-seen-goal":
                    synthetic_goal_event = True
                    synthetic_goal_id = -(20_000_000 + action_idx)
                    synthetic_parent_id = (
                        action.trigger_node_id
                        if action.trigger_node_id is not None
                        else action.parent_node_id
                    )
                    if (
                        synthetic_parent_id is not None
                        and synthetic_parent_id not in nodes
                    ):
                        parent_cfg = (
                            {}
                            if action.parent_configuration is None
                            else dict(action.parent_configuration)
                        )
                        nodes[synthetic_parent_id] = NodeState(
                            node_id=synthetic_parent_id,
                            display_id=self._get_or_assign_display_id(
                                synthetic_parent_id
                            ),
                            parent_id=None,
                            depth=max(0, action.depth - 1),
                            configuration=parent_cfg,
                        )
                    nodes[synthetic_goal_id] = NodeState(
                        node_id=synthetic_goal_id,
                        display_id=self._get_or_assign_display_id(synthetic_goal_id),
                        parent_id=synthetic_parent_id,
                        depth=action.depth,
                        entropy=action.entropy,
                        fill_color="#C8E6C9",
                        outline_color="#2E7D32",
                        outline_width=1.8,
                        configuration=dict(action.configuration),
                        is_goal=True,
                    )
                    if synthetic_parent_id is not None:
                        edge_map[(synthetic_parent_id, synthetic_goal_id)] = EdgeState(
                            parent_id=synthetic_parent_id,
                            child_id=synthetic_goal_id,
                            color="#66BB6A",
                            width=1.8,
                            dashed=False,
                        )
                    current_node_id = synthetic_goal_id
                    hardware_configuration = dict(action.configuration)
                    hardware_parent_configuration = (
                        None
                        if action.parent_configuration is None
                        else dict(action.parent_configuration)
                    )
                    if synthetic_parent_id is not None and synthetic_parent_id in nodes:
                        hardware_move_source_configuration = dict(
                            nodes[synthetic_parent_id].configuration
                        )
                    else:
                        hardware_move_source_configuration = dict(action.configuration)
                    hardware_moveset = action.moveset
            if action.event == "goal" or (
                action.event != "candidate_expand" and action.unresolved_count == 0
            ):
                node.is_goal = True
                node.fill_color = "#C8E6C9"

            seen_link = None
            seen_target_highlight_node_id = None
            active_candidate_node_id = None
            active_candidate_index = None
            if not synthetic_goal_event:
                hardware_moveset = None
                hardware_configuration = dict(node.configuration)
                hardware_move_source_configuration = dict(node.configuration)
                hardware_parent_configuration = (
                    None
                    if action.parent_configuration is None
                    else dict(action.parent_configuration)
                )
                if action.moveset is not None:
                    hardware_moveset = action.moveset

            if action.event == "descend":
                parent_id = (
                    action.parent_node_id
                    if action.parent_node_id is not None
                    else action.node_id
                )
                parent = nodes.get(parent_id)
                if parent is None:
                    parent = NodeState(
                        node_id=parent_id,
                        display_id=self._get_or_assign_display_id(parent_id),
                        parent_id=None,
                        depth=max(0, action.depth - 1),
                    )
                    nodes[parent_id] = parent
                self._register_candidates(parent, action.candidate_movesets)
                edge_color = "lightgray"
                if (
                    action.candidate_index is not None
                    and 0 <= action.candidate_index < len(parent.candidates)
                ):
                    candidate = parent.candidates[action.candidate_index]
                    candidate.attempted = True
                    candidate.selected = True
                    edge_color = candidate.color
                    hardware_moveset = candidate.moveset
                    hardware_move_source_configuration = dict(parent.configuration)
                edge_map[(parent_id, action.node_id)] = EdgeState(
                    parent_id=parent_id,
                    child_id=action.node_id,
                    color=edge_color,
                    width=1.8,
                    dashed=False,
                )
                if action.moveset_score is not None:
                    best_scores_by_node[action.node_id] = action.moveset_score

                # Pre-populate descended node candidates from the nearest future
                # snapshot relevant to that node, avoiding empty candidate boxes.
                future_movesets = self._infer_future_candidate_movesets(
                    node_id=action.node_id,
                    start_idx=action_idx,
                )
                if future_movesets:
                    self._register_candidates(node, future_movesets)
            elif action.event == "candidate_expand":
                self._register_candidates(node, action.candidate_movesets)

            elif action.event == "entropy_bump":
                self._register_candidates(node, action.candidate_movesets)
                if (
                    action.candidate_index is not None
                    and 0 <= action.candidate_index < len(node.candidates)
                ):
                    candidate = node.candidates[action.candidate_index]
                    candidate.attempted = True
                    hardware_moveset = candidate.moveset
                    fail_id = -(10_000_000 + action_idx)
                    fail_configuration = dict(node.configuration)
                    if (
                        action.reason == "state-seen"
                        and action.state_seen_node_id is not None
                        and action.state_seen_node_id in nodes
                    ):
                        fail_configuration = dict(
                            nodes[action.state_seen_node_id].configuration
                        )
                        seen_target_highlight_node_id = action.state_seen_node_id

                    failed_candidate = CandidateState(
                        moveset=candidate.moveset,
                        color=candidate.color,
                        attempted=True,
                        selected=True,
                    )
                    nodes[fail_id] = NodeState(
                        node_id=fail_id,
                        display_id=self._get_or_assign_display_id(fail_id),
                        parent_id=action.node_id,
                        depth=node.depth + 1,
                        entropy=node.entropy,
                        show_entropy=False,
                        fill_color=self._failure_color(action.reason),
                        terminal_reason=action.reason,
                        terminal_parent_entropy=node.entropy,
                        candidates=[failed_candidate],
                        last_candidate_movesets=(candidate.moveset,),
                        configuration=fail_configuration,
                    )
                    edge_map[(action.node_id, fail_id)] = EdgeState(
                        parent_id=action.node_id,
                        child_id=fail_id,
                        color=candidate.color,
                        width=1.2,
                        dashed=True,
                    )

                    # Focus the synthetic failed child so active-view candidate
                    # rendering is anchored to the failed branch node itself.
                    current_node_id = fail_id
                    active_candidate_node_id = fail_id
                    active_candidate_index = 0
                    hardware_configuration = dict(fail_configuration)
                    hardware_move_source_configuration = dict(node.configuration)

            elif action.event == "revert" and action.trigger_node_id is not None:
                trigger = nodes.get(action.trigger_node_id)
                if trigger is not None:
                    trigger.fill_color = "#BDBDBD"

        for node in nodes.values():
            node.active_candidate_index = None
            if (
                active_candidate_node_id is not None
                and node.node_id == active_candidate_node_id
            ):
                node.active_candidate_index = active_candidate_index
            if (
                seen_target_highlight_node_id is not None
                and node.node_id == seen_target_highlight_node_id
            ):
                node.outline_color = "#D32F2F"
                node.outline_width = 2.4
            elif node.node_id == current_node_id:
                node.outline_color = "#FFD54F"
                node.outline_width = 2.2
            elif node.is_goal:
                node.outline_color = "#2E7D32"
                node.outline_width = 1.8
            else:
                node.outline_color = "black"
                node.outline_width = 1.0

        if traced_best_buffer_ids is not None:
            best_display_ids: list[int | None] = [
                self._get_or_assign_display_id(node_id)
                for node_id in traced_best_buffer_ids[: self.best_buffer_size]
            ]
            if len(best_display_ids) < self.best_buffer_size:
                best_display_ids.extend(
                    [None] * (self.best_buffer_size - len(best_display_ids))
                )
        else:
            ranked_best = sorted(
                best_scores_by_node.items(),
                key=lambda item: (-item[1], self._get_or_assign_display_id(item[0])),
            )
            best_display_ids = [
                self._get_or_assign_display_id(node_id)
                for node_id, _score in ranked_best[: self.best_buffer_size]
            ]
            if len(best_display_ids) < self.best_buffer_size:
                best_display_ids.extend(
                    [None] * (self.best_buffer_size - len(best_display_ids))
                )

        return TreeFrameState(
            step_index=frame_index,
            event=current_event,
            current_node_id=current_node_id,
            current_node_display_id=self._get_or_assign_display_id(current_node_id),
            nodes=nodes,
            edges=list(edge_map.values()),
            event_reason=last_reason,
            event_state_seen_node_id=last_state_seen_node_id,
            event_state_seen_display_id=(
                None
                if last_state_seen_node_id is None
                else self._get_or_assign_display_id(last_state_seen_node_id)
            ),
            event_no_valid_moves_qubit=last_no_valid_moves_qubit,
            hardware_configuration=hardware_configuration,
            hardware_parent_configuration=hardware_parent_configuration,
            hardware_move_source_configuration=hardware_move_source_configuration,
            hardware_moveset=hardware_moveset,
            seen_link=seen_link,
            best_buffer_node_display_ids=tuple(best_display_ids),
            best_goal_depth=best_goal_depth,
        )
