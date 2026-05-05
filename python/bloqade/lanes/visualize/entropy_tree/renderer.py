"""Matplotlib renderer for reduced entropy-tree frame state."""

from __future__ import annotations

import textwrap
from collections import defaultdict

from matplotlib.patches import FancyBboxPatch, Rectangle  # type: ignore[import-untyped]

from bloqade.lanes.bytecode._native import (
    EntropyScorer,
    LocationAddress as NativeLocationAddress,
)
from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress
from bloqade.lanes.visualize.arch import ArchVisualizer
from bloqade.lanes.visualize.entropy_tree.state import NodeState, TreeFrameState

_DEPTH_Y_STEP = 0.34
_TREE_Y_PAD = 0.16
_X_STEP = 0.32
_TREE_X_PAD = 0.08
_NODE_W_MAX = 0.40
_NODE_W_MIN = 0.30
_NODE_TEXT_BASE_PAD = 0.05
_NODE_TEXT_CHAR_UNIT = 0.015
_EDGE_WIDTH_MIN_SCALE = 0.45


def format_entropy_reason(
    frame: TreeFrameState,
    qid_label_map: dict[int, int] | None = None,
) -> str | None:
    """Human-readable description of why an entropy bump fired at this frame.

    Returns ``None`` when the frame does not describe an entropy-bump event.
    """
    reason = frame.event_reason
    if reason is None:
        return None
    if reason == "state-seen":
        seen_id = frame.event_state_seen_display_id
        if seen_id is None:
            return "entropy bump reason: encountered previously seen configuration"
        return f"entropy bump reason: seen configuration at display node {seen_id}"
    if reason == "no-valid-moves":
        qid = frame.event_no_valid_moves_qubit
        if qid is None:
            return "entropy bump reason: no valid moves available"
        label = qid_label_map.get(qid, qid) if qid_label_map else qid
        return f"entropy bump reason: no valid moves for qubit {label}"
    if reason == "entropy":
        return "entropy bump reason: reached entropy threshold/reversion condition"
    return f"entropy bump reason: {reason}"


def _moveset_to_tuples(
    moveset: frozenset[LaneAddress],
) -> list[tuple[int, int, int, int, int, int]]:
    """Convert a Python ``LaneAddress`` moveset into the `(dir, mt, zone, word, site, bus)`
    tuple format the Rust scorer consumes."""
    return [
        (
            int(lane.direction),
            int(lane.move_type),
            lane.zone_id,
            lane.word_id,
            lane.site_id,
            lane.bus_id,
        )
        for lane in moveset
    ]


def _configuration_to_native(
    config: dict[int, LocationAddress],
) -> dict[int, NativeLocationAddress]:
    """Unwrap Python ``LocationAddress`` values into their ``_inner`` native handles."""
    return {qid: loc._inner for qid, loc in config.items()}


def _path_bounds(arch_spec) -> tuple[float, float, float, float]:  # type: ignore[no-untyped-def]
    return ArchVisualizer(arch_spec).path_bounds()


def _stable_focus_bounds(
    *,
    arch_spec,  # type: ignore[no-untyped-def]
    root_configuration,  # type: ignore[no-untyped-def]
    target,  # type: ignore[no-untyped-def]
) -> tuple[float, float, float, float]:
    """Compute a fixed viewport for candidate hardware snapshots."""
    arch_x_min, arch_x_max, arch_y_min, arch_y_max = _path_bounds(arch_spec)
    arch_x_width = arch_x_max - arch_x_min
    arch_y_width = arch_y_max - arch_y_min
    full_bounds = (
        arch_x_min - 0.06 * arch_x_width,
        arch_x_max + 0.06 * arch_x_width,
        arch_y_min - 0.08 * arch_y_width,
        arch_y_max + 0.08 * arch_y_width,
    )

    moving_qids = [
        qid
        for qid, root_loc in root_configuration.items()
        if target.get(qid) is not None and target[qid] != root_loc
    ]
    positions: list[tuple[float, float]] = []
    for qid in moving_qids:
        root_loc = root_configuration.get(qid)
        target_loc = target.get(qid)
        if root_loc is not None:
            positions.append(arch_spec.get_position(root_loc))
        if target_loc is not None:
            positions.append(arch_spec.get_position(target_loc))

    if not positions:
        return full_bounds

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    x_min_local = min(xs)
    x_max_local = max(xs)
    y_min_local = min(ys)
    y_max_local = max(ys)
    x_width = max(1e-6, x_max_local - x_min_local)
    y_width = max(1e-6, y_max_local - y_min_local)
    local_bounds = (
        x_min_local - 0.12 * x_width,
        x_max_local + 0.12 * x_width,
        y_min_local - 0.22 * y_width,
        y_max_local + 0.22 * y_width,
    )

    # Never crop the architecture extent in focus panels.
    return (
        min(local_bounds[0], full_bounds[0]),
        max(local_bounds[1], full_bounds[1]),
        min(local_bounds[2], full_bounds[2]),
        max(local_bounds[3], full_bounds[3]),
    )


def _format_qid_list_lines(prefix: str, qids: list[int], width: int = 18) -> list[str]:
    items = ", ".join(str(q) for q in qids) if qids else "-"
    wrapped = textwrap.wrap(items, width=max(4, width))
    if not wrapped:
        return [f"{prefix}: -"]
    lines = [f"{prefix}: {wrapped[0]}"]
    lines.extend([f"  {cont}" for cont in wrapped[1:]])
    return lines


def _layout(nodes: dict[int, NodeState]) -> dict[int, tuple[float, float]]:
    if not nodes:
        return {}

    children: dict[int, list[int]] = defaultdict(list)
    roots: list[int] = []
    for node_id, node in nodes.items():
        parent_id = getattr(node, "parent_id", None)
        if parent_id is None or parent_id not in nodes:
            roots.append(node_id)
        else:
            children[parent_id].append(node_id)

    for child_ids in children.values():
        child_ids.sort(key=lambda nid: (getattr(nodes[nid], "depth"), nid))
    roots.sort(key=lambda nid: (getattr(nodes[nid], "depth"), nid))

    x_raw: dict[int, float] = {}
    next_leaf_x = 0.0
    leaf_step = 1.0
    root_gap = 2.0

    def assign_subtree_x_iterative(node_id: int) -> None:
        nonlocal next_leaf_x
        stack = [(node_id, False)]
        while stack:
            current_id, visited = stack.pop()
            child_ids = [cid for cid in children.get(current_id, []) if cid in nodes]
            if visited:
                if not child_ids:
                    x_raw[current_id] = next_leaf_x
                    next_leaf_x += leaf_step
                    continue
                x_raw[current_id] = 0.5 * (x_raw[child_ids[0]] + x_raw[child_ids[-1]])
                continue

            stack.append((current_id, True))
            for child_id in reversed(child_ids):
                stack.append((child_id, False))

    for idx, root_id in enumerate(roots):
        if idx > 0:
            next_leaf_x += root_gap
        assign_subtree_x_iterative(root_id)

    positions: dict[int, tuple[float, float]] = {}
    for node_id, node in nodes.items():
        x = x_raw[node_id] * _X_STEP
        y = -float(getattr(node, "depth")) * _DEPTH_Y_STEP
        positions[node_id] = (x, y)
    return positions


def _min_horizontal_gap_per_depth(
    positions: dict[int, tuple[float, float]],
    nodes: dict[int, NodeState],
) -> float | None:
    by_depth: dict[int, list[float]] = defaultdict(list)
    for node_id, node in nodes.items():
        if node_id not in positions:
            continue
        depth = int(getattr(node, "depth"))
        by_depth[depth].append(positions[node_id][0])

    min_gap: float | None = None
    for xs in by_depth.values():
        if len(xs) < 2:
            continue
        xs.sort()
        for left, right in zip(xs, xs[1:]):
            gap = right - left
            if gap <= 0:
                continue
            if min_gap is None or gap < min_gap:
                min_gap = gap
    return min_gap


def _target_node_width(nodes: dict[int, NodeState]) -> float:
    max_chars = 0
    for node in nodes.values():
        left = f"E={getattr(node, 'entropy', 0)}"
        right = str(getattr(node, "display_id", getattr(node, "node_id", 0)))
        max_chars = max(max_chars, len(left) + 1 + len(right))
    if max_chars == 0:
        max_chars = 6
    width = _NODE_TEXT_BASE_PAD + _NODE_TEXT_CHAR_UNIT * max_chars
    return min(_NODE_W_MAX, max(_NODE_W_MIN, width))


def draw_tree_frame(ax, frame: TreeFrameState) -> None:  # type: ignore[no-untyped-def]
    ax.clear()
    positions = _layout(frame.nodes)
    best_display_ids = {
        display_id
        for display_id in frame.best_buffer_node_display_ids
        if display_id is not None
    }
    node_w = _target_node_width(frame.nodes)
    node_h = 0.15
    edge_width_scale = 1.0
    min_gap = _min_horizontal_gap_per_depth(positions, frame.nodes)
    if min_gap is not None:
        # Shrink nodes as density increases, but never below readable minimum.
        allowed_w = 0.90 * min_gap
        if allowed_w < node_w:
            if allowed_w >= _NODE_W_MIN:
                node_w = allowed_w
            else:
                node_w = _NODE_W_MIN
                # Once width hits floor, reduce edge thickness gradually.
                pressure = min(1.0, max(0.0, (_NODE_W_MIN - allowed_w) / _NODE_W_MIN))
                edge_width_scale = max(
                    _EDGE_WIDTH_MIN_SCALE,
                    1.0 - (1.0 - _EDGE_WIDTH_MIN_SCALE) * pressure,
                )

    for edge in frame.edges:
        if edge.parent_id not in positions or edge.child_id not in positions:
            continue
        x1, y1 = positions[edge.parent_id]
        x2, y2 = positions[edge.child_id]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=edge.color,
            linewidth=edge.width * edge_width_scale,
            linestyle="--" if edge.dashed else "-",
            zorder=1,
        )

    for node_id, node in frame.nodes.items():
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        node_patch = FancyBboxPatch(
            (x - node_w / 2.0, y - node_h / 2.0),
            node_w,
            node_h,
            boxstyle="round,pad=0.004,rounding_size=0.02",
            facecolor=node.fill_color,
            edgecolor=node.outline_color,
            linewidth=node.outline_width,
            zorder=3,
        )
        ax.add_patch(node_patch)
        ax.text(
            x,
            y,
            str(getattr(node, "display_id", node.node_id)),
            ha="center",
            va="center",
            fontsize=8.3,
            family="monospace",
            zorder=5,
        )
        if node.display_id in best_display_ids:
            best_outline = FancyBboxPatch(
                (x - node_w / 2.0 - 0.008, y - node_h / 2.0 - 0.008),
                node_w + 0.016,
                node_h + 0.016,
                boxstyle="round,pad=0.004,rounding_size=0.02",
                facecolor="none",
                edgecolor="#1E88E5",
                linewidth=2.0,
                zorder=6,
            )
            ax.add_patch(best_outline)

    xs = [xy[0] for xy in positions.values()] or [0.5]
    ys = [xy[1] for xy in positions.values()] or [0.0]
    # Keep horizontal limits wide enough for full node boxes; otherwise, with
    # very few nodes (e.g., a single root) side borders can be clipped.
    x_pad = max(_TREE_X_PAD, node_w / 2.0 + 0.03)
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - _TREE_Y_PAD, max(ys) + _TREE_Y_PAD)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Entropy-Guided Search Tree (minimized nodes)")
    ax.axis("off")


# ── Hardware snapshot: split from a 145-line monolith into 5 focused helpers. ──


def _draw_architecture_background(ax, arch_spec) -> None:  # type: ignore[no-untyped-def]
    """Plot the architecture's site lattice as the hardware panel background."""
    word_ids = list(range(len(arch_spec.words)))
    ArchVisualizer(arch_spec).plot(
        ax=ax,
        show_words=word_ids,
        facecolors="none",
        edgecolors="#8e919c",
        linewidths=0.8,
    )


def _draw_blocked_locations(  # type: ignore[no-untyped-def]
    ax,
    arch_spec,
    blocked_locations,
    *,
    blocked_location_labels: dict[object, int] | None,
    show_qubit_ids: bool,
) -> list[tuple[float, float]]:
    """Scatter blocked locations as gray dots; return their plotted positions."""
    positions: list[tuple[float, float]] = []
    for loc in blocked_locations:
        x, y = arch_spec.get_position(loc)
        positions.append((x, y))
        ax.scatter(
            x,
            y,
            c="#B0BEC5",
            s=78,
            zorder=7,
            edgecolors="#455A64",
            linewidths=1.0,
            alpha=0.95,
        )
        if show_qubit_ids and blocked_location_labels is not None:
            blocked_qid = blocked_location_labels.get(loc)
            if blocked_qid is not None:
                ax.annotate(
                    str(blocked_qid),
                    (x, y),
                    color="#263238",
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    fontweight="bold",
                    zorder=8,
                )
    return positions


def _draw_qubits(  # type: ignore[no-untyped-def]
    ax,
    arch_spec,
    configuration,
    *,
    moving_qids,
    target,
    qid_label_map: dict[int, int] | None,
    show_qubit_ids: bool,
) -> list[tuple[float, float]]:
    """Scatter qubits with color-coded borders; return their plotted positions."""
    positions: list[tuple[float, float]] = []
    for qid, loc in configuration.items():
        if qid in moving_qids:
            at_destination = target.get(qid) == loc
            edge_color = "#2E7D32" if at_destination else "#C62828"
            edge_width = 1.8
        else:
            edge_color = "black"
            edge_width = 1.2
        x, y = arch_spec.get_position(loc)
        ax.scatter(
            x,
            y,
            c="#6437FF",
            s=95,
            zorder=10,
            edgecolors=edge_color,
            linewidths=edge_width,
        )
        positions.append((x, y))
        if show_qubit_ids:
            qid_label = (
                qid_label_map.get(qid, qid) if qid_label_map is not None else qid
            )
            ax.annotate(
                str(qid_label),
                (x, y),
                color="white",
                ha="center",
                va="center",
                fontsize=5.5,
                fontweight="bold",
                zorder=11,
            )
    return positions


def _draw_moveset_path(  # type: ignore[no-untyped-def]
    ax,
    arch_spec,
    moveset,
    source_config,
) -> list[tuple[float, float]]:
    """Draw lane paths for an active moveset; return points along each path."""
    positions: list[tuple[float, float]] = []
    if not moveset:
        return positions
    occupied_locs = set(source_config.values())
    for lane in moveset:
        src, _dst = arch_spec.get_endpoints(lane)
        if src in occupied_locs:
            path = arch_spec.get_path(lane)
            if len(path) < 2:
                continue
            x_vals = [x for x, _ in path]
            y_vals = [y for _, y in path]
            ax.plot(x_vals, y_vals, color="#C2477F", linewidth=1.8, zorder=9)
            positions.extend(path)
            ax.annotate(
                "",
                xy=path[-1],
                xytext=path[-2],
                arrowprops=dict(arrowstyle="->", color="#C2477F", lw=1.8),
                zorder=10,
            )
    return positions


def _apply_view_bounds(  # type: ignore[no-untyped-def]
    ax,
    arch_spec,
    *,
    plotted_positions: list[tuple[float, float]],
    fixed_bounds: tuple[float, float, float, float] | None,
) -> None:
    """Set axis limits: fixed -> local-fit -> full-arch fallback."""
    if fixed_bounds is not None:
        x_min, x_max, y_min, y_max = fixed_bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    elif plotted_positions:
        xs = [p[0] for p in plotted_positions]
        ys = [p[1] for p in plotted_positions]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        x_width = max(1e-6, x_max - x_min)
        y_width = max(1e-6, y_max - y_min)
        ax.set_xlim(x_min - 0.12 * x_width, x_max + 0.12 * x_width)
        ax.set_ylim(y_min - 0.22 * y_width, y_max + 0.22 * y_width)
    else:
        x_min, x_max, y_min, y_max = _path_bounds(arch_spec)
        x_width = x_max - x_min
        y_width = y_max - y_min
        ax.set_xlim(x_min - 0.06 * x_width, x_max + 0.06 * x_width)
        ax.set_ylim(y_min - 0.08 * y_width, y_max + 0.08 * y_width)


def _draw_hardware_snapshot(
    ax,  # type: ignore[no-untyped-def]
    *,
    arch_spec,  # type: ignore[no-untyped-def]
    target,  # type: ignore[no-untyped-def]
    configuration,  # type: ignore[no-untyped-def]
    blocked_locations=(),  # type: ignore[no-untyped-def]
    moveset=None,  # type: ignore[no-untyped-def]
    move_source_configuration=None,  # type: ignore[no-untyped-def]
    root_configuration=None,  # type: ignore[no-untyped-def]
    qid_label_map: dict[int, int] | None = None,
    blocked_location_labels: dict[object, int] | None = None,
    show_qubit_ids: bool = True,
    fixed_bounds: tuple[float, float, float, float] | None = None,
) -> None:
    """Compose a hardware-state snapshot: arch + blockers + qubits + moveset."""
    ax.clear()
    _draw_architecture_background(ax, arch_spec)

    if root_configuration is None:
        root_configuration = configuration
    moving_qids = {
        qid
        for qid, root_loc in root_configuration.items()
        if target.get(qid) is not None and target[qid] != root_loc
    }

    plotted: list[tuple[float, float]] = []
    plotted.extend(
        _draw_blocked_locations(
            ax,
            arch_spec,
            blocked_locations,
            blocked_location_labels=blocked_location_labels,
            show_qubit_ids=show_qubit_ids,
        )
    )
    plotted.extend(
        _draw_qubits(
            ax,
            arch_spec,
            configuration,
            moving_qids=moving_qids,
            target=target,
            qid_label_map=qid_label_map,
            show_qubit_ids=show_qubit_ids,
        )
    )
    source_config = (
        move_source_configuration
        if move_source_configuration is not None
        else configuration
    )
    plotted.extend(_draw_moveset_path(ax, arch_spec, moveset, source_config))
    for qid, target_loc in target.items():
        if qid in configuration:
            plotted.append(arch_spec.get_position(target_loc))

    _apply_view_bounds(
        ax, arch_spec, plotted_positions=plotted, fixed_bounds=fixed_bounds
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_focus_panel(
    ax,  # type: ignore[no-untyped-def]
    frame: TreeFrameState,
    arch_spec,  # type: ignore[no-untyped-def]
    target: dict[int, LocationAddress],
    scorer: EntropyScorer,
    *,
    blocked_locations: tuple[LocationAddress, ...] = (),
    qid_label_map: dict[int, int] | None = None,
    blocked_location_labels: dict[object, int] | None = None,
) -> None:
    ax.clear()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("auto")
    ax.set_title("Active Node (candidate hardware views)")
    panel_box = FancyBboxPatch(
        (0.01, 0.02),
        0.98,
        0.96,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        transform=ax.transAxes,
        facecolor="none",
        edgecolor="#37474F",
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(panel_box)

    node = frame.nodes.get(frame.current_node_id)
    if node is None:
        ax.text(
            0.5,
            0.5,
            "No active node",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.axis("off")
        return

    alpha = scorer.alpha
    beta = scorer.beta
    gamma = scorer.gamma
    score_text = (
        "n/a" if getattr(node, "move_score", None) is None else f"{node.move_score:.3f}"
    )
    root_node = next((n for n in frame.nodes.values() if n.parent_id is None), None)
    root_configuration = (
        root_node.configuration
        if root_node is not None
        else frame.hardware_configuration
    )
    stable_bounds = _stable_focus_bounds(
        arch_spec=arch_spec,
        root_configuration=root_configuration,
        target=target,
    )
    # Current state + up to three candidate moves.
    container_x = 0.015
    container_y = 0.03
    container_w = 0.965
    container_h = 0.92

    slot_gap = 0.016
    slot_pad = 0.01
    slot_w = container_w - 2 * slot_pad
    total_slots = 4
    slot_h = (container_h - 2 * slot_pad - (total_slots - 1) * slot_gap) / total_slots
    info_w = 0.31

    # Slot 0: current state (same height as candidate slots).
    current_slot_x = container_x + slot_pad
    current_slot_y = container_y + container_h - slot_pad - slot_h
    current_slot_rect = Rectangle(
        (current_slot_x, current_slot_y),
        slot_w,
        slot_h,
        transform=ax.transAxes,
        facecolor="none",
        edgecolor="#455A64",
        linewidth=1.4,
        zorder=8,
    )
    ax.add_patch(current_slot_rect)
    current_info_x = current_slot_x + 0.010
    current_info_y_top = current_slot_y + slot_h - 0.015
    ax.text(
        current_info_x,
        current_info_y_top,
        "current state",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        family="monospace",
        zorder=9,
    )
    ax.text(
        current_info_x,
        current_info_y_top - 0.040,
        f"E={node.entropy}   s={score_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        family="monospace",
        zorder=9,
    )
    ax.text(
        current_info_x,
        current_info_y_top - 0.078,
        f"depth={node.depth}   node={node.display_id}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        family="monospace",
        zorder=9,
    )
    current_hw_x = current_slot_x + info_w - 0.015
    current_hw_w = slot_w - info_w - 0.008
    current_inset = ax.inset_axes(
        [current_hw_x, current_slot_y + 0.010, current_hw_w, slot_h - 0.024]
    )
    _draw_hardware_snapshot(
        current_inset,
        arch_spec=arch_spec,
        target=target,
        configuration=frame.hardware_configuration,
        blocked_locations=blocked_locations,
        root_configuration=root_configuration,
        moveset=None,
        move_source_configuration=frame.hardware_move_source_configuration,
        qid_label_map=qid_label_map,
        blocked_location_labels=blocked_location_labels,
        show_qubit_ids=True,
        fixed_bounds=stable_bounds,
    )

    candidates = list(node.candidates)
    max_slots = 3

    # Dedicated candidate group box, separated from current state.
    section_gap = 0.02
    candidate_group_x = container_x + slot_pad - 0.004
    candidate_group_y = container_y + slot_pad
    candidate_group_w = slot_w + 0.008
    candidate_group_h = current_slot_y - section_gap - candidate_group_y
    candidate_group_rect = Rectangle(
        (candidate_group_x, candidate_group_y),
        candidate_group_w,
        candidate_group_h,
        transform=ax.transAxes,
        facecolor="none",
        edgecolor="#78909C",
        linewidth=1.25,
        zorder=7,
    )
    ax.add_patch(candidate_group_rect)

    candidate_group_pad = 0.010
    candidate_slot_gap = 0.012
    candidate_slot_h = (
        candidate_group_h
        - 2 * candidate_group_pad
        - (max_slots - 1) * candidate_slot_gap
    ) / max_slots

    for idx in range(max_slots):
        slot_x = container_x + slot_pad
        slot_y = (
            candidate_group_y
            + candidate_group_h
            - candidate_group_pad
            - (idx + 1) * candidate_slot_h
            - idx * candidate_slot_gap
        )
        candidate = candidates[idx] if idx < len(candidates) else None
        if candidate is not None:
            border_color = (
                "#FFD54F" if node.active_candidate_index == idx else candidate.color
            )
            border_width = 2.0 if node.active_candidate_index == idx else 1.4
        else:
            border_color = "#90A4AE"
            border_width = 1.1
        slot_rect = Rectangle(
            (slot_x, slot_y),
            slot_w,
            candidate_slot_h,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor=border_color,
            linewidth=border_width,
            zorder=8,
        )
        ax.add_patch(slot_rect)
        info_x = slot_x + 0.010
        info_y_top = slot_y + candidate_slot_h - 0.015
        if candidate is not None:
            metrics = scorer.metrics(
                _configuration_to_native(node.configuration),
                _moveset_to_tuples(candidate.moveset),
            )
            info_lines = [
                f"score={metrics.score:.3f}",
                f"f={alpha:.2f}*D + {beta:.2f}*A + {gamma:.2f}*M",
                (
                    f"D={metrics.distance_progress:.2f} "
                    f"A={metrics.arrived} M={metrics.mobility_gain:.2f}"
                ),
            ]
            closer_labels: list[int] = [
                (
                    qid_label_map[qid]
                    if qid_label_map is not None and qid in qid_label_map
                    else qid
                )
                for qid in metrics.closer
            ]
            further_labels: list[int] = [
                (
                    qid_label_map[qid]
                    if qid_label_map is not None and qid in qid_label_map
                    else qid
                )
                for qid in metrics.further
            ]
            info_lines.extend(_format_qid_list_lines("closer", closer_labels, width=18))
            info_lines.extend(
                _format_qid_list_lines("further", further_labels, width=18)
            )
        else:
            info_lines = [
                "score=-",
                f"f={alpha:.2f}*D + {beta:.2f}*A + {gamma:.2f}*M",
                "D=- A=- M=-",
            ]
            info_lines.extend(_format_qid_list_lines("closer", [], width=18))
            info_lines.extend(_format_qid_list_lines("further", [], width=18))
        line_step = min(
            0.034,
            max(0.024, (candidate_slot_h - 0.03) / max(1, len(info_lines))),
        )
        for line_idx, line in enumerate(info_lines):
            ax.text(
                info_x,
                info_y_top - line_idx * line_step,
                line,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.8,
                family="monospace",
                zorder=9,
            )
        if candidate is not None:
            hw_x = slot_x + info_w - 0.015
            hw_w = slot_w - info_w - 0.008
            inset = ax.inset_axes(
                [hw_x, slot_y + 0.010, hw_w, candidate_slot_h - 0.024]
            )
            _draw_hardware_snapshot(
                inset,
                arch_spec=arch_spec,
                target=target,
                configuration=node.configuration,
                blocked_locations=blocked_locations,
                root_configuration=root_configuration,
                moveset=candidate.moveset,
                move_source_configuration=node.configuration,
                qid_label_map=qid_label_map,
                blocked_location_labels=blocked_location_labels,
                show_qubit_ids=True,
                fixed_bounds=stable_bounds,
            )

    terminal_reason = getattr(node, "terminal_reason", None)
    if terminal_reason is not None:
        parent_entropy = getattr(node, "terminal_parent_entropy", None)
        if parent_entropy is None:
            terminal_text = f"terminal: {terminal_reason}"
        else:
            terminal_text = f"terminal: {terminal_reason} (parent E={parent_entropy})"
        ax.text(
            0.5,
            0.14,
            terminal_text,
            ha="center",
            va="center",
            fontsize=10.0,
            family="monospace",
            color="#424242",
            transform=ax.transAxes,
            zorder=6,
        )
    ax.axis("off")


def draw_metadata_panel(
    ax,  # type: ignore[no-untyped-def]
    frame: TreeFrameState,
    info_line: str,
    unresolved_qids: list[int],
) -> None:
    ax.clear()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    panel_box = FancyBboxPatch(
        (0.01, 0.03),
        0.98,
        0.94,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=ax.transAxes,
        facecolor="#FAFAFA",
        edgecolor="#455A64",
        linewidth=1.4,
        zorder=1,
    )
    ax.add_patch(panel_box)
    ax.text(
        0.03,
        0.95,
        "Algorithm Metadata",
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        transform=ax.transAxes,
        zorder=3,
    )
    info_wrapped = textwrap.fill(info_line, width=62)
    ax.text(
        0.03,
        0.80,
        info_wrapped,
        ha="left",
        va="top",
        fontsize=9.0,
        family="monospace",
        transform=ax.transAxes,
        zorder=3,
    )
    unresolved_text = (
        ", ".join(str(qid) for qid in unresolved_qids) if unresolved_qids else "none"
    )
    unresolved_line = f"Unresolved qubits ({len(unresolved_qids)}): {unresolved_text}"
    unresolved_wrapped = textwrap.fill(unresolved_line, width=62)
    ax.text(
        0.03,
        0.58,
        unresolved_wrapped,
        ha="left",
        va="top",
        fontsize=9.0,
        family="monospace",
        transform=ax.transAxes,
        zorder=3,
    )

    buffer_ids = frame.best_buffer_node_display_ids
    if buffer_ids:
        # Wrap candidate boxes by row so they stay inside metadata panel.
        box_w = 0.072
        box_h = 0.14
        col_gap = 0.010
        row_gap = 0.020
        start_x = 0.04
        panel_right = 0.96
        panel_bottom = 0.05
        available_w = panel_right - start_x
        cols = max(1, int((available_w + col_gap) // (box_w + col_gap)))
        rows = (len(buffer_ids) + cols - 1) // cols
        row0_y = panel_bottom + (rows - 1) * (box_h + row_gap)
        total_w = (
            min(cols, len(buffer_ids)) * box_w
            + max(0, min(cols, len(buffer_ids)) - 1) * col_gap
        )
        total_h = rows * box_h + max(0, rows - 1) * row_gap
        outer_pad = 0.014
        outer_rect = Rectangle(
            (
                start_x - outer_pad,
                (row0_y - (rows - 1) * (box_h + row_gap)) - outer_pad,
            ),
            total_w + 2 * outer_pad,
            total_h + 2 * outer_pad,
            transform=ax.transAxes,
            facecolor="#E3F2FD",
            edgecolor="#1E88E5",
            linewidth=1.8,
            zorder=2,
        )
        ax.add_patch(outer_rect)
        ax.text(
            start_x,
            row0_y + box_h + 0.016,
            "Best candidate buffer",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )
        for i, display_id in enumerate(buffer_ids):
            row = i // cols
            col = i % cols
            x = start_x + col * (box_w + col_gap)
            y = row0_y - row * (box_h + row_gap)
            rect = Rectangle(
                (x, y),
                box_w,
                box_h,
                transform=ax.transAxes,
                facecolor="#FFFFFF",
                edgecolor="#111111",
                linewidth=1.1,
                zorder=3,
            )
            ax.add_patch(rect)
            if display_id is not None:
                ax.text(
                    x + box_w / 2.0,
                    y + box_h / 2.0,
                    str(display_id),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    family="monospace",
                    zorder=4,
                )
