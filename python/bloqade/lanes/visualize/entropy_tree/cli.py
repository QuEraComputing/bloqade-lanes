"""Command-line entry for the entropy-tree visualizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from bloqade.lanes.visualize.entropy_tree.controller import EntropyTreeController
from bloqade.lanes.visualize.entropy_tree.state import TreeStateReducer
from bloqade.lanes.visualize.entropy_tree.tracer import (
    EntropyTraceBundle,
    build_entropy_trace,
    load_kernel_from_file,
)


def _build_default_kernel():  # type: ignore[no-untyped-def]
    """Small fallback kernel used when --kernel-file is not provided."""
    from kirin.dialects import ilist

    from bloqade import qubit, squin

    @squin.kernel(typeinfer=True, fold=True)
    def default_kernel():
        reg = qubit.qalloc(4)
        squin.broadcast.cz(
            ilist.IList([reg[0], reg[1]]),
            ilist.IList([reg[2], reg[3]]),
        )

    return default_kernel


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m bloqade.lanes.visualize.entropy_tree",
        description="Interactive visualizer for the Rust entropy-guided search.",
    )
    p.add_argument(
        "--kernel-file",
        type=Path,
        default=None,
        help=(
            "Path to a Python file exporting a squin kernel. The kernel "
            "function must be named after the file stem (e.g. "
            "kernels/small/ghz_4.py must export a symbol named 'ghz_4')."
        ),
    )
    p.add_argument("--layer-index", type=int, default=0)
    p.add_argument("--max-expansions", type=int, default=1000)
    p.add_argument("--max-goal-candidates", type=int, default=None)
    p.add_argument("--export-animation", type=Path, default=None)
    p.add_argument("--export-fps", type=float, default=1.5)
    p.add_argument("--no-interactive", action="store_true")
    return p.parse_args(argv)


def _resolve_kernel(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    if args.kernel_file is None:
        return _build_default_kernel(), "default_kernel"
    symbol = args.kernel_file.stem
    return load_kernel_from_file(args.kernel_file, symbol), symbol


def _export_animation(
    bundle: EntropyTraceBundle,
    output_path: Path,
    fps: float,
) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from bloqade.lanes.visualize.entropy_tree.renderer import (
        draw_metadata_panel,
        draw_tree_frame,
        format_entropy_reason,
    )

    reducer = TreeStateReducer(
        steps=bundle.steps,
        root_node_id=bundle.root_node_id,
        best_buffer_size=bundle.best_buffer_size,
    )
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.65, 2.35], height_ratios=[2.2, 7.8])
    ax_meta = fig.add_subplot(gs[0, 0])
    ax_tree = fig.add_subplot(gs[1, 0])
    ax_focus = fig.add_subplot(gs[:, 1])  # reserved but unused in animation
    ax_focus.axis("off")
    ax_meta.axis("off")

    def animate(i: int):
        frame = reducer.frame_at(i)
        draw_tree_frame(ax_tree, frame)
        reason = format_entropy_reason(frame, bundle.local_to_global_qid or None)
        best_depth = (
            "-" if frame.best_goal_depth is None else str(frame.best_goal_depth)
        )
        info_line = (
            f"Step {i}/{reducer.frame_count - 1} | event={frame.event} "
            f"| current_node={frame.current_node_display_id} "
            f"| Best goal depth: {best_depth}"
        )
        if reason:
            info_line = f"{info_line} | {reason}"
        unresolved = sorted(
            qid
            for qid, tloc in bundle.traced_target.items()
            if frame.hardware_configuration.get(qid) != tloc
        )
        draw_metadata_panel(ax_meta, frame, info_line, unresolved)
        return []

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=reducer.frame_count,
        interval=max(1, int(round(1000 / max(fps, 0.1)))),
        blit=False,
        repeat=True,
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    writer_fps = max(1, int(round(fps)))
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=writer_fps)
    elif suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=writer_fps)
    else:
        raise ValueError(
            f"Unsupported animation extension '{output_path.suffix}'. Use .gif or .mp4."
        )
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    print(f"Saved animation to: {output_path}")


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    kernel, kernel_name = _resolve_kernel(args)

    max_expansions = None if args.max_expansions < 0 else args.max_expansions
    bundle = build_entropy_trace(
        kernel=kernel,
        kernel_name=kernel_name,
        layer_index=args.layer_index,
        max_expansions=max_expansions,
        max_goal_candidates=args.max_goal_candidates,
    )
    print(
        f"Captured {len(bundle.steps)} search steps for {bundle.kernel_name} "
        f"(CZ stage {args.layer_index})."
    )

    if args.export_animation is not None:
        _export_animation(bundle, args.export_animation, args.export_fps)

    if not args.no_interactive:
        from bloqade.lanes.bytecode._native import EntropyScorer

        reducer = TreeStateReducer(
            steps=bundle.steps,
            root_node_id=bundle.root_node_id,
            best_buffer_size=bundle.best_buffer_size,
        )
        scorer = EntropyScorer(
            bundle.arch_spec._inner,
            {qid: loc._inner for qid, loc in bundle.traced_target.items()},
            [loc._inner for loc in bundle.blocked_locations],
        )
        controller = EntropyTreeController(
            reducer=reducer,
            arch_spec=bundle.arch_spec,
            target=bundle.traced_target,
            root_node_id=bundle.root_node_id,
            best_buffer_size=bundle.best_buffer_size,
            scorer=scorer,
            blocked_locations=bundle.blocked_locations,
            qid_label_map=bundle.local_to_global_qid or None,
            blocked_location_labels=bundle.location_to_global_qid or None,
        )
        controller.attach()

    return 0
