"""Interactive controller for the entropy-tree visualizer.

Extends :class:`bloqade.lanes.visualize.app.DebuggerController` so the
slider, Prev/Next buttons, and keyboard shortcuts come for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import figure
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

from bloqade.lanes.bytecode._native import EntropyScorer
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.visualize.app import DebuggerController
from bloqade.lanes.visualize.entropy_tree.renderer import (
    draw_focus_panel,
    draw_metadata_panel,
    draw_tree_frame,
    format_entropy_reason,
)
from bloqade.lanes.visualize.entropy_tree.state import (
    TreeFrameState,
    TreeStateReducer,
)


@dataclass
class EntropyTreeController(DebuggerController):
    """Drives the interactive entropy-tree view."""

    reducer: TreeStateReducer
    arch_spec: Any
    target: dict[int, LocationAddress]
    root_node_id: int
    best_buffer_size: int
    scorer: EntropyScorer | None = None
    blocked_locations: tuple[LocationAddress, ...] = ()
    qid_label_map: dict[int, int] | None = None
    blocked_location_labels: dict[Any, int] | None = None

    step_index: int = 0
    num_steps: int = field(init=False)
    _fig: Any = field(default=None, init=False, repr=False)
    _ax_tree: Any = field(default=None, init=False, repr=False)
    _ax_focus: Any = field(default=None, init=False, repr=False)
    _ax_meta: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.num_steps = self.reducer.frame_count

    def run(self) -> None:
        plt.show()

    def reset(self) -> None:
        self._render(0)

    def on_next(self, event) -> None:  # type: ignore[no-untyped-def]
        _ = event
        if self.step_index + 1 < self.num_steps:
            self.step_index += 1
            self.sync_slider(self.step_index)
            self._render(self.step_index)

    def on_prev(self, event) -> None:  # type: ignore[no-untyped-def]
        _ = event
        if self.step_index > 0:
            self.step_index -= 1
            self.sync_slider(self.step_index)
            self._render(self.step_index)

    def on_exit(self, event) -> None:  # type: ignore[no-untyped-def]
        _ = event
        if self._fig is not None:
            plt.close(self._fig)

    def on_slider_change(self, value) -> None:  # type: ignore[no-untyped-def]
        new_idx = int(value)
        if new_idx != self.step_index:
            self.step_index = new_idx
            self._render(new_idx)

    def attach(self) -> Any:
        """Build the figure and wire the event loop. Returns the Figure."""
        fig = plt.figure(figsize=(16, 9))
        # 3 content rows; a single slider lives below the grid in the area
        # reserved by `subplots_adjust(bottom=...)`. Keyboard shortcuts
        # (left / right / escape) replace the Prev / Next / Exit buttons.
        gs = GridSpec(
            3,
            2,
            figure=fig,
            width_ratios=[1.65, 2.35],
            height_ratios=[2.2, 3.8, 3.0],
        )
        self._ax_meta = fig.add_subplot(gs[0, 0])
        self._ax_tree = fig.add_subplot(gs[1:3, 0])
        self._ax_focus = fig.add_subplot(gs[0:3, 1])
        self._ax_meta.axis("off")
        fig.subplots_adjust(bottom=0.12, top=0.97, left=0.04, right=0.98)

        self._fig = fig
        self.run_mpl_event_loop(self._ax_tree, fig)
        return fig

    def run_mpl_event_loop(self, ax, fig) -> None:  # type: ignore[override,no-untyped-def]
        """Slider-only event loop — no Prev/Next/Exit buttons.

        Overrides :meth:`DebuggerController.run_mpl_event_loop` to reclaim the
        figure area those buttons would otherwise occupy. Keyboard shortcuts
        (``left`` / ``right`` / ``escape``) still drive navigation via the
        inherited :meth:`DebuggerController.on_key` dispatcher.
        """
        _ = ax  # kept for base-class signature compatibility
        self.slider = None

        num_steps = getattr(self, "num_steps", 1)
        if num_steps > 1:
            slider_ax = fig.add_axes((0.1, 0.04, 0.8, 0.04))
            initial_step = max(0, min(self.step_index, num_steps - 1))
            self.slider = Slider(
                ax=slider_ax,
                label="Step",
                valmin=0,
                valmax=num_steps - 1,
                valinit=initial_step,
                valstep=1,
                valfmt="%d",
            )
            self.slider.on_changed(self.on_slider_change)

        fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.reset()
        self.run()

        if isinstance(fig, figure.Figure):
            plt.close(fig)

    def _render(self, idx: int) -> None:
        frame: TreeFrameState = self.reducer.frame_at(idx)
        draw_tree_frame(self._ax_tree, frame)
        if self.scorer is not None:
            draw_focus_panel(
                self._ax_focus,
                frame,
                self.arch_spec,
                self.target,
                self.scorer,
                blocked_locations=self.blocked_locations,
                qid_label_map=self.qid_label_map,
                blocked_location_labels=self.blocked_location_labels,
            )

        reason = format_entropy_reason(frame, qid_label_map=self.qid_label_map)
        best_depth = (
            "-" if frame.best_goal_depth is None else str(frame.best_goal_depth)
        )
        info_line = (
            f"Step {idx}/{self.num_steps - 1} | event={frame.event} "
            f"| current_node={frame.current_node_display_id} "
            f"| Best goal depth: {best_depth}"
        )
        if reason is not None:
            info_line = f"{info_line} | {reason}"
        unresolved = sorted(
            qid
            for qid, target_loc in self.target.items()
            if frame.hardware_configuration.get(qid) != target_loc
        )
        unresolved_labels = sorted(
            (self.qid_label_map.get(qid, qid) if self.qid_label_map else qid)
            for qid in unresolved
        )
        draw_metadata_panel(self._ax_meta, frame, info_line, unresolved_labels)
        if self._fig is not None:
            self._fig.canvas.draw_idle()
