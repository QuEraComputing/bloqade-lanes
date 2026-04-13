from dataclasses import dataclass, field
from typing import Callable

from kirin import ir
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from bloqade.lanes.layout import ArchSpec

from .app import DebuggerController
from .artist import get_drawer, render_generator


@dataclass
class StaticDebuggerController(DebuggerController):
    ax: Axes
    num_steps: int
    draw: Callable[[int], None]
    step_index: int = field(default=0, init=False)
    running: bool = field(default=True, init=False)
    waiting: bool = field(default=True, init=False)
    updated: bool = field(default=False, init=False)

    def on_exit(self, event):
        self.running = False
        self.waiting = False
        if not self.updated:
            self.updated = True

    def on_next(self, event):
        self.waiting = False
        if not self.updated:
            self.step_index = min(self.step_index + 1, self.num_steps - 1)
            self.sync_slider(self.step_index)
            self.ax.cla()
            self.updated = True

    def on_prev(self, event):
        self.waiting = False
        if not self.updated:
            self.step_index = max(self.step_index - 1, 0)
            self.sync_slider(self.step_index)
            self.ax.cla()
            self.updated = True

    def on_slider_change(self, value):
        new_index = max(0, min(int(value), self.num_steps - 1))
        if new_index == self.step_index:
            return
        self.step_index = new_index
        self.ax.cla()
        self.updated = True
        self.waiting = False

    def on_key(self, event):
        match event.key:
            case "left":
                self.on_prev(event)
            case "right":
                self.on_next(event)
            case "escape":
                self.on_exit(event)

    def reset(self):
        self.step_index = 0
        self.sync_slider(0)
        self.running = True
        self.waiting = True
        self.updated = False

    def run(self):
        while self.running:
            self.draw(self.step_index)
            while self.waiting:
                plt.pause(0.01)
            self.waiting = True
            self.updated = False


@dataclass
class AnimatorController(DebuggerController):
    ax: Axes
    num_steps: int
    get_renderer: Callable[[int], tuple[int, Callable[[int], None]]]
    step_index: int = field(default=0, init=False)
    animation_step: int = field(default=1, init=False)
    num_frames: int = field(default=0, init=False)
    running: bool = field(default=True, init=False)
    waiting: bool = field(default=True, init=False)
    updated: bool = field(default=False, init=False)
    animation_step_index: int = field(default=0, init=False)

    def on_exit(self, event):
        self.running = False
        self.waiting = False
        if not self.updated:
            self.updated = True

    def on_next(self, event):
        if self.animation_step == 1:
            self.waiting = False
            if not self.updated:
                self.step_index = min(self.step_index + 1, self.num_steps - 1)
                self.sync_slider(self.step_index)
                self.ax.cla()
                self.updated = True
        else:
            self.animation_step = 1

    def on_prev(self, event):
        if self.animation_step == -1:
            self.waiting = False
            if not self.updated:
                self.step_index = max(self.step_index - 1, 0)
                self.sync_slider(self.step_index)
                self.ax.cla()
                self.updated = True
        else:
            self.animation_step = -1

    def on_slider_change(self, value):
        new_index = max(0, min(int(value), self.num_steps - 1))
        if new_index == self.step_index:
            return
        self.step_index = new_index
        # Slider jumps go forward into the new step's animation.
        self.animation_step = 1
        self.ax.cla()
        self.updated = True
        self.waiting = False

    def reset(self):
        self.step_index = 0
        self.sync_slider(0)
        self.animation_step = 1
        self.running = True
        self.waiting = True
        self.updated = False

    def run(self):
        while self.running:
            self.num_frames, renderer = self.get_renderer(self.step_index)
            self.animation_step_index = (
                0 if self.animation_step == 1 else self.num_frames
            )
            while self.waiting:
                renderer(self.animation_step_index)
                self.animation_step_index += self.animation_step
                self.animation_step_index = max(0, self.animation_step_index)
                self.animation_step_index = min(
                    self.animation_step_index, self.num_frames
                )
                plt.pause(0.01)

            self.waiting = True
            self.updated = False


def debugger(
    mt: ir.Method,
    arch_spec: ArchSpec,
    interactive: bool = True,
    pause_time: float = 1.0,
    atom_marker: str = "o",
    ax: Axes | None = None,
):
    # set up matplotlib figure with buttons
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = ax.figure

    fig.subplots_adjust(bottom=0.2)

    draw, num_steps = get_drawer(mt, arch_spec, ax, atom_marker)
    if interactive:
        controller = StaticDebuggerController(ax, num_steps, draw)
        controller.run_mpl_event_loop(ax, fig)
    else:
        for step_index in range(num_steps):
            draw(step_index)
            plt.pause(pause_time)
            ax.cla()


def animated_debugger(
    mt: ir.Method,
    arch_spec: ArchSpec,
    interactive: bool = True,
    atom_marker: str = "o",
    ax: Axes | None = None,
    fps: int = 30,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = ax.figure

    fig.subplots_adjust(bottom=0.2)

    get_renderer, num_steps = render_generator(mt, arch_spec, ax, atom_marker, fps)
    if interactive:
        controller = AnimatorController(ax, num_steps, get_renderer)
        controller.run_mpl_event_loop(ax, fig)
    else:
        for step_index in range(num_steps):
            num_frames, renderer = get_renderer(step_index)
            for animation_step_index in range(num_frames):
                renderer(animation_step_index)
                plt.pause(1.0 / fps)
            ax.cla()
