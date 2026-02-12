from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Callable

import numpy as np
from kirin import ir
from matplotlib import colormaps, figure, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Button
from scipy.interpolate import interp1d

from bloqade.lanes.analysis.atom import AtomInterpreter, AtomState, MoveExecution, Value
from bloqade.lanes.dialects import move
from bloqade.lanes.layout import ArchSpec


class quera_color_code(str, Enum):
    # TODO: for python 3.11+, replace traits with StrEnum
    purple = "#6437FF"
    red = "#C2477F"
    yellow = "#EADD08"
    aod_line_color = "#FFE8E9"


@dataclass
class PlotParameters:
    scale: float

    atom_marker: str = "o"
    aod_marker: str = "+"

    local_r_color: str = "tab:blue"
    local_rz_color: str = "tab:green"
    global_r_color: str = "tab:blue"
    global_rz_color: str = "tab:green"
    cz_color: str = "tab:red"
    aod_line_color: str = quera_color_code.aod_line_color
    atom_color: str = quera_color_code.purple
    aod_line_style: str = "dashed"

    @property
    def atom_plot_args(self) -> dict:
        return {
            "color": self.atom_color,
            "marker": self.atom_marker,
            "linestyle": "",
            "s": self.scale * 65,
        }

    @property
    def gate_spot_args(self) -> dict:
        return {
            "marker": self.atom_marker,
            "s": self.scale * 160,
            "alpha": 0.3,
        }

    @property
    def slm_plot_args(self) -> dict:
        return {
            "facecolors": "none",
            "edgecolors": "k",
            "linestyle": "-",
            "s": self.scale * 80,
            "alpha": 1.0,
            "linewidth": 0.5 * np.sqrt(self.scale),
            "marker": self.atom_marker,
        }

    @property
    def aod_line_args(self) -> dict:
        return {
            "alpha": 1.0,
            "colors": self.aod_line_color,
            "linestyles": self.aod_line_style,
            "zorder": -101,
        }

    @property
    def aod_marker_args(self) -> dict:
        return {
            "color": quera_color_code.red,
            "marker": self.aod_marker,
            "s": self.scale * 260,
            "linewidth": np.sqrt(self.scale),
            "alpha": 0.7,
            "zorder": -100,
        }


@dataclass
class MoveRenderer:
    ax: Axes
    aod_x: list[interp1d]
    aod_y: list[interp1d]
    moving_atoms_x_indices: list[int]
    moving_atoms_y_indices: list[int]
    stationary_atoms_x: list[float]
    stationary_atoms_y: list[float]
    plot_params: PlotParameters

    @cached_property
    def total_time(self) -> float:
        return max((func.x[-1] for func in self.aod_x + self.aod_y), default=0.0)

    def __post_init__(self):
        aod_x = [func(0) for func in self.aod_x]
        aod_y = [func(0) for func in self.aod_y]
        moving_atom_x = list(map(aod_x.__getitem__, self.moving_atoms_x_indices))
        moving_atom_y = list(map(aod_y.__getitem__, self.moving_atoms_y_indices))

        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        self.aod_x_lines = self.ax.vlines(
            aod_x, y_min, y_max, **self.plot_params.aod_line_args
        )
        self.aod_y_lines = self.ax.hlines(
            aod_y, x_min, x_max, **self.plot_params.aod_line_args
        )
        aod_x_positions, aod_y_positions = np.meshgrid(aod_x, aod_y)

        self.aod_positions = self.ax.scatter(
            aod_x_positions.ravel(),
            aod_y_positions.ravel(),
            **self.plot_params.aod_marker_args,
        )
        self.moving_atoms_scatter = self.ax.scatter(
            moving_atom_x, moving_atom_y, **self.plot_params.atom_plot_args
        )
        self.ax.scatter(
            self.stationary_atoms_x,
            self.stationary_atoms_y,
            **self.plot_params.atom_plot_args,
        )

    def update(self, time: float):
        aod_x = [func(time) for func in self.aod_x]
        aod_y = [func(time) for func in self.aod_y]
        moving_atom_x = list(map(aod_x.__getitem__, self.moving_atoms_x_indices))
        moving_atom_y = list(map(aod_y.__getitem__, self.moving_atoms_y_indices))
        ymin, ymax = self.ax.get_ylim()
        xmin, xmax = self.ax.get_xlim()
        self.aod_x_lines.set_segments([[(x, ymin), (x, ymax)] for x in aod_x])
        self.aod_y_lines.set_segments([[(xmin, y), (xmax, y)] for y in aod_y])
        aod_x_positions, aod_y_positions = np.meshgrid(aod_x, aod_y)
        self.aod_positions.set_offsets(
            np.column_stack([aod_x_positions.ravel(), aod_y_positions.ravel()])
        )
        self.moving_atoms_scatter.set_offsets(
            np.column_stack([moving_atom_x, moving_atom_y])
        )


@dataclass
class StateArtist:
    ax: Axes
    arch_spec: ArchSpec
    plot_params: PlotParameters

    def _get_aod_paths(self, speed, move_execution: AtomState):
        waypoints: list[tuple[set[float], set[float]]] = []
        path_len = None

        for lane in move_execution.data.prev_lanes.values():
            path = self.arch_spec.get_path(lane)
            if path_len is not None:
                if len(path) != path_len:
                    raise ValueError(
                        "All paths must have the same length for animation. "
                        f"Expected length {path_len}, but got path of length {len(path)}"
                        f"for lane {lane}"
                    )
            else:
                path_len = len(path)
                waypoints = [(set(), set()) for _ in range(path_len)]

            for (x, y), (xs, ys) in zip(path, waypoints):
                xs.add(x)
                ys.add(y)

        if path_len is None:
            return ([], []), ([], [])

        sorted_waypoints = [(sorted(xs), sorted(ys)) for (xs, ys) in waypoints]

        aod_x_positions = np.asarray([xs for (xs, _) in sorted_waypoints])
        aod_y_positions = np.asarray([ys for (_, ys) in sorted_waypoints])

        x_max_diffs = np.abs(np.diff(aod_x_positions, axis=0)).max(axis=1)
        y_max_diffs = np.abs(np.diff(aod_y_positions, axis=0)).max(axis=1)
        time_diffs = np.hypot(x_max_diffs, y_max_diffs) / speed
        time = np.insert(np.cumsum(time_diffs, axis=0), 0, 0, axis=0)

        aod_x_funcs = [
            interp1d(time, aod_x_position, kind="linear")
            for aod_x_position in aod_x_positions.T[:]
        ]
        aod_y_funcs = [
            interp1d(time, aod_y_position, kind="linear")
            for aod_y_position in aod_y_positions.T[:]
        ]
        last_xs, last_ys = sorted_waypoints[-1]
        return (aod_x_funcs, aod_y_funcs), (last_xs, last_ys)

    def _get_moving_indices(
        self, move_execution: AtomState, last_xs: list[float], last_ys: list[float]
    ) -> tuple[list[int], list[int]]:
        src_locs = [
            self.arch_spec.get_position(move_execution.data.qubit_to_locations[qubit])
            for qubit in move_execution.data.prev_lanes.keys()
        ]

        moving_atom_indices = [
            (last_xs.index(x), last_ys.index(y))
            for x, y in src_locs
            if x in last_xs and y in last_ys
        ]
        x_indices, y_indices = (
            zip(*moving_atom_indices) if len(moving_atom_indices) > 0 else ([], [])
        )
        return list(x_indices), list(y_indices)

    def _get_stationary_positions(
        self, move_execution: AtomState
    ) -> tuple[list[float], list[float]]:
        stationary_atom_positions = [
            self.arch_spec.get_position(location)
            for qubit, location in move_execution.data.qubit_to_locations.items()
            if qubit not in move_execution.data.prev_lanes.keys()
        ]
        stationary_atom_positions_x, stationary_atom_positions_y = (
            zip(*stationary_atom_positions)
            if len(stationary_atom_positions) > 0
            else ([], [])
        )

        return list(stationary_atom_positions_x), list(stationary_atom_positions_y)

    def move_renderer(
        self, move_execution: MoveExecution, speed: float
    ) -> MoveRenderer | None:
        if not isinstance(move_execution, AtomState):
            return None

        if len(move_execution.data.prev_lanes) == 0:
            return None

        (aod_x_funcs, aod_y_funcs), (first_xs, first_ys) = self._get_aod_paths(
            speed, move_execution
        )
        moving_atom_indices_x, moving_atom_indices_y = self._get_moving_indices(
            move_execution, first_xs, first_ys
        )
        stationary_atom_positions_x, stationary_atom_positions_y = (
            self._get_stationary_positions(move_execution)
        )
        return MoveRenderer(
            self.ax,
            aod_x=aod_x_funcs,
            aod_y=aod_y_funcs,
            moving_atoms_x_indices=moving_atom_indices_x,
            moving_atoms_y_indices=moving_atom_indices_y,
            stationary_atoms_x=stationary_atom_positions_x,
            stationary_atoms_y=stationary_atom_positions_y,
            plot_params=self.plot_params,
        )

    def draw_atoms(
        self,
        state: MoveExecution,
    ):
        if not isinstance(state, AtomState):
            return

        locations = list(state.data.locations_to_qubit)

        x, y = (
            zip(*map(self.arch_spec.get_position, locations))
            if len(locations) > 0
            else ([], [])
        )
        self.ax.scatter(x, y, **self.plot_params.atom_plot_args)

    def draw_moves(
        self,
        state: MoveExecution,
        **kwargs,
    ):
        if not isinstance(state, AtomState):
            return

        cmap = colormaps["viridis"]
        for lane in state.data.prev_lanes.values():
            path = self.arch_spec.get_path(lane)
            steps = list(zip(path, path[1:]))
            color_indices = np.linspace(0, 1, len(steps))
            for cl_val, ((x_start, y_start), (x_end, y_end)) in zip(
                color_indices, steps
            ):
                self.ax.quiver(
                    [x_start],
                    [y_start],
                    [x_end - x_start],
                    [y_end - y_start],
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    color=cmap(cl_val),
                    **kwargs,
                )

    def _show_local(self, stmt: move.LocalR | move.LocalRz, color: str):
        positions = list(
            self.arch_spec.words[location.word_id].site_position(location.site_id)
            for location in stmt.location_addresses
        )
        x_pos, y_pos = zip(*positions) if len(positions) > 0 else ([], [])
        self.ax.scatter(x_pos, y_pos, color=color, **self.plot_params.gate_spot_args)

    def show_local_r(self, stmt: move.LocalR):
        self._show_local(stmt, color="blue")

    def show_local_rz(self, stmt: move.LocalRz):
        self._show_local(stmt, color="green")

    def _show_global(self, stmt: move.GlobalR | move.GlobalRz, color: str):
        x_min, x_max = self.arch_spec.x_bounds
        x_width = x_max - x_min
        x_min -= 0.5 * x_width
        x_max += 0.5 * x_width

        y_min, y_max = self.arch_spec.y_bounds
        y_width = y_max - y_min
        y_min -= 0.5 * y_width
        y_max += 0.5 * y_width

        self.ax.fill_between(
            [x_min, x_max],
            [y_min, y_min],
            [y_max, y_max],
            color=color,
            alpha=0.3,
        )

    def show_global_r(self, stmt: move.GlobalR):
        self._show_global(stmt, color="blue")

    def show_global_rz(self, stmt: move.GlobalRz):
        self._show_global(stmt, color="green")

    def show_cz(self, stmt: move.CZ):
        words = tuple(
            self.arch_spec.words[word_id]
            for word_id in self.arch_spec.zones[stmt.zone_address.zone_id]
        )

        y_min = float("inf")
        y_max = float("-inf")

        for word in words:
            for _, y_pos in word.all_positions():
                y_min = min(y_min, y_pos)
                y_max = max(y_max, y_pos)

        x_min, x_max = self.arch_spec.x_bounds
        y_width = y_max - y_min
        y_min -= 0.1 * y_width
        y_max += 0.1 * y_width

        self.ax.fill_between(
            [x_min - 10, x_max + 10],
            [y_min, y_min],
            [y_max, y_max],
            color="red",
            alpha=0.3,
        )

    def show_slm(self, stmt: ir.Statement, atom_marker: str):
        self.arch_spec.plot(
            self.ax,
            show_words=range(len(self.arch_spec.words)),
            **self.plot_params.slm_plot_args,
        )


def get_state_artist(
    arch_spec: ArchSpec, ax: Axes, atom_marker: str = "o"
) -> StateArtist:
    x_min, x_max, y_min, y_max = arch_spec.path_bounds()
    x_width = x_max - x_min
    y_width = y_max - y_min

    x_min -= 0.1 * x_width
    x_max += 0.1 * x_width
    y_min -= 0.1 * y_width
    y_max += 0.1 * y_width

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    nsites = len(arch_spec.words) * len(arch_spec.words[0].site_indices)

    scale = (
        np.sqrt(44.0 / nsites) * 2.0 * plt.rcParams["figure.dpi"] / 100
    )  # scale the size of the figure

    plot_params = PlotParameters(scale, atom_marker=atom_marker)

    return StateArtist(ax, arch_spec, plot_params=plot_params)


def get_drawer(mt: ir.Method, arch_spec: ArchSpec, ax: Axes, atom_marker: str = "o"):
    artist = get_state_artist(arch_spec, ax)

    frame, _ = AtomInterpreter(mt.dialects, arch_spec=arch_spec).run(mt)

    methods: dict = {
        move.LocalR: artist.show_local_r,
        move.LocalRz: artist.show_local_rz,
        move.GlobalR: artist.show_global_r,
        move.GlobalRz: artist.show_global_rz,
        move.CZ: artist.show_cz,
    }

    steps: list[tuple[ir.Statement, AtomState]] = []
    constants = {}
    for stmt in mt.callable_region.walk():
        results = frame.get_values(stmt.results)
        match results:
            case (AtomState() as state,):
                steps.append((stmt, state))
            case (Value(value),) if isinstance(value, (float, int)):
                constants[stmt.results[0]] = value

    def stmt_text(stmt: ir.Statement) -> str:
        stmt_str = f"{type(stmt).__name__}("
        if len(stmt.args) != 0:
            stmt_str = stmt_str + (
                ", ".join(f"{constants[arg]}" for arg in stmt.args if arg in constants)
            )
        stmt_str = stmt_str + ")"
        return stmt_str

    def draw(step_index: int):
        if len(steps) == 0:
            return
        stmt, curr_state = steps[step_index]
        artist.show_slm(stmt, atom_marker)

        visualize_fn = methods.get(type(stmt), lambda stmt: None)
        visualize_fn(stmt)
        artist.draw_atoms(curr_state)
        artist.draw_moves(curr_state)

        ax.set_title(f"Step {step_index+1} / {len(steps)}")
        ax.text(
            0.5, 1.01, stmt_text(stmt), ha="center", va="bottom", transform=ax.transAxes
        )

        plt.draw()

    return draw, len(steps)


def interactive_debugger(
    draw: Callable[[int], None],
    num_steps: int,
    fig: figure.Figure | figure.SubFigure,
):

    ax = plt.gca()
    step_index = 0
    running = True
    waiting = True
    updated = False

    prev_ax = fig.add_axes((0.01, 0.01, 0.1, 0.075))
    exit_ax = fig.add_axes((0.21, 0.01, 0.1, 0.075))
    next_ax = fig.add_axes((0.41, 0.01, 0.1, 0.075))

    prev_button = Button(prev_ax, "Prev (<)")
    next_button = Button(next_ax, "Next (>)")
    exit_button = Button(exit_ax, "Exit(Esc)")

    def on_exit(event):
        nonlocal running, waiting, updated
        running = False
        waiting = False
        if not updated:
            updated = True

    def on_next(event):
        nonlocal waiting, step_index, updated
        waiting = False
        if not updated:
            step_index = min(step_index + 1, num_steps - 1)
            ax.cla()
            updated = True

    def on_prev(event):
        nonlocal waiting, step_index, updated
        waiting = False
        if not updated:
            step_index = max(step_index - 1, 0)
            ax.cla()
            updated = True

    # connect buttons to callbacks
    next_button.on_clicked(on_next)
    prev_button.on_clicked(on_prev)
    exit_button.on_clicked(on_exit)

    # connect keyboard shortcuts to callbacks
    def on_key(event):
        match event.key:
            case "left":
                on_prev(event)
            case "right":
                on_next(event)
            case "escape":
                on_exit(event)

    fig.canvas.mpl_connect("key_press_event", on_key)

    while running:
        draw(step_index)

        while waiting:
            plt.pause(0.01)

        waiting = True
        updated = False

    if isinstance(fig, figure.Figure):
        plt.close(fig)


def render_generator(
    mt: ir.Method, arch_spec: ArchSpec, ax: Axes, atom_marker: str = "o", fps: int = 30
) -> tuple[Callable[[int], tuple[int, Callable[[int], None]]], int]:

    artist = get_state_artist(arch_spec, ax, atom_marker)

    methods: dict = {
        move.LocalR: artist.show_local_r,
        move.LocalRz: artist.show_local_rz,
        move.GlobalR: artist.show_global_r,
        move.GlobalRz: artist.show_global_rz,
        move.CZ: artist.show_cz,
    }

    frame, _ = AtomInterpreter(mt.dialects, arch_spec=arch_spec).run(mt)

    x_min, x_max, y_min, y_max = arch_spec.path_bounds()
    x_width = x_max - x_min
    y_width = y_max - y_min

    x_min -= 0.1 * x_width
    x_max += 0.1 * x_width
    y_min -= 0.1 * y_width
    y_max += 0.1 * y_width

    steps: list[tuple[ir.Statement, AtomState]] = []
    constants = {}
    for stmt in mt.callable_region.walk():
        results = frame.get_values(stmt.results)
        match results:
            case (AtomState() as state,):
                steps.append((stmt, state))
            case (Value(value),) if isinstance(value, (float, int)):
                constants[stmt.results[0]] = value

    def stmt_text(stmt: ir.Statement) -> str:
        stmt_str = f"{type(stmt).__name__}("
        if len(stmt.args) != 0:
            stmt_str = stmt_str + (
                ", ".join(f"{constants[arg]}" for arg in stmt.args if arg in constants)
            )
        stmt_str = stmt_str + ")"
        return stmt_str

    def _no_op(ani_step_index: int):
        pass

    def get_renderer(step_index: int) -> tuple[int, Callable[[int], None]]:
        if len(steps) == 0:
            return 3 * fps, _no_op

        stmt, curr_state = steps[step_index]
        artist.show_slm(stmt, atom_marker)
        ax.set_title(f"Step {step_index+1} / {len(steps)}: {stmt_text(stmt)}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")

        visualize_fn = methods.get(type(stmt))
        if visualize_fn is not None:
            artist.draw_atoms(curr_state)
            visualize_fn(stmt)
            return 3 * fps, _no_op

        move_renderer = artist.move_renderer(curr_state, speed=2.0)
        if move_renderer is not None:
            operation_time = min(5.0, move_renderer.total_time)
            total_frames = int(operation_time * fps)

            def _move_renderer(ani_step_index: int):
                if ani_step_index > total_frames or ani_step_index < 0:
                    return
                t = ani_step_index / total_frames * move_renderer.total_time
                move_renderer.update(t)

            return total_frames, _move_renderer

        artist.draw_atoms(curr_state)
        return 3 * fps, _no_op

    return get_renderer, len(steps)


def interactive_animator(
    get_renderer: Callable[[int], tuple[int, Callable[[int], None]]],
    num_steps: int,
    ax: Axes,
):
    fig = ax.get_figure(True)

    if fig is None:
        raise ValueError("Could not get figure from axes for interactive animator.")

    step_index = 0
    animation_step = 1
    num_frames = 0

    running = True
    waiting = True
    updated = False

    prev_ax = fig.add_axes((0.01, 0.01, 0.1, 0.075))
    exit_ax = fig.add_axes((0.21, 0.01, 0.1, 0.075))
    next_ax = fig.add_axes((0.41, 0.01, 0.1, 0.075))

    prev_button = Button(prev_ax, "Prev (<)")
    next_button = Button(next_ax, "Next (>)")
    exit_button = Button(exit_ax, "Exit(Esc)")

    def on_exit(event):
        nonlocal running, waiting, updated
        running = False
        waiting = False
        if not updated:
            updated = True

    def on_next(event):
        nonlocal waiting, step_index, updated, animation_step

        if animation_step == 1:
            waiting = False
            if not updated:
                step_index = min(step_index + 1, num_steps - 1)
                ax.cla()
                updated = True
        else:
            animation_step = 1

    def on_prev(event):
        nonlocal waiting, step_index, updated, animation_step
        if animation_step == -1:
            waiting = False
            if not updated:
                step_index = max(step_index - 1, 0)
                ax.cla()
                updated = True
        else:
            animation_step = -1

    # connect buttons to callbacks
    next_button.on_clicked(on_next)
    prev_button.on_clicked(on_prev)
    exit_button.on_clicked(on_exit)

    # connect keyboard shortcuts to callbacks
    def on_key(event):
        match event.key:
            case "left":
                on_prev(event)
            case "right":
                on_next(event)
            case "escape":
                on_exit(event)

    fig.canvas.mpl_connect("key_press_event", on_key)

    while running:
        num_frames, renderer = get_renderer(step_index)
        animation_step_index = 0 if animation_step == 1 else num_frames

        while waiting:
            renderer(animation_step_index)
            animation_step_index += animation_step
            animation_step_index = max(0, animation_step_index)
            animation_step_index = min(animation_step_index, num_frames)
            plt.pause(0.01)

        waiting = True
        updated = False

    if isinstance(fig, figure.Figure):
        plt.close(fig)


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
        interactive_debugger(draw, num_steps, fig)
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
        interactive_animator(get_renderer, num_steps, ax)
    else:
        for step_index in range(num_steps):
            num_frames, renderer = get_renderer(step_index)
            for animation_step_index in range(num_frames):
                renderer(animation_step_index)
                plt.pause(1.0 / fps)
            ax.cla()
