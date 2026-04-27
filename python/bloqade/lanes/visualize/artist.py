from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from types import MappingProxyType
from typing import Any, Callable, Sequence

import numpy as np
from kirin import ir
from matplotlib import colormaps, pyplot as plt
from matplotlib.axes import Axes
from scipy.interpolate import interp1d

from bloqade.lanes.analysis.atom import AtomInterpreter, AtomState, MoveExecution, Value
from bloqade.lanes.dialects import move
from bloqade.lanes.layout import ArchSpec
from bloqade.lanes.visualize.arch import ArchVisualizer


class QuEraColorCode(str, Enum):
    # TODO: for python 3.11+, replace traits with StrEnum
    Purple = "#6437FF"
    Red = "#C2477F"
    Yellow = "#EADD08"
    AodLineColor = "#FFE8E9"


@dataclass(frozen=True)
class PlotParameters:
    scale: float

    atom_marker: str = "o"
    aod_marker: str = "+"

    local_r_color: str = "tab:blue"
    local_rz_color: str = "tab:green"
    global_r_color: str = "tab:blue"
    global_rz_color: str = "tab:green"
    cz_color: str = "tab:red"
    aod_line_color: str = QuEraColorCode.AodLineColor
    atom_color: str = QuEraColorCode.Purple
    aod_line_style: str = "dashed"

    @cached_property
    def atom_plot_args(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(
            {
                "color": self.atom_color,
                "marker": self.atom_marker,
                "linestyle": "",
                "s": self.scale * 65,
            }
        )

    @cached_property
    def gate_spot_args(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(
            {
                "marker": self.atom_marker,
                "s": self.scale * 160,
                "alpha": 0.3,
            }
        )

    @cached_property
    def slm_plot_args(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(
            {
                "facecolors": "none",
                "edgecolors": "k",
                "linestyle": "-",
                "s": self.scale * 80,
                "alpha": 1.0,
                "linewidth": 0.5 * np.sqrt(self.scale),
                "marker": self.atom_marker,
            }
        )

    @cached_property
    def aod_line_args(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(
            {
                "alpha": 1.0,
                "colors": self.aod_line_color,
                "linestyles": self.aod_line_style,
                "zorder": -101,
            }
        )

    @cached_property
    def aod_marker_args(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(
            {
                "color": QuEraColorCode.Red,
                "marker": self.aod_marker,
                "s": self.scale * 260,
                "linewidth": np.sqrt(self.scale),
                "alpha": 0.7,
                "zorder": -100,
            }
        )

    @cached_property
    def atom_label_args(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(
            {
                "color": "white",
                "fontsize": max(4.0, self.scale * 6.0),
                "ha": "center",
                "va": "center",
                "zorder": 200,
            }
        )


def init_aod_x_lines(ax: Axes, aod_x: Sequence[float], plot_params: PlotParameters):
    y_min, y_max = ax.get_ylim()
    return ax.vlines(aod_x, y_min, y_max, **plot_params.aod_line_args)


def init_aod_y_lines(ax: Axes, aod_y: Sequence[float], plot_params: PlotParameters):
    x_min, x_max = ax.get_xlim()
    return ax.hlines(aod_y, x_min, x_max, **plot_params.aod_line_args)


def init_aod_positions(
    ax: Axes,
    aod_x: Sequence[float],
    aod_y: Sequence[float],
    plot_params: PlotParameters,
):
    aod_x_positions, aod_y_positions = np.meshgrid(aod_x, aod_y)
    return ax.scatter(
        aod_x_positions.ravel(),
        aod_y_positions.ravel(),
        **plot_params.aod_marker_args,
    )


def init_moving_atoms(
    ax: Axes,
    moving_atoms_x_indices,
    moving_atoms_y_indices,
    aod_x,
    aod_y,
    plot_params: PlotParameters,
):
    moving_atom_x = list(map(aod_x.__getitem__, moving_atoms_x_indices))
    moving_atom_y = list(map(aod_y.__getitem__, moving_atoms_y_indices))
    return ax.scatter(moving_atom_x, moving_atom_y, **plot_params.atom_plot_args)


def init_stationary_atoms(
    ax: Axes, stationary_atoms_x, stationary_atoms_y, plot_params: PlotParameters
):
    return ax.scatter(
        stationary_atoms_x,
        stationary_atoms_y,
        **plot_params.atom_plot_args,
    )


class Renderer(ABC):

    def max_time(self) -> float:
        raise NotImplementedError

    def update(self, time: float):
        raise NotImplementedError


@dataclass
class AODXLines(Renderer):
    ax: Axes
    aod_x: list[interp1d]
    plot_params: PlotParameters

    def max_time(self) -> float:
        if self.aod_x:
            return max((f.x[-1] for f in self.aod_x), default=0.0)
        return 0.0

    def __post_init__(self):
        x0 = [f(0) for f in self.aod_x]
        self.lines = init_aod_x_lines(self.ax, x0, self.plot_params)

    def update(self, time: float):
        x = [f(time) for f in self.aod_x]
        y_min, y_max = self.ax.get_ylim()
        self.lines.set_segments([[(xi, y_min), (xi, y_max)] for xi in x])


@dataclass
class AODYLines(Renderer):
    ax: Axes
    aod_y: list[interp1d]
    plot_params: PlotParameters

    def max_time(self) -> float:
        if self.aod_y:
            return max((f.x[-1] for f in self.aod_y), default=0.0)
        return 0.0

    def __post_init__(self):
        y0 = [f(0) for f in self.aod_y]
        self.lines = init_aod_y_lines(self.ax, y0, self.plot_params)

    def update(self, time: float):
        y = [f(time) for f in self.aod_y]
        x_min, x_max = self.ax.get_xlim()
        self.lines.set_segments([[(x_min, yi), (x_max, yi)] for yi in y])


@dataclass
class AODPositions:
    ax: Axes
    aod_x: list[interp1d]
    aod_y: list[interp1d]
    plot_params: PlotParameters

    def max_time(self) -> float:
        max_x = max((f.x[-1] for f in self.aod_x), default=0.0) if self.aod_x else 0.0
        max_y = max((f.x[-1] for f in self.aod_y), default=0.0) if self.aod_y else 0.0
        return max(max_x, max_y)

    def __post_init__(self):
        x0 = [f(0) for f in self.aod_x]
        y0 = [f(0) for f in self.aod_y]
        self.positions = init_aod_positions(self.ax, x0, y0, self.plot_params)

    def update(self, time: float):
        x = [f(time) for f in self.aod_x]
        y = [f(time) for f in self.aod_y]
        x_pos, y_pos = np.meshgrid(x, y)
        self.positions.set_offsets(np.column_stack([x_pos.ravel(), y_pos.ravel()]))


@dataclass
class MovingAtomsScatter:
    ax: Axes
    moving_atoms_x_indices: list[int]
    moving_atoms_y_indices: list[int]
    moving_atom_labels: list[int]
    aod_x: list[interp1d]
    aod_y: list[interp1d]
    plot_params: PlotParameters

    def max_time(self) -> float:
        max_x = max((f.x[-1] for f in self.aod_x), default=0.0) if self.aod_x else 0.0
        max_y = max((f.x[-1] for f in self.aod_y), default=0.0) if self.aod_y else 0.0
        return max(max_x, max_y)

    def __post_init__(self):
        x0 = [f(0) for f in self.aod_x]
        y0 = [f(0) for f in self.aod_y]
        moving_atom_x = list(map(x0.__getitem__, self.moving_atoms_x_indices))
        moving_atom_y = list(map(y0.__getitem__, self.moving_atoms_y_indices))
        self.scatter = self.ax.scatter(
            moving_atom_x, moving_atom_y, **self.plot_params.atom_plot_args
        )
        self.labels = [
            self.ax.text(
                x_pos,
                y_pos,
                str(label),
                **self.plot_params.atom_label_args,
            )
            for x_pos, y_pos, label in zip(
                moving_atom_x, moving_atom_y, self.moving_atom_labels
            )
        ]

    def update(self, time: float):
        x = [f(time) for f in self.aod_x]
        y = [f(time) for f in self.aod_y]
        moving_atom_x = list(map(x.__getitem__, self.moving_atoms_x_indices))
        moving_atom_y = list(map(y.__getitem__, self.moving_atoms_y_indices))
        self.scatter.set_offsets(np.column_stack([moving_atom_x, moving_atom_y]))
        for label, x_pos, y_pos in zip(self.labels, moving_atom_x, moving_atom_y):
            label.set_position((x_pos, y_pos))


@dataclass
class StationaryAtomsScatter:
    ax: Axes
    stationary_atoms_x: list[float]
    stationary_atoms_y: list[float]
    stationary_atom_labels: list[int]
    plot_params: PlotParameters

    def max_time(self) -> float:
        return 0.0

    def __post_init__(self):
        self.scatter = init_stationary_atoms(
            self.ax, self.stationary_atoms_x, self.stationary_atoms_y, self.plot_params
        )
        self.labels = [
            self.ax.text(
                x_pos,
                y_pos,
                str(label),
                **self.plot_params.atom_label_args,
            )
            for x_pos, y_pos, label in zip(
                self.stationary_atoms_x,
                self.stationary_atoms_y,
                self.stationary_atom_labels,
            )
        ]

    def update(self, time: float):
        pass  # Stationary atoms do not change position


@dataclass
class MoveRenderer:
    renderers: list[Renderer]

    @property
    def total_time(self) -> float:
        # Return the max time from all renderers, always as float
        return float(max((r.max_time() for r in self.renderers), default=0.0))

    def update(self, time: float):
        for renderer in self.renderers:
            renderer.update(time)


@dataclass
class StateArtist:
    ax: Axes
    arch_spec: ArchSpec
    plot_params: PlotParameters
    x_min: float
    x_max: float
    y_min: float
    y_max: float

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

    def _draw_atom_labels(
        self,
        x_positions: Sequence[float],
        y_positions: Sequence[float],
        atom_labels: Sequence[int],
    ):
        for x_pos, y_pos, atom_label in zip(x_positions, y_positions, atom_labels):
            self.ax.text(
                x_pos,
                y_pos,
                str(atom_label),
                **self.plot_params.atom_label_args,
            )

    def _get_moving_atom_data(
        self, move_execution: AtomState, last_xs: list[float], last_ys: list[float]
    ) -> tuple[list[int], list[int], list[int]]:
        moving_atom_data: list[tuple[int, int, int]] = []
        for qubit in move_execution.data.prev_lanes.keys():
            loc = move_execution.data.qubit_to_locations[qubit]
            pos = self.arch_spec.get_position(loc)
            assert (
                pos is not None
            ), f"location {loc!r} has no position in this architecture"
            x, y = pos
            if x in last_xs and y in last_ys:
                moving_atom_data.append((last_xs.index(x), last_ys.index(y), qubit))

        if len(moving_atom_data) == 0:
            return [], [], []

        x_indices, y_indices, labels = zip(*moving_atom_data)
        return list(x_indices), list(y_indices), list(labels)

    def _get_stationary_positions(
        self, move_execution: AtomState
    ) -> tuple[list[float], list[float], list[int]]:
        stationary_atom_data = []
        for qubit, location in move_execution.data.qubit_to_locations.items():
            if qubit in move_execution.data.prev_lanes.keys():
                continue
            pos = self.arch_spec.get_position(location)
            assert (
                pos is not None
            ), f"location {location!r} has no position in this architecture"
            stationary_atom_data.append((*pos, qubit))
        if len(stationary_atom_data) == 0:
            return [], [], []

        stationary_atom_positions_x, stationary_atom_positions_y, labels = zip(
            *stationary_atom_data
        )
        return (
            list(stationary_atom_positions_x),
            list(stationary_atom_positions_y),
            list(labels),
        )

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
        moving_atom_indices_x, moving_atom_indices_y, moving_atom_labels = (
            self._get_moving_atom_data(move_execution, first_xs, first_ys)
        )
        stationary_atom_positions_x, stationary_atom_positions_y, stationary_labels = (
            self._get_stationary_positions(move_execution)
        )
        renderers = [
            AODXLines(self.ax, aod_x_funcs, self.plot_params),
            AODYLines(self.ax, aod_y_funcs, self.plot_params),
            AODPositions(self.ax, aod_x_funcs, aod_y_funcs, self.plot_params),
            MovingAtomsScatter(
                self.ax,
                moving_atom_indices_x,
                moving_atom_indices_y,
                moving_atom_labels,
                aod_x_funcs,
                aod_y_funcs,
                self.plot_params,
            ),
            StationaryAtomsScatter(
                self.ax,
                stationary_atom_positions_x,
                stationary_atom_positions_y,
                stationary_labels,
                self.plot_params,
            ),
        ]
        return MoveRenderer(renderers)

    def draw_atoms(
        self,
        state: MoveExecution,
    ):
        if not isinstance(state, AtomState):
            return

        atoms = list(state.data.locations_to_qubit.items())
        locations = [location for location, _ in atoms]
        labels = [qubit for _, qubit in atoms]

        x, y = (
            zip(*map(self.arch_spec.get_position, locations))
            if len(locations) > 0
            else ([], [])
        )
        self.ax.scatter(x, y, **self.plot_params.atom_plot_args)
        self._draw_atom_labels(x, y, labels)

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
        positions = []
        for location in stmt.location_addresses:
            pos = self.arch_spec.get_position(location)
            assert (
                pos is not None
            ), f"location {location!r} has no position in this architecture"
            positions.append(pos)
        x_pos, y_pos = zip(*positions) if len(positions) > 0 else ([], [])
        self.ax.scatter(x_pos, y_pos, color=color, **self.plot_params.gate_spot_args)

    def show_local_r(self, stmt: move.LocalR):
        self._show_local(stmt, color=self.plot_params.local_r_color)

    def show_local_rz(self, stmt: move.LocalRz):
        self._show_local(stmt, color=self.plot_params.local_rz_color)

    def _show_global(self, stmt: move.GlobalR | move.GlobalRz, color: str):
        _viz = ArchVisualizer(self.arch_spec)
        x_min, x_max = _viz.x_bounds
        x_width = x_max - x_min
        x_min -= 0.5 * x_width
        x_max += 0.5 * x_width

        y_min, y_max = _viz.y_bounds
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
        self._show_global(stmt, color=self.plot_params.global_r_color)

    def show_global_rz(self, stmt: move.GlobalRz):
        self._show_global(stmt, color=self.plot_params.global_rz_color)

    def show_cz(self, stmt: move.CZ):
        zone_id = stmt.zone_address.zone_id

        y_min = float("inf")
        y_max = float("-inf")

        for word_id in range(len(self.arch_spec.words)):
            word = self.arch_spec.words[word_id]
            for site_id in range(len(word.site_indices)):
                from bloqade.lanes.layout.encoding import LocationAddress

                loc = LocationAddress(word_id, site_id, zone_id)
                pos = self.arch_spec.get_position(loc)
                assert (
                    pos is not None
                ), f"location {loc!r} has no position in this architecture"
                y_min = min(y_min, pos[1])
                y_max = max(y_max, pos[1])

        x_min, x_max = ArchVisualizer(self.arch_spec).x_bounds
        y_width = y_max - y_min
        y_min -= 0.1 * y_width
        y_max += 0.1 * y_width

        self.ax.fill_between(
            [x_min - 10, x_max + 10],
            [y_min, y_min],
            [y_max, y_max],
            color=self.plot_params.cz_color,
            alpha=0.3,
        )

    def show_slm(self, stmt: ir.Statement, atom_marker: str):
        ArchVisualizer(self.arch_spec).plot(
            self.ax,
            show_words=range(len(self.arch_spec.words)),
            **self.plot_params.slm_plot_args,
        )


def get_state_artist(
    arch_spec: ArchSpec, ax: Axes, atom_marker: str = "o"
) -> StateArtist:
    x_min, x_max, y_min, y_max = ArchVisualizer(arch_spec).path_bounds()
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

    return StateArtist(
        ax,
        arch_spec,
        plot_params=plot_params,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


def get_drawer(mt: ir.Method, arch_spec: ArchSpec, ax: Axes, atom_marker: str = "o"):
    artist = get_state_artist(arch_spec, ax, atom_marker)

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

        stmt, _ = steps[step_index]
        ax.set_title(f"Step {step_index+1} / {len(steps)}: {stmt_text(stmt)}")
        ax.set_aspect("equal", adjustable="box")
        stmt, curr_state = steps[step_index]
        artist.show_slm(stmt, atom_marker)

        visualize_fn = methods.get(type(stmt), lambda stmt: None)
        visualize_fn(stmt)
        artist.draw_atoms(curr_state)
        artist.draw_moves(curr_state)
        ax.set_xlim(artist.x_min, artist.x_max)
        ax.set_ylim(artist.y_min, artist.y_max)
        plt.draw()

    return draw, len(steps)


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
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(artist.x_min, artist.x_max)
        ax.set_ylim(artist.y_min, artist.y_max)

        visualize_fn = methods.get(type(stmt))
        if visualize_fn is not None:
            artist.draw_atoms(curr_state)
            visualize_fn(stmt)
            return 3 * fps, _no_op

        move_renderer = artist.move_renderer(curr_state, speed=2.0)
        if move_renderer is not None:
            operation_time = min(5.0, max(1.0, move_renderer.total_time))
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
