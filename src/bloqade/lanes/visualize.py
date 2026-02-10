from dataclasses import dataclass

from kirin import ir
from matplotlib import figure, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Button

from bloqade.lanes.analysis.atom import AtomInterpreter, AtomState, MoveExecution, Value
from bloqade.lanes.dialects import move
from bloqade.lanes.layout import ArchSpec


@dataclass
class StateArtist:
    ax: Axes
    arch_spec: ArchSpec

    def draw_atoms(
        self,
        state: MoveExecution,
        **kwargs,
    ):
        if not isinstance(state, AtomState):
            return

        locations = list(state.data.locations_to_qubit)

        x, y = (
            zip(*map(self.arch_spec.get_position, locations))
            if len(locations) > 0
            else ([], [])
        )
        self.ax.scatter(x, y, **kwargs)

    def draw_moves(
        self,
        state: MoveExecution,
        **kwargs,
    ):
        if not isinstance(state, AtomState):
            return

        for lane in state.data.prev_lanes.values():
            path = self.arch_spec.get_path(lane)
            for (x_start, y_start), (x_end, y_end) in zip(path, path[1:]):
                self.ax.quiver(
                    [x_start],
                    [y_start],
                    [x_end - x_start],
                    [y_end - y_start],
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    **kwargs,
                )

    def _show_local(self, stmt: move.LocalR | move.LocalRz, color: str):
        positions = (
            self.arch_spec.words[location.word_id].site_position(location.site_id)
            for location in stmt.location_addresses
        )
        x_pos, y_pos = zip(*positions)
        self.ax.plot(
            x_pos,
            y_pos,
            color=color,
            marker="o",
            linestyle="",
            alpha=0.3,
            markersize=15,
        )

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
        slm_plt_arg: dict = {
            "facecolors": "none",
            "edgecolors": "k",
            "linestyle": "-",
            "s": 80,
            "alpha": 0.3,
            "linewidth": 0.5,
            "marker": atom_marker,
        }
        self.arch_spec.plot(
            self.ax, show_words=range(len(self.arch_spec.words)), **slm_plt_arg
        )


def get_drawer(mt: ir.Method, arch_spec: ArchSpec, ax: Axes, atom_marker: str = "o"):
    artist = StateArtist(ax, arch_spec)

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
        if len(stmt.args) == 0:
            return f"{type(stmt).__name__}"
        return (
            f"{type(stmt).__name__}("
            + ", ".join(f"{constants.get(arg,'missing')}" for arg in stmt.args)
            + ")"
        )

    def draw(step_index: int):
        if len(steps) == 0:
            return
        stmt, curr_state = steps[step_index]
        artist.show_slm(stmt, atom_marker)

        visualize_fn = methods.get(type(stmt), lambda stmt: None)
        visualize_fn(stmt)
        artist.draw_atoms(curr_state, color="#6437FF", s=80, marker=atom_marker)
        artist.draw_moves(curr_state, color="orange")

        ax.set_title(f"Step {step_index+1} / {len(steps)}: {stmt_text(stmt)}")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        plt.draw()

    return draw, len(steps)


def interactive_debugger(
    draw,
    num_steps: int,
    fig: figure.Figure,
):

    ax = plt.gca()
    step_index = 0
    running = True
    waiting = True
    updated = False

    prev_ax = fig.add_axes((0.01, 0.01, 0.1, 0.075))
    exit_ax = fig.add_axes((0.21, 0.01, 0.1, 0.075))
    next_ax = fig.add_axes((0.41, 0.01, 0.1, 0.075))

    prev_button = Button(prev_ax, "Previous (<)")
    next_button = Button(next_ax, "Next (>)")
    exit_button = Button(exit_ax, "Exit (Esc)")

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

    plt.close(fig)


def debugger(
    mt: ir.Method,
    arch_spec: ArchSpec,
    interactive: bool = True,
    pause_time: float = 1.0,
    atom_marker: str = "o",
):
    # set up matplotlib figure with buttons
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)

    draw, num_steps = get_drawer(mt, arch_spec, ax, atom_marker)

    if interactive:
        interactive_debugger(draw, num_steps, fig)
    else:
        for step_index in range(num_steps):
            draw(step_index)
            plt.pause(pause_time)
            ax.cla()
