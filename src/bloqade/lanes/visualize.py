from itertools import chain

from kirin import ir
from matplotlib import figure, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Button

from bloqade.lanes.analysis.atom import AtomInterpreter, AtomState, MoveExecution, Value
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.arch import ArchSpec


class StateArtist:

    def draw_atoms(
        self,
        state: MoveExecution,
        arch_spec: ArchSpec,
        ax: Axes | None = None,
        **kwargs,
    ):
        import matplotlib.pyplot as plt

        if not isinstance(state, AtomState):
            return

        if ax is None:
            ax = plt.gca()

        for location in state.data.locations_to_qubit:
            x_pos, y_pos = zip(
                *arch_spec.words[location.word_id].site_positions(location.site_id)
            )
            ax.scatter(x_pos, y_pos, **kwargs)

    def draw_moves(
        self,
        state: MoveExecution,
        arch_spec: ArchSpec,
        ax: Axes | None = None,
        **kwargs,
    ):
        import matplotlib.pyplot as plt

        if not isinstance(state, AtomState):
            return

        if ax is None:
            ax = plt.gca()

        for lane in state.data.prev_lanes.values():
            start, end = arch_spec.get_endpoints(lane)
            start_pos = arch_spec.words[start.word_id].site_positions(start.site_id)
            end_pos = arch_spec.words[end.word_id].site_positions(end.site_id)
            for (x_start, y_start), (x_end, y_end) in zip(start_pos, end_pos):
                ax.quiver(
                    [x_start],
                    [y_start],
                    [x_end - x_start],
                    [y_end - y_start],
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    **kwargs,
                )


def show_local(
    ax: Axes, stmt: move.LocalR | move.LocalRz, arch_spec: ArchSpec, color: str
):
    positions = chain.from_iterable(
        arch_spec.words[location.word_id].site_positions(location.site_id)
        for location in stmt.location_addresses
    )
    x_pos, y_pos = zip(*positions)
    ax.plot(
        x_pos, y_pos, color=color, marker="o", linestyle="", alpha=0.3, markersize=15
    )


def show_local_r(ax: Axes, stmt: move.LocalR, arch_spec: ArchSpec):
    show_local(ax, stmt, arch_spec, color="blue")


def show_local_rz(ax: Axes, stmt: move.LocalRz, arch_spec: ArchSpec):
    show_local(ax, stmt, arch_spec, color="green")


def show_global(
    ax: Axes, stmt: move.GlobalR | move.GlobalRz, arch_spec: ArchSpec, color: str
):
    x_min, x_max = arch_spec.x_bounds
    x_width = x_max - x_min
    x_min -= 0.5 * x_width
    x_max += 0.5 * x_width

    y_min, y_max = arch_spec.y_bounds
    y_width = y_max - y_min
    y_min -= 0.5 * y_width
    y_max += 0.5 * y_width

    ax.fill_between(
        [x_min, x_max],
        [y_min, y_min],
        [y_max, y_max],
        color=color,
        alpha=0.3,
    )


def show_global_r(ax: Axes, stmt: move.GlobalR, arch_spec: ArchSpec):
    show_global(ax, stmt, arch_spec, color="blue")


def show_global_rz(ax: Axes, stmt: move.GlobalRz, arch_spec: ArchSpec):
    show_global(ax, stmt, arch_spec, color="green")


def show_cz(ax: Axes, stmt: move.CZ, arch_spec: ArchSpec):
    words = tuple(
        arch_spec.words[word_id]
        for word_id in arch_spec.zones[stmt.zone_address.zone_id]
    )

    y_min = float("inf")
    y_max = float("-inf")

    for word in words:
        for site_id in range(len(word.sites)):
            _, y_pos = zip(*word.site_positions(site_id))
            y_min = min(y_min, min(y_pos))
            y_max = max(y_max, max(y_pos))

    x_min, x_max = arch_spec.x_bounds
    y_width = y_max - y_min
    y_min -= 0.1 * y_width
    y_max += 0.1 * y_width

    ax.fill_between(
        [x_min - 10, x_max + 10],
        [y_min, y_min],
        [y_max, y_max],
        color="red",
        alpha=0.3,
    )


def show_slm(ax, stmt: ir.Statement, arch_spec: ArchSpec, atom_marker: str):
    slm_plt_arg: dict = {
        "facecolors": "none",
        "edgecolors": "k",
        "linestyle": "-",
        "s": 80,
        "alpha": 0.3,
        "linewidth": 0.5,
        "marker": atom_marker,
    }
    arch_spec.plot(ax, show_words=range(len(arch_spec.words)), **slm_plt_arg)


def get_drawer(mt: ir.Method, arch_spec: ArchSpec, ax: Axes, atom_marker: str = "o"):
    methods: dict = {
        move.LocalR: show_local_r,
        move.LocalRz: show_local_rz,
        move.GlobalR: show_global_r,
        move.GlobalRz: show_global_rz,
        move.CZ: show_cz,
    }

    frame, _ = AtomInterpreter(mt.dialects, arch_spec=arch_spec).run(mt)

    x_min, x_max = arch_spec.x_bounds
    y_min, y_max = arch_spec.y_bounds

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

    artist = StateArtist()

    def draw(step_index: int):
        if len(steps) == 0:
            return
        stmt, curr_state = steps[step_index]
        show_slm(ax, stmt, arch_spec, atom_marker)

        visualize_fn = methods.get(type(stmt), lambda a, b, c: None)
        visualize_fn(ax, stmt, arch_spec)
        artist.draw_atoms(
            curr_state, arch_spec, ax=ax, color="#6437FF", s=80, marker=atom_marker
        )
        artist.draw_moves(curr_state, arch_spec, ax=ax, color="orange")

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

    prev_button = Button(prev_ax, "Previous")
    next_button = Button(next_ax, "Next")
    exit_button = Button(exit_ax, "Exit")

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
