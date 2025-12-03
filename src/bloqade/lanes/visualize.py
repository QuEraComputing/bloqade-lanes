from itertools import chain

from kirin import ir
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from bloqade.lanes.analysis.atom.analysis import AtomInterpreter, AtomState
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import MoveType


def show_move(ax: Axes, stmt: move.Move, arch_spec: ArchSpec):
    word_bus = []
    site_bus = []

    for lane in stmt.lanes:
        if lane.move_type == MoveType.WORD:
            word_bus.append(lane.bus_id)
        else:
            site_bus.append(lane.bus_id)

    arch_spec.plot(
        ax,
        show_words=range(len(arch_spec.words)),
        show_word_bus=word_bus,
        show_site_bus=site_bus,
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
    default(ax, stmt, arch_spec)


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
    default(ax, stmt, arch_spec)


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
        color="purple",
        alpha=0.3,
    )

    default(ax, stmt, arch_spec)


def default(ax, stmt: ir.Statement, arch_spec: ArchSpec):
    arch_spec.plot(ax, show_words=range(len(arch_spec.words)))


def animate(mt: ir.Method, arch_spec: ArchSpec, pause_time: float = 1.0):

    methods: dict = {
        move.Move: show_move,
        move.LocalR: show_local_r,
        move.LocalRz: show_local_rz,
        move.GlobalR: show_global_r,
        move.GlobalRz: show_global_rz,
        move.CZ: show_cz,
    }

    frame, _ = AtomInterpreter(mt.dialects, arch_spec=arch_spec).run(mt)
    prev_state = None

    x_min, x_max = arch_spec.x_bounds
    y_min, y_max = arch_spec.y_bounds

    x_width = x_max - x_min
    y_width = y_max - y_min

    x_min -= 0.1 * x_width
    x_max += 0.1 * x_width
    y_min -= 0.1 * y_width
    y_max += 0.1 * y_width

    for stmt in mt.callable_region.walk():
        curr_state = frame.atom_state_map.get(stmt)

        if not isinstance(curr_state, AtomState):
            continue

        ax = plt.gca()

        if prev_state is None:
            prev_state = curr_state
            continue

        visualize_fn = methods.get(type(stmt), default)
        visualize_fn(ax, stmt, arch_spec)

        if isinstance(curr_state, AtomState):
            curr_state.plot(arch_spec, color="grey", ax=ax, s=150)
            prev_state.plot(arch_spec, color="grey", ax=ax, alpha=0.25, s=150)

            prev_state = curr_state

        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))

        plt.pause(pause_time)
        plt.cla()
