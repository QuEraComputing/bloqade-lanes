from kirin import ir
from matplotlib import pyplot as plt

from bloqade.lanes.analysis.atom.analysis import AtomInterpreter, AtomState
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import MoveType


def show_move(ax, stmt: move.Move, arch_spec: ArchSpec):
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


def show_local_r(ax, stmt: move.LocalR, arch_spec: ArchSpec):
    arch_spec.plot(ax, show_words=range(len(arch_spec.words)))


def show_local_rz(ax, stmt: move.LocalRz, arch_spec: ArchSpec):
    arch_spec.plot(ax, show_words=range(len(arch_spec.words)))


def show_global_r(ax, stmt: move.GlobalR, arch_spec: ArchSpec):
    arch_spec.plot(ax, show_words=range(len(arch_spec.words)))


def show_global_rz(ax, stmt: move.GlobalRz, arch_spec: ArchSpec):
    arch_spec.plot(ax, show_words=range(len(arch_spec.words)))


def show_cz(ax, stmt: move.CZ, arch_spec: ArchSpec):
    arch_spec.plot(ax, show_words=range(len(arch_spec.words)))


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
            curr_state.plot(arch_spec, atom_color="red", ax=ax)
            prev_state.plot(arch_spec, atom_color="green", ax=ax)

            prev_state = curr_state

        plt.pause(pause_time)
        plt.cla()
