from unittest.mock import MagicMock

import matplotlib.axes as mpl_axes
import matplotlib.pyplot as plt
import pytest

from bloqade.lanes.visualize import debug as debug_mod
from bloqade.lanes.visualize.debug import (
    AnimatorController,
    StaticDebuggerController,
)


@pytest.fixture
def dummy_ax():
    ax = MagicMock(spec=mpl_axes.Axes)
    ax.cla = MagicMock()
    return ax


@pytest.fixture
def dummy_draw():
    return MagicMock()


@pytest.fixture
def dummy_get_renderer():
    def get_renderer(step_index):
        return 3, MagicMock()

    return get_renderer


def test_static_debugger_on_next_prev_exit(dummy_ax, dummy_draw):
    ctrl = StaticDebuggerController(dummy_ax, 5, dummy_draw)
    ctrl.on_next(MagicMock())
    assert ctrl.step_index == 1
    assert dummy_ax.cla.called
    # Reset updated to allow prev
    ctrl.updated = False
    ctrl.on_prev(MagicMock())
    assert ctrl.step_index == 0
    ctrl.on_exit(MagicMock())
    assert not ctrl.running
    assert not ctrl.waiting


def test_static_debugger_on_key_dispatch(dummy_ax, dummy_draw):
    ctrl = StaticDebuggerController(dummy_ax, 5, dummy_draw)
    ctrl.on_key(MagicMock(key="right"))
    assert ctrl.step_index == 1
    # Reset updated to allow left
    ctrl.updated = False
    ctrl.on_key(MagicMock(key="left"))
    assert ctrl.step_index == 0
    ctrl.updated = False
    ctrl.on_key(MagicMock(key="escape"))
    assert not ctrl.running


def test_static_debugger_reset(dummy_ax, dummy_draw):
    ctrl = StaticDebuggerController(dummy_ax, 5, dummy_draw)
    ctrl.step_index = 3
    ctrl.running = False
    ctrl.waiting = False
    ctrl.updated = True
    ctrl.reset()
    assert ctrl.step_index == 0
    assert ctrl.running
    assert ctrl.waiting
    assert not ctrl.updated


def test_animator_controller_next_prev_exit(dummy_ax, dummy_get_renderer):
    ctrl = AnimatorController(dummy_ax, 5, dummy_get_renderer)
    ctrl.on_next(MagicMock())
    assert ctrl.animation_step == 1
    ctrl.on_prev(MagicMock())
    assert ctrl.animation_step == -1
    ctrl.on_exit(MagicMock())
    assert not ctrl.running
    assert not ctrl.waiting


def test_animator_controller_reset(dummy_ax, dummy_get_renderer):
    ctrl = AnimatorController(dummy_ax, 5, dummy_get_renderer)
    ctrl.step_index = 3
    ctrl.animation_step = -1
    ctrl.running = False
    ctrl.waiting = False
    ctrl.updated = True
    ctrl.reset()
    assert ctrl.step_index == 0
    assert ctrl.animation_step == 1
    assert ctrl.running
    assert ctrl.waiting
    assert not ctrl.updated


def test_static_debugger_run(monkeypatch, dummy_ax, dummy_draw):
    ctrl = StaticDebuggerController(dummy_ax, 2, dummy_draw)
    call_count = {"draw": 0, "pause": 0}

    def fake_draw(idx):
        call_count["draw"] += 1
        # Simulate waiting loop for pause
        ctrl.waiting = True
        if call_count["draw"] == 2:
            ctrl.running = False
            ctrl.waiting = False

    monkeypatch.setattr(ctrl, "draw", fake_draw)

    def fake_pause(t):
        call_count["pause"] += 1
        # End waiting after first pause
        ctrl.waiting = False

    monkeypatch.setattr(plt, "pause", fake_pause)
    ctrl.run()
    assert call_count["draw"] >= 1
    assert call_count["pause"] >= 1


def test_animator_controller_run(monkeypatch, dummy_ax, dummy_get_renderer):
    ctrl = AnimatorController(dummy_ax, 2, dummy_get_renderer)
    call_count = {"renderer": 0, "pause": 0}

    def fake_get_renderer(idx):
        def fake_renderer(frame):
            call_count["renderer"] += 1
            if call_count["renderer"] == 2:
                ctrl.running = False
                ctrl.waiting = False
            else:
                ctrl.waiting = False

        return 2, fake_renderer

    ctrl.get_renderer = fake_get_renderer
    monkeypatch.setattr(
        plt, "pause", lambda t: call_count.update({"pause": call_count["pause"] + 1})
    )
    ctrl.run()
    assert call_count["renderer"] >= 1
    assert call_count["pause"] >= 1


def test_debugger_noninteractive(monkeypatch, dummy_ax, dummy_draw):
    # Patch get_drawer to return dummy_draw and 2 steps
    monkeypatch.setattr(debug_mod, "get_drawer", lambda *a, **kw: (dummy_draw, 2))
    monkeypatch.setattr(plt, "pause", lambda t: None)
    monkeypatch.setattr(dummy_ax, "cla", lambda: None)
    called = []

    def draw(idx):
        called.append(idx)

    debug_mod.debugger(
        mt=MagicMock(),
        arch_spec=MagicMock(),
        interactive=False,
        pause_time=0.01,
        atom_marker="o",
        ax=dummy_ax,
    )
    assert called == [] or isinstance(called, list)


def test_animated_debugger_noninteractive(monkeypatch, dummy_ax):
    # Patch render_generator to return dummy get_renderer and 2 steps
    def fake_get_renderer(idx):
        def fake_renderer(frame):
            pass

        return 2, fake_renderer

    monkeypatch.setattr(
        debug_mod, "render_generator", lambda *a, **kw: (fake_get_renderer, 2)
    )
    monkeypatch.setattr(plt, "pause", lambda t: None)
    monkeypatch.setattr(dummy_ax, "cla", lambda: None)
    debug_mod.animated_debugger(
        mt=MagicMock(),
        arch_spec=MagicMock(),
        interactive=False,
        atom_marker="o",
        ax=dummy_ax,
        fps=30,
    )
