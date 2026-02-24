from unittest.mock import MagicMock

import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_fig
import matplotlib.pyplot as plt
import matplotlib.widgets as mpl_widgets
import pytest

from bloqade.lanes.visualize.app import DebuggerController


class DummyController(DebuggerController):
    def __init__(self):
        self.run_called = False
        self.exit_called = False
        self.next_called = False
        self.prev_called = False
        self.key_called = False
        self.reset_called = False

    def run(self):
        self.run_called = True

    def on_exit(self, event):
        self.exit_called = True

    def on_next(self, event):
        self.next_called = True

    def on_prev(self, event):
        self.prev_called = True

    def on_key(self, event):
        self.key_called = True

    def reset(self):
        self.reset_called = True

    def run_mpl_event_loop(self, ax, fig):
        self.run()
        self.reset()


def test_protocol_methods():
    controller = DummyController()
    controller.run()
    controller.on_exit(None)
    controller.on_next(None)
    controller.on_prev(None)
    controller.on_key(MagicMock(key="left"))
    controller.reset()
    assert controller.run_called
    assert controller.exit_called
    assert controller.next_called
    assert controller.prev_called
    assert controller.key_called
    assert controller.reset_called


def test_run_mpl_event_loop_calls_methods(monkeypatch):
    controller = DummyController()
    ax = MagicMock(spec=mpl_axes.Axes)
    fig = MagicMock(spec=mpl_fig.Figure)
    monkeypatch.setattr(
        controller, "run", lambda: setattr(controller, "run_called", True)
    )
    monkeypatch.setattr(
        controller, "reset", lambda: setattr(controller, "reset_called", True)
    )
    controller.run_mpl_event_loop(ax, fig)
    assert controller.run_called
    assert controller.reset_called


class TestableController(DummyController):
    def __init__(self):
        super().__init__()
        self.on_next_event = None
        self.on_prev_event = None
        self.on_exit_event = None
        self.on_key_event = None
        self.reset_event = False

    def on_next(self, event):
        self.on_next_event = event
        super().on_next(event)

    def on_prev(self, event):
        self.on_prev_event = event
        super().on_prev(event)

    def on_exit(self, event):
        self.on_exit_event = event
        super().on_exit(event)

    def on_key(self, event):
        self.on_key_event = event
        super().on_key(event)

    def reset(self):
        self.reset_event = True
        super().reset()


def test_run_mpl_event_loop_button_callbacks(monkeypatch):
    controller = TestableController()
    ax = MagicMock(spec=mpl_axes.Axes)
    fig = MagicMock(spec=mpl_fig.Figure)
    # Patch add_axes to return a MagicMock for each button
    fig.add_axes = MagicMock(side_effect=[MagicMock(), MagicMock(), MagicMock()])
    # Patch Button to allow callback registration
    monkeypatch.setattr(
        mpl_widgets, "Button", lambda ax, label: MagicMock(on_clicked=MagicMock())
    )
    # Patch fig.canvas.mpl_connect
    fig.canvas = MagicMock()
    fig.canvas.mpl_connect = MagicMock()
    monkeypatch.setattr(
        controller, "run", lambda: setattr(controller, "run_called", True)
    )
    monkeypatch.setattr(
        controller, "reset", lambda: setattr(controller, "reset_called", True)
    )
    # Patch plt.close to avoid closing figures
    monkeypatch.setattr("matplotlib.pyplot.close", lambda fig: None)
    DebuggerController.run_mpl_event_loop(controller, ax, fig)
    assert controller.run_called
    assert controller.reset_called
    fig.add_axes.assert_called()
    fig.canvas.mpl_connect.assert_called_with("key_press_event", controller.on_key)


def test_on_key_dispatch(monkeypatch):
    controller = TestableController()
    # Patch prev/next/exit to track calls
    monkeypatch.setattr(
        controller, "on_prev", lambda event: setattr(controller, "prev_called", True)
    )
    monkeypatch.setattr(
        controller, "on_next", lambda event: setattr(controller, "next_called", True)
    )
    monkeypatch.setattr(
        controller, "on_exit", lambda event: setattr(controller, "exit_called", True)
    )

    # Simulate key events
    class Event:
        def __init__(self, key: str):
            self.key = key

    left_event = Event("left")
    right_event = Event("right")
    escape_event = Event("escape")
    DebuggerController.on_key(controller, left_event)
    assert controller.prev_called
    DebuggerController.on_key(controller, right_event)
    assert controller.next_called
    DebuggerController.on_key(controller, escape_event)
    assert controller.exit_called


def test_debuggercontroller_notimplemented():
    class IncompleteController(DebuggerController):
        pass

    ctrl = IncompleteController()  # type: ignore
    with pytest.raises(NotImplementedError):
        ctrl.run()
    with pytest.raises(NotImplementedError):
        ctrl.on_exit(None)
    with pytest.raises(NotImplementedError):
        ctrl.on_next(None)
    with pytest.raises(NotImplementedError):
        ctrl.on_prev(None)
    with pytest.raises(NotImplementedError):
        ctrl.reset()


def test_run_mpl_event_loop_closes_figure(monkeypatch):
    controller = DummyController()
    ax = MagicMock(spec=mpl_axes.Axes)
    fig = MagicMock(spec=mpl_fig.Figure)
    fig.add_axes = MagicMock(side_effect=[MagicMock(), MagicMock(), MagicMock()])
    fig.canvas = MagicMock()
    fig.canvas.mpl_connect = MagicMock()
    monkeypatch.setattr(
        "matplotlib.widgets.Button", lambda ax, label: MagicMock(on_clicked=MagicMock())
    )
    monkeypatch.setattr(
        controller, "run", lambda: setattr(controller, "run_called", True)
    )
    monkeypatch.setattr(
        controller, "reset", lambda: setattr(controller, "reset_called", True)
    )
    close_called = {}

    def fake_close(f):
        close_called["closed"] = f

    monkeypatch.setattr(plt, "close", fake_close)
    DebuggerController.run_mpl_event_loop(controller, ax, fig)
    assert close_called["closed"] == fig


def test_run_mpl_event_loop_subfigure(monkeypatch):
    controller = DummyController()
    ax = MagicMock(spec=mpl_axes.Axes)

    class SubFigure:
        def add_axes(self, *args, **kwargs):
            return MagicMock()

        @property
        def canvas(self):
            return MagicMock()

    fig = SubFigure()
    fig.canvas.mpl_connect = MagicMock()
    monkeypatch.setattr(
        "matplotlib.widgets.Button", lambda ax, label: MagicMock(on_clicked=MagicMock())
    )
    monkeypatch.setattr(
        controller, "run", lambda: setattr(controller, "run_called", True)
    )
    monkeypatch.setattr(
        controller, "reset", lambda: setattr(controller, "reset_called", True)
    )
    DebuggerController.run_mpl_event_loop(controller, ax, fig)  # type: ignore
    assert controller.run_called
    assert controller.reset_called
