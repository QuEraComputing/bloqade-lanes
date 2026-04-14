from unittest.mock import MagicMock

import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_fig
import matplotlib.pyplot as plt
import matplotlib.widgets as mpl_widgets
import pytest

from bloqade.lanes.visualize.app import DebuggerController


class DummyController(DebuggerController):
    def __init__(self):
        self.run_called: bool = False
        self.exit_called: bool = False
        self.next_called: bool = False
        self.prev_called: bool = False
        self.key_called: bool = False
        self.reset_called: bool = False

    def run(self) -> None:
        self.run_called = True

    def on_exit(self, event: object) -> None:
        self.exit_called = True

    def on_next(self, event: object) -> None:
        self.next_called = True

    def on_prev(self, event: object) -> None:
        self.prev_called = True

    def on_key(self, event: object) -> None:
        self.key_called = True

    def reset(self) -> None:
        self.reset_called = True

    def run_mpl_event_loop(
        self, ax: mpl_axes.Axes, fig: mpl_fig.Figure | mpl_fig.SubFigure
    ) -> None:
        self.run()
        self.reset()


def test_protocol_methods() -> None:
    controller: DummyController = DummyController()
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


def test_run_mpl_event_loop_calls_methods(monkeypatch) -> None:
    controller: DummyController = DummyController()
    ax: mpl_axes.Axes = MagicMock(spec=mpl_axes.Axes)
    fig: mpl_fig.Figure = MagicMock(spec=mpl_fig.Figure)
    monkeypatch.setattr(
        controller, "run", lambda: setattr(controller, "run_called", True)
    )
    monkeypatch.setattr(
        controller, "reset", lambda: setattr(controller, "reset_called", True)
    )
    controller.run_mpl_event_loop(ax, fig)
    assert controller.run_called
    assert controller.reset_called


@pytest.mark.usefixtures("monkeypatch")
class _TestableController(DummyController):
    def __init__(self):
        super().__init__()
        self.on_next_event: object | None = None
        self.on_prev_event: object | None = None
        self.on_exit_event: object | None = None
        self.on_key_event: object | None = None
        self.reset_event: bool = False

    def on_next(self, event: object) -> None:
        self.on_next_event = event
        super().on_next(event)

    def on_prev(self, event: object) -> None:
        self.on_prev_event = event
        super().on_prev(event)

    def on_exit(self, event: object) -> None:
        self.on_exit_event = event
        super().on_exit(event)

    def on_key(self, event: object) -> None:
        self.on_key_event = event
        super().on_key(event)

    def reset(self) -> None:
        self.reset_event = True
        super().reset()


def test_run_mpl_event_loop_button_callbacks(monkeypatch) -> None:
    controller: _TestableController = _TestableController()
    ax: mpl_axes.Axes = MagicMock(spec=mpl_axes.Axes)
    fig: mpl_fig.Figure = MagicMock(spec=mpl_fig.Figure)
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

    controller = _TestableController()
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
        def __init__(self, key: str) -> None:
            self.key: str = key

    left_event: Event = Event("left")
    right_event: Event = Event("right")
    escape_event: Event = Event("escape")
    DebuggerController.on_key(controller, left_event)
    assert controller.prev_called
    DebuggerController.on_key(controller, right_event)
    assert controller.next_called
    DebuggerController.on_key(controller, escape_event)
    assert controller.exit_called


def test_debuggercontroller_notimplemented() -> None:
    class IncompleteController(DebuggerController):
        pass

    ctrl: DebuggerController = IncompleteController()  # type: ignore
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


def test_run_mpl_event_loop_closes_figure(monkeypatch) -> None:
    controller: DummyController = DummyController()
    ax: mpl_axes.Axes = MagicMock(spec=mpl_axes.Axes)
    fig: mpl_fig.Figure = MagicMock(spec=mpl_fig.Figure)
    fig.add_axes = MagicMock(side_effect=[MagicMock(), MagicMock(), MagicMock()])
    fig.canvas = MagicMock()
    fig.canvas.mpl_connect = MagicMock()
    monkeypatch.setattr(
        "bloqade.lanes.visualize.app.Button",
        lambda ax, label: MagicMock(on_clicked=MagicMock()),
    )
    monkeypatch.setattr(
        controller, "run", lambda: setattr(controller, "run_called", True)
    )
    monkeypatch.setattr(
        controller, "reset", lambda: setattr(controller, "reset_called", True)
    )
    close_called: dict[str, object] = {}

    def fake_close(f: object) -> None:
        close_called["closed"] = f

    monkeypatch.setattr(plt, "close", fake_close)
    DebuggerController.run_mpl_event_loop(controller, ax, fig)
    assert close_called["closed"] == fig


def test_slider_created_when_num_steps_greater_than_one(monkeypatch) -> None:
    """When the controller exposes ``num_steps > 1``, a Slider widget is
    added to the figure and wired to ``on_slider_change``."""
    controller: DummyController = DummyController()
    controller.num_steps = 5  # type: ignore[attr-defined]
    controller.step_index = 0  # type: ignore[attr-defined]

    ax: mpl_axes.Axes = MagicMock(spec=mpl_axes.Axes)
    fig: mpl_fig.Figure = MagicMock(spec=mpl_fig.Figure)
    fig.add_axes = MagicMock(side_effect=[MagicMock() for _ in range(4)])
    fig.canvas = MagicMock()
    fig.canvas.mpl_connect = MagicMock()

    created_sliders: list[MagicMock] = []

    def fake_slider(**kwargs):
        slider = MagicMock(name="Slider", _kwargs=kwargs)
        created_sliders.append(slider)
        return slider

    monkeypatch.setattr(
        "bloqade.lanes.visualize.app.Button",
        lambda ax, label: MagicMock(on_clicked=MagicMock()),
    )
    monkeypatch.setattr("bloqade.lanes.visualize.app.Slider", fake_slider)
    monkeypatch.setattr(plt, "close", lambda fig: None)

    DebuggerController.run_mpl_event_loop(controller, ax, fig)

    assert len(created_sliders) == 1
    slider_kwargs = created_sliders[0]._kwargs
    assert slider_kwargs["valmin"] == 0
    assert slider_kwargs["valmax"] == 4  # num_steps - 1
    assert slider_kwargs["valinit"] == 0
    assert slider_kwargs["valstep"] == 1
    created_sliders[0].on_changed.assert_called_once_with(controller.on_slider_change)
    assert controller.slider is created_sliders[0]


def test_no_slider_when_only_one_step(monkeypatch) -> None:
    """A single-step debugger session should not get a slider — there's
    nowhere to navigate to."""
    controller: DummyController = DummyController()
    controller.num_steps = 1  # type: ignore[attr-defined]

    ax: mpl_axes.Axes = MagicMock(spec=mpl_axes.Axes)
    fig: mpl_fig.Figure = MagicMock(spec=mpl_fig.Figure)
    fig.add_axes = MagicMock(side_effect=[MagicMock() for _ in range(3)])
    fig.canvas = MagicMock()
    fig.canvas.mpl_connect = MagicMock()

    slider_called: list[object] = []
    monkeypatch.setattr(
        "bloqade.lanes.visualize.app.Button",
        lambda ax, label: MagicMock(on_clicked=MagicMock()),
    )
    monkeypatch.setattr(
        "bloqade.lanes.visualize.app.Slider",
        lambda **kwargs: slider_called.append(kwargs) or MagicMock(),
    )
    monkeypatch.setattr(plt, "close", lambda fig: None)

    DebuggerController.run_mpl_event_loop(controller, ax, fig)

    assert slider_called == []
    assert controller.slider is None


def test_sync_slider_suppresses_callbacks() -> None:
    """``sync_slider`` must temporarily disable the slider's eventson flag
    so programmatic ``set_val`` calls don't recursively re-enter
    ``on_slider_change``."""
    controller: DummyController = DummyController()
    slider = MagicMock()
    slider.eventson = True
    controller.slider = slider

    seen_eventson: list[bool] = []
    slider.set_val = MagicMock(
        side_effect=lambda val: seen_eventson.append(slider.eventson)
    )

    controller.sync_slider(3)

    slider.set_val.assert_called_once_with(3)
    assert seen_eventson == [False], "eventson must be False during set_val"
    assert slider.eventson is True, "eventson must be restored after set_val"


def test_sync_slider_no_op_without_slider() -> None:
    """When no slider was created, sync_slider must be safe to call."""
    controller: DummyController = DummyController()
    controller.slider = None
    controller.sync_slider(7)  # must not raise


def test_default_on_slider_change_is_noop() -> None:
    """The base controller's on_slider_change is a no-op so legacy
    controllers don't have to implement it."""
    controller: DummyController = DummyController()
    controller.on_slider_change(5)  # must not raise


def test_run_mpl_event_loop_clears_stale_slider_on_reuse(monkeypatch) -> None:
    """If the same controller instance is invoked twice (or only the
    second invocation has multiple steps) ``self.slider`` must not still
    point at a slider belonging to a closed figure. ``run_mpl_event_loop``
    clears it at the start of every run."""
    controller: DummyController = DummyController()
    controller.num_steps = 1  # type: ignore[attr-defined]
    # Pretend an earlier run left a stale slider lying around.
    controller.slider = MagicMock(name="StaleSliderFromPriorRun")

    ax: mpl_axes.Axes = MagicMock(spec=mpl_axes.Axes)
    fig: mpl_fig.Figure = MagicMock(spec=mpl_fig.Figure)
    fig.add_axes = MagicMock(side_effect=[MagicMock() for _ in range(3)])
    fig.canvas = MagicMock()
    fig.canvas.mpl_connect = MagicMock()

    monkeypatch.setattr(
        "bloqade.lanes.visualize.app.Button",
        lambda ax, label: MagicMock(on_clicked=MagicMock()),
    )
    monkeypatch.setattr(plt, "close", lambda fig: None)

    DebuggerController.run_mpl_event_loop(controller, ax, fig)

    assert controller.slider is None, "stale slider must be cleared on reuse"


def test_slider_initial_step_clamped_into_range(monkeypatch) -> None:
    """If the controller's ``step_index`` happens to be out of range when
    ``run_mpl_event_loop`` runs (e.g. set externally before reset), the
    slider's ``valinit`` must be clamped so Slider construction does not
    raise."""
    controller: DummyController = DummyController()
    controller.num_steps = 3  # type: ignore[attr-defined]
    controller.step_index = 99  # type: ignore[attr-defined]

    ax: mpl_axes.Axes = MagicMock(spec=mpl_axes.Axes)
    fig: mpl_fig.Figure = MagicMock(spec=mpl_fig.Figure)
    fig.add_axes = MagicMock(side_effect=[MagicMock() for _ in range(4)])
    fig.canvas = MagicMock()
    fig.canvas.mpl_connect = MagicMock()

    captured_kwargs: dict[str, object] = {}

    def fake_slider(**kwargs):
        captured_kwargs.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(
        "bloqade.lanes.visualize.app.Button",
        lambda ax, label: MagicMock(on_clicked=MagicMock()),
    )
    monkeypatch.setattr("bloqade.lanes.visualize.app.Slider", fake_slider)
    monkeypatch.setattr(plt, "close", lambda fig: None)

    DebuggerController.run_mpl_event_loop(controller, ax, fig)

    # initial_step was 99 but valmax is 2 (num_steps-1); must clamp to 2.
    assert captured_kwargs["valinit"] == 2


def test_run_mpl_event_loop_subfigure(monkeypatch) -> None:
    controller: DummyController = DummyController()
    ax: mpl_axes.Axes = MagicMock(spec=mpl_axes.Axes)

    class SubFigure:
        def add_axes(self, *args, **kwargs) -> MagicMock:
            return MagicMock()

        @property
        def canvas(self) -> MagicMock:
            return MagicMock()

    fig: SubFigure = SubFigure()
    fig.canvas.mpl_connect = MagicMock()
    monkeypatch.setattr(
        "bloqade.lanes.visualize.app.Button",
        lambda ax, label: MagicMock(on_clicked=MagicMock()),
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
