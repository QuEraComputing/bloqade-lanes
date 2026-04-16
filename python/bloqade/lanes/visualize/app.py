from abc import ABC

from matplotlib import axes, figure, pyplot as plt
from matplotlib.widgets import Button, Slider


class DebuggerController(ABC):
    #: Slider attached to this controller, or None if no slider was created.
    #: Set by :meth:`run_mpl_event_loop` and consulted by subclasses to
    #: keep the slider visual in sync when ``step_index`` changes via
    #: button or keyboard events.
    slider: Slider | None = None

    def run(self):
        raise NotImplementedError

    def on_exit(self, event):
        raise NotImplementedError

    def on_next(self, event):
        raise NotImplementedError

    def on_prev(self, event):
        raise NotImplementedError

    def on_slider_change(self, value):
        """Handle a slider drag/click event. Default is a no-op so legacy
        controllers without slider support continue to work."""
        _ = value

    def on_key(self, event):
        match event.key:
            case "left":
                self.on_prev(event)
            case "right":
                self.on_next(event)
            case "escape":
                self.on_exit(event)

    def reset(self):
        raise NotImplementedError

    def sync_slider(self, step_index: int) -> None:
        """Update the slider's displayed value to ``step_index`` without
        re-triggering :meth:`on_slider_change` (avoids infinite recursion
        when buttons/keys move the step). Safe to call when no slider was
        created."""
        slider = self.slider
        if slider is None:
            return
        previous = slider.eventson
        slider.eventson = False
        try:
            slider.set_val(step_index)
        finally:
            slider.eventson = previous

    def run_mpl_event_loop(
        self,
        ax: axes.Axes,
        fig: figure.Figure | figure.SubFigure,
    ):
        # Always clear any slider reference left over from a previous run on
        # this same controller instance. Without this, ``sync_slider`` could
        # poke a stale widget belonging to a closed figure.
        self.slider = None

        prev_ax = fig.add_axes((0.01, 0.01, 0.1, 0.075))
        exit_ax = fig.add_axes((0.21, 0.01, 0.1, 0.075))
        next_ax = fig.add_axes((0.41, 0.01, 0.1, 0.075))

        prev_button = Button(prev_ax, "Prev (<)")
        next_button = Button(next_ax, "Next (>)")
        exit_button = Button(exit_ax, "Exit(Esc)")

        next_button.on_clicked(self.on_next)
        prev_button.on_clicked(self.on_prev)
        exit_button.on_clicked(self.on_exit)

        # Add a slider for jumping to any step directly when there are
        # multiple steps to navigate. The slider sits above the row of
        # buttons inside the bottom controls area reserved by the caller
        # via ``fig.subplots_adjust(bottom=0.2)``.
        num_steps = getattr(self, "num_steps", 1)
        if num_steps > 1:
            slider_ax = fig.add_axes((0.1, 0.12, 0.8, 0.04))
            # Clamp initial_step into the valid slider range; out-of-band
            # step_index values would otherwise cause Slider to raise.
            initial_step = max(0, min(getattr(self, "step_index", 0), num_steps - 1))
            self.slider = Slider(
                ax=slider_ax,
                label="Step",
                valmin=0,
                valmax=num_steps - 1,
                valinit=initial_step,
                valstep=1,
                valfmt="%d",
            )
            self.slider.on_changed(self.on_slider_change)

        fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.reset()
        self.run()

        if isinstance(fig, figure.Figure):
            plt.close(fig)
