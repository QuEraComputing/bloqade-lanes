from typing import Protocol

from matplotlib import axes, figure, pyplot as plt
from matplotlib.widgets import Button


class DebuggerController(Protocol):
    def run(self):
        raise NotImplementedError

    def on_exit(self, event):
        raise NotImplementedError

    def on_next(self, event):
        raise NotImplementedError

    def on_prev(self, event):
        raise NotImplementedError

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

    def run_mpl_event_loop(
        self,
        ax: axes.Axes,
        fig: figure.Figure | figure.SubFigure,
    ):

        prev_ax = fig.add_axes((0.01, 0.01, 0.1, 0.075))
        exit_ax = fig.add_axes((0.21, 0.01, 0.1, 0.075))
        next_ax = fig.add_axes((0.41, 0.01, 0.1, 0.075))

        prev_button = Button(prev_ax, "Prev (<)")
        next_button = Button(next_ax, "Next (>)")
        exit_button = Button(exit_ax, "Exit(Esc)")

        next_button.on_clicked(self.on_next)
        prev_button.on_clicked(self.on_prev)
        exit_button.on_clicked(self.on_exit)
        fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.reset()
        self.run()

        if isinstance(fig, figure.Figure):
            plt.close(fig)
