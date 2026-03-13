from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from bloqade.lanes.visualize.artist import (
    AODPositions,
    AODXLines,
    AODYLines,
    MoveRenderer,
    MovingAtomsScatter,
    PlotParameters,
    StationaryAtomsScatter,
)


def test_plot_parameters_properties():
    params = PlotParameters(scale=1.0)
    assert isinstance(params.atom_plot_args, dict)
    assert isinstance(params.atom_label_args, dict)
    assert isinstance(params.gate_spot_args, dict)
    assert isinstance(params.slm_plot_args, dict)
    assert isinstance(params.aod_line_args, dict)
    assert isinstance(params.aod_marker_args, dict)


def test_aod_x_lines():
    fig, ax = plt.subplots()
    interp = interp1d([0, 1], [0, 1])
    lines = AODXLines(ax, [interp], PlotParameters(1.0))
    assert isinstance(lines, AODXLines)
    lines.update(0.5)
    plt.close(fig)


def test_aod_y_lines():
    fig, ax = plt.subplots()
    interp = interp1d([0, 1], [0, 1])
    lines = AODYLines(ax, [interp], PlotParameters(1.0))
    assert isinstance(lines, AODYLines)
    lines.update(0.5)
    plt.close(fig)


def test_aod_positions():
    fig, ax = plt.subplots()
    interp = interp1d([0, 1], [0, 1])
    positions = AODPositions(ax, [interp], [interp], PlotParameters(1.0))
    assert isinstance(positions, AODPositions)
    positions.update(0.5)
    plt.close(fig)


def test_moving_atoms_scatter():
    fig, ax = plt.subplots()
    interp = interp1d([0, 1], [0, 1])
    scatter = MovingAtomsScatter(
        ax, [0], [0], [0], [interp], [interp], PlotParameters(1.0)
    )
    assert isinstance(scatter, MovingAtomsScatter)
    scatter.update(0.5)
    plt.close(fig)


def test_stationary_atoms_scatter():
    fig, ax = plt.subplots()
    scatter = StationaryAtomsScatter(ax, [0.0], [0.0], [0], PlotParameters(1.0))
    assert isinstance(scatter, StationaryAtomsScatter)
    scatter.update(0.5)
    plt.close(fig)


def test_move_renderer():
    fig, ax = plt.subplots()
    interp = interp1d([0, 1], [0, 1])
    xlines = AODXLines(ax, [interp], PlotParameters(1.0))
    ylines = AODYLines(ax, [interp], PlotParameters(1.0))
    renderer = MoveRenderer([xlines, ylines])
    assert isinstance(renderer.total_time, float)
    renderer.update(0.5)
    plt.close(fig)
