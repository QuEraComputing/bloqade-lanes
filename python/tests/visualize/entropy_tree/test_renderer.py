from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402

from bloqade.lanes.visualize.entropy_tree import (  # noqa: E402
    renderer as renderer_module,
)
from bloqade.lanes.visualize.entropy_tree.renderer import (  # noqa: E402
    _apply_view_bounds,
    _draw_architecture_background,
    _draw_moveset_path,
    _layout,
    _stable_focus_bounds,
    draw_metadata_panel,
    draw_tree_frame,
    format_entropy_reason,
)
from bloqade.lanes.visualize.entropy_tree.state import (  # noqa: E402
    NodeState,
    TreeStateReducer,
)


def test_format_entropy_reason_returns_none_without_reason():
    class _Frame:
        event_reason = None
        event_state_seen_display_id = None
        event_no_valid_moves_qubit = None

    assert format_entropy_reason(_Frame()) is None  # type: ignore[arg-type]


def test_format_entropy_reason_describes_state_seen():
    class _Frame:
        event_reason = "state-seen"
        event_state_seen_display_id = 7
        event_no_valid_moves_qubit = None

    text = format_entropy_reason(_Frame())  # type: ignore[arg-type]
    assert text is not None
    assert "seen configuration" in text
    assert "7" in text


def test_format_entropy_reason_describes_no_valid_moves():
    class _Frame:
        event_reason = "no-valid-moves"
        event_state_seen_display_id = None
        event_no_valid_moves_qubit = 3

    text = format_entropy_reason(_Frame(), qid_label_map={3: 42})  # type: ignore[arg-type]
    assert text is not None
    assert "no valid moves" in text
    assert "42" in text


def test_draw_tree_frame_on_empty_reducer_does_not_raise():
    reducer = TreeStateReducer(steps=(), root_node_id=0, best_buffer_size=0)
    fig, ax = plt.subplots()
    try:
        draw_tree_frame(ax, reducer.frame_at(0))
    finally:
        plt.close(fig)


def test_draw_metadata_panel_on_empty_reducer_does_not_raise():
    reducer = TreeStateReducer(steps=(), root_node_id=0, best_buffer_size=0)
    fig, ax = plt.subplots()
    ax.axis("off")
    try:
        draw_metadata_panel(ax, reducer.frame_at(0), "info line", [])
    finally:
        plt.close(fig)


def test_draw_moveset_path_builds_source_location_set_once():
    source_loc = object()
    other_loc = object()
    lanes = ("lane-a", "lane-b")

    class _SourceConfig:
        values_calls = 0

        def values(self):
            self.values_calls += 1
            return (source_loc,)

    class _ArchSpec:
        def get_endpoints(self, lane):
            return (source_loc if lane == "lane-a" else other_loc, other_loc)

        def get_path(self, lane):
            assert lane == "lane-a"
            return [(0.0, 0.0), (1.0, 1.0)]

    class _Ax:
        def plot(self, *args, **kwargs):
            pass

        def annotate(self, *args, **kwargs):
            pass

    source_config = _SourceConfig()

    positions = _draw_moveset_path(_Ax(), _ArchSpec(), lanes, source_config)

    assert positions == [(0.0, 0.0), (1.0, 1.0)]
    assert source_config.values_calls == 1


def test_layout_handles_deep_single_child_tree_without_recursion_error():
    nodes = {
        idx: NodeState(
            node_id=idx,
            display_id=idx,
            parent_id=None if idx == 0 else idx - 1,
            depth=idx,
        )
        for idx in range(5000)
    }

    positions = _layout(nodes)

    assert len(positions) == len(nodes)
    assert positions[0][0] == positions[4999][0]
    assert positions[4999][1] < positions[0][1]


def test_stable_focus_bounds_uses_arch_visualizer(monkeypatch):
    class _ArchSpec:
        def get_position(self, loc):
            return loc

    class _ArchVisualizer:
        def __init__(self, arch_spec):
            self.arch_spec = arch_spec

        def path_bounds(self):
            return (0.0, 10.0, 0.0, 10.0)

    monkeypatch.setattr(renderer_module, "ArchVisualizer", _ArchVisualizer)

    bounds = _stable_focus_bounds(
        arch_spec=_ArchSpec(),
        root_configuration={0: (2.0, 2.0)},
        target={0: (8.0, 8.0)},
    )

    assert bounds == (-0.6, 10.6, -0.8, 10.8)


def test_apply_view_bounds_uses_arch_visualizer_for_full_bounds(monkeypatch):
    class _ArchSpec:
        pass

    class _ArchVisualizer:
        def __init__(self, arch_spec):
            self.arch_spec = arch_spec

        def path_bounds(self):
            return (1.0, 3.0, 2.0, 6.0)

    monkeypatch.setattr(renderer_module, "ArchVisualizer", _ArchVisualizer)
    fig, ax = plt.subplots()
    try:
        _apply_view_bounds(
            ax,
            _ArchSpec(),
            plotted_positions=[],
            fixed_bounds=None,
        )
        assert ax.get_xlim() == (0.88, 3.12)
        assert ax.get_ylim() == (1.68, 6.32)
    finally:
        plt.close(fig)


def test_draw_architecture_background_uses_arch_visualizer(monkeypatch):
    class _ArchSpec:
        words = (object(), object())

    calls = []

    class _ArchVisualizer:
        def __init__(self, arch_spec):
            self.arch_spec = arch_spec

        def plot(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(renderer_module, "ArchVisualizer", _ArchVisualizer)
    fig, ax = plt.subplots()
    try:
        _draw_architecture_background(ax, _ArchSpec())
    finally:
        plt.close(fig)

    assert calls
    assert calls[0]["show_words"] == [0, 1]
