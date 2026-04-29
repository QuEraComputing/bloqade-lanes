from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402

from bloqade.lanes.visualize.entropy_tree.renderer import (  # noqa: E402
    draw_metadata_panel,
    draw_tree_frame,
    format_entropy_reason,
)
from bloqade.lanes.visualize.entropy_tree.state import TreeStateReducer  # noqa: E402


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
