import pytest
from kirin import ir
from kirin.interp.exceptions import InterpreterError

from bloqade.lanes.analysis.placement import PlacementAnalysis
from bloqade.lanes.dialects import place
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategy
from bloqade.lanes.layout import LocationAddress


def _build_analysis() -> PlacementAnalysis:
    analysis = object.__new__(PlacementAnalysis)
    analysis.cz_lookahead_buffers = {}
    analysis.cz_lookahead_stmt_positions = {}
    analysis.placement_strategy = LogicalPlacementStrategy()
    analysis.initial_layout = (LocationAddress(0, 0), LocationAddress(0, 1))
    analysis.address_analysis = {}
    return analysis


def test_build_cz_buffer_records_order_and_positions():
    """Test that the CZ buffer is built correctly and that the statement positions are recorded correctly."""
    analysis = _build_analysis()
    block = ir.Block(
        [
            cz0 := place.CZ(ir.TestValue(), qubits=(0, 1)),
            place.Rz(cz0.state_after, ir.TestValue(), qubits=(0,)),
            cz1 := place.CZ(ir.TestValue(), qubits=(2, 3)),
            cz2 := place.CZ(ir.TestValue(), qubits=(4, 5)),
        ]
    )

    buffer = analysis.build_cz_buffer(block)
    assert buffer == (((0,), (1,)), ((2,), (3,)), ((4,), (5,)))
    assert analysis.cz_lookahead_stmt_positions[block][cz0] == 0
    assert analysis.cz_lookahead_stmt_positions[block][cz1] == 1
    assert analysis.cz_lookahead_stmt_positions[block][cz2] == 2


def test_buffered_future_cz_layers_returns_suffix_including_current():
    """Test that buffered CZ layers return a shrinking suffix including the current CZ."""
    analysis = _build_analysis()
    block = ir.Block(
        [
            cz0 := place.CZ(ir.TestValue(), qubits=(0, 1)),
            cz1 := place.CZ(ir.TestValue(), qubits=(2, 3)),
            cz2 := place.CZ(ir.TestValue(), qubits=(4, 5)),
        ]
    )
    analysis.cz_lookahead_buffers[block] = analysis.build_cz_buffer(block)

    assert analysis.buffered_future_cz_layers(cz0) == (
        ((0,), (1,)),
        ((2,), (3,)),
        ((4,), (5,)),
    )
    assert analysis.buffered_future_cz_layers(cz1) == (((2,), (3,)), ((4,), (5,)))
    assert analysis.buffered_future_cz_layers(cz2) == (((4,), (5,)),)


def test_buffered_future_cz_layers_raises_on_missing_stmt_mapping():
    """Test that the buffered future CZ layers raises an error if the current CZ statement is not in the statement positions."""
    analysis = _build_analysis()
    block = ir.Block([cz0 := place.CZ(ir.TestValue(), qubits=(0, 1))])
    analysis.cz_lookahead_buffers[block] = analysis.build_cz_buffer(block)
    analysis.cz_lookahead_stmt_positions[block].pop(cz0)

    with pytest.raises(
        InterpreterError,
        match="Lookahead CZ buffer missing current CZ statement mapping",
    ):
        analysis.buffered_future_cz_layers(cz0)


def test_buffered_future_cz_layers_raises_on_out_of_sync_buffer():
    """Test that the buffered future CZ layers raises an error if the buffer is out of sync with the executed CZ order."""
    analysis = _build_analysis()
    block = ir.Block([cz0 := place.CZ(ir.TestValue(), qubits=(0, 1))])
    analysis.build_cz_buffer(block)
    analysis.cz_lookahead_buffers[block] = (((9,), (10,)),)

    with pytest.raises(
        InterpreterError, match="Lookahead CZ buffer out of sync with executed CZ order"
    ):
        analysis.buffered_future_cz_layers(cz0)
