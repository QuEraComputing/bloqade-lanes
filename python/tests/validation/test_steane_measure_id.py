"""Measurement-ID analysis handles unresolved registers without inventing IDs."""

import pytest
from bloqade.analysis.measure_id import MeasurementIDAnalysis
from bloqade.analysis.measure_id.lattice import (
    AnyMeasureId,
    MeasureIdTuple,
    RawMeasureId,
)
from bloqade.types import QubitType
from kirin import types
from kirin.dialects import ilist
from kirin.prelude import structural_no_opt

from bloqade.lanes.dialects import steane713


def _measurement_method(register_type, width=None):
    @structural_no_opt.add(steane713)
    def measure(register):
        return steane713.terminal_measure(register)

    stmt = next(
        stmt
        for stmt in measure.code.walk()
        if isinstance(stmt, steane713.stmts.TerminalLogicalMeasurement)
    )
    # Model IR before type inference has established a usable register shape.
    stmt.qubits.type = register_type
    stmt.num_physical_qubits = width
    return measure, stmt


@pytest.mark.parametrize(
    "register_type",
    [
        types.Bottom,
        types.Any,
        ilist.IListType[QubitType, types.Any],
        ilist.IListType[QubitType, types.Literal("unknown")],
    ],
)
def test_unknown_register_shape_does_not_allocate_measurement_ids(register_type):
    method, stmt = _measurement_method(register_type, width=7)
    analysis = MeasurementIDAnalysis(method.dialects)
    frame, result = analysis.run(method)

    assert isinstance(result, AnyMeasureId)
    assert frame.entries[stmt.result] == result
    assert analysis.measure_count == 0


def test_unstamped_measurement_preserves_only_logical_register_shape():
    method, _ = _measurement_method(ilist.IListType[QubitType, types.Literal(2)])
    analysis = MeasurementIDAnalysis(method.dialects)
    _, result = analysis.run(method)

    assert result == MeasureIdTuple((AnyMeasureId(), AnyMeasureId()), ilist.IList)
    assert analysis.measure_count == 0


def test_stamped_measurement_allocates_distinct_physical_ids():
    method, _ = _measurement_method(
        ilist.IListType[QubitType, types.Literal(2)], width=7
    )
    analysis = MeasurementIDAnalysis(method.dialects)
    _, result = analysis.run(method)

    assert result == MeasureIdTuple(
        (
            MeasureIdTuple(tuple(RawMeasureId(i) for i in range(1, 8)), ilist.IList),
            MeasureIdTuple(tuple(RawMeasureId(i) for i in range(8, 15)), ilist.IList),
        ),
        ilist.IList,
    )
    assert analysis.measure_count == 14


@pytest.mark.parametrize("register_type", [types.Bottom, types.Int])
def test_type_inference_rejects_non_registers(register_type):
    from kirin.analysis.typeinfer import TypeInference

    method, _ = _measurement_method(register_type)
    _, result = TypeInference(method.dialects).run(method, register_type)
    assert result.is_structurally_equal(types.Bottom)


@pytest.mark.parametrize("width", [None, 7])
def test_type_inference_preserves_register_and_physical_width(width):
    from bloqade.types import MeasurementResultType
    from kirin.analysis.typeinfer import TypeInference

    register_type = ilist.IListType[QubitType, types.Literal(2)]
    method, _ = _measurement_method(register_type, width=width)
    _, result = TypeInference(method.dialects).run(method, register_type)

    inner_width = types.Any if width is None else types.Literal(width)
    expected = ilist.IListType[
        ilist.IListType[MeasurementResultType, inner_width], types.Literal(2)
    ]
    assert result.is_structurally_equal(expected)
