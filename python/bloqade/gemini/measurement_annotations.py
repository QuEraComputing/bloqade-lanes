from __future__ import annotations

from functools import cache
from typing import Callable, Literal, TypeVar

from bloqade.analysis.address import AddressAnalysis, AddressQubit
from bloqade.decoders.dialects.annotate.stmts import SetDetector, SetObservable
from bloqade.rewrite.passes import AggressiveUnroll
from kirin import ir
from kirin.dialects import func, ilist, py

from bloqade import qubit
from bloqade.gemini.logical.dialects.operations.stmts import (
    TerminalLogicalMeasurement,
)

MeasurementLevel = Literal["logical", "physical"]
MeasurementResolver = Callable[[int, int], ir.SSAValue]
_S = TypeVar("_S", bound=ir.Statement)


def _find_qubit_ssas(mt: ir.Method) -> list[ir.SSAValue]:
    """Return one qubit SSA value per concrete qubit address."""
    address_analysis = AddressAnalysis(mt.dialects)
    frame, _ = address_analysis.run(mt)
    qubits_by_address: dict[int, ir.SSAValue] = {}

    for stmt in mt.callable_region.walk():
        for result in stmt.results:
            address = frame.get(result)
            if isinstance(address, AddressQubit):
                qubits_by_address.setdefault(address.data, result)

    return [
        qubits_by_address[address] for address in range(address_analysis.qubit_count)
    ]


def _find_return_stmt(mt: ir.Method) -> func.Return:
    """Find the final return statement in a single-block kernel."""
    block = mt.callable_region.blocks[0]
    last = block.last_stmt
    assert isinstance(last, func.Return), f"Expected func.Return, got {type(last)}"
    return last


def _insert_before(stmt: _S, anchor: ir.Statement) -> _S:
    """Insert ``stmt`` before ``anchor`` and return it for chaining."""
    stmt.insert_before(anchor)
    return stmt


def _validate_m2_matrix(matrix: list[list[int]], name: str) -> int:
    if len(matrix) == 0:
        raise ValueError(f"{name} must have at least one row")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError(f"{name} rows must all have the same length")
    return len(matrix)


def _validate_m2_matrices(
    m2dets: list[list[int]] | None,
    m2obs: list[list[int]] | None,
) -> int:
    if m2dets is None and m2obs is None:
        raise ValueError("At least one of m2dets or m2obs must be provided")

    num_rows: int | None = None
    if m2dets is not None:
        num_rows = _validate_m2_matrix(m2dets, "m2dets")
    if m2obs is not None:
        m2obs_rows = _validate_m2_matrix(m2obs, "m2obs")
        if num_rows is not None and num_rows != m2obs_rows:
            raise ValueError("m2dets and m2obs must have the same number of rows")
        num_rows = m2obs_rows

    assert num_rows is not None
    return num_rows


def _prepare_logical_measurements(
    mt: ir.Method,
    num_total_measurements: int,
) -> tuple[int, MeasurementResolver, func.Return]:
    qubit_ssas = _find_qubit_ssas(mt)
    num_qubits = len(qubit_ssas)
    if num_qubits == 0:
        raise ValueError("No qubit allocations found in the kernel")

    measurements_per_qubit, remainder = divmod(num_total_measurements, num_qubits)
    if remainder != 0:
        raise ValueError("Incompatible shape of m2dets or m2obs")

    return_stmt = _find_return_stmt(mt)
    terminal_measurement = next(
        (
            stmt
            for stmt in mt.callable_region.walk()
            if isinstance(stmt, TerminalLogicalMeasurement)
        ),
        None,
    )
    if terminal_measurement is None:
        qubit_list = _insert_before(ilist.New(qubit_ssas), return_stmt)
        terminal_measurement = _insert_before(
            TerminalLogicalMeasurement(qubit_list.result), return_stmt
        )

    @cache
    def get_logical_measurement(qubit_index: int) -> ir.SSAValue:
        index = _insert_before(py.Constant(qubit_index), return_stmt)
        item = _insert_before(
            py.GetItem(terminal_measurement.result, index.result), return_stmt
        )
        return item.result

    @cache
    def resolve_measurement(_group_index: int, row_index: int) -> ir.SSAValue:
        qubit_index, measurement_index = divmod(row_index, measurements_per_qubit)
        index = _insert_before(py.Constant(measurement_index), return_stmt)
        item = _insert_before(
            py.GetItem(get_logical_measurement(qubit_index), index.result),
            return_stmt,
        )
        return item.result

    return 1, resolve_measurement, return_stmt


def _prepare_physical_measurements(
    mt: ir.Method,
    physical_qubits_per_logical_qubit: int,
) -> tuple[int, MeasurementResolver, func.Return]:
    AggressiveUnroll(mt.dialects, no_raise=True).fixpoint(mt)

    num_physical_qubits = len(_find_qubit_ssas(mt))
    if num_physical_qubits == 0:
        raise ValueError("No physical qubit allocations found in the kernel")

    num_logical_qubits, remainder = divmod(
        num_physical_qubits, physical_qubits_per_logical_qubit
    )
    if remainder != 0:
        raise ValueError(
            "The number of physical qubits must be divisible by the number of "
            "physical qubits per logical qubit"
        )

    measure_stmts = [
        stmt
        for stmt in mt.callable_region.walk()
        if isinstance(stmt, qubit.stmts.Measure)
    ]
    if len(measure_stmts) != 1:
        raise ValueError("Expected exactly one physical qubit.Measure statement")
    measure_stmt = measure_stmts[0]
    return_stmt = _find_return_stmt(mt)

    @cache
    def resolve_measurement(group_index: int, row_index: int) -> ir.SSAValue:
        flat_index = group_index * physical_qubits_per_logical_qubit + row_index
        index = _insert_before(py.Constant(flat_index), return_stmt)
        item = _insert_before(
            py.GetItem(measure_stmt.result, index.result), return_stmt
        )
        return item.result

    return num_logical_qubits, resolve_measurement, return_stmt


def _append_annotations(
    matrix: list[list[int]],
    *,
    num_groups: int,
    resolve_measurement: MeasurementResolver,
    return_stmt: func.Return,
    level: MeasurementLevel,
    detector: bool,
) -> None:
    for group_index in range(num_groups):
        for annotation_index in range(len(matrix[0])):
            measurements = [
                resolve_measurement(group_index, row_index)
                for row_index, row in enumerate(matrix)
                if row[annotation_index]
            ]
            measurement_list = _insert_before(ilist.New(measurements), return_stmt)

            if detector:
                coordinate_group = group_index if level == "physical" else 0
                coord_0 = _insert_before(
                    py.Constant(float(coordinate_group)), return_stmt
                )
                coord_1 = _insert_before(
                    py.Constant(float(annotation_index)), return_stmt
                )
                coords = _insert_before(
                    ilist.New([coord_0.result, coord_1.result]), return_stmt
                )
                _insert_before(
                    SetDetector(measurement_list.result, coords.result), return_stmt
                )
            else:
                _insert_before(SetObservable(measurement_list.result), return_stmt)


def append_measurements_and_annotations(
    mt: ir.Method,
    m2dets: list[list[int]] | None,
    m2obs: list[list[int]] | None,
    *,
    level: MeasurementLevel = "logical",
) -> None:
    """Append measurement-backed detector and observable annotations.

    Logical matrices describe the complete flattened measurement output.
    Physical matrices describe one logical block and are repeated for every
    physical-qubit block in the kernel. The method is mutated in-place and its
    existing return value is preserved.
    """
    if level not in ("logical", "physical"):
        raise ValueError(f"Unknown measurement annotation level: {level!r}")

    num_matrix_rows = _validate_m2_matrices(m2dets, m2obs)
    if level == "logical":
        num_groups, resolver, return_stmt = _prepare_logical_measurements(
            mt, num_matrix_rows
        )
    else:
        num_groups, resolver, return_stmt = _prepare_physical_measurements(
            mt, num_matrix_rows
        )

    if m2dets is not None:
        _append_annotations(
            m2dets,
            num_groups=num_groups,
            resolve_measurement=resolver,
            return_stmt=return_stmt,
            level=level,
            detector=True,
        )
    if m2obs is not None:
        _append_annotations(
            m2obs,
            num_groups=num_groups,
            resolve_measurement=resolver,
            return_stmt=return_stmt,
            level=level,
            detector=False,
        )
