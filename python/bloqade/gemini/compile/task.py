from collections.abc import Callable
from functools import cache
from typing import Any, TypeVar

from bloqade.analysis.address import AddressAnalysis, AddressQubit
from bloqade.analysis.validation.simple_nocloning import FlatKernelNoCloningValidation
from bloqade.decoders.dialects.annotate.stmts import SetDetector, SetObservable
from kirin import ir
from kirin.dialects import func, ilist, py
from kirin.validation import ValidationSuite

from bloqade.gemini.cudaq import cudaq_to_squin, is_cudaq_kernel
from bloqade.gemini.logical.dialects.operations.stmts import TerminalLogicalMeasurement
from bloqade.gemini.logical.validation.clifford.analysis import GeminiLogicalValidation
from bloqade.gemini.logical.validation.measurement.analysis import (
    GeminiTerminalMeasurementValidation,
)
from bloqade.gemini.steane_defaults import steane7_m2dets, steane7_m2obs
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.transform import LogicalPipeline

__all__ = [
    "_find_qubit_ssas",
    "_find_return_stmt",
    "_insert_before",
    "append_measurements_and_annotations",
    "compile_task",
    "run_squin_kernel_validation",
]


def run_squin_kernel_validation(mt: ir.Method):
    """
    Run validation checks on a Squin kernel method.

    Args:
        mt (ir.Method): The Squin kernel method to validate.

    Returns:
        ValidationResult: A validation result object containing the
            validation errors, if they exist

    Note: To trigger an error run `run_squin_kernel_validation(mt).raise_if_invalid()`.

    """
    validator = ValidationSuite(
        [
            GeminiLogicalValidation,
            GeminiTerminalMeasurementValidation,
            FlatKernelNoCloningValidation,
        ]
    )
    return validator.validate(mt)


_S = TypeVar("_S", bound=ir.Statement)


def _find_qubit_ssas(mt: ir.Method) -> list[ir.SSAValue]:
    """Return one qubit SSA value per concrete qubit address.

    ``qalloc`` calls must be aggressively unrolled so each allocation has a
    corresponding SSA value in ``mt``.
    """
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
    """Find the func.Return statement at the end of the function body."""
    block = mt.callable_region.blocks[0]
    last = block.last_stmt
    assert isinstance(last, func.Return), f"Expected func.Return, got {type(last)}"
    return last


def _insert_before(stmt: _S, anchor: ir.Statement) -> _S:
    """Insert stmt before anchor and return stmt for chaining."""
    stmt.insert_before(anchor)
    return stmt


def append_measurements_and_annotations(
    mt: ir.Method,
    m2dets: list[list[int]] | None,
    m2obs: list[list[int]] | None,
) -> None:
    """Append terminal measurement, detector, and observable IR statements to a squin kernel.

    The method is mutated in-place.

    Args:
        mt: A squin ``ir.Method`` whose body returns ``None``.
        m2dets: Binary matrix of shape ``(num_total_meas, num_detectors)``.
            Each column defines a detector by its non-zero row indices.
        m2obs: Binary matrix of shape ``(num_total_meas, num_observables)``.
            Each column defines an observable by its non-zero row indices.
    """

    if m2dets is None and m2obs is None:
        raise ValueError("At least one of m2dets or m2obs must be provided")

    qubit_ssas = _find_qubit_ssas(mt)
    num_qubits = len(qubit_ssas)
    if num_qubits == 0:
        raise ValueError("No qubit allocations found in the kernel")

    m2 = m2dets if m2dets is not None else m2obs
    assert m2 is not None
    num_total_meas = len(m2)
    meas_per_qubit = num_total_meas // num_qubits
    assert (
        meas_per_qubit * num_qubits == num_total_meas
    ), "Incompatible shape of m2dets or m2obs"

    return_stmt = _find_return_stmt(mt)

    # insert TerminalLogicalMeasurement if not present
    terminal_measurement = next(
        (
            s
            for s in mt.callable_region.walk()
            if isinstance(s, TerminalLogicalMeasurement)
        ),
        None,
    )
    if terminal_measurement is not None:
        term_meas = terminal_measurement
    else:
        qlist_stmt = _insert_before(ilist.New(qubit_ssas), return_stmt)
        term_meas = _insert_before(
            TerminalLogicalMeasurement(qlist_stmt.result), return_stmt
        )

    @cache
    def _get_logical_measurement(q_idx: int) -> ir.SSAValue:
        (idx := py.Constant(q_idx)).insert_before(return_stmt)
        (getitem := py.GetItem(term_meas.result, idx.result)).insert_before(return_stmt)
        return getitem.result

    @cache
    def _get_physical_measurement(q_idx: int, m_idx: int) -> ir.SSAValue:
        (idx := py.Constant(m_idx)).insert_before(return_stmt)
        (
            getitem := py.GetItem(_get_logical_measurement(q_idx), idx.result)
        ).insert_before(return_stmt)
        return getitem.result

    # insert detectors
    if m2dets is not None:
        for j in range(len(m2dets[0])):
            indices = [i for i, row in enumerate(m2dets) if row[j]]
            meas_ssas = [
                _get_physical_measurement(*divmod(idx, meas_per_qubit))
                for idx in indices
            ]
            meas_list = _insert_before(ilist.New(meas_ssas), return_stmt)

            coord_0 = _insert_before(py.Constant(0.0), return_stmt)
            coord_1 = _insert_before(py.Constant(float(j)), return_stmt)
            coords = _insert_before(
                ilist.New([coord_0.result, coord_1.result]), return_stmt
            )

            _insert_before(SetDetector(meas_list.result, coords.result), return_stmt)

    # insert observables
    if m2obs is not None:
        for j in range(len(m2obs[0])):
            indices = [i for i, row in enumerate(m2obs) if row[j]]
            meas_ssas = [
                _get_physical_measurement(*divmod(idx, meas_per_qubit))
                for idx in indices
            ]
            meas_list = _insert_before(ilist.New(meas_ssas), return_stmt)
            _insert_before(SetObservable(meas_list.result), return_stmt)


def compile_task(
    logical_kernel: ir.Method | Callable[..., Any],
    m2dets: list[list[int]] | None = None,
    m2obs: list[list[int]] | None = None,
):
    """Compile a logical kernel into physical move artifacts.

    Handles CUDAQ kernel detection/conversion, squin kernel validation,
    squin-to-move compilation, architecture spec generation, and
    post-processing extraction.

    Args:
        logical_kernel: A squin ``ir.Method`` or a CUDA-Q kernel to compile.
        m2dets: Binary measurement-to-detector matrix. For CUDA-Q kernels,
            defaults to Steane [[7,1,3]] detectors if ``None``.
        m2obs: Binary measurement-to-observable matrix. For CUDA-Q kernels,
            defaults to Steane [[7,1,3]] observables if ``None``.

    Returns:
        A tuple of ``(logical_squin_kernel, physical_arch_spec,
        physical_move_kernel, post_processing)``.

    """
    if is_cudaq_kernel(logical_kernel):
        logical_squin_kernel: ir.Method = cudaq_to_squin(logical_kernel)

        if m2dets is None and m2obs is None:
            num_qubits = len(_find_qubit_ssas(logical_squin_kernel))
            m2dets = steane7_m2dets(num_qubits)
            m2obs = steane7_m2obs(num_qubits)

        append_measurements_and_annotations(logical_squin_kernel, m2dets, m2obs)
    elif isinstance(logical_kernel, ir.Method):
        # Compilation and annotation rewrites are in-place. Work on an owned
        # copy so creating a task never changes a caller-owned kernel.
        logical_squin_kernel = logical_kernel.similar()
        if m2dets is not None or m2obs is not None:
            append_measurements_and_annotations(logical_squin_kernel, m2dets, m2obs)
    else:
        raise ValueError(f"Unknown kernel type {type(logical_kernel)}")

    run_squin_kernel_validation(logical_squin_kernel).raise_if_invalid()

    physical_arch_spec = physical.get_arch_spec()
    physical_move_kernel = LogicalPipeline(transversal_rewrite=True).emit(
        logical_squin_kernel
    )
    post_processing = atom.AtomInterpreter(
        physical_move_kernel.dialects, arch_spec=physical_arch_spec
    ).get_post_processing(physical_move_kernel)

    return (
        logical_squin_kernel,
        physical_arch_spec,
        physical_move_kernel,
        post_processing,
    )
