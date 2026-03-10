from __future__ import annotations

from functools import cache
from typing import TypeVar

from bloqade.decoders.dialects.annotate.stmts import SetDetector, SetObservable
from bloqade.gemini.logical.dialects.operations.stmts import (
    TerminalLogicalMeasurement,
)
from kirin import ir, passes
from kirin.dialects import func, ilist, py

from bloqade import qubit

_S = TypeVar("_S", bound=ir.Statement)


def _find_qubit_ssas(mt: ir.Method) -> list[ir.SSAValue]:
    """Walk the squin IR and collect SSA values for all qubit allocations."""
    qubit_ssas: list[ir.SSAValue] = []
    for stmt in mt.callable_region.walk():
        if isinstance(stmt, func.Invoke) and stmt.callee is qubit.new:
            qubit_ssas.append(stmt.result)
        elif isinstance(stmt, qubit.stmts.New):
            qubit_ssas.append(stmt.result)
    return qubit_ssas


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

            obs_idx = _insert_before(py.Constant(j), return_stmt)
            _insert_before(SetObservable(meas_list.result, obs_idx.result), return_stmt)

    # TODO: remove this once post-processing of None return values is supported
    if isinstance(return_stmt.value.owner, func.ConstantNone):
        none_owner = return_stmt.value.owner
        return_stmt.replace_by(func.Return(term_meas.result))
        none_owner.delete()
        passes.TypeInfer(mt.dialects, no_raise=True)(mt)
