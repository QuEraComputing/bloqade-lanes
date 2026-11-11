"""Interpreter method tables for the ``lanes.steane713`` dialect."""

from bloqade.analysis.measure_id.analysis import MeasureIDFrame, MeasurementIDAnalysis
from bloqade.analysis.measure_id.lattice import (
    AnyMeasureId,
    MeasureId,
    MeasureIdTuple,
    RawMeasureId,
)
from bloqade.types import MeasurementResultType, QubitType
from kirin import interp, types, types as kirin_types
from kirin.analysis.typeinfer import TypeInference
from kirin.dialects import ilist
from kirin.interp import Frame, MethodTable, impl

from . import stmts
from ._dialect import dialect


@dialect.register(key="typeinfer")
class TypeInfer(MethodTable):
    @impl(stmts.TerminalLogicalMeasurement)
    def terminal_logical_measurement(
        self,
        interp_: TypeInference,
        frame: Frame[types.TypeAttribute],
        stmt: stmts.TerminalLogicalMeasurement,
    ):
        qubits_type = frame.get(stmt.qubits)
        if qubits_type.is_structurally_equal(
            types.Bottom
        ) or not qubits_type.is_subseteq(ilist.IListType[QubitType, types.Any]):
            return (types.Bottom,)

        assert isinstance(qubits_type, types.Generic)
        outer_len_type = qubits_type.vars[1]
        if stmt.num_physical_qubits is not None:
            inner_len_type = types.Literal(stmt.num_physical_qubits)
        else:
            inner_len_type = types.Any

        inner_type = ilist.IListType[MeasurementResultType, inner_len_type]
        return (ilist.IListType[inner_type, outer_len_type],)


@dialect.register(key="measure_id")
class LogicalQubit(interp.MethodTable):
    @interp.impl(stmts.TerminalLogicalMeasurement)
    def terminal_measurement(
        self,
        interp_: MeasurementIDAnalysis,
        frame: MeasureIDFrame,
        stmt: stmts.TerminalLogicalMeasurement,
    ):

        qubits_type = stmt.qubits.type
        if qubits_type.is_structurally_equal(kirin_types.Bottom):
            return (AnyMeasureId(),)

        # NOTE: an unresolved register type is a *reason to give up*, not a bug,
        # and every other unknowable case below says so by returning
        # `AnyMeasureId`. An `assert` here instead turned an ordinary untyped
        # parameter -- `def sub(reg): return terminal_measure(reg)` -- into an
        # `AssertionError` that `ValidationSuite` caught and re-reported as
        # "Validation pass '...' failed:" plus a traceback. The arity check
        # guards `vars[1]` below, which would raise `IndexError` on a `Generic`
        # that is not an `IList[T, Len]`.
        if (
            not isinstance(qubits_type, kirin_types.Generic)
            or len(qubits_type.vars) < 2
        ):
            return (AnyMeasureId(),)

        if not isinstance(len_var := qubits_type.vars[1], kirin_types.Literal):
            return (AnyMeasureId(),)

        if not isinstance(num_logical_qubits := len_var.data, int):
            return (AnyMeasureId(),)

        if (num_physical_qubits := stmt.num_physical_qubits) is not None:

            def logical_to_physical(
                logical_address: int,
            ) -> MeasureId:
                raw_measure_ids = map(
                    RawMeasureId,
                    range(
                        interp_.measure_count + 1,
                        interp_.measure_count + num_physical_qubits + 1,
                    ),
                )
                interp_.measure_count += num_physical_qubits
                return MeasureIdTuple(tuple(raw_measure_ids), ilist.IList)

        else:

            def logical_to_physical(
                logical_address: int,
            ) -> MeasureId:
                return AnyMeasureId()

        return (
            MeasureIdTuple(
                tuple(map(logical_to_physical, range(num_logical_qubits))), ilist.IList
            ),
        )
