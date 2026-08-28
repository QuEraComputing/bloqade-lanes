from typing import TYPE_CHECKING

from bloqade.analysis.measure_id.lattice import MeasureIdTuple
from kirin import interp as _interp, ir
from kirin.analysis import ForwardFrame

from ..dialects import operations

if TYPE_CHECKING:
    from ..validation.measurement.analysis import (
        _GeminiTerminalMeasurementValidationAnalysis,
    )


@operations.dialect.register(key="gemini.validate.terminal_measurement")
class __GeminiLogicalMeasurementValidation(_interp.MethodTable):
    @_interp.impl(operations.stmts.TerminalLogicalMeasurement)
    def terminal_measure(
        self,
        interp: "_GeminiTerminalMeasurementValidationAnalysis",
        frame: ForwardFrame,
        stmt: operations.stmts.TerminalLogicalMeasurement,
    ):

        # should only be one terminal measurement EVER
        if not interp.terminal_measurement_encountered:
            interp.terminal_measurement_encountered = True
        else:
            interp.add_validation_error(
                stmt,
                ir.ValidationError(
                    stmt,
                    "Multiple terminal measurements are not allowed in Gemini logical programs!",
                ),
            )
            return (interp.lattice.bottom(),)

        measurement_analysis_results = interp.measurement_analysis_results
        total_qubits_allocated = interp.unique_qubits_allocated

        # NOTE: `Frame.get` *raises* on a missing key rather than returning
        # None, and the analysis has no entry for this value when the statement
        # sits inside a callee that was not inlined -- it was evaluated in a
        # nested frame. The branch below already handles "no usable result", so
        # route the miss into it instead of letting an `InterpreterError` escape
        # for `ValidationSuite` to re-report as "Validation pass '...' failed:"
        # plus a traceback.
        try:
            measure_lattice_element = measurement_analysis_results.get(stmt.result)
        except _interp.InterpreterError:
            measure_lattice_element = None

        if not isinstance(measure_lattice_element, MeasureIdTuple):
            interp.add_validation_error(
                stmt,
                ir.ValidationError(
                    stmt,
                    "Measurement ID Analysis failed to produce the necessary results needed for validation.",
                ),
            )
            return (interp.lattice.bottom(),)

        if len(measure_lattice_element.data) != total_qubits_allocated:
            interp.add_validation_error(
                stmt,
                ir.ValidationError(
                    stmt,
                    "The number of qubits in the terminal measurement does not match the number of total qubits allocated! "
                    + f"{total_qubits_allocated} qubits were allocated but only {len(measure_lattice_element.data)} were measured.",
                ),
            )
            return (interp.lattice.bottom(),)

        return (interp.lattice.bottom(),)
