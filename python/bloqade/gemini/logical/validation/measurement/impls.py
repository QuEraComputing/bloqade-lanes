import bloqade.qubit as qubit
from bloqade.analysis.address.impls import Func as AddressFuncMethodTable
from kirin import interp as _interp, ir
from kirin.analysis import ForwardFrame
from kirin.dialects import func

from .analysis import _GeminiTerminalMeasurementValidationAnalysis


@qubit.dialect.register(key="gemini.validate.terminal_measurement")
class _QubitGeminiMeasurementValidation(_interp.MethodTable):
    # This is a non-logical measurement, can safely flag as invalid
    @_interp.impl(qubit.stmts.Measure)
    def measure(
        self,
        interp: _GeminiTerminalMeasurementValidationAnalysis,
        frame: ForwardFrame,
        stmt: qubit.stmts.Measure,
    ):

        interp.add_validation_error(
            stmt,
            ir.ValidationError(
                stmt,
                "Non-terminal measurements are not allowed in Gemini programs!",
            ),
        )

        return (interp.lattice.bottom(),)


@func.dialect.register(key="gemini.validate.terminal_measurement")
class Func(AddressFuncMethodTable):
    pass
