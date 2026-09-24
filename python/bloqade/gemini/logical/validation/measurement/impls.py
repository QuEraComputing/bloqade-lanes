import bloqade.qubit as qubit
from bloqade.analysis.address.impls import Func as AddressFuncMethodTable
from bloqade.squin import gate
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


@gate.dialect.register(key="gemini.validate.terminal_measurement")
class _GateGeminiMeasurementValidation(_interp.MethodTable):
    # NOTE: every concrete statement of the gate dialect must be listed here;
    # a gate missing from this list falls through to `eval_fallback` and is
    # silently accepted after the terminal measurement. The test suite checks
    # this list against `gate.dialect.stmts`.
    @_interp.impl(gate.stmts.X)
    @_interp.impl(gate.stmts.Y)
    @_interp.impl(gate.stmts.Z)
    @_interp.impl(gate.stmts.H)
    @_interp.impl(gate.stmts.T)
    @_interp.impl(gate.stmts.S)
    @_interp.impl(gate.stmts.SqrtX)
    @_interp.impl(gate.stmts.SqrtY)
    @_interp.impl(gate.stmts.Rx)
    @_interp.impl(gate.stmts.Ry)
    @_interp.impl(gate.stmts.Rz)
    @_interp.impl(gate.stmts.U3)
    @_interp.impl(gate.stmts.PhasedXZ)
    @_interp.impl(gate.stmts.CX)
    @_interp.impl(gate.stmts.CY)
    @_interp.impl(gate.stmts.CZ)
    @_interp.impl(gate.stmts.CCZ)
    @_interp.impl(gate.stmts.Swap)
    def gate_after_measurement(
        self,
        interp: _GeminiTerminalMeasurementValidationAnalysis,
        frame: ForwardFrame,
        stmt: gate.stmts.Gate,
    ):
        interp.check_gate_after_measurement(stmt)
        return ()


@func.dialect.register(key="gemini.validate.terminal_measurement")
class Func(AddressFuncMethodTable):
    pass
