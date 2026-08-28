from bloqade.analysis import address
from bloqade.squin import gate
from kirin import interp as _interp, ir
from kirin.analysis import ForwardFrame, const
from kirin.dialects import scf

from bloqade import qubit
from bloqade.gemini.logical.dialects import operations

from .analysis import _GeminiLogicalValidationAnalysis


@scf.dialect.register(key="gemini.validate.logical")
class __ScfGeminiLogicalValidation(_interp.MethodTable):
    @_interp.impl(scf.IfElse)
    def if_else(
        self,
        interp: _GeminiLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: scf.IfElse,
    ):
        interp.add_validation_error(
            stmt,
            ir.ValidationError(
                stmt, "If statements are not supported in logical Gemini programs!"
            ),
        )
        return (interp.lattice.bottom(),)

    @_interp.impl(scf.For)
    def for_loop(
        self,
        interp: _GeminiLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: scf.For,
    ):
        if not isinstance(stmt.iterable.hints.get("const"), const.Value):
            interp.add_validation_error(
                stmt,
                ir.ValidationError(
                    stmt,
                    "Non-constant iterable in for loop is not supported in Gemini logical programs!",
                ),
            )

        return (interp.lattice.bottom(),)


# NOTE: `func.Invoke` used to be reported by an impl here. `GeminiLogicalValidation`
# still reports it, but delegates to `common.validation.static_call` instead --
# an impl only sees what this `Forward` analysis reaches, and the `scf.For` impl
# above returns bottom without descending into the loop body, so an invoke nested
# in a loop was never visited. A syntactic `walk()` sees all of them.


@gate.dialect.register(key="gemini.validate.logical")
class __GateGeminiLogicalValidation(_interp.MethodTable):
    @_interp.impl(gate.stmts.U3)
    @_interp.impl(gate.stmts.T)
    @_interp.impl(gate.stmts.Rx)
    @_interp.impl(gate.stmts.Ry)
    @_interp.impl(gate.stmts.Rz)
    def non_clifford(
        self,
        interp: _GeminiLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: gate.stmts.SingleQubitGate | gate.stmts.RotationGate,
    ):
        if interp.check_first_gate(stmt.qubits):
            return ()

        interp.add_validation_error(
            stmt,
            ir.ValidationError(
                stmt,
                f"Non-clifford gate {stmt.name} can only be used for initial state preparation, i.e. as the first gate!",
            ),
        )
        return ()

    @_interp.impl(gate.stmts.X)
    @_interp.impl(gate.stmts.Y)
    @_interp.impl(gate.stmts.SqrtX)
    @_interp.impl(gate.stmts.SqrtY)
    @_interp.impl(gate.stmts.Z)
    @_interp.impl(gate.stmts.H)
    @_interp.impl(gate.stmts.S)
    def clifford(
        self,
        interp: _GeminiLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: gate.stmts.SingleQubitGate,
    ):
        # NOTE: ignore result, but make sure the first gate flag is set to False
        interp.check_first_gate(stmt.qubits)

        return ()

    @_interp.impl(gate.stmts.CX)
    @_interp.impl(gate.stmts.CY)
    @_interp.impl(gate.stmts.CZ)
    @_interp.impl(gate.stmts.Swap)
    def two_qubit_gate(
        self,
        interp: _GeminiLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: gate.stmts.ControlledGate | gate.stmts.TwoQubitGate,
    ):
        # NOTE: both register operands are acted on; iterate over `args` so this
        # covers the (controls, targets) and (qubits1, qubits2) spellings alike
        for qubits in stmt.args:
            interp.check_first_gate(qubits)

        return ()


@operations.dialect.register(key="gemini.validate.logical")
class __OperationsGeminiLogicalValidation(_interp.MethodTable):
    @_interp.impl(operations.stmts.StarRz)
    def star_rz(
        self,
        interp: _GeminiLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: operations.stmts.StarRz,
    ):
        interp.check_first_gate(stmt.qubits)
        return ()


@qubit.dialect.register(key="gemini.validate.logical")
class QubitMethods(_interp.MethodTable):
    @_interp.impl(qubit.stmts.New)
    def check_qubit_allocation(
        self,
        interp: _GeminiLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: qubit.stmts.New,
    ):
        qubit_val = interp.addr_frame.get(stmt.result)
        if not isinstance(qubit_val, address.AddressQubit):
            interp.add_validation_error(
                stmt,
                ir.ValidationError(
                    stmt,
                    "Cannot determine qubit address location.",
                ),
            )
            return (interp.lattice.bottom(),)

        if qubit_val.data >= interp.max_qubits:
            interp.add_validation_error(
                stmt,
                ir.ValidationError(
                    stmt,
                    f"Qubit allocations exceeded {interp.max_qubits}.",
                ),
            )

        return (interp.lattice.bottom(),)
