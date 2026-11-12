from dataclasses import dataclass, field
from typing import Any, ClassVar

from bloqade.analysis import address
from bloqade.analysis.address import Address, AddressAnalysis, AddressReg
from bloqade.squin import gate
from kirin import interp as _interp, ir
from kirin.analysis import Forward, ForwardFrame, const
from kirin.dialects import scf
from kirin.lattice import EmptyLattice
from kirin.validation import ValidationPass

from bloqade import qubit, squin
from bloqade.lanes.dialects.steane713 import extensions as extensions
from bloqade.lanes.validation.static_call import (
    NoStaticCallValidation,
    UnrollFixableError,
)


@dataclass
class _SteaneLogicalValidationAnalysis(Forward[EmptyLattice]):
    keys = ("lanes.validate.logical",)

    lattice = EmptyLattice
    addr_frame: ForwardFrame[Address]

    first_gates: dict[int, bool] = field(init=False, default_factory=dict)

    # Gemini capacity policy, not an intrinsic limit of the Steane code.
    max_qubits: ClassVar[int] = 10

    def eval_fallback(self, frame: ForwardFrame, node: ir.Statement):
        if isinstance(node, squin.gate.stmts.Gate):
            # NOTE: report instead of raising so an unsupported gate is listed
            # alongside every other validation error rather than aborting the run
            self.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"Gate {node.name} is not supported in logical Gemini programs!",
                ),
            )

        return tuple(self.lattice.bottom() for _ in range(len(node.results)))

    def check_first_gate(self, qubits: ir.SSAValue) -> bool:
        address = self.addr_frame.get(qubits)

        if not isinstance(address, AddressReg):
            # NOTE: we should have a flat kernel with simple address analysis, so in case we don't
            # get concrete addresses, we might as well error here since something's wrong
            return False

        is_first = True
        for addr_int in address.data:
            is_first = is_first and self.first_gates.get(addr_int, True)
            self.first_gates[addr_int] = False

        return is_first

    def method_self(self, method: ir.Method) -> EmptyLattice:
        return self.lattice.bottom()


@dataclass
class SteaneLogicalValidation(ValidationPass):
    """Validate a flattened Steane logical program under Gemini capacity limits.

    Unresolved calls are reported by delegating to ``NoStaticCallValidation``
    rather than by an impl in ``impls.py``. A ``func.Invoke`` impl only sees the
    statements this ``Forward`` analysis reaches, and the ``scf.For`` impl
    returns bottom without descending into the loop body -- so a call nested in a
    loop was never visited. The delegate is a syntactic ``walk()`` and sees all
    of them.

    Delegating rather than composing the two passes side by side in every suite
    keeps this pass a drop-in for callers who assemble their own
    ``ValidationSuite``, and stops a single unresolved call being counted twice.
    """

    def name(self) -> str:
        return "Gemini Logical Validation"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        address_analysis = AddressAnalysis(method.dialects)
        addr_frame, _ = address_analysis.run(method)

        analysis = _SteaneLogicalValidationAnalysis(
            method.dialects, addr_frame=addr_frame
        )

        # Before the walk, so an unresolved call leads the report: the other
        # violations are often a consequence of the program not being flat.
        # `_validation_errors` is keyed by node and ordered by first insertion,
        # so seeding it here puts these errors first.
        _, call_errors = NoStaticCallValidation().run(method)
        for error in call_errors:
            analysis.add_validation_error(error.node, error)

        frame, _ = analysis.run(method)

        # NOTE: applies to sub-kernels too. One that is handed its qubits rather
        # than allocating them opts out with `verify=False`; it is re-validated
        # as part of its caller once inlined.
        if address_analysis.qubit_count < 1:
            analysis.add_validation_error(
                method.code,
                ir.ValidationError(
                    method.code,
                    "kernel allocates no qubits; at least 1 is required",
                ),
            )
        elif address_analysis.qubit_count > analysis.max_qubits:
            analysis.add_validation_error(
                method.code,
                ir.ValidationError(
                    method.code,
                    f"kernel allocates {address_analysis.qubit_count} qubits, "
                    f"exceeding the maximum of {analysis.max_qubits}",
                ),
            )

        return frame, analysis.get_validation_errors()


@scf.dialect.register(key="lanes.validate.logical")
class __ScfSteaneLogicalValidation(_interp.MethodTable):
    @_interp.impl(scf.IfElse)
    def if_else(
        self,
        interp: _SteaneLogicalValidationAnalysis,
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
        interp: _SteaneLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: scf.For,
    ):
        if not isinstance(stmt.iterable.hints.get("const"), const.Value):
            interp.add_validation_error(
                stmt,
                UnrollFixableError(
                    stmt,
                    "Non-constant iterable in for loop is not supported in Gemini logical programs!",
                ),
            )

        return (interp.lattice.bottom(),)


@gate.dialect.register(key="lanes.validate.logical")
class __GateSteaneLogicalValidation(_interp.MethodTable):
    @_interp.impl(gate.stmts.U3)
    @_interp.impl(gate.stmts.T)
    @_interp.impl(gate.stmts.Rx)
    @_interp.impl(gate.stmts.Ry)
    @_interp.impl(gate.stmts.Rz)
    def non_clifford(
        self,
        interp: _SteaneLogicalValidationAnalysis,
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
        interp: _SteaneLogicalValidationAnalysis,
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
        interp: _SteaneLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: gate.stmts.ControlledGate | gate.stmts.TwoQubitGate,
    ):
        # NOTE: both register operands are acted on; iterate over `args` so this
        # covers the (controls, targets) and (qubits1, qubits2) spellings alike
        for qubits in stmt.args:
            interp.check_first_gate(qubits)

        return ()


@extensions.dialect.register(key="lanes.validate.logical")
class __ExtensionsSteaneLogicalValidation(_interp.MethodTable):
    @_interp.impl(extensions.stmts.StarRz)
    def star_rz(
        self,
        interp: _SteaneLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: extensions.stmts.StarRz,
    ):
        interp.check_first_gate(stmt.qubits)
        return ()


@qubit.dialect.register(key="lanes.validate.logical")
class QubitMethods(_interp.MethodTable):
    @_interp.impl(qubit.stmts.New)
    def check_qubit_allocation(
        self,
        interp: _SteaneLogicalValidationAnalysis,
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
