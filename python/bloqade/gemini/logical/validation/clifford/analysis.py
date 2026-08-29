from dataclasses import dataclass, field
from typing import Any, ClassVar

from bloqade.analysis.address import Address, AddressAnalysis, AddressReg
from kirin import ir
from kirin.analysis import Forward, ForwardFrame
from kirin.lattice import EmptyLattice
from kirin.validation import ValidationPass

from bloqade import squin

# NOTE: safe at module level -- `bloqade.gemini.__init__` imports `common`
# before `logical`, and `common` never imports back into `logical`.
from bloqade.gemini.common.validation.static_call import NoStaticCallValidation


@dataclass
class _GeminiLogicalValidationAnalysis(Forward[EmptyLattice]):
    keys = ("gemini.validate.logical",)

    lattice = EmptyLattice
    addr_frame: ForwardFrame[Address]

    first_gates: dict[int, bool] = field(init=False, default_factory=dict)

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
class GeminiLogicalValidation(ValidationPass):
    """Validates a logical gemini program.

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

        analysis = _GeminiLogicalValidationAnalysis(
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

        if address_analysis.qubit_count > analysis.max_qubits:
            analysis.add_validation_error(
                method.code,
                ir.ValidationError(
                    method.code,
                    f"kernel allocates {address_analysis.qubit_count} qubits, "
                    f"exceeding the maximum of {analysis.max_qubits}",
                ),
            )

        return frame, analysis.get_validation_errors()
