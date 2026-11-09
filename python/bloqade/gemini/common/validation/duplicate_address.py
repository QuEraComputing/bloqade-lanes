"""Cross-statement validation: each ``NewAt``'s address must be unique.

Implemented as a Forward dataflow analysis with a method-table impl for
``stmts.NewAt``. The impl pulls each address arg via ``maybe_const``, builds a
``LocationAddress``, and accumulates a seen-map on the interpreter. A second
NewAt pinning the same address records a ``ValidationError``.

Const-foldability of the address args is ``ConstAddressValidation``'s job; a
non-const ``NewAt`` is skipped here rather than reported or raised on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kirin import ir
from kirin.analysis.forward import Forward, ForwardFrame
from kirin.lattice.empty import EmptyLattice
from kirin.validation import ValidationPass

if TYPE_CHECKING:
    from bloqade.gemini.common.dialects.qubit.stmts import NewAt
    from bloqade.lanes.bytecode.encoding import LocationAddress


@dataclass
class _DuplicateAddressValidationAnalysis(Forward[EmptyLattice]):
    keys = ("gemini.common.qubit.duplicates",)
    lattice = EmptyLattice

    seen: dict[LocationAddress, NewAt] = field(init=False, default_factory=dict)

    def initialize(self):
        self.seen.clear()
        return super().initialize()

    def method_self(self, method: ir.Method) -> EmptyLattice:
        return self.lattice.bottom()

    def eval_fallback(self, frame: ForwardFrame[EmptyLattice], node: ir.Statement):
        return tuple(self.lattice.bottom() for _ in node.results)


@dataclass
class DuplicateAddressValidation(ValidationPass):
    """Report any pair of ``gemini.common.NewAt`` statements that pin the
    same physical address.
    """

    def name(self) -> str:
        return "gemini.common.qubit.duplicates"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        analysis = _DuplicateAddressValidationAnalysis(method.dialects)
        # `run`, not `run_no_raise`: nothing in this analysis raises by design,
        # and swallowing one that does would truncate the walk and drop every
        # duplicate after it -- silently, which is the failure mode this pass
        # exists to prevent. ValidationSuite already turns an unexpected raise
        # into a reported error rather than a crash.
        frame, _ = analysis.run(method)
        return frame, analysis.get_validation_errors()
