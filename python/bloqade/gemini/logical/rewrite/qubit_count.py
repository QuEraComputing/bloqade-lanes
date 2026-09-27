from dataclasses import dataclass

from kirin.ir.nodes.stmt import Statement
from kirin.rewrite.abc import RewriteResult, RewriteRule

from ..dialects.operations.stmts import TerminalLogicalMeasurement


@dataclass
class InsertQubitCount(RewriteRule):
    """Stamp every terminal measurement with its physical-qubits-per-logical width.

    ``@logical.kernel`` always passes Steane [[7,1,3]]'s seven -- see the note
    there. The width stays a field so tests and other pipelines can stamp a
    different one, but nothing in the shipped lowering honours another value.
    """

    num_physical_qubits: int

    def rewrite_Statement(self, node: Statement) -> RewriteResult:
        if not isinstance(node, TerminalLogicalMeasurement):
            return RewriteResult()

        node.num_physical_qubits = self.num_physical_qubits
        return RewriteResult(has_done_something=True)
