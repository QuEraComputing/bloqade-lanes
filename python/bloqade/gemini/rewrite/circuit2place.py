from dataclasses import dataclass

from bloqade.native.dialects.gate import stmts as gate
from kirin import ir
from kirin.dialects import ilist
from kirin.rewrite import abc

from bloqade.gemini.logical.dialects.operations import stmts as gemini_stmts
from bloqade.lanes.dialects import place
from bloqade.lanes.rewrite.circuit2place import RewritePlaceOperations


class RewriteInitializeToLogicalInitialize(abc.RewriteRule):
    """Rewrite gemini.logical.Initialize statements to place.LogicalInitialize statement."""

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not (
            isinstance(node, gemini_stmts.Initialize)
            and isinstance(qubit_list := node.qubits.owner, ilist.New)
        ):
            return abc.RewriteResult()

        node.replace_by(
            place.LogicalInitialize(
                phi=node.phi,
                theta=node.theta,
                lam=node.lam,
                qubits=qubit_list.values,
            )
        )
        return abc.RewriteResult(has_done_something=True)


@dataclass
class GeminiRewritePlaceOperations(RewritePlaceOperations):
    """
    Gemini-specific rewrite rule to convert native operations to place operations.
    Extends the generic RewritePlaceOperations with handling for Gemini-specific
    statement types (Initialize and TerminalLogicalMeasurement).
    """

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not isinstance(
            node,
            (
                gemini_stmts.TerminalLogicalMeasurement,
                gemini_stmts.Initialize,
                gate.CZ,
                gate.R,
                gate.Rz,
            ),
        ):
            return abc.RewriteResult()
        rewrite_method_name = f"rewrite_{type(node).__name__}"
        rewrite_method = getattr(self, rewrite_method_name)
        return rewrite_method(node)

    def rewrite_Initialize(self, node: gemini_stmts.Initialize) -> abc.RewriteResult:
        if not isinstance(args_list := node.qubits.owner, ilist.New):
            return abc.RewriteResult()

        inputs = args_list.values
        body, block, entry_state = self.prep_region()
        gate_stmt = place.Initialize(
            entry_state,
            phi=node.phi,
            theta=node.theta,
            lam=node.lam,
            qubits=tuple(range(len(inputs))),
        )
        node.replace_by(
            self.construct_execute(gate_stmt, qubits=inputs, body=body, block=block)
        )

        return abc.RewriteResult(has_done_something=True)

    def rewrite_TerminalLogicalMeasurement(
        self, node: gemini_stmts.TerminalLogicalMeasurement
    ) -> abc.RewriteResult:
        if not isinstance(args_list := node.qubits.owner, ilist.New):
            return abc.RewriteResult()

        inputs = args_list.values
        body, block, entry_state = self.prep_region()
        gate_stmt = place.EndMeasure(
            entry_state,
            qubits=tuple(range(len(inputs))),
        )
        new_node = self.construct_execute(
            gate_stmt, qubits=inputs, body=body, block=block
        )
        new_node.insert_before(node)
        node.replace_by(
            place.ConvertToPhysicalMeasurements(
                tuple(new_node.results),
            )
        )

        return abc.RewriteResult(has_done_something=True)
