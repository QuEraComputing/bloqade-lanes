"""Lower native statements to qmove, each wrapped as ``load; qmove.X; store``.

Every rule is local. The output is an intermediate form, not valid qmove: each
statement carries its own ``load``/``store`` placeholders against the state
cell, which validation (V2) rejects. ``qmove_state.thread_method`` turns it
into qmove normal form, joining the chains into one ``enter``..``exit`` chain
threaded through ``scf``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from bloqade.native.dialects.gate import stmts as gate
from kirin import ir
from kirin.dialects import func
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade import qubit
from bloqade.gemini.common.dialects.arrange import stmts as arrange
from bloqade.lanes.dialects import move, qmove


def _wrap(
    node: ir.Statement, make: Callable[[ir.SSAValue], ir.Statement]
) -> ir.Statement:
    load = move.Load()
    load.insert_before(node)
    stateful = make(load.result)
    stateful.insert_before(node)
    move.Store(stateful.results[0]).insert_before(node)
    return stateful


@dataclass
class RewriteNativeToQMove(RewriteRule):
    subroutines: frozenset[ir.Method] = field(default_factory=frozenset)
    """Lowered subroutine clones; invokes of these become ``qmove.invoke``."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if isinstance(node, gate.CZ):
            _wrap(node, lambda s: qmove.CZ(s, node.controls, node.targets))
        elif isinstance(node, gate.R):
            _wrap(
                node,
                lambda s: qmove.R(s, node.axis_angle, node.rotation_angle, node.qubits),
            )
        elif isinstance(node, gate.Rz):
            _wrap(node, lambda s: qmove.Rz(s, node.rotation_angle, node.qubits))
        elif isinstance(node, arrange.MoveTo):
            _wrap(
                node,
                lambda s: qmove.MoveTo(
                    s,
                    node.qubits,
                    node.locations,
                    multi_move_warning=node.multi_move_warning,
                ),
            )
        elif isinstance(node, arrange.Permute):
            _wrap(
                node,
                lambda s: qmove.Permute(
                    s, node.qubits, node.perm, insert_moves=node.insert_moves
                ),
            )
        elif isinstance(node, qubit.stmts.Measure):
            measure = _wrap(node, lambda s: qmove.Measure(s, node.qubits))
            assert isinstance(measure, qmove.Measure)
            measure.measurements.type = node.result.type
            node.result.replace_by(measure.measurements)
        elif isinstance(node, func.Invoke) and node.callee in self.subroutines:
            invoke = _wrap(
                node, lambda s: qmove.Invoke(s, tuple(node.inputs), callee=node.callee)
            )
            assert isinstance(invoke, qmove.Invoke)
            invoke.value.type = node.result.type
            node.result.replace_by(invoke.value)
        else:
            return RewriteResult()
        node.delete()
        return RewriteResult(has_done_something=True)
