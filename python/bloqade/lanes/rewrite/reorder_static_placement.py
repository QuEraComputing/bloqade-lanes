from dataclasses import dataclass
from typing import Callable

import rustworkx
from kirin import ir
from kirin.rewrite import abc

from bloqade.lanes.dialects import place
from bloqade.lanes.types import StateType

_BARRIERS: tuple[type, ...] = (place.Initialize, place.EndMeasure)

# Union of all concrete QuantumStmt subclasses that carry a .qubits attribute.
_GateStmt = place.R | place.Rz | place.CZ | place.Initialize | place.EndMeasure


def _asap_schedule(stmts: list[_GateStmt]) -> list[_GateStmt]:
    """Return stmts sorted into ASAP layers. Stable within each layer."""
    if len(stmts) <= 1:
        return list(stmts)

    dag = rustworkx.PyDAG()
    node_for: list[int] = [dag.add_node(i) for i in range(len(stmts))]
    last_touch: dict[int, int] = {}

    for i, stmt in enumerate(stmts):
        for q in stmt.qubits:
            if q in last_touch:
                dag.add_edge(last_touch[q], node_for[i], None)
            last_touch[q] = node_for[i]

    layer: dict[int, int] = {}
    for node_idx in rustworkx.topological_sort(dag):
        preds = dag.predecessor_indices(node_idx)
        layer[node_idx] = 0 if not preds else max(layer[p] for p in preds) + 1

    sorted_indices = sorted(range(len(stmts)), key=lambda i: layer[node_for[i]])
    return [stmts[i] for i in sorted_indices]


def asap_reorder_policy(
    stmts: list[_GateStmt],
) -> list[_GateStmt]:
    """ASAP scheduling policy. Hard barriers (Initialize, EndMeasure) divide
    the body into independent segments; each segment is scheduled separately."""
    result: list[_GateStmt] = []
    segment: list[_GateStmt] = []

    for stmt in stmts:
        if isinstance(stmt, _BARRIERS):
            result.extend(_asap_schedule(segment))
            result.append(stmt)
            segment = []
        else:
            segment.append(stmt)
    result.extend(_asap_schedule(segment))
    return result


@dataclass
class ReorderStaticPlacement(abc.RewriteRule):
    """Reorder quantum statements within a StaticPlacement using a pluggable policy.

    The policy receives the gate statements of the body (excluding the trailing
    Yield and any hard barriers) and returns them in the desired order. Hard
    barriers are preserved in their original positions; the policy only reorders
    the gate statements within each segment between barriers.
    """

    reorder_policy: Callable[[list[_GateStmt]], list[_GateStmt]]

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not isinstance(node, place.StaticPlacement):
            return abc.RewriteResult()

        body_block = node.body.blocks[0]
        old_yield = body_block.last_stmt
        assert isinstance(old_yield, place.Yield)

        stmts: list[_GateStmt] = [
            s
            for s in body_block.stmts
            if isinstance(
                s, (place.R, place.Rz, place.CZ, place.Initialize, place.EndMeasure)
            )
        ]
        if not stmts:
            return abc.RewriteResult()

        new_stmts = self.reorder_policy(stmts)

        if [id(s) for s in new_stmts] == [id(s) for s in stmts]:
            return abc.RewriteResult()

        new_body = ir.Region(new_block := ir.Block())
        curr_state = new_block.args.append_from(StateType, "entry_state")

        for stmt in new_stmts:
            remapped = stmt.from_stmt(stmt, args=(curr_state, *stmt.args[1:]))
            new_block.stmts.append(remapped)
            curr_state = remapped.state_after
            for old_r, new_r in zip(stmt.results[1:], remapped.results[1:]):
                old_r.replace_by(new_r)

        new_block.stmts.append(place.Yield(curr_state, *old_yield.classical_results))

        new_sp = place.StaticPlacement(node.qubits, new_body)
        new_sp.insert_before(node)

        for old_r, new_r in zip(node.results, new_sp.results, strict=True):
            old_r.replace_by(new_r)

        node.delete()
        return abc.RewriteResult(has_done_something=True)
