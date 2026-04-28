"""FuseAdjacentGates: fuse adjacent same-op same-params R/Rz/CZ statements.

A place-dialect → place-dialect rewrite that operates on the body of a
``place.StaticPlacement``. Within that body, runs of textually-adjacent
quantum statements with the same opcode, identical non-qubit SSA arguments,
and pairwise-disjoint qubit sets are collapsed into a single statement
covering the union of the qubits.

See ``docs/superpowers/specs/2026-04-28-place-stage-gate-fusion-design.md``
for the full design.
"""

from dataclasses import dataclass

from kirin import ir
from kirin.rewrite import abc as rewrite_abc

from bloqade.lanes.dialects import place

# Opcodes that are eligible for fusion. Other QuantumStmts (Initialize,
# EndMeasure) and non-quantum statements (Yield, etc.) flush the current
# group and do not start a new one.
_FUSABLE_TYPES = (place.R, place.Rz)


@dataclass
class FuseAdjacentGates(rewrite_abc.RewriteRule):
    """Fuse adjacent same-op same-params R/Rz/CZ statements with disjoint qubits."""

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not isinstance(node, place.StaticPlacement):
            return rewrite_abc.RewriteResult()
        body_block = node.body.blocks[0]
        changed = self._fuse_block(body_block)
        return rewrite_abc.RewriteResult(has_done_something=changed)

    def _fuse_block(self, block: ir.Block) -> bool:
        changed = False
        group: list[place.R | place.Rz] = []

        def flush() -> bool:
            if len(group) >= 2:
                _merge_group(group)
                group.clear()
                return True
            group.clear()
            return False

        for stmt in list(block.stmts):
            if not isinstance(stmt, _FUSABLE_TYPES):
                if flush():
                    changed = True
                continue
            if not group:
                group.append(stmt)
                continue
            if _can_extend(group, stmt):
                group.append(stmt)
            else:
                if flush():
                    changed = True
                group.append(stmt)
        if flush():
            changed = True
        return changed


def _can_extend(group: list[place.R | place.Rz], stmt: place.R | place.Rz) -> bool:
    head = group[0]
    tail = group[-1]
    if type(stmt) is not type(head):
        return False
    # State-chain adjacency.
    if stmt.state_before is not tail.state_after:
        return False
    if not _same_non_qubit_args(head, stmt):
        return False
    existing_qubits = {q for s in group for q in s.qubits}
    return existing_qubits.isdisjoint(stmt.qubits)


def _same_non_qubit_args(a: place.R | place.Rz, b: place.R | place.Rz) -> bool:
    """SSA-identity comparison of non-qubit args. Assumes type(a) is type(b)."""
    if isinstance(a, place.R):
        assert isinstance(b, place.R)
        return a.axis_angle is b.axis_angle and a.rotation_angle is b.rotation_angle
    if isinstance(a, place.Rz):
        assert isinstance(b, place.Rz)
        return a.rotation_angle is b.rotation_angle
    raise AssertionError(f"unfusable opcode in predicate: {type(a)}")


def _merge_group(group: list[place.R | place.Rz]) -> None:
    head = group[0]
    tail = group[-1]
    if isinstance(head, place.R):
        all_qubits = tuple(q for s in group for q in s.qubits)
        merged: ir.Statement = place.R(
            head.state_before,
            axis_angle=head.axis_angle,
            rotation_angle=head.rotation_angle,
            qubits=all_qubits,
        )
    elif isinstance(head, place.Rz):
        all_qubits = tuple(q for s in group for q in s.qubits)
        merged = place.Rz(
            head.state_before,
            rotation_angle=head.rotation_angle,
            qubits=all_qubits,
        )
    else:
        raise AssertionError(f"unfusable opcode in merge: {type(head)}")
    tail.replace_by(merged)
    for stmt in reversed(group[:-1]):
        stmt.delete()
