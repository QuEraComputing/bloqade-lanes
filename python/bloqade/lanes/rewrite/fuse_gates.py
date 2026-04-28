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
        group: list[place.R] = []

        def flush() -> bool:
            if len(group) >= 2:
                self._merge_group(group)
                group.clear()
                return True
            group.clear()
            return False

        for stmt in list(block.stmts):
            if not isinstance(stmt, place.R):
                if flush():
                    changed = True
                continue
            if not group:
                group.append(stmt)
                continue
            if self._can_extend_r(group, stmt):
                group.append(stmt)
            else:
                if flush():
                    changed = True
                group.append(stmt)
        if flush():
            changed = True
        return changed

    @staticmethod
    def _can_extend_r(group: list[place.R], stmt: place.R) -> bool:
        head = group[0]
        tail = group[-1]
        if stmt.axis_angle is not head.axis_angle:
            return False
        if stmt.rotation_angle is not head.rotation_angle:
            return False
        # State-chain adjacency: the stmt's state input must be the tail's state output.
        if stmt.state_before is not tail.state_after:
            return False
        existing_qubits = {q for s in group for q in s.qubits}
        return existing_qubits.isdisjoint(stmt.qubits)

    @staticmethod
    def _merge_group(group: list[place.R]) -> None:
        head = group[0]
        tail = group[-1]
        all_qubits = tuple(q for s in group for q in s.qubits)
        merged = place.R(
            head.state_before,
            axis_angle=head.axis_angle,
            rotation_angle=head.rotation_angle,
            qubits=all_qubits,
        )
        tail.replace_by(merged)
        for stmt in reversed(group[:-1]):
            stmt.delete()
