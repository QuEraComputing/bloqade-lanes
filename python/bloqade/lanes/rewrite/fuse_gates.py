"""FuseAdjacentGates: fuse adjacent same-op same-params R/Rz/StarRz/CZ statements.

A place-dialect → place-dialect rewrite that operates on the body of a
``place.StaticPlacement``. Within that body, runs of textually-adjacent
quantum statements with the same opcode, identical non-qubit SSA arguments,
and pairwise-disjoint qubit sets are collapsed into a single statement
covering the union of the qubits.

See ``docs/superpowers/specs/2026-04-28-place-stage-gate-fusion-design.md``
for the full design.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from kirin import ir
from kirin.rewrite import abc as rewrite_abc

from bloqade.lanes.dialects import place

T = TypeVar("T", place.R, place.Rz, place.StarRz, place.CZ)


class GateGroup(ABC, Generic[T]):
    """A run of textually-adjacent fusable statements of one opcode.

    Maintains ``all_qubits`` incrementally so disjointness checks are O(1)
    per statement. Subclasses implement opcode-specific predicate bits
    (non-qubit SSA arg comparison) and the merged-statement construction.
    """

    def __init__(self) -> None:
        self.statements: list[T] = []
        self.all_qubits: set[int] = set()

    def append(self, stmt: T) -> None:
        self.statements.append(stmt)
        self.all_qubits.update(stmt.qubits)

    def merge_in_place(self) -> bool:
        """If the group has ≥2 stmts, replace the tail with one merged op
        covering all qubits and delete the earlier statements. Returns True
        iff a merge was performed.
        """
        if len(self.statements) < 2:
            return False
        merged = self._build_merged()
        self.statements[-1].replace_by(merged)
        for stmt in reversed(self.statements[:-1]):
            stmt.delete()
        return True

    def _state_chain_ok(self, stmt: T) -> bool:
        return stmt.state_before is self.statements[-1].state_after

    def _qubits_disjoint(self, stmt: T) -> bool:
        return self.all_qubits.isdisjoint(stmt.qubits)

    @abstractmethod
    def can_extend(self, stmt: T) -> bool: ...

    @abstractmethod
    def _build_merged(self) -> ir.Statement: ...


class RGroup(GateGroup[place.R]):
    def can_extend(self, stmt: place.R) -> bool:
        head = self.statements[0]
        return (
            self._state_chain_ok(stmt)
            and self._qubits_disjoint(stmt)
            and stmt.axis_angle is head.axis_angle
            and stmt.rotation_angle is head.rotation_angle
        )

    def _build_merged(self) -> place.R:
        head = self.statements[0]
        return place.R(
            head.state_before,
            axis_angle=head.axis_angle,
            rotation_angle=head.rotation_angle,
            qubits=tuple(q for s in self.statements for q in s.qubits),
        )


class RzGroup(GateGroup[place.Rz]):
    def can_extend(self, stmt: place.Rz) -> bool:
        head = self.statements[0]
        return (
            self._state_chain_ok(stmt)
            and self._qubits_disjoint(stmt)
            and stmt.rotation_angle is head.rotation_angle
        )

    def _build_merged(self) -> place.Rz:
        head = self.statements[0]
        return place.Rz(
            head.state_before,
            rotation_angle=head.rotation_angle,
            qubits=tuple(q for s in self.statements for q in s.qubits),
        )


class StarRzGroup(GateGroup[place.StarRz]):
    def can_extend(self, stmt: place.StarRz) -> bool:
        head = self.statements[0]
        return (
            self._state_chain_ok(stmt)
            and self._qubits_disjoint(stmt)
            and stmt.rotation_angle is head.rotation_angle
            and stmt.qubit_indices == head.qubit_indices
        )

    def _build_merged(self) -> place.StarRz:
        head = self.statements[0]
        return place.StarRz(
            head.state_before,
            head.rotation_angle,
            qubits=tuple(q for s in self.statements for q in s.qubits),
            qubit_indices=head.qubit_indices,
        )


class CZGroup(GateGroup[place.CZ]):
    def can_extend(self, stmt: place.CZ) -> bool:
        # CZ has no non-qubit SSA args.
        return self._state_chain_ok(stmt) and self._qubits_disjoint(stmt)

    def _build_merged(self) -> place.CZ:
        head = self.statements[0]
        # Re-interleave so place.CZ.controls (first half of qubits) and
        # place.CZ.targets (second half) keep returning the right halves.
        controls = tuple(c for s in self.statements for c in s.controls)
        targets = tuple(t for s in self.statements for t in s.targets)
        return place.CZ(head.state_before, qubits=controls + targets)


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
        group: GateGroup | None = None

        for stmt in list(block.stmts):
            if isinstance(stmt, place.R):
                if isinstance(group, RGroup) and group.can_extend(stmt):
                    group.append(stmt)
                    continue
                if group is not None and group.merge_in_place():
                    changed = True
                group = RGroup()
                group.append(stmt)
            elif isinstance(stmt, place.Rz):
                if isinstance(group, RzGroup) and group.can_extend(stmt):
                    group.append(stmt)
                    continue
                if group is not None and group.merge_in_place():
                    changed = True
                group = RzGroup()
                group.append(stmt)
            elif isinstance(stmt, place.StarRz):
                if isinstance(group, StarRzGroup) and group.can_extend(stmt):
                    group.append(stmt)
                    continue
                if group is not None and group.merge_in_place():
                    changed = True
                group = StarRzGroup()
                group.append(stmt)
            elif isinstance(stmt, place.CZ):
                if isinstance(group, CZGroup) and group.can_extend(stmt):
                    group.append(stmt)
                    continue
                if group is not None and group.merge_in_place():
                    changed = True
                group = CZGroup()
                group.append(stmt)
            else:
                if group is not None and group.merge_in_place():
                    changed = True
                group = None

        if group is not None and group.merge_in_place():
            changed = True
        return changed
