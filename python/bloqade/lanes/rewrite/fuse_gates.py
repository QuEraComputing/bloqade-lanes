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
        # StaticPlacement.check() guarantees a single block.
        body_block = node.body.blocks[0]
        changed = self._fuse_block(body_block)
        return rewrite_abc.RewriteResult(has_done_something=changed)

    def _fuse_block(self, block: ir.Block) -> bool:
        # Skeleton — no fusion logic yet. Filled in by Task 2.
        _ = block
        return False
