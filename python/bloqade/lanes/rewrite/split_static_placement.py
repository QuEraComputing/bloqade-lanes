from dataclasses import dataclass
from typing import Callable

from kirin import ir
from kirin.rewrite import abc

from bloqade.lanes.dialects import place
from bloqade.lanes.types import StateType

_GateStmt = place.R | place.Rz | place.CZ


def cz_layer_split_policy(body_block: ir.Block) -> list[ir.Block]:
    """Split a StaticPlacement body into CZ-anchored groups (policy A).

    Accumulates SQ gates (R, Rz); when a CZ is encountered, flushes
    [accumulated SQ + CZ] as one group. Remaining SQ after the last CZ
    forms the final group. Returns the original block unchanged if there
    is at most one CZ (no split needed).
    """
    old_yield = body_block.last_stmt
    assert isinstance(old_yield, place.Yield)
    classical_results = tuple(old_yield.classical_results)

    _supported = (place.R, place.Rz, place.CZ)
    for stmt in body_block.stmts:
        if not isinstance(stmt, (*_supported, place.Yield)):
            return [body_block]

    stmts: list[_GateStmt] = [s for s in body_block.stmts if isinstance(s, _supported)]

    if not any(isinstance(s, place.CZ) for s in stmts):
        return [body_block]

    groups: list[list[_GateStmt]] = []
    sq_accum: list[_GateStmt] = []

    for stmt in stmts:
        if isinstance(stmt, place.CZ):
            groups.append(sq_accum + [stmt])
            sq_accum = []
        else:
            sq_accum.append(stmt)

    if sq_accum:
        groups.append(sq_accum)

    if len(groups) <= 1:
        return [body_block]

    new_blocks: list[ir.Block] = []
    for group_idx, group in enumerate(groups):
        new_block = ir.Block()
        curr_state = new_block.args.append_from(StateType, "entry_state")

        for stmt in group:
            remapped = stmt.from_stmt(stmt, args=(curr_state, *stmt.args[1:]))
            new_block.stmts.append(remapped)
            curr_state = remapped.state_after

        is_last = group_idx == len(groups) - 1
        extra = classical_results if is_last else ()
        new_block.stmts.append(place.Yield(curr_state, *extra))
        new_blocks.append(new_block)

    return new_blocks


@dataclass
class SplitStaticPlacement(abc.RewriteRule):
    """Split a StaticPlacement body into multiple StaticPlacement statements.

    The policy receives the body block (with its fully-threaded state chain)
    and returns a list of new blocks. Each block becomes one StaticPlacement.
    If the policy returns ≤1 block the rewriter is a no-op.

    The policy is responsible for state threading: each output block must
    start with a block argument of StateType and end with place.Yield.
    The last block must carry the classical results from the original Yield.
    """

    split_policy: Callable[[ir.Block], list[ir.Block]]

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not isinstance(node, place.StaticPlacement):
            return abc.RewriteResult()

        body_block = node.body.blocks[0]
        new_blocks = self.split_policy(body_block)

        if len(new_blocks) <= 1:
            return abc.RewriteResult()

        new_sps: list[place.StaticPlacement] = []
        for block in new_blocks:
            # 1. Scan which local qubit indices this block's statements reference.
            used: set[int] = set()
            for stmt in block.stmts:
                if isinstance(stmt, (place.R, place.Rz, place.CZ)):
                    used.update(stmt.qubits)
            qubit_indices = sorted(used)
            remap = {orig: new for new, orig in enumerate(qubit_indices)}

            # 2. Remap qubit indices in-place: replace each gate statement with a
            #    clone that has updated qubit attributes, propagating the result SSA
            #    values so the state chain and Yield remain valid.
            gate_stmts = [
                s for s in block.stmts if isinstance(s, (place.R, place.Rz, place.CZ))
            ]
            for stmt in gate_stmts:
                new_stmt = stmt.from_stmt(
                    stmt,
                    attributes={
                        "qubits": ir.PyAttr(tuple(remap[q] for q in stmt.qubits))
                    },
                )
                stmt.replace_by(new_stmt)

            # 3. Build the narrowed qubit tuple and wrap in a new StaticPlacement.
            new_qubits = tuple(node.qubits[i] for i in qubit_indices)
            new_sps.append(place.StaticPlacement(new_qubits, ir.Region(block)))

        for sp in new_sps:
            sp.insert_before(node)

        for old_r, new_r in zip(node.results, new_sps[-1].results, strict=True):
            old_r.replace_by(new_r)

        node.delete()
        return abc.RewriteResult(has_done_something=True)
