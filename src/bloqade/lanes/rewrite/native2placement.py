from dataclasses import dataclass

from bloqade.internal.dialects import move, placement
from bloqade.native.dialects import gates
from kirin import ir
from kirin.analysis import const
from kirin.dialects import ilist
from kirin.rewrite import abc


@dataclass
class RewriteNativeToPlacement(abc.RewriteRule):
    """
    Rewrite rule to convert native operations to placement operations.
    This is a placeholder for the actual implementation.
    """

    gate: str

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if isinstance(node, gates.CZ):
            return self.rewrite_CZ(node)
        elif isinstance(node, gates.R):
            return self.rewrite_R(node)
        elif isinstance(node, gates.Rz):
            return self.rewrite_Rz(node)
        elif isinstance(node, move.Move):
            return self.rewrite_Move(node)

        return abc.RewriteResult()

    def rewrite_CZ(self, node: gates.CZ) -> abc.RewriteResult:
        if not isinstance(args_list := node.qargs.owner, ilist.New) or not isinstance(
            ctrls_list := node.ctrls.owner, ilist.New
        ):
            return abc.RewriteResult()

        qargs = args_list.values
        ctrls = ctrls_list.values

        body = ir.Region(block := ir.Block())
        entry_state = block.args.append_from(placement.StateType, name="entry_state")
        block.stmts.append(
            gate_stmt := placement.CZ(
                entry_state,
                pairs=[(i, i + len(qargs)) for i in range(len(qargs))],
            )
        )
        block.stmts.append(placement.ExitRegion(state=gate_stmt.result))

        node.replace_by(
            placement.ShuttleAtoms(qubits=qargs + ctrls, body=body, gate=self.gate)
        )

        return abc.RewriteResult(has_done_something=True)

    def rewrite_R(self, node: gates.R) -> abc.RewriteResult:
        if not isinstance(args_list := node.inputs.owner, ilist.New):
            return abc.RewriteResult()

        inputs = args_list.values

        body = ir.Region(block := ir.Block())
        entry_state = block.args.append_from(placement.StateType, name="entry_state")

        block.stmts.append(
            gate_stmt := placement.R(
                entry_state,
                inputs=list(range(len(inputs))),
                axis_angle=node.axis_angle,
                rotation_angle=node.rotation_angle,
            )
        )
        block.stmts.append(placement.ExitRegion(state=gate_stmt.result))

        node.replace_by(
            placement.ShuttleAtoms(qubits=inputs, body=body, gate=self.gate)
        )

        return abc.RewriteResult(has_done_something=True)

    def rewrite_Rz(self, node: gates.Rz) -> abc.RewriteResult:
        if not isinstance(args_list := node.inputs.owner, ilist.New):
            return abc.RewriteResult()

        inputs = args_list.values

        body = ir.Region(block := ir.Block())
        entry_state = block.args.append_from(placement.StateType, name="entry_state")

        block.stmts.append(
            gate_stmt := placement.Rz(
                entry_state,
                inputs=list(range(len(inputs))),
                rotation_angle=node.rotation_angle,
            )
        )
        block.stmts.append(placement.ExitRegion(state=gate_stmt.result))

        node.replace_by(
            placement.ShuttleAtoms(qubits=inputs, body=body, gate=self.gate)
        )

        return abc.RewriteResult(has_done_something=True)

    def rewrite_Move(self, node: move.Move):
        if not isinstance(args_list := node.inputs.owner, ilist.New) or not isinstance(
            pos_list := node.positions.hints.get("const"), const.Value
        ):
            return abc.RewriteResult()

        inputs = args_list.values

        body = ir.Region(block := ir.Block())
        entry_state = block.args.append_from(placement.StateType, name="entry_state")

        block.stmts.append(
            gate_stmt := placement.Move(
                entry_state,
                inputs=list(range(len(inputs))),
                positions=list(pos_list.data),
                zone_id=node.zone_id,
            )
        )
        block.stmts.append(placement.ExitRegion(state=gate_stmt.result))

        node.replace_by(
            placement.ShuttleAtoms(qubits=inputs, body=body, gate=self.gate)
        )

        return abc.RewriteResult(has_done_something=True)


class MergePlacementRegions(abc.RewriteRule):
    """
    Merge adjacent placement regions into a single region.
    This is a placeholder for the actual implementation.
    """

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not (
            isinstance(node, placement.ShuttleAtoms)
            and isinstance(next_node := node.next_stmt, placement.ShuttleAtoms)
        ):
            return abc.RewriteResult()

        new_qubits = node.qubits
        new_input_map = {}
        for old_qid, qbit in enumerate(next_node.qubits):
            if qbit not in new_qubits:
                new_input_map[old_qid] = len(new_qubits)
                new_qubits = new_qubits + (qbit,)
            else:
                new_input_map[old_qid] = new_qubits.index(qbit)

        new_body = node.body.clone()

        curr_state = None
        stmt = (curr_block := new_body.blocks[0]).last_stmt
        assert isinstance(stmt, placement.ExitRegion)
        curr_state = stmt.state
        stmt.delete()

        # make sure to copy list of blocks since the loop body will
        # mutate the list contained inside of `next_node.body.blocks`
        next_block = next_node.body.blocks[0]
        stmt = next_block.first_stmt
        while stmt:
            next_stmt = stmt.next_stmt
            stmt.detach()
            if isinstance(stmt, placement.CZ):
                curr_block.stmts.append(
                    new_stmt := placement.CZ(
                        curr_state,
                        pairs=[
                            (new_input_map[i], new_input_map[j]) for i, j in stmt.pairs
                        ],
                    )
                )
                curr_state = new_stmt.result
            elif isinstance(stmt, placement.R):
                curr_block.stmts.append(
                    new_stmt := placement.R(
                        curr_state,
                        inputs=[new_input_map[i] for i in stmt.inputs],
                        axis_angle=stmt.axis_angle,
                        rotation_angle=stmt.rotation_angle,
                    )
                )
                curr_state = new_stmt.result
            elif isinstance(stmt, placement.Rz):
                curr_block.stmts.append(
                    new_stmt := placement.Rz(
                        curr_state,
                        inputs=[new_input_map[i] for i in stmt.inputs],
                        rotation_angle=stmt.rotation_angle,
                    )
                )
                curr_state = new_stmt.result
            elif isinstance(stmt, placement.Move):
                curr_block.stmts.append(
                    new_stmt := placement.Move(
                        curr_state,
                        inputs=[new_input_map[i] for i in stmt.inputs],
                        positions=stmt.positions,
                        zone_id=stmt.zone_id,
                    )
                )
                curr_state = new_stmt.result
            elif isinstance(stmt, placement.ExitRegion):
                curr_block.stmts.append(placement.ExitRegion(state=curr_state))
            else:
                curr_block.stmts.append(stmt)

            stmt = next_stmt

        # replace next node with the new merged region
        next_node.replace_by(
            placement.ShuttleAtoms(
                qubits=new_qubits,
                body=new_body,
                gate=node.gate,
            )
        )
        # delete the node
        node.delete()

        return abc.RewriteResult(has_done_something=True)
