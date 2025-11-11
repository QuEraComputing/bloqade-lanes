from dataclasses import dataclass

from bloqade.native.dialects.gate import stmts as gate
from kirin import ir
from kirin.dialects import ilist, py
from kirin.rewrite import abc

from bloqade.lanes.dialects import circuit, execute
from bloqade.lanes.types import StateType


@dataclass
class RewriteLowLevelCircuit(abc.RewriteRule):
    """
    Rewrite rule to convert native operations to placement operations.
    This is a placeholder for the actual implementation.
    """

    def default_(self, node: ir.Statement) -> abc.RewriteResult:
        return abc.RewriteResult()

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        rewrite_method_name = f"rewrite_{type(node).__name__}"
        rewrite_method = getattr(self, rewrite_method_name, self.default_)
        return rewrite_method(node)

    def prep_region(self) -> tuple[ir.Region, ir.Block, ir.SSAValue]:
        body = ir.Region(block := ir.Block())
        entry_state = block.args.append_from(StateType, name="entry_state")
        return body, block, entry_state

    def construct_execute(
        self,
        gate_stmt: circuit.QuantumStmt,
        *,
        qubits: tuple[ir.SSAValue, ...],
        body: ir.Region,
        block: ir.Block,
    ) -> circuit.StaticCircuit:
        block.stmts.append(gate_stmt)
        block.stmts.append(circuit.Exit(state=gate_stmt.state_after))

        return circuit.StaticCircuit(qubits=qubits, body=body)

    def rewrite_CZ(self, node: gate.CZ) -> abc.RewriteResult:
        if not isinstance(
            targets_list := node.targets.owner, ilist.New
        ) or not isinstance(controls_list := node.controls.owner, ilist.New):
            return abc.RewriteResult()

        targets = targets_list.values
        controls = controls_list.values
        if len(targets) != len(controls):
            return abc.RewriteResult()

        all_qubits = tuple(range(len(targets) + len(controls)))
        n_controls = len(controls)

        body, block, entry_state = self.prep_region()
        stmt = circuit.CZ(
            entry_state,
            controls=all_qubits[:n_controls],
            targets=all_qubits[n_controls:],
        )

        node.replace_by(
            self.construct_execute(
                stmt, qubits=controls + targets, body=body, block=block
            )
        )

        return abc.RewriteResult(has_done_something=True)

    def rewrite_R(self, node: gate.R) -> abc.RewriteResult:
        if not isinstance(args_list := node.qubits.owner, ilist.New):
            return abc.RewriteResult()

        inputs = args_list.values

        body, block, entry_state = self.prep_region()
        gate_stmt = circuit.R(
            entry_state,
            qubits=tuple(range(len(inputs))),
            axis_angle=node.axis_angle,
            rotation_angle=node.rotation_angle,
        )
        node.replace_by(
            self.construct_execute(gate_stmt, qubits=inputs, body=body, block=block)
        )

        return abc.RewriteResult(has_done_something=True)

    def rewrite_Rz(self, node: gate.Rz) -> abc.RewriteResult:
        if not isinstance(args_list := node.qubits.owner, ilist.New):
            return abc.RewriteResult()

        inputs = args_list.values

        body = ir.Region(block := ir.Block())
        entry_state = block.args.append_from(StateType, name="entry_state")

        gate_stmt = circuit.Rz(
            entry_state,
            qubits=tuple(range(len(inputs))),
            rotation_angle=node.rotation_angle,
        )

        node.replace_by(
            self.construct_execute(gate_stmt, qubits=inputs, body=body, block=block)
        )

        return abc.RewriteResult(has_done_something=True)


class RewriteConstantToStatic(abc.RewriteRule):
    """
    Rewrite rule to convert constant values to static float values.
    """

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not (
            isinstance(node, py.Constant)
            and isinstance(value := node.value.unwrap(), float)
        ):
            return abc.RewriteResult()

        node.replace_by(circuit.StaticFloat(value=value))

        return abc.RewriteResult(has_done_something=True)


class MergePlacementRegions(abc.RewriteRule):
    """
    Merge adjacent placement regions into a single region.
    This is a placeholder for the actual implementation.
    """

    def remap_qubits(
        self, node: circuit.R | circuit.Rz | circuit.CZ, input_map: dict[int, int]
    ) -> circuit.R | circuit.Rz | circuit.CZ:
        if isinstance(node, circuit.CZ):
            return circuit.CZ(
                node.state_before,
                targets=tuple(input_map[i] for i in node.targets),
                controls=tuple(input_map[i] for i in node.controls),
            )
        else:
            return node.from_stmt(
                node,
                attributes={
                    "qubits": ir.PyAttr(tuple(input_map[i] for i in node.qubits))
                },
            )

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not (
            isinstance(node, circuit.StaticCircuit)
            and isinstance(next_node := node.next_stmt, circuit.StaticCircuit)
        ):
            return abc.RewriteResult()

        new_qubits = node.qubits
        new_input_map: dict[int, int] = {}
        for old_qid, qbit in enumerate(next_node.qubits):
            if qbit not in new_qubits:
                new_input_map[old_qid] = len(new_qubits)
                new_qubits = new_qubits + (qbit,)
            else:
                new_input_map[old_qid] = new_qubits.index(qbit)

        new_body = node.body.clone()

        curr_state = None
        stmt = (curr_block := new_body.blocks[0]).last_stmt
        assert isinstance(stmt, circuit.Exit)
        
        if (exit_qubits := stmt.qubits) is None:
            exit_qubits = tuple(range(len(node.qubits)))

        exit_qubits = tuple(new_input_map[q] for q in exit_qubits)

        curr_state = stmt.state
        stmt.delete()

        # make sure to copy list of blocks since the loop body will
        # mutate the list contained inside of `next_node.body.blocks`
        next_block = next_node.body.blocks[0]
        stmt = next_block.first_stmt
        while stmt:
            next_stmt = stmt.next_stmt
            stmt.detach()
            if isinstance(stmt, (circuit.CZ, circuit.R, circuit.Rz)):
                curr_block.stmts.append(
                    new_stmt := self.remap_qubits(stmt, new_input_map)
                )
                curr_state = new_stmt.state_after
            elif isinstance(stmt, circuit.Exit):
                if (other_exit_qubits := stmt.qubits) is None:
                    other_exit_qubits = tuple(range(len(next_node.qubits)))

                exit_qubits = exit_qubits + tuple(new_input_map[q] for q in other_exit_qubits)

                curr_block.stmts.append(type(stmt)(state=curr_state, qubits=exit_qubits))
            else:
                curr_block.stmts.append(stmt)

            stmt = next_stmt

        # replace next node with the new merged region
        next_node.replace_by(
            execute.StaticCircuit(
                qubits=new_qubits,
                body=new_body,
            )
        )
        # delete the node
        node.delete()

        return abc.RewriteResult(has_done_something=True)
