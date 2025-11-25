from dataclasses import dataclass
from typing import Callable

from bloqade.gemini.dialects.logical import stmts as gemini_stmts
from bloqade.native.dialects.gate import stmts as gate
from kirin import ir
from kirin.dialects import ilist
from kirin.rewrite import abc

from bloqade.lanes.dialects import circuit
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
        quantum_stmt: circuit.QuantumStmt,
        *,
        qubits: tuple[ir.SSAValue, ...],
        body: ir.Region,
        block: ir.Block,
    ) -> circuit.StaticCircuit:
        block.stmts.append(quantum_stmt)
        block.stmts.append(
            circuit.Yield(quantum_stmt.state_after, *quantum_stmt.results[1:])
        )

        return circuit.StaticCircuit(qubits=qubits, body=body)

    def rewrite_TerminalLogicalMeasurement(
        self, node: gemini_stmts.TerminalLogicalMeasurement
    ) -> abc.RewriteResult:
        if not isinstance(args_list := node.qubits.owner, ilist.New):
            return abc.RewriteResult()

        inputs = args_list.values
        body, block, entry_state = self.prep_region()
        gate_stmt = circuit.EndMeasure(
            entry_state,
            qubits=tuple(range(len(inputs))),
        )
        new_node = self.construct_execute(
            gate_stmt, qubits=inputs, body=body, block=block
        )
        new_node.insert_before(node)
        node.replace_by(
            circuit.ConvertToPhysicalMeasurements(
                tuple(new_node.results),
            )
        )

        return abc.RewriteResult(has_done_something=True)

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


def _default_merge_heuristic(r1: ir.Region, r2: ir.Region) -> bool:
    # placeholder heuristic: always merge
    return True


@dataclass
class MergePlacementRegions(abc.RewriteRule):
    """
    Merge adjacent placement regions into a single region.
    This is a placeholder for the actual implementation.
    """

    merge_heuristic: Callable[[ir.Region, ir.Region], bool] = _default_merge_heuristic
    """Heuristic function to decide whether to merge two circuit regions."""

    def remap_qubits(
        self,
        curr_state: ir.SSAValue,
        node: circuit.R | circuit.Rz | circuit.CZ | circuit.EndMeasure,
        input_map: dict[int, int],
    ):
        if isinstance(node, circuit.CZ):
            return circuit.CZ(
                curr_state,
                targets=tuple(input_map[i] for i in node.targets),
                controls=tuple(input_map[i] for i in node.controls),
            )
        else:
            return node.from_stmt(
                node,
                args=(curr_state,),
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

        if not self.merge_heuristic(node.body, next_node.body):
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
        new_block = new_body.blocks[0]

        curr_yield = new_block.last_stmt
        assert isinstance(curr_yield, circuit.Yield)

        curr_state = curr_yield.final_state
        current_yields = list(curr_yield.classical_results)
        curr_yield.delete()

        for stmt in next_node.body.blocks[0].stmts:
            if isinstance(
                stmt, (circuit.R, circuit.Rz, circuit.CZ, circuit.EndMeasure)
            ):
                remapped_stmt = self.remap_qubits(curr_state, stmt, new_input_map)
                curr_state = remapped_stmt.results[0]
                new_block.stmts.append(remapped_stmt)
                new_block.stmts.append(remapped_stmt)
                for old_result, new_result in zip(
                    stmt.results[1:], remapped_stmt.results[1:]
                ):
                    current_yields.append(new_result)

        new_yield = circuit.Yield(
            curr_state,
            *current_yields,
        )
        new_block.stmts.append(new_yield)

        # create the new static circuit
        new_static_circuit = circuit.StaticCircuit(
            new_qubits,
            new_body,
        )
        new_static_circuit.insert_before(node)

        # replace old results with new results from new static circuit
        old_results = list(node.results) + list(next_node.results)
        for old_result, new_result in zip(old_results, new_static_circuit.results):
            old_result.replace_by(new_result)

        # delete the old nodes
        node.delete()
        next_node.delete()  # this will be skipped if it is the next item in the Walk worklist

        return abc.RewriteResult(has_done_something=True)
