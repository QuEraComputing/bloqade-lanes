from dataclasses import dataclass
from typing import Callable, TypeGuard

from bloqade.gemini.dialects.logical import stmts as gemini_stmts
from bloqade.native.dialects.gate import stmts as gate
from kirin import ir, types
from kirin.dialects import ilist, py
from kirin.rewrite import abc

from bloqade import qubit
from bloqade.lanes.dialects import place
from bloqade.lanes.types import StateType


class RewriteInitializeToLogicalInitialize(abc.RewriteRule):
    """Rewrite gemini.logical.Initialize statements to place.LogicalInitialize statement."""

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not (
            isinstance(node, gemini_stmts.Initialize)
            and isinstance(qubit_list := node.qubits.owner, ilist.New)
        ):
            return abc.RewriteResult()

        node.replace_by(
            place.LogicalInitialize(
                phi=node.phi,
                theta=node.theta,
                lam=node.lam,
                qubits=qubit_list.values,
            )
        )
        return abc.RewriteResult(has_done_something=True)


class RewriteLogicalInitializeToNewLogical(abc.RewriteRule):
    """Rewrite qubit references in place.LogicalInitialize statements to place.NewLogicalQubit allocations."""

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not isinstance(node, place.LogicalInitialize):
            return abc.RewriteResult()

        def is_qubit_new(owner: ir.Statement | ir.Block) -> TypeGuard[qubit.stmts.New]:
            return isinstance(owner, qubit.stmts.New)

        qubit_stmts = tuple(
            filter(is_qubit_new, (qubit.owner for qubit in node.qubits))
        )

        for qubit_stmt in qubit_stmts:
            qubit_stmt.replace_by(place.NewLogicalQubit(node.theta, node.phi, node.lam))

        return abc.RewriteResult(has_done_something=len(qubit_stmts) > 0)


class CleanUpLogicalInitialize(abc.RewriteRule):
    """Clean up any remaining LogicalInitialize statements."""

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not isinstance(node, place.LogicalInitialize):
            return abc.RewriteResult()

        qubit_stmts = tuple(qubit.owner for qubit in node.qubits)
        if not types.is_tuple_of(qubit_stmts, place.NewLogicalQubit):
            return abc.RewriteResult()

        node.delete()
        return abc.RewriteResult(has_done_something=True)


class InitializeNewQubits(abc.RewriteRule):
    """Rewrite NewLogical qubit allocations to Initialize statements."""

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not isinstance(node, qubit.stmts.New):
            return abc.RewriteResult()

        (zero := py.Constant(0.0)).insert_before(node)
        node.replace_by(
            place.NewLogicalQubit(
                theta=zero.result,
                phi=zero.result,
                lam=zero.result,
            )
        )
        return abc.RewriteResult(has_done_something=True)


@dataclass
class RewritePlaceOperations(abc.RewriteRule):
    """
    Rewrite rule to convert native operations to place operations.
    This is a placeholder for the actual implementation.
    """

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not isinstance(
            node,
            (
                gemini_stmts.TerminalLogicalMeasurement,
                gemini_stmts.Initialize,
                gate.CZ,
                gate.R,
                gate.Rz,
            ),
        ):
            return abc.RewriteResult()
        rewrite_method_name = f"rewrite_{type(node).__name__}"
        rewrite_method = getattr(self, rewrite_method_name)
        return rewrite_method(node)

    def prep_region(self) -> tuple[ir.Region, ir.Block, ir.SSAValue]:
        body = ir.Region(block := ir.Block())
        entry_state = block.args.append_from(StateType, name="entry_state")
        return body, block, entry_state

    def construct_execute(
        self,
        quantum_stmt: place.QuantumStmt,
        *,
        qubits: tuple[ir.SSAValue, ...],
        body: ir.Region,
        block: ir.Block,
    ) -> place.StaticPlacement:
        block.stmts.append(quantum_stmt)
        block.stmts.append(
            place.Yield(quantum_stmt.state_after, *quantum_stmt.results[1:])
        )

        return place.StaticPlacement(qubits=qubits, body=body)

    def rewrite_Initialize(self, node: gemini_stmts.Initialize) -> abc.RewriteResult:
        if not isinstance(args_list := node.qubits.owner, ilist.New):
            return abc.RewriteResult()

        inputs = args_list.values
        body, block, entry_state = self.prep_region()
        gate_stmt = place.Initialize(
            entry_state,
            phi=node.phi,
            theta=node.theta,
            lam=node.lam,
            qubits=tuple(range(len(inputs))),
        )
        node.replace_by(
            self.construct_execute(gate_stmt, qubits=inputs, body=body, block=block)
        )

        return abc.RewriteResult(has_done_something=True)

    def rewrite_TerminalLogicalMeasurement(
        self, node: gemini_stmts.TerminalLogicalMeasurement
    ) -> abc.RewriteResult:
        if not isinstance(args_list := node.qubits.owner, ilist.New):
            return abc.RewriteResult()

        inputs = args_list.values
        body, block, entry_state = self.prep_region()
        gate_stmt = place.EndMeasure(
            entry_state,
            qubits=tuple(range(len(inputs))),
        )
        new_node = self.construct_execute(
            gate_stmt, qubits=inputs, body=body, block=block
        )
        new_node.insert_before(node)
        node.replace_by(
            place.ConvertToPhysicalMeasurements(
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

        body, block, entry_state = self.prep_region()
        stmt = place.CZ(
            entry_state,
            qubits=all_qubits,
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
        gate_stmt = place.R(
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

        gate_stmt = place.Rz(
            entry_state,
            qubits=tuple(range(len(inputs))),
            rotation_angle=node.rotation_angle,
        )

        node.replace_by(
            self.construct_execute(gate_stmt, qubits=inputs, body=body, block=block)
        )

        return abc.RewriteResult(has_done_something=True)


class HoistConstants(abc.RewriteRule):
    """This rewrite rule hoists all constant values to the top of the kernel."""

    def is_pure(self, node: ir.Statement) -> bool:
        return (
            node.has_trait(ir.Pure)
            or (maybe_pure := node.get_trait(ir.MaybePure)) is not None
            and maybe_pure.is_pure(node)
        )

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not (
            isinstance(node, py.Constant)
            and (parent_block := node.parent_block) is not None
            and (first_stmt := parent_block.first_stmt) is not None
            and node is not first_stmt
        ):
            return abc.RewriteResult()

        node.detach()
        node.insert_before(first_stmt)

        return abc.RewriteResult(has_done_something=True)


class HoistNewQubitsUp(abc.RewriteRule):
    """This rewrite rule shifts all qubit allocations before all gate operations."""

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not (
            isinstance(node, place.NewLogicalQubit)
            and (parent_block := node.parent_block) is not None
            and (first_stmt := parent_block.first_stmt) is not None
            and node is not first_stmt
        ):
            return abc.RewriteResult()

        stmt = node.prev_stmt
        node.detach()

        def loop_cond(stmt: ir.Statement | None) -> TypeGuard[ir.Statement]:
            return (
                stmt is not None
                and not isinstance(stmt, place.NewLogicalQubit)
                and set(stmt.results).isdisjoint(node.args)
            )

        while loop_cond(stmt):
            stmt = stmt.prev_stmt

        if stmt is None:
            node.insert_before(first_stmt)
        else:
            node.insert_after(stmt)

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

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not (
            isinstance(node, place.StaticPlacement)
            and isinstance(next_node := node.next_stmt, place.StaticPlacement)
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
        assert isinstance(curr_yield, place.Yield)

        curr_state = curr_yield.final_state
        current_yields = list(curr_yield.classical_results)
        curr_yield.delete()

        for stmt in next_node.body.blocks[0].stmts:
            if isinstance(
                stmt, (place.R, place.Rz, place.CZ, place.EndMeasure, place.Initialize)
            ):
                remapped_stmt = stmt.from_stmt(
                    stmt,
                    args=(curr_state, *stmt.args[1:]),
                    attributes={
                        "qubits": ir.PyAttr(
                            tuple(new_input_map[i] for i in stmt.qubits)
                        )
                    },
                )
                curr_state = remapped_stmt.results[0]
                new_block.stmts.append(remapped_stmt)
                for old_result, new_result in zip(
                    stmt.results[1:], remapped_stmt.results[1:]
                ):
                    old_result.replace_by(new_result)
                    current_yields.append(new_result)

        new_yield = place.Yield(
            curr_state,
            *current_yields,
        )
        new_block.stmts.append(new_yield)

        # create the new static circuit
        new_static_circuit = place.StaticPlacement(
            new_qubits,
            new_body,
        )
        new_static_circuit.insert_before(node)

        # replace old results with new results from new static circuit
        old_results = list(node.results) + list(next_node.results)
        for old_result, new_result in zip(
            old_results, new_static_circuit.results, strict=True
        ):
            old_result.replace_by(new_result)

        # delete the old nodes
        node.delete()
        next_node.delete()  # this will be skipped if it is the next item in the Walk worklist

        return abc.RewriteResult(has_done_something=True)
