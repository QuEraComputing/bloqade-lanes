from dataclasses import dataclass

from bloqade.internal.analysis.placement.lattice import AtomsState, ConcreteState
from bloqade.internal.dialects import placement
from bloqade.shuttle.dialects import filled, gate, init, spec
from kirin import ir
from kirin.dialects import func, ilist, py
from kirin.rewrite.abc import RewriteResult, RewriteRule


class LiftStatements(RewriteRule):
    """This rule lifts statements out of shuttle atoms."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, placement.ShuttleAtoms):
            return RewriteResult()

        assert (
            len(node.body.blocks) == 1
        ), "ShuttleAtoms body must have exactly one block"

        has_done_something = False
        block = node.body.blocks[0]
        stmt = block.first_stmt
        while stmt:
            next_stmt = stmt.next_stmt
            if not isinstance(stmt, (placement.GateOperation, placement.ExitRegion)):
                has_done_something = True
                stmt.detach()
                stmt.insert_before(node)
            stmt = next_stmt

        assert isinstance(
            block.last_stmt, placement.ExitRegion
        ), "ShuttleAtoms body must end with an ExitRegion"

        if len(block.stmts) == 1:
            has_done_something = True
            node.delete()

        return RewriteResult(has_done_something=has_done_something)


@dataclass
class RewriteCZ(RewriteRule):
    """This rule rewrites gate statements inside shuttle atoms."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, placement.CZ):
            return RewriteResult()

        assert isinstance(
            parent_node := node.parent_stmt, placement.ShuttleAtoms
        ), "CZ must be inside ShuttleAtoms"

        (zone := spec.GetStaticTrap(zone_id=parent_node.gate)).insert_before(node)
        gate.TopHatCZ(zone=zone.result).insert_before(node)
        node.result.replace_by(node.state_before)
        node.delete()

        return RewriteResult(has_done_something=True)


@dataclass
class RewriteRz(RewriteRule):
    """This rule rewrites gate statements inside shuttle atoms."""

    placement_analysis: dict[ir.SSAValue, AtomsState]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, placement.Rz):
            return RewriteResult()

        assert isinstance(
            parent_node := node.parent_stmt, placement.ShuttleAtoms
        ), "Rz must be inside ShuttleAtoms"

        if not isinstance(
            curr_state := self.placement_analysis.get(node.result),
            ConcreteState,
        ):
            return RewriteResult()

        gate_positions = ilist.IList([curr_state.layout[qid] for qid in node.inputs])

        if len(gate_positions) == len(curr_state.layout):
            # Apply globally
            gate.GlobalRz(
                rotation_angle=node.rotation_angle,
            ).insert_before(node)
        else:
            (gate_zone := spec.GetStaticTrap(zone_id=parent_node.gate)).insert_before(
                node
            )
            (positions := py.Constant(gate_positions)).insert_before(node)
            (
                locations := filled.Fill(gate_zone.result, positions.result)
            ).insert_before(node)
            gate.LocalRz(
                rotation_angle=node.rotation_angle,
                zone=locations.result,
            ).insert_before(node)

        node.result.replace_by(node.state_before)
        node.delete()
        return RewriteResult(has_done_something=True)


@dataclass
class RewriteR(RewriteRule):
    """This rule rewrites gate statements inside shuttle atoms."""

    placement_analysis: dict[ir.SSAValue, AtomsState]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, placement.R):
            return RewriteResult()

        assert isinstance(
            parent_node := node.parent_stmt, placement.ShuttleAtoms
        ), "R must be inside ShuttleAtoms"

        if not isinstance(
            curr_state := self.placement_analysis.get(node.result),
            ConcreteState,
        ):
            return RewriteResult()

        gate_positions = ilist.IList([curr_state.layout[qid] for qid in node.inputs])

        if len(gate_positions) == len(curr_state.layout):
            # Apply globally
            gate.GlobalR(
                axis_angle=node.axis_angle,
                rotation_angle=node.rotation_angle,
            ).insert_before(node)
        else:
            (gate_zone := spec.GetStaticTrap(zone_id=parent_node.gate)).insert_before(
                node
            )
            (positions := py.Constant(gate_positions)).insert_before(node)
            (
                locations := filled.Fill(gate_zone.result, positions.result)
            ).insert_before(node)
            gate.LocalR(
                axis_angle=node.axis_angle,
                rotation_angle=node.rotation_angle,
                zone=locations.result,
            ).insert_before(node)

        node.result.replace_by(node.state_before)
        node.delete()
        return RewriteResult(has_done_something=True)


@dataclass
class RemoveMoves(RewriteRule):
    """This rule removes move statements inside shuttle atoms."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, placement.Move):
            return RewriteResult()

        assert isinstance(
            node.parent_stmt, placement.ShuttleAtoms
        ), "Move must be inside ShuttleAtoms"

        node.result.replace_by(node.state_before)
        node.delete()
        return RewriteResult(has_done_something=True)


@dataclass
class InsertLibMoves(RewriteRule):

    placement_analysis: dict[ir.SSAValue, AtomsState]
    move_function: ir.Method

    def get_state(self, node: ir.Statement) -> ir.SSAValue | None:
        if isinstance(node, placement.GateOperation):
            return node.state_before
        elif isinstance(node, placement.ExitRegion):
            return node.state
        else:
            return None

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if (state_before := self.get_state(node)) is None or not isinstance(
            node.parent_stmt, placement.ShuttleAtoms
        ):
            return RewriteResult(has_done_something=False)

        state = self.placement_analysis.get(state_before)
        if not isinstance(state, ConcreteState):
            return RewriteResult(has_done_something=False)

        for layer in state.move_layers:
            x_src = sorted(set(move.src[0] for move in layer))
            y_src = sorted(set(move.src[1] for move in layer))
            x_dst = sorted(set(move.dst[0] for move in layer))
            y_dst = sorted(set(move.dst[1] for move in layer))

            (x_src_stmt := py.Constant(ilist.IList(x_src))).insert_before(node)
            (y_src_stmt := py.Constant(ilist.IList(y_src))).insert_before(node)
            (x_dst_stmt := py.Constant(ilist.IList(x_dst))).insert_before(node)
            (y_dst_stmt := py.Constant(ilist.IList(y_dst))).insert_before(node)

            (
                func.Invoke(
                    (
                        x_src_stmt.result,
                        y_src_stmt.result,
                        x_dst_stmt.result,
                        y_dst_stmt.result,
                    ),
                    callee=self.move_function,
                    kwargs=(),
                )
            ).insert_before(node)

        return RewriteResult(has_done_something=len(state.move_layers) > 0)


def insert_initial_layout(
    entry_method: ir.Method,
    initial_layout: dict[int, tuple[int, int]],
    mem: str,
):
    """Inserts the initial layout into the entry function.

    This function is responsible for inserting the initial layout of qubits
    into the entry function's body. It creates the necessary statements to
    set up the initial state of the qubits.

    Args:
        entry_function (func.Function): The entry function where the initial layout is to be inserted.
        initial_layout (dict[int, tuple[int, int]]): A mapping from qubit IDs to their initial positions.
        mem (str): The zone identifier for the memory zone where the qubits are located.

    Raises:
        AssertionError: If the function body is empty.

    Returns:
        None

    """
    first_stmt = entry_method.callable_region.blocks[0].first_stmt

    assert first_stmt is not None, "Function body must not be empty"
    filled_value = ilist.IList(list(initial_layout.values()))

    (zone_stmt := spec.GetStaticTrap(zone_id=mem)).insert_before(first_stmt)
    (filled_stmt := py.Constant(filled_value)).insert_before(first_stmt)
    (
        ele_stmt := filled.Fill(zone=zone_stmt.result, filled=filled_stmt.result)
    ).insert_before(first_stmt)
    (input_zones := ilist.New((ele_stmt.result,))).insert_before(first_stmt)
    (init.Fill(input_zones.result)).insert_before(first_stmt)

    return RewriteResult(has_done_something=True)
