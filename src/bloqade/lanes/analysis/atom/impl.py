from typing import TypeVar

from kirin import interp, ir
from kirin.dialects import cf, ssacfg
from kirin.lattice.empty import EmptyLattice

from ...dialects import move
from .analysis import (
    AtomFrame,
    AtomInterpreter,
    AtomState,
    AtomStateType,
    UnknownAtomState,
)


@ssacfg.dialect.register(key="atom")
class SsaCfg(interp.MethodTable):

    FrameType = TypeVar("FrameType", bound=interp.AbstractFrame)

    @interp.impl(ir.SSACFG())
    def ssacfg(
        self,
        interp_: AtomInterpreter,
        frame: AtomFrame,
        node: ir.Region,
    ):
        result = None
        frame.worklist.append(
            interp.Successor(node.blocks[0], *frame.get_values(node.blocks[0].args))
        )
        frame.atom_states.append(UnknownAtomState())

        while (succ := frame.worklist.pop()) is not None:
            atom_state = frame.atom_states.pop()
            if atom_state is None:
                raise interp.InterpreterError("Missing atom state for successor")

            visited = frame.atom_visited.setdefault(succ.block, set())
            if (atom_state, succ) in visited:
                continue

            block_result = self.run_succ(interp_, atom_state, frame, succ)
            if len(visited) < 128:
                visited.add((atom_state, succ))
            else:
                continue

            if isinstance(block_result, interp.Successor):
                raise interp.InterpreterError(
                    "unexpected successor, successors should be in worklist"
                )

            result = interp_.join_results(result, block_result)

        if isinstance(result, interp.YieldValue):
            return result.values
        return result

    def run_succ(
        self,
        interp_: AtomInterpreter,
        atom_state: AtomStateType,
        frame: AtomFrame,
        succ: interp.Successor,
    ) -> interp.SpecialValue[EmptyLattice]:
        frame.current_block = succ.block
        frame.set_values(succ.block.args, succ.block_args)
        frame.current_state = atom_state

        for stmt in succ.block.stmts:
            frame.current_stmt = stmt
            stmt_results = interp_.frame_eval(frame, stmt)
            if isinstance(stmt_results, tuple):
                frame.set_values(stmt._results, stmt_results)
            elif stmt_results is None:
                continue  # empty result
            else:  # terminate
                return stmt_results
        return None


@cf.dialect.register(key="atom")
class Cf(interp.MethodTable):
    @interp.impl(cf.Branch)
    def branch(
        self,
        interp_: AtomInterpreter,
        frame: AtomFrame,
        stmt: cf.Branch,
    ):
        frame.worklist.append(
            interp.Successor(stmt.successor, *frame.get_values(stmt.arguments))
        )
        frame.atom_states.append(frame.current_state)
        return ()

    @interp.impl(cf.ConditionalBranch)
    def conditional_branch(
        self,
        interp_: AtomInterpreter,
        frame: AtomFrame,
        stmt: cf.ConditionalBranch,
    ):

        frame.worklist.append(
            interp.Successor(
                stmt.then_successor, *frame.get_values(stmt.then_arguments)
            )
        )
        frame.atom_states.append(frame.current_state)

        frame.worklist.append(
            interp.Successor(
                stmt.else_successor, *frame.get_values(stmt.else_arguments)
            )
        )
        frame.atom_states.append(frame.current_state)


@move.dialect.register(key="atom")
class Move(interp.MethodTable):
    @interp.impl(move.Move)
    def move_impl(
        self,
        interp_: AtomInterpreter,
        frame: AtomFrame,
        stmt: move.Move,
    ):
        current_state = frame.current_state
        if not isinstance(current_state, AtomState):
            return

        qubits_to_move = {}
        for move_lane in stmt.lanes:
            src, dst = interp_.arch_spec.get_endpoints(move_lane)
            if src is None or dst is None:
                frame.current_state = UnknownAtomState()
                return

            qubit = current_state.get_qubit(src)
            if qubit is None:
                continue

            qubits_to_move[qubit] = dst

        frame.current_state = current_state.update(qubits_to_move)

    @interp.impl(move.Initialize)
    def initialize_impl(
        self, interp_: AtomInterpreter, frame: AtomFrame, stmt: move.Initialize
    ):
        frame.current_state = AtomState(stmt.location_addresses)
