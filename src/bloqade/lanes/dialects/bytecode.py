from kirin import exception, ir, types
from kirin.decl import info, statement

from bloqade import types as bloqade_types

dialect = ir.Dialect("bytecode")


class State:
    pass


StateType = types.PyClass(State)


@statement
class ByteCodeStmt(ir.Statement):
    """This is a base class for all byte code statements."""

    state_before: ir.SSAValue = info.argument(StateType)
    result: ir.ResultValue = info.result(StateType)


@statement
class ExitRegion(ir.Statement):
    """This is base class for terminal statements in the byte code."""

    traits = frozenset({ir.IsTerminator()})
    state: ir.SSAValue = info.argument(StateType)


@statement(dialect=dialect)
class ByteCodeRegion(ir.Statement):
    traits = frozenset({ir.SSACFG()})
    qubits: tuple[ir.SSAValue, ...] = info.argument(bloqade_types.QubitType)
    body: ir.Region = info.region(multi=False)
    starting_addresses: tuple[tuple[int, int], ...] = info.attribute()
    measure_result: tuple[ir.ResultValue, ...] = info.result()

    def __init__(
        self,
        qubits: tuple[ir.SSAValue, ...],
        body: ir.Region,
        *,
        starting_addresses: tuple[tuple[int, int], ...],
    ):

        result_types = tuple(bloqade_types.MeasurementResultType for _ in qubits)

        super().__init__(
            args=qubits,
            args_slice={"qubits": slice(0, None)},
            regions=(body,),
            result_types=result_types,
            attributes={"starting_addresses": ir.PyAttr(starting_addresses)},
        )

    def check(self) -> None:
        if len(self.qubits) != len(self.starting_addresses):
            raise exception.StaticCheckError(
                "Number of qubits must match number of physical addresses"
            )

        if len(self.body.blocks) != 1:
            raise exception.StaticCheckError(
                "ShuttleAtoms body must have exactly one block"
            )

        if not isinstance((self.body.blocks[0].last_stmt), ExitRegion):
            raise exception.StaticCheckError(
                "ShuttleAtoms body must end with an ExitRegion"
            )

        if not all(
            isinstance(stmt, (ByteCodeStmt, ExitRegion))
            for stmt in self.body.blocks[0].stmts
        ):
            raise exception.StaticCheckError(
                "ShuttleAtoms body can only contain GateOperations and ExitRegion"
            )
