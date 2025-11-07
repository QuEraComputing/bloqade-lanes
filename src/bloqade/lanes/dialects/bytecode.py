from kirin import exception, ir
from kirin.decl import info, statement

from bloqade import types as bloqade_types
from bloqade.lanes.layout.encoding import LocationAddress
from bloqade.lanes.types import StateType

dialect = ir.Dialect("bytecode")


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
class ExecuteCode(ir.Statement):
    """This statement represents executing a block of byte code instructions.

    The body region contains the byte code instructions to be executed.
    The inputs are the squin qubits to be used in the execution.

    The region always terminates with an ExitRegion statement, which provides the
    the measurement results for the qubits depending on which byte code was executed.
    """

    traits = frozenset({ir.SSACFG()})
    qubits: tuple[ir.SSAValue, ...] = info.argument(bloqade_types.QubitType)
    body: ir.Region = info.region(multi=False)
    starting_addresses: tuple[LocationAddress, ...] | None = info.attribute()
    measure_result: tuple[ir.ResultValue, ...] = info.result()

    def __init__(
        self,
        qubits: tuple[ir.SSAValue, ...],
        body: ir.Region,
        *,
        starting_addresses: tuple[LocationAddress, ...] | None = None,
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
        if self.starting_addresses is not None and len(self.qubits) != len(
            self.starting_addresses
        ):
            raise exception.StaticCheckError(
                "Number of qubits must match number of physical addresses"
            )

        if len(self.body.blocks) != 1:
            raise exception.StaticCheckError(
                "ShuttleAtoms body must have exactly one block"
            )

        body_block = self.body.blocks[0]
        last_stmt = body_block.last_stmt
        if not isinstance(last_stmt, ExitRegion):
            raise exception.StaticCheckError(
                "ShuttleAtoms body must end with an ExitRegion statement"
            )

        stmt = body_block.first_stmt
        while stmt is not last_stmt:
            if not isinstance(stmt, ByteCodeStmt):
                raise exception.StaticCheckError(
                    "All statements in ShuttleAtoms body must be ByteCodeStmt"
                )
            stmt = stmt.next_stmt
