from kirin import exception, interp, ir, types
from kirin.decl import info, statement

from bloqade.lanes.analysis.placement import PlacementAnalysis, LocalID


from bloqade import types as bloqade_types
from bloqade.lanes.layout.encoding import LocationAddress
from bloqade.lanes.types import StateType

dialect = ir.Dialect(name="lowlevel.circuit")

@statement(dialect=dialect)
class StaticFloat(ir.Statement):
    traits = frozenset({ir.ConstantLike()})

    value: float = info.attribute(type=types.Float)
    result: ir.ResultValue = info.result(type=types.Float)


@statement
class QuantumStmt(ir.Statement):
    """This is a base class for all low level statements."""

    state_before: ir.SSAValue = info.argument(StateType)
    state_after: ir.ResultValue = info.result(StateType)


@statement(dialect=dialect)
class CZ(QuantumStmt):
    targets: tuple[int, ...] = info.attribute()
    controls: tuple[int, ...] = info.attribute()


@statement(dialect=dialect)
class R(QuantumStmt):
    qubits: tuple[int, ...] = info.attribute()

    axis_angle: ir.SSAValue = info.argument(type=types.Float)
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class Rz(QuantumStmt):
    qubits: tuple[int, ...] = info.attribute()
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class Exit(ir.Statement):
    traits = frozenset({ir.IsTerminator()})

    qubits: tuple[int, ...] | None = info.attribute(default=None)
    state: ir.SSAValue = info.argument(StateType)
    


@statement(dialect=dialect)
class StaticCircuit(ir.Statement):
    """This statement represents A static circuit to be executed on the hardware.

    The body region contains the low-level instructions to be executed.
    The inputs are the squin qubits to be used in the execution.

    The region always terminates with an ExitRegion statement, which provides the
    the measurement results for the qubits depending on which low-level code was executed.
    """

    traits = frozenset({ir.SSACFG()})
    qubits: tuple[ir.SSAValue, ...] = info.argument(bloqade_types.QubitType)
    body: ir.Region = info.region(multi=False)

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
        if len(self.body.blocks) != 1:
            raise exception.StaticCheckError(
                "ShuttleAtoms body must have exactly one block"
            )

        body_block = self.body.blocks[0]
        last_stmt = body_block.last_stmt
        if not isinstance(last_stmt, Exit):
            raise exception.StaticCheckError(
                "ShuttleAtoms body must end with an EndMeasure statement"
            )

        stmt = body_block.first_stmt
        while stmt is not last_stmt:
            if not isinstance(stmt, QuantumStmt):
                raise exception.StaticCheckError(
                    "All statements in ShuttleAtoms body must be ByteCodeStmt"
                )
            stmt = stmt.next_stmt







@dialect.register(key="runtime.placement")
class PlacementMethods(interp.MethodTable):
    
    @interp.impl(CZ)
    def impl_cz(self, _interp: PlacementAnalysis, frame, stmt: CZ):

        current_state = frame.get(stmt.state_before)
        new_state = _interp.get_placement_cz(
            current_state,
            tuple(map(LocalID, stmt.controls)),
            tuple(map(LocalID, stmt.targets)),
        )
        return (new_state,)

    @interp.impl(R)
    @interp.impl(Rz)
    def impl_single_qubit_gate(self, _interp: PlacementAnalysis, frame, stmt: R | Rz):
        current_state = frame.get(stmt.state_before)
        return (
            _interp.get_placement_sq(
                current_state,
                tuple(map(LocalID, stmt.qubits)),
            ),
        )
