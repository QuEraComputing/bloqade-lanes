from kirin import exception, interp, ir, types
from kirin.analysis.forward import ForwardFrame
from kirin.decl import info, statement

from bloqade import types as bloqade_types
from bloqade.lanes.analysis.placement import AtomState, PlacementAnalysis
from bloqade.lanes.types import StateType

dialect = ir.Dialect(name="lowlevel.circuit")


@statement(dialect=dialect)
class ConstantFloat(ir.Statement):
    traits = frozenset({ir.ConstantLike()})

    value: float = info.attribute(type=types.Float)
    result: ir.ResultValue = info.result(type=types.Float)


@statement
class QuantumStmt(ir.Statement):
    """This is a base class for all low level statements."""

    state_before: ir.SSAValue = info.argument(StateType)
    state_after: ir.ResultValue = info.result(StateType)

    def __init__(
        self, state_before: ir.SSAValue, *extra_result_types: types.TypeAttribute
    ):
        super().__init__(
            args=(state_before,),
            args_slice={"state_before": 0},
            result_types=(StateType,) + extra_result_types,
        )


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
class EndMeasure(QuantumStmt):
    state_before: ir.SSAValue = info.argument(StateType)

    qubits: tuple[int, ...] = info.attribute()

    def __init__(self, state: ir.SSAValue, *, qubits: tuple[int, ...]):
        result_types = tuple(bloqade_types.MeasurementResultType for _ in qubits)
        super().__init__(state, *result_types)
        self.qubits = qubits


@statement(dialect=dialect)
class Yield(ir.Statement):
    traits = frozenset({ir.IsTerminator()})

    final_state: ir.SSAValue = info.argument(StateType)
    classical_results: tuple[ir.SSAValue, ...] = info.argument(
        bloqade_types.MeasurementResultType
    )

    def __init__(self, final_state: ir.SSAValue, *classical_results: ir.SSAValue):
        super().__init__(
            args=(final_state,),
            args_slice={"final_state": 0, "classical_results": slice(1, None)},
        )


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
    ):
        if isinstance(last_stmt := body.blocks[0].last_stmt, Yield):
            result_types = tuple(value.type for value in last_stmt.classical_results)
        else:
            result_types = ()

        super().__init__(
            args=qubits,
            args_slice={"qubits": slice(0, None)},
            regions=(body,),
            result_types=result_types,
        )

    def check(self) -> None:
        if len(self.body.blocks) != 1:
            raise exception.StaticCheckError(
                "ShuttleAtoms body must have exactly one block"
            )

        body_block = self.body.blocks[0]
        last_stmt = body_block.last_stmt
        if not isinstance(last_stmt, Yield):
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
    def impl_cz(
        self, _interp: PlacementAnalysis, frame: ForwardFrame[AtomState], stmt: CZ
    ):
        return (
            _interp.placement_strategy.cz_placements(
                frame.get(stmt.state_before),
                stmt.controls,
                stmt.targets,
            ),
        )

    @interp.impl(R)
    @interp.impl(Rz)
    def impl_single_qubit_gate(
        self, _interp: PlacementAnalysis, frame: ForwardFrame[AtomState], stmt: R | Rz
    ):
        return (
            _interp.placement_strategy.sq_placements(
                frame.get(stmt.state_before),
                stmt.qubits,
            ),
        )

    @interp.impl(Yield)
    def impl_yield(
        self, _interp: PlacementAnalysis, frame: ForwardFrame[AtomState], stmt: Yield
    ):
        return interp.YieldValue((frame.get(stmt.final_state),))

    @interp.impl(StaticCircuit)
    def impl_static_circuit(
        self,
        _interp: PlacementAnalysis,
        frame: ForwardFrame[AtomState],
        stmt: StaticCircuit,
    ):
        initial_state = _interp.get_inintial_state(stmt.qubits)
        _interp.frame_call_region(frame, stmt, stmt.body, initial_state)
