from bloqade.analysis import address
from kirin import exception, interp, ir, types
from kirin.analysis.forward import ForwardFrame
from kirin.decl import info, statement
from kirin.dialects import ilist
from kirin.lattice.empty import EmptyLattice

from bloqade import types as bloqade_types
from bloqade.lanes.analysis.layout import LayoutAnalysis
from bloqade.lanes.analysis.placement import AtomState, ConcreteState, PlacementAnalysis
from bloqade.lanes.types import StateType

dialect = ir.Dialect(name="lanes.place")


@statement(dialect=dialect)
class LogicalInitialize(ir.Statement):
    """Initialize logical qubits in the |0> state.

    Args:
        qubits (tuple[int, ...]): The logical qubit IDs to initialize.

    """

    theta: ir.SSAValue = info.argument(type=types.Float)
    phi: ir.SSAValue = info.argument(type=types.Float)
    lam: ir.SSAValue = info.argument(type=types.Float)
    qubits: tuple[ir.SSAValue, ...] = info.argument(bloqade_types.QubitType)


@statement(dialect=dialect)
class NewLogicalQubit(ir.Statement):
    """Allocate new logical qubits with initial state u3(theta, phi, lam)|0>.

    Args:
        theta (float): Angle for rotation around the Y axis
        phi (float): angle for rotation around the Z axis
        lam (float): angle for rotation around the Z axis

    """

    traits = frozenset()
    theta: ir.SSAValue = info.argument(types.Float)
    phi: ir.SSAValue = info.argument(types.Float)
    lam: ir.SSAValue = info.argument(types.Float)
    result: ir.ResultValue = info.result(bloqade_types.QubitType)


@statement(init=False)
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
class Initialize(QuantumStmt):
    qubits: tuple[int, ...] = info.attribute()

    theta: ir.SSAValue = info.argument(type=types.Float)
    phi: ir.SSAValue = info.argument(type=types.Float)
    lam: ir.SSAValue = info.argument(type=types.Float)


@statement(dialect=dialect)
class CZ(QuantumStmt):
    qubits: tuple[int, ...] = info.attribute()

    @property
    def controls(self) -> tuple[int, ...]:
        return self.qubits[: len(self.qubits) // 2]

    @property
    def targets(self) -> tuple[int, ...]:
        return self.qubits[len(self.qubits) // 2 :]


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
class ConvertToPhysicalMeasurements(ir.Statement):
    """Convert logical measurement results to physical measurement results.

    This is a placeholder for a rewrite pass that will explicitly extract physical
    measurement results from the future returned, in the move dialect.
    """

    logical_measurements: tuple[ir.SSAValue, ...] = info.argument(
        type=bloqade_types.MeasurementResultType
    )

    result: ir.ResultValue = info.result(
        type=ilist.IListType[ilist.IListType[bloqade_types.MeasurementResultType]]
    )


@statement(dialect=dialect)
class Yield(ir.Statement):
    traits = frozenset({ir.IsTerminator()})

    final_state: ir.SSAValue = info.argument(StateType)
    classical_results: tuple[ir.SSAValue, ...] = info.argument(
        bloqade_types.MeasurementResultType
    )

    def __init__(self, final_state: ir.SSAValue, *classical_results: ir.SSAValue):
        super().__init__(
            args=(final_state, *classical_results),
            args_slice={"final_state": 0, "classical_results": slice(1, None)},
        )


@statement(dialect=dialect)
class StaticPlacement(ir.Statement):
    """This statement represents A static circuit to be executed on the hardware.

    The body region contains the low-level instructions to be executed.
    The inputs are the squin qubits to be used in the execution.

    The region always terminates with an Yield statement, which provides the
    the measurement results for the qubits depending on which low-level code was executed.
    """

    traits = frozenset({ir.SSACFG(), ir.HasCFG()})
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
                "StaticCircuit body must have exactly one block"
            )

        body_block = self.body.blocks[0]
        last_stmt = body_block.last_stmt
        if not isinstance(last_stmt, Yield):
            raise exception.StaticCheckError(
                "StaticCircuit body must end with an EndMeasure statement"
            )

        if len(last_stmt.classical_results) != len(self.results):
            raise exception.StaticCheckError(
                "Number of yielded classical results in Yield does not match number of results in StaticCircuit"
            )


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
        return interp.YieldValue(frame.get_values(stmt.args))

    @interp.impl(StaticPlacement)
    def impl_static_circuit(
        self,
        _interp: PlacementAnalysis,
        frame: ForwardFrame[AtomState],
        stmt: StaticPlacement,
    ):

        initial_state = _interp.get_inintial_state(stmt.qubits)
        with _interp.new_frame(stmt, has_parent_access=True) as circuit_frame:

            frame_call_result = _interp.frame_call_region(
                circuit_frame, stmt, stmt.body, initial_state
            )

            frame.set_values(
                circuit_frame.entries.keys(), circuit_frame.entries.values()
            )

        match frame_call_result:
            case (ConcreteState() as final_state, *ret):
                for qid, qubit in enumerate(stmt.qubits):
                    _interp.move_count[qubit] += final_state.move_count[qid]

                return tuple(ret)
            case _:
                raise interp.InterpreterError(
                    "StaticPlacement body did not return a ConcreteState"
                )

    @interp.impl(EndMeasure)
    def end_measure(
        self,
        _interp: PlacementAnalysis,
        frame: ForwardFrame[AtomState],
        stmt: EndMeasure,
    ):
        new_state = _interp.placement_strategy.measure_placements(
            frame.get(stmt.state_before), stmt.qubits
        )
        return (new_state,) + (AtomState.bottom(),) * len(stmt.qubits)


@dialect.register(key="place.layout")
class InitialLayoutMethods(interp.MethodTable):

    @interp.impl(CZ)
    def cz(
        self,
        _interp: LayoutAnalysis,
        frame: ForwardFrame[EmptyLattice],
        stmt: CZ,
    ):
        _interp.add_stage(stmt.controls, stmt.targets)

        return ()

    @interp.impl(StaticPlacement)
    def static_circuit(
        self,
        _interp: LayoutAnalysis,
        frame: ForwardFrame[EmptyLattice],
        stmt: StaticPlacement,
    ):
        initial_addresses = tuple(
            _interp.address_entries[qubit] for qubit in stmt.qubits
        )

        if not types.is_tuple_of(initial_addresses, address.AddressQubit):
            raise interp.InterpreterError(
                "All qubits in StaticCircuit must have a valid address"
            )

        _interp.global_address_stack.extend(addr.data for addr in initial_addresses)
        _interp.frame_call_region(frame, stmt, stmt.body, EmptyLattice.top())
        # no nested circuits, so we can clear the stack here
        _interp.global_address_stack.clear()

        return tuple(EmptyLattice.bottom() for _ in stmt.results)


@dialect.register(key="qubit.address")
class QubitAddressAnalysis(interp.MethodTable):

    @interp.impl(NewLogicalQubit)
    def new_logical_qubit(
        self,
        _interp: address.AddressAnalysis,
        frame: ForwardFrame[address.Address],
        stmt: NewLogicalQubit,
    ):
        addr = address.AddressQubit(_interp.next_address)
        _interp.next_address += 1
        return (addr,)
