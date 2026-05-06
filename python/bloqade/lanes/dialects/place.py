from bloqade.analysis import address
from kirin import exception, interp, ir, types
from kirin.analysis.forward import ForwardFrame
from kirin.decl import info, statement
from kirin.dialects import ilist
from kirin.lattice.empty import EmptyLattice

from bloqade import types as bloqade_types
from bloqade.gemini.star import validate_steane_star_support
from bloqade.lanes.analysis.layout import LayoutAnalysis
from bloqade.lanes.analysis.placement import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    PlacementAnalysis,
)
from bloqade.lanes.bytecode.encoding import LocationAddress
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


@statement(init=False)
class _NewQubitBase(ir.Statement):
    """Shared abstract base for qubit-allocation statements.

    Holds the two fields common to both ``NewPinnedQubit`` (physical) and
    ``NewLogicalQubit`` (logical) so that isinstance guards throughout the
    codebase can match "any qubit allocation" without one type being a subtype
    of the other.  Neither class should be used where only one flavour is
    intended; use the concrete subclass directly in those cases.
    """

    traits = frozenset()
    location_address: LocationAddress | None = info.attribute(default=None)
    result: ir.ResultValue = info.result(bloqade_types.QubitType)


@statement(dialect=dialect)
class NewPinnedQubit(_NewQubitBase):
    """Allocate a physical qubit at an optional pinned location.

    No initialization angles — physical qubits have a location but no associated
    initialization sequence.  location_address=None means the layout heuristic will
    assign a site; after ResolvePinnedAddresses runs this is never None in well-formed IR.
    """


@statement(dialect=dialect)
class NewLogicalQubit(_NewQubitBase):
    """Allocate a logical qubit with initial state u3(theta, phi, lam)|0>.

    location_address and result are inherited from _NewQubitBase.

    Args:
        theta (float): Angle for rotation around the Y axis
        phi (float): angle for rotation around the Z axis
        lam (float): angle for rotation around the Z axis
        location_address (LocationAddress | None): Pinned physical address; None means
            the layout heuristic chooses. After ResolvePinnedAddresses runs, this is
            never None in well-formed IR.

    """

    theta: ir.SSAValue = info.argument(types.Float)
    phi: ir.SSAValue = info.argument(types.Float)
    lam: ir.SSAValue = info.argument(types.Float)


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
class StarRz(QuantumStmt):
    qubits: tuple[int, ...] = info.attribute()
    qubit_indices: tuple[int, int, int] = info.attribute()
    rotation_angle: ir.SSAValue = info.argument(type=types.Float)

    def check(self) -> None:
        try:
            validate_steane_star_support(self.qubit_indices)
        except ValueError as exc:
            raise exception.StaticCheckError(str(exc)) from exc


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
        lookahead_cz_layers = _interp.buffered_future_cz_layers(stmt)

        state = _interp.placement_strategy.cz_placements(
            frame.get(stmt.state_before),
            stmt.controls,
            stmt.targets,
            lookahead_cz_layers,
        )
        if isinstance(state, ExecuteCZ) and not state.verify(
            _interp.placement_strategy.arch_spec, stmt.targets, stmt.controls
        ):
            raise interp.InterpreterError("Invalid moves detected")
        return (state,)

    @interp.impl(R)
    @interp.impl(Rz)
    @interp.impl(StarRz)
    def impl_single_qubit_gate(
        self,
        _interp: PlacementAnalysis,
        frame: ForwardFrame[AtomState],
        stmt: R | Rz | StarRz,
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
        body_block = stmt.body.blocks[0]
        _interp.cz_lookahead_buffers[body_block] = _interp.build_cz_buffer(body_block)
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
                _interp.cz_lookahead_buffers.pop(body_block, None)
                _interp.cz_lookahead_stmt_positions.pop(body_block, None)
                return tuple(ret)
            case _:
                _interp.cz_lookahead_buffers.pop(body_block, None)
                _interp.cz_lookahead_stmt_positions.pop(body_block, None)
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

    @interp.impl(NewLogicalQubit)
    @interp.impl(NewPinnedQubit)
    def new_qubit_layout(
        self,
        _interp: LayoutAnalysis,
        frame: ForwardFrame[EmptyLattice],
        stmt: _NewQubitBase,
    ):
        if stmt.location_address is None:
            return (EmptyLattice.bottom(),)
        addr_entry = _interp.address_entries.get(stmt.result)
        if not isinstance(addr_entry, address.AddressQubit):
            return (EmptyLattice.bottom(),)
        qubit_id = addr_entry.data
        pinned_values = _interp.location_addresses.values()
        if stmt.location_address in pinned_values:
            raise interp.InterpreterError(
                f"Duplicate pinned location address: {stmt.location_address}"
            )
        _interp.location_addresses[qubit_id] = stmt.location_address
        return (EmptyLattice.bottom(),)

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
    @interp.impl(NewPinnedQubit)
    def new_qubit_address(
        self,
        _interp: address.AddressAnalysis,
        frame: ForwardFrame[address.Address],
        stmt: _NewQubitBase,
    ):
        addr = address.AddressQubit(_interp.next_address)
        _interp.next_address += 1
        return (addr,)
