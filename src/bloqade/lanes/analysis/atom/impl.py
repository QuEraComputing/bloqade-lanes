from kirin import interp
from kirin.analysis.forward import ForwardFrame
from kirin.dialects import func, ilist, py

from bloqade.lanes import layout

from ...dialects import move
from .analysis import (
    AtomInterpreter,
)
from .lattice import (
    AtomState,
    Bottom,
    IListResult,
    MeasureFuture,
    MeasureResult,
    MoveExecution,
    Value,
)


@move.dialect.register(key="atom")
class Move(interp.MethodTable):
    @interp.impl(move.Move)
    def move_impl(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: move.Move,
    ):
        current_state = frame.get(stmt.current_state)

        if isinstance(current_state, AtomState):
            new_data = current_state.data.apply_moves(stmt.lanes, interp_.path_finder)
        else:
            new_data = None

        if new_data is None:
            return (MoveExecution.bottom(),)
        else:
            return (AtomState(new_data),)

    @interp.impl(move.CZ)
    @interp.impl(move.LocalR)
    @interp.impl(move.LocalRz)
    @interp.impl(move.GlobalR)
    @interp.impl(move.GlobalRz)
    @interp.impl(move.LogicalInitialize)
    @interp.impl(move.PhysicalInitialize)
    def noop_impl(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: move.StatefulStatement,
    ):
        return (frame.get(stmt.current_state).copy(),)

    @interp.impl(move.Load)
    def load_impl(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: move.Load,
    ):
        return (interp_.current_state,)

    @interp.impl(move.Fill)
    def fill_impl(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: move.Fill,
    ):
        current_state = frame.get(stmt.current_state)
        if not isinstance(current_state, AtomState):
            return (MoveExecution.bottom(),)

        new_locations = {i: addr for i, addr in enumerate(stmt.location_addresses)}
        new_data = current_state.data.add_atoms(new_locations)
        return (AtomState(new_data),)

    @interp.impl(move.EndMeasure)
    def end_measure_impl(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: move.EndMeasure,
    ):
        current_state = frame.get(stmt.current_state)
        interp_.current_state = current_state

        if not isinstance(current_state, AtomState):
            return (MoveExecution.bottom(),)

        results: dict[layout.ZoneAddress, dict[layout.LocationAddress, int]] = {}
        for zone_address in stmt.zone_addresses:
            result = results.setdefault(zone_address, {})
            for loc_addr in interp_.arch_spec.yield_zone_locations(zone_address):
                if (qubit_id := current_state.data.get_qubit(loc_addr)) is not None:
                    result[loc_addr] = qubit_id

        return (MeasureFuture(results),)

    @interp.impl(move.Store)
    def store_impl(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: move.Store,
    ):
        current_state = frame.get(stmt.current_state)
        interp_.current_state = current_state
        return ()

    @interp.impl(move.GetFutureResult)
    def get_future_result_impl(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: move.GetFutureResult,
    ):

        future = frame.get(stmt.measurement_future)

        if not isinstance(future, MeasureFuture):
            print("GetFutureResult: future is not MeasureFuture")
            return (Bottom(),)

        result = future.results.get(stmt.zone_address)

        if result is None:
            print(f"GetFutureResult: no result for zone address {stmt.zone_address}")
            return (Bottom(),)

        qubit_id = result.get(stmt.location_address)

        if qubit_id is None:
            print(
                f"GetFutureResult: no qubit id for location address {stmt.zone_address} {stmt.location_address}"
            )
            return (Bottom(),)

        return (MeasureResult(qubit_id),)


@py.constant.dialect.register(key="atom")
class PyConstantMethods(interp.MethodTable):

    @interp.impl(py.Constant)
    def constant(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: py.Constant,
    ):
        return (Value(stmt.value.unwrap()),)


@py.indexing.dialect.register(key="atom")
class PyIndexingMethods(interp.MethodTable):

    @interp.impl(py.GetItem)
    def index(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: py.GetItem,
    ):
        obj = frame.get(stmt.obj)
        index = frame.get(stmt.index)
        match (obj, index):
            case (IListResult(values), Value(i)) if isinstance(i, int):
                try:
                    return (values[i],)
                except IndexError:
                    return (Bottom(),)
            case _:
                return (Bottom(),)


@ilist.dialect.register(key="atom")
class IListMethods(interp.MethodTable):

    @interp.impl(ilist.New)
    def ilist_new(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: ilist.New,
    ):
        return (IListResult(frame.get_values(stmt.values)),)


@func.dialect.register(key="atom")
class FuncMethods(interp.MethodTable):

    @interp.impl(func.Return)
    def func_return(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: func.Return,
    ):
        return interp.ReturnValue(frame.get(stmt.value))
