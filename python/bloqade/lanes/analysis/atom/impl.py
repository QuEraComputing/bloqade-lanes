from bloqade.decoders.dialects import annotate
from kirin import interp
from kirin.analysis.forward import ForwardFrame
from kirin.dialects import func, ilist, py
from kirin.interp import InterpreterError

from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.bytecode.exceptions import MoveValidationError

from ...dialects import move
from .analysis import (
    AtomInterpreter,
)
from .lattice import (
    AtomState,
    Bottom,
    DetectorResult,
    IListResult,
    MeasureFuture,
    MeasureResult,
    MoveExecution,
    ObservableResult,
    TupleResult,
    Value,
)


@annotate.dialect.register(key="atom")
class Annotate(interp.MethodTable):
    @interp.impl(annotate.stmts.SetDetector)
    def set_detector(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: annotate.stmts.SetDetector,
    ):
        result = DetectorResult(frame.get(stmt.measurements))
        interp_._detectors.append(result)
        return (result,)

    @interp.impl(annotate.stmts.SetObservable)
    def set_observable(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: annotate.stmts.SetObservable,
    ):
        result = ObservableResult(frame.get(stmt.measurements))
        interp_._observables.append(result)
        return (result,)


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

        if not isinstance(current_state, AtomState):
            return (MoveExecution.bottom(),)

        # Canonical validate/apply: an inexecutable lane group (occupied
        # destination not vacated by the group, invalid lanes, ...) is a
        # program error, reported with every individual problem — never a
        # silently repaired state. Note this requires the interpreter's
        # arch spec to match the IR's address space: post-transversal move
        # IR is physically addressed and must be analyzed against the
        # physical spec.
        try:
            validated = current_state.data.validate_moves(stmt.lanes, interp_.arch_spec)
            new_data = current_state.data.apply_validated(validated)
        except MoveValidationError as e:
            details = "\n".join(f"  - {err}" for err in e.errors) or f"  - {e}"
            raise InterpreterError(
                f"move statement is not executable:\n{details}"
            ) from e

        return (AtomState(new_data),)

    @interp.impl(move.CZ)
    @interp.impl(move.LocalR)
    @interp.impl(move.LocalRz)
    @interp.impl(move.StarRz)
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
        interp_.final_measurement_count += 1

        if not isinstance(current_state, AtomState):
            return (MoveExecution.bottom(),)

        results: dict[ZoneAddress, dict[LocationAddress, int]] = {}
        for zone_address in stmt.zone_addresses:
            result = results.setdefault(zone_address, {})
            for loc_addr in interp_.arch_spec.yield_zone_locations(zone_address):
                if (qubit_id := current_state.data.get_qubit(loc_addr)) is not None:
                    result[loc_addr] = qubit_id

        return (
            MeasureFuture(
                results=results,
                measurement_count=interp_.final_measurement_count,
            ),
        )

    @interp.impl(move.ConstZone)
    def const_zone_impl(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: move.ConstZone,
    ):
        return (Value(stmt.value),)

    @interp.impl(move.Measure)
    def measure_impl(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: move.Measure,
    ):
        current_state = frame.get(stmt.current_state)
        interp_.current_state = current_state

        # Read zones directly from the compile-time attribute — no frame
        # lookup needed now that zones are an attribute tuple.
        zone_addresses = list(stmt.zone_addresses)

        interp_.final_measurement_count += 1

        if not isinstance(current_state, AtomState):
            return (MoveExecution.bottom(), MoveExecution.bottom())

        # Build the MeasurementFuture by mirroring end_measure_impl: for
        # each zone, walk every location in the zone, and record any qubit
        # currently at that location.
        results: dict[ZoneAddress, dict[LocationAddress, int]] = {}
        for zone_address in zone_addresses:
            result = results.setdefault(zone_address, {})
            for loc_addr in interp_.arch_spec.yield_zone_locations(zone_address):
                if (qubit_id := current_state.data.get_qubit(loc_addr)) is not None:
                    result[loc_addr] = qubit_id

        # move.Measure has two results: (new_state, future). Measurement
        # observes the state but does not reshape it on the Python
        # analysis side, so we thread ``current_state`` forward unchanged.
        return (
            current_state,
            MeasureFuture(
                results=results,
                measurement_count=interp_.final_measurement_count,
            ),
        )

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
            return (Bottom(),)

        result = future.results.get(stmt.zone_address)

        if result is None:
            return (Bottom(),)

        qubit_id = result.get(stmt.location_address)

        if qubit_id is None:
            return (Bottom(),)

        # Assign the global measurement-record index. Each GetFutureResult
        # that resolves to a real qubit corresponds to exactly one
        # ``qubit.measure`` emitted downstream by ``InsertMeasurements``
        # (in this same IR order), which in turn is one column of the raw
        # per-shot measurement array. Incrementing here — only on the
        # branch that yields a MeasureResult — keeps the record index in
        # lockstep with that emission order. GetFutureResults that resolve
        # to Bottom emit no measurement and must not consume an index.
        measurement_id = interp_.measurement_record_count
        interp_.measurement_record_count += 1

        return (MeasureResult(measurement_id, qubit_id, stmt.location_address),)


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
            case (IListResult(values), Value(i)) | (
                TupleResult(values),
                Value(i),
            ) if isinstance(i, int):
                try:
                    return (values[i],)
                except IndexError:
                    return (Bottom(),)
            case (IListResult(values), Value(i)) if isinstance(i, slice):
                return (IListResult(values[i]),)
            case (TupleResult(values), Value(i)) if isinstance(i, slice):
                return (TupleResult(values[i]),)
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


@py.tuple.dialect.register(key="atom")
class TupleMethods(interp.MethodTable):
    @interp.impl(py.tuple.New)
    def tuple_new(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: py.tuple.New,
    ):
        return (TupleResult(frame.get_values(stmt.args)),)


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

    @interp.impl(func.ConstantNone)
    def const_none(
        self,
        interp_: AtomInterpreter,
        frame: ForwardFrame[MoveExecution],
        stmt: func.ConstantNone,
    ):
        return (Value(None),)
