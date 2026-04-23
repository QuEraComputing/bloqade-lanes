from bloqade.decoders.dialects import annotate
from kirin import interp
from kirin.analysis.forward import ForwardFrame
from kirin.dialects import func, ilist, py

from bloqade.lanes import layout
from bloqade.lanes.analysis.atom.atom_state_data import AtomStateData
from bloqade.lanes.layout.encoding import LocationAddress

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


def _restore_collisions_to_pre_move(
    pre_data: AtomStateData, post_data: AtomStateData
) -> AtomStateData:
    """Restore collided atoms to their pre-move positions.

    When ``apply_moves`` detects a collision (two atoms at the same site),
    it removes both from the location maps and records them in the collision
    dict.  For the abstract-interpretation use case the atoms must remain
    trackable, so this helper puts newly collided atoms back at their
    *original* (pre-move) positions so that subsequent ``get_qubit_pairing``
    calls and return-move ``apply_moves`` calls work correctly.
    """
    pre_collision = pre_data.collision
    new_collisions = {
        mover: displaced
        for mover, displaced in post_data.collision.items()
        if mover not in pre_collision
    }

    if not new_collisions:
        return post_data

    # Build updated location maps: start from post_data maps, then add back
    # each newly collided atom at its pre-move position.
    locations_to_qubit: dict[LocationAddress, int] = dict(post_data.locations_to_qubit)
    qubit_to_locations: dict[int, LocationAddress] = dict(post_data.qubit_to_locations)

    for mover, displaced in new_collisions.items():
        # Restore mover to its pre-move position
        if mover in pre_data.qubit_to_locations:
            mover_loc = pre_data.qubit_to_locations[mover]
            qubit_to_locations[mover] = mover_loc
            locations_to_qubit[mover_loc] = mover

        # Restore displaced to its pre-move position (which is its home)
        if displaced in pre_data.qubit_to_locations:
            displaced_loc = pre_data.qubit_to_locations[displaced]
            qubit_to_locations[displaced] = displaced_loc
            locations_to_qubit[displaced_loc] = displaced

    return AtomStateData.from_fields(
        locations_to_qubit=locations_to_qubit,
        qubit_to_locations=qubit_to_locations,
        collision=dict(post_data.collision),
        prev_lanes=dict(post_data.prev_lanes),
        move_count=dict(post_data.move_count),
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

        if isinstance(current_state, AtomState):
            new_data = current_state.data.apply_moves(stmt.lanes, interp_.arch_spec)
            if new_data is not None and len(new_data.collision) > len(
                current_state.data.collision
            ):
                # New collisions: restore collided atoms to pre-move positions
                # so they remain trackable through CZ + return move patterns.
                new_data = _restore_collisions_to_pre_move(current_state.data, new_data)
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

        # Track site + count for the measure_lower rewrite downstream.
        interp_.measure_sites.append({"stmt": stmt, "zones": tuple(zone_addresses)})
        interp_.final_measurement_count += 1

        if not isinstance(current_state, AtomState):
            return (MoveExecution.bottom(), MoveExecution.bottom())

        # Build the MeasurementFuture by mirroring end_measure_impl: for
        # each zone, walk every location in the zone, and record any qubit
        # currently at that location.
        results: dict[layout.ZoneAddress, dict[layout.LocationAddress, int]] = {}
        for zone_address in zone_addresses:
            result = results.setdefault(zone_address, {})
            for loc_addr in interp_.arch_spec.yield_zone_locations(zone_address):
                if (qubit_id := current_state.data.get_qubit(loc_addr)) is not None:
                    result[loc_addr] = qubit_id

        # move.Measure has two results: (new_state, future). Measurement
        # observes the state but does not reshape it on the Python
        # analysis side, so we thread ``current_state`` forward unchanged.
        return (current_state, MeasureFuture(results))

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
            case (IListResult(values), Value(i)) | (
                TupleResult(values),
                Value(i),
            ) if isinstance(i, int):
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
