from bloqade.decoders.dialects import annotate
from kirin import ir, types
from kirin.dialects import func, ilist

from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.bytecode.encoding import WordLaneAddress
from bloqade.lanes.dialects import move
from bloqade.lanes.prelude import kernel

kernel = kernel.add(annotate)


def test_atom_interpreter_simple():
    @kernel
    def main():
        state0 = move.load()
        state1 = move.fill(state0, location_addresses=(move.LocationAddress(0, 0),))
        state2 = move.logical_initialize(
            state1,
            thetas=(0.0,),
            phis=(0.0,),
            lams=(0.0,),
            location_addresses=(move.LocationAddress(0, 0),),
        )

        state3 = move.local_r(
            state2,
            axis_angle=0.0,
            rotation_angle=1.57,
            location_addresses=(move.LocationAddress(0, 0),),
        )

        state4 = move.move(state3, lanes=(WordLaneAddress(0, 0, 0),))
        future = move.end_measure(state4, zone_addresses=(move.ZoneAddress(0),))
        results = move.get_future_result(
            future,
            zone_address=move.ZoneAddress(0),
            location_address=move.LocationAddress(1, 0),
        )

        return results

    interp = atom.AtomInterpreter(kernel, arch_spec=get_arch_spec())
    _frame, result = interp.run(main)
    assert result == atom.MeasureResult(
        measurement_id=0, qubit_id=0, location_address=move.LocationAddress(1, 0)
    )


def test_atom_interpreter_rejects_inexecutable_move():
    """A move whose destination holds an atom that does not move in the same
    group is a program error: the analysis raises with the diagnostic rather
    than silently repairing the state (previously the collision was masked
    by restoring both atoms to their pre-move positions)."""
    import pytest

    @kernel
    def main():
        state0 = move.load()
        state1 = move.fill(
            state0,
            location_addresses=(
                move.LocationAddress(0, 0),
                move.LocationAddress(1, 0),
            ),
        )
        # Word 1 is occupied by a stationary atom.
        state2 = move.move(state1, lanes=(WordLaneAddress(0, 0, 0),))
        future = move.end_measure(state2, zone_addresses=(move.ZoneAddress(0),))
        return move.get_future_result(
            future,
            zone_address=move.ZoneAddress(0),
            location_address=move.LocationAddress(1, 0),
        )

    interp = atom.AtomInterpreter(kernel, arch_spec=get_arch_spec())
    with pytest.raises(Exception, match="not executable"):
        interp.run(main)


def test_atom_interpreter_tracks_ilist_slice_getitem():
    @kernel
    def main():
        state0 = move.load()
        state1 = move.fill(
            state0,
            location_addresses=(
                move.LocationAddress(0, 0),
                move.LocationAddress(1, 0),
                move.LocationAddress(2, 0),
            ),
        )
        future = move.end_measure(state1, zone_addresses=(move.ZoneAddress(0),))
        result_0 = move.get_future_result(
            future,
            zone_address=move.ZoneAddress(0),
            location_address=move.LocationAddress(0, 0),
        )
        result_1 = move.get_future_result(
            future,
            zone_address=move.ZoneAddress(0),
            location_address=move.LocationAddress(1, 0),
        )
        result_2 = move.get_future_result(
            future,
            zone_address=move.ZoneAddress(0),
            location_address=move.LocationAddress(2, 0),
        )
        results = ilist.IList([result_0, result_1, result_2])
        return results[1:]

    interp = atom.AtomInterpreter(kernel, arch_spec=get_arch_spec())
    _, result = interp.run(main)

    assert result == atom.IListResult(
        (
            atom.MeasureResult(
                measurement_id=1,
                qubit_id=1,
                location_address=move.LocationAddress(1, 0),
            ),
            atom.MeasureResult(
                measurement_id=2,
                qubit_id=2,
                location_address=move.LocationAddress(2, 0),
            ),
        )
    )


def _build_measure_method(zones: tuple[move.ZoneAddress, ...]) -> ir.Method:
    """Build a move IR method by hand exercising move.Measure.

    There is no Python-level callable exposed for move.Measure (it is
    emitted by stack_move2move, not written by users), so we assemble
    the IR directly.
    """
    block = ir.Block(argtypes=(types.MethodType,))
    load = move.Load()
    block.stmts.append(load)
    fill = move.Fill(
        load.result,
        location_addresses=(move.LocationAddress(0, 0),),
    )
    block.stmts.append(fill)
    measure = move.Measure(current_state=fill.result, zone_addresses=zones)
    block.stmts.append(measure)
    block.stmts.append(move.Store(measure.result))
    none_stmt = func.ConstantNone()
    block.stmts.append(none_stmt)
    block.stmts.append(func.Return(none_stmt.result))

    region = ir.Region(blocks=block)
    function = func.Function(
        sym_name="main",
        signature=func.Signature((), types.NoneType),
        slots=(),
        body=region,
    )
    return ir.Method(
        dialects=kernel,
        code=function,
        sym_name="main",
        arg_names=[],
    )


def test_atom_interpreter_tracks_measure_zones_and_count():
    method = _build_measure_method((move.ZoneAddress(0),))
    interp = atom.AtomInterpreter(method.dialects, arch_spec=get_arch_spec())
    result = interp.get_measurement_positions(method)

    assert interp.final_measurement_count == 1
    assert len(result.measurements) == 1
    assert result.measurements[0].index == 0
    assert result.measurements[0].zone_addresses == (move.ZoneAddress(0),)


def test_atom_interpreter_tracks_multi_zone_measure():
    zones = (move.ZoneAddress(0), move.ZoneAddress(1))
    method = _build_measure_method(zones)
    interp = atom.AtomInterpreter(method.dialects, arch_spec=get_arch_spec())
    result = interp.get_measurement_positions(method)

    assert interp.final_measurement_count == 1
    assert len(result.measurements) == 1
    assert result.measurements[0].zone_addresses == zones
