from bloqade.decoders.dialects import annotate
from kirin import ir, types
from kirin.dialects import func

from bloqade import squin
from bloqade.lanes._prelude import kernel
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import WordLaneAddress

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
    frame, result = interp.run(main)
    assert result == atom.MeasureResult(
        qubit_id=0, location_address=move.LocationAddress(1, 0)
    )


def test_get_post_processing():
    # Define a simple kernel for testing
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
        future = move.end_measure(state1, zone_addresses=(move.ZoneAddress(0),))
        results_1 = move.get_future_result(
            future,
            zone_address=move.ZoneAddress(0),
            location_address=move.LocationAddress(1, 0),
        )
        results_2 = move.get_future_result(
            future,
            zone_address=move.ZoneAddress(0),
            location_address=move.LocationAddress(0, 0),
        )
        return squin.set_detector([results_1, results_2], [0, 1]), squin.set_observable(
            [results_1, results_2]
        )

    interp = atom.AtomInterpreter(kernel, arch_spec=get_arch_spec())
    post_proc = interp.get_post_processing(main)

    # Simulate measurement results: 2 shots, 1 qubit
    measurement_results = [[True, True], [False, False]]

    # Test emit_return
    returns = list(post_proc.emit_return(measurement_results))
    assert len(returns) == 2

    # Test emit_detectors
    detectors = list(post_proc.emit_detectors(measurement_results))
    assert isinstance(detectors, list)

    # Test emit_observables
    observables = list(post_proc.emit_observables(measurement_results))
    assert isinstance(observables, list)

    # Optionally, check the structure of the outputs
    for det in detectors:
        assert isinstance(det, list)
    for obs in observables:
        assert isinstance(obs, list)

    assert returns[0] == (False, False)
    assert returns[1] == (False, False)


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
    interp.run(method)

    assert interp.final_measurement_count == 1
    assert len(interp.measure_sites) == 1
    site = interp.measure_sites[0]
    assert isinstance(site["stmt"], move.Measure)
    assert site["zones"] == (move.ZoneAddress(0),)


def test_atom_interpreter_tracks_multi_zone_measure():
    zones = (move.ZoneAddress(0), move.ZoneAddress(1))
    method = _build_measure_method(zones)
    interp = atom.AtomInterpreter(method.dialects, arch_spec=get_arch_spec())
    interp.run(method)

    assert interp.final_measurement_count == 1
    assert len(interp.measure_sites) == 1
    assert interp.measure_sites[0]["zones"] == zones
