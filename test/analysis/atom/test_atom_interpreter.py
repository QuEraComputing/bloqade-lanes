from bloqade.decoders.dialects import annotate

from bloqade import squin
from bloqade.lanes import kernel
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import SiteLaneAddress

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

        state4 = move.move(state3, lanes=(SiteLaneAddress(0, 0, 0),))
        future = move.end_measure(state4, zone_addresses=(move.ZoneAddress(0),))
        results = move.get_future_result(
            future,
            zone_address=move.ZoneAddress(0),
            location_address=move.LocationAddress(0, 5),
        )

        return results

    interp = atom.AtomInterpreter(kernel, arch_spec=get_arch_spec())
    frame, result = interp.run(main)
    assert result == atom.MeasureResult(qubit_id=0)


def test_get_post_processing():
    # Define a simple kernel for testing
    @kernel
    def main():
        state0 = move.load()
        state1 = move.fill(
            state0,
            location_addresses=(
                move.LocationAddress(0, 0),
                move.LocationAddress(0, 1),
            ),
        )
        future = move.end_measure(state1, zone_addresses=(move.ZoneAddress(0),))
        results_1 = move.get_future_result(
            future,
            zone_address=move.ZoneAddress(0),
            location_address=move.LocationAddress(0, 1),
        )
        results_2 = move.get_future_result(
            future,
            zone_address=move.ZoneAddress(0),
            location_address=move.LocationAddress(0, 0),
        )
        return squin.set_detector([results_1, results_2], [0, 1]), squin.set_observable(
            [results_1, results_2], 0
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
