from bloqade.lanes import kernel
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import SiteLaneAddress


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
