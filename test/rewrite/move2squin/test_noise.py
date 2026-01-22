from typing import Any

from bloqade.test_utils import assert_nodes
from bloqade.types import Qubit as Qubit
from kirin import ir, rewrite
from kirin.analysis import forward
from kirin.dialects import func, ilist

from bloqade import qubit, squin
from bloqade.lanes import layout
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.rewrite.move2squin import noise


@squin.kernel
def lane_noise_kernel(qubit: qubit.Qubit):
    return


@squin.kernel
def bus_idle_noise_kernel(qubits: ilist.IList[qubit.Qubit, Any]):
    return


@squin.kernel
def cz_unpaired_noise_kernel(qubits: ilist.IList[qubit.Qubit, Any]):
    return


@squin.kernel
def cz_paired_noise_kernel(
    controls: ilist.IList[qubit.Qubit, Any], targets: ilist.IList[qubit.Qubit, Any]
):
    return


@squin.kernel
def global_rz_noise_kernel(qubits: ilist.IList[qubit.Qubit, Any], angle: float):
    return


@squin.kernel
def local_rz_noise_kernel(qubits: ilist.IList[qubit.Qubit, Any], angle: float):
    return


@squin.kernel
def global_r_noise_kernel(
    qubits: ilist.IList[qubit.Qubit, Any], theta: float, phi: float
):
    return


@squin.kernel
def local_r_noise_kernel(
    qubits: ilist.IList[qubit.Qubit, Any], theta: float, phi: float
):
    return


MODEL = noise.SimpleNoiseModel(
    lane_noise=lane_noise_kernel,
    idle_noise=bus_idle_noise_kernel,
    cz_unpaired_noise=cz_unpaired_noise_kernel,
    cz_paired_noise=cz_paired_noise_kernel,
    global_rz_noise=global_rz_noise_kernel,
    local_rz_noise=local_rz_noise_kernel,
    global_r_noise=global_r_noise_kernel,
    local_r_noise=local_r_noise_kernel,
)


def test_simple_noise_model_methods():

    assert (
        MODEL.get_bus_idle_noise(layout.MoveType.SITE, 1) is bus_idle_noise_kernel
    ), "bus idle noise lookup failed"
    assert (
        MODEL.get_cz_paired_noise(layout.ZoneAddress(0)) is cz_paired_noise_kernel
    ), "cz paired noise lookup failed"
    assert (
        MODEL.get_cz_unpaired_noise(layout.ZoneAddress(0)) is cz_unpaired_noise_kernel
    ), "cz unpaired noise lookup failed"
    assert MODEL.get_lane_noise(layout.SiteLaneAddress(0, 1, 1)) is lane_noise_kernel
    assert MODEL.get_global_rz_noise() is global_rz_noise_kernel
    assert (
        MODEL.get_local_rz_noise((layout.LocationAddress(0, 1),))
        is local_rz_noise_kernel
    )
    assert MODEL.get_global_r_noise() is global_r_noise_kernel
    assert (
        MODEL.get_local_r_noise((layout.LocationAddress(0, 1),)) is local_r_noise_kernel
    )


def test_insert_move_noise_no_op():
    state = ir.TestValue()
    test_block = ir.Block([node := move.Move(state, lanes=())])

    physical_ssa_values = {
        0: (ir.TestValue()),
        1: (ir.TestValue()),
    }

    atom_state: Any = atom.AtomState(
        data=atom.AtomStateData.new(
            {0: layout.LocationAddress(0, 0), 1: layout.LocationAddress(0, 1)}
        )
    )

    atom_state_map = forward.ForwardFrame(node, entries={node.result: atom_state})

    rewriter = rewrite.Walk(
        noise.InsertNoise(
            arch_spec=get_arch_spec(),
            physical_ssa_values=physical_ssa_values,  # type: ignore
            atom_state_map=atom_state_map,
            noise_model=MODEL,
        )
    )

    rewriter.rewrite(test_block)

    expected_block = ir.Block(
        [
            move.Move(state, lanes=()),
        ]
    )

    assert_nodes(test_block, expected_block)


def test_insert_move_noise_lane_noise():
    state = ir.TestValue()
    test_block = ir.Block(
        [node := move.Move(state, lanes=(layout.SiteLaneAddress(0, 0, 1),))]
    )

    physical_ssa_values = {
        0: (zero := ir.TestValue()),
        1: (one := ir.TestValue()),
    }
    atom_state: Any = atom.AtomState(
        data=atom.AtomStateData.new(
            {0: layout.LocationAddress(0, 6), 1: layout.LocationAddress(0, 1)}
        )
    )

    atom_state_map = forward.ForwardFrame(node, entries={node.result: atom_state})

    rewriter = rewrite.Walk(
        noise.InsertNoise(
            arch_spec=get_arch_spec(),
            physical_ssa_values=physical_ssa_values,  # type: ignore
            atom_state_map=atom_state_map,
            noise_model=MODEL,
        )
    )

    rewriter.rewrite(test_block)

    expected_block = ir.Block(
        [
            func.Invoke(inputs=(zero,), callee=lane_noise_kernel),
            reg := ilist.New((one,)),
            func.Invoke(inputs=(reg.result,), callee=bus_idle_noise_kernel),
            move.Move(state, lanes=(layout.SiteLaneAddress(0, 0, 1),)),
        ]
    )

    assert_nodes(test_block, expected_block)


def test_insert_cz_noise():
    state = ir.TestValue()
    test_block = ir.Block([node := move.CZ(state, zone_address=layout.ZoneAddress(0))])

    physical_ssa_values = {
        0: (zero := ir.TestValue()),
        1: (one := ir.TestValue()),
        2: (two := ir.TestValue()),
        3: (three := ir.TestValue()),
    }
    atom_state: Any = atom.AtomState(
        data=atom.AtomStateData.new(
            {
                0: layout.LocationAddress(0, 6),
                1: layout.LocationAddress(0, 1),
                2: layout.LocationAddress(0, 2),
                3: layout.LocationAddress(0, 3),
            }
        )
    )

    atom_state_map = forward.ForwardFrame(node, entries={node.result: atom_state})

    rewriter = rewrite.Walk(
        noise.InsertNoise(
            arch_spec=get_arch_spec(),
            physical_ssa_values=physical_ssa_values,  # type: ignore
            atom_state_map=atom_state_map,
            noise_model=MODEL,
        )
    )

    rewriter.rewrite(test_block)
    test_block.print()
    expected_block = ir.Block(
        [
            move.CZ(state, zone_address=layout.ZoneAddress(0)),
            controls_reg := ilist.New((one,)),
            targets_reg := ilist.New((two,)),
            func.Invoke(
                inputs=(controls_reg.result, targets_reg.result),
                callee=cz_paired_noise_kernel,
            ),
            unpaired_reg := ilist.New((zero, three)),
            func.Invoke(inputs=(unpaired_reg.result,), callee=cz_unpaired_noise_kernel),
        ]
    )
    assert test_block.print_str() == expected_block.print_str() or assert_nodes(
        test_block, expected_block
    )


def test_insert_local_gate_noise():
    state = ir.TestValue()
    axis_angle = ir.TestValue()
    rotation_angle = ir.TestValue()
    test_block = ir.Block(
        [
            node := move.LocalR(
                state,
                axis_angle,
                rotation_angle,
                location_addresses=(layout.LocationAddress(0, 1),),
            )
        ]
    )

    physical_ssa_values = {
        0: (ir.TestValue()),
        1: (one := ir.TestValue()),
        2: (ir.TestValue()),
        3: (ir.TestValue()),
    }
    atom_state: Any = atom.AtomState(
        data=atom.AtomStateData.new(
            {
                0: layout.LocationAddress(0, 0),
                1: layout.LocationAddress(0, 1),
                2: layout.LocationAddress(0, 2),
                3: layout.LocationAddress(0, 3),
            }
        )
    )

    atom_state_map = forward.ForwardFrame(node, entries={node.result: atom_state})

    rewriter = rewrite.Walk(
        noise.InsertNoise(
            arch_spec=get_arch_spec(),
            physical_ssa_values=physical_ssa_values,  # type: ignore
            atom_state_map=atom_state_map,
            noise_model=MODEL,
        )
    )

    rewriter.rewrite(test_block)

    expected_block = ir.Block(
        [
            move.LocalR(
                state,
                axis_angle,
                rotation_angle,
                location_addresses=(layout.LocationAddress(0, 1),),
            ),
            reg := ilist.New((one,)),
            func.Invoke(
                inputs=(reg.result, axis_angle, rotation_angle),
                callee=local_r_noise_kernel,
            ),
        ]
    )

    assert_nodes(test_block, expected_block)


def test_insert_global_gate_noise():
    state = ir.TestValue()
    axis_angle = ir.TestValue()
    rotation_angle = ir.TestValue()
    test_block = ir.Block(
        [
            node := move.GlobalR(
                state,
                axis_angle,
                rotation_angle,
            )
        ]
    )

    physical_ssa_values = {
        0: (zero := ir.TestValue()),
        1: (one := ir.TestValue()),
        2: (two := ir.TestValue()),
        3: (three := ir.TestValue()),
    }
    atom_state: Any = atom.AtomState(
        data=atom.AtomStateData.new(
            {
                0: layout.LocationAddress(0, 0),
                1: layout.LocationAddress(0, 1),
                2: layout.LocationAddress(0, 2),
                3: layout.LocationAddress(0, 3),
            }
        )
    )

    atom_state_map = forward.ForwardFrame(node, entries={node.result: atom_state})

    rewriter = rewrite.Walk(
        noise.InsertNoise(
            arch_spec=get_arch_spec(),
            physical_ssa_values=physical_ssa_values,  # type: ignore
            atom_state_map=atom_state_map,
            noise_model=MODEL,
        )
    )

    rewriter.rewrite(test_block)

    expected_block = ir.Block(
        [
            move.GlobalR(
                state,
                axis_angle,
                rotation_angle,
            ),
            reg := ilist.New((zero, one, two, three)),
            func.Invoke(
                inputs=(reg.result, axis_angle, rotation_angle),
                callee=global_r_noise_kernel,
            ),
        ]
    )

    assert_nodes(test_block, expected_block)
