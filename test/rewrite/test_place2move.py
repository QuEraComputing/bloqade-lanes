from dataclasses import dataclass, field

from bloqade.test_utils import assert_nodes
from kirin import ir, rewrite
from kirin.dialects import py

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement.lattice import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    ExecuteMeasure,
)
from bloqade.lanes.dialects import move, place
from bloqade.lanes.layout import word
from bloqade.lanes.rewrite import place2move

ARCH_SPEC = layout.ArchSpec(
    (word.Word(((0, 0), (0, 1))),),
    ((0,),),
    (0,),
    frozenset([0]),
    frozenset(),
    frozenset(),
    (),
    (),
    (),
)


@dataclass
class MoveSchedulerTester(place2move.MoveSchedulerABC):
    arch_spec: layout.ArchSpec = field(init=False, default=ARCH_SPEC)
    lanes: list[tuple[layout.LaneAddress, ...]]

    def compute_moves(self, state_before: AtomState, state_after: AtomState):
        return self.lanes


def test_insert_move_no_op():

    move_scheduler = MoveSchedulerTester([])

    placement_analysis: dict[ir.SSAValue, AtomState] = {}

    rule = rewrite.Walk(place2move.InsertMoves(move_scheduler, placement_analysis))

    test_block = ir.Block(
        [
            place.CZ(ir.TestValue(), qubits=(0, 1, 2, 3)),
        ]
    )
    result = rule.rewrite(test_block)

    assert not result.has_done_something


def test_insert_move():

    lane_group = (
        layout.SiteLaneAddress(0, 0, 0, layout.Direction.FORWARD),
        layout.SiteLaneAddress(0, 1, 0, layout.Direction.FORWARD),
    )

    state_before = ir.TestValue()
    move_scheduler = MoveSchedulerTester([lane_group])

    placement_analysis: dict[ir.SSAValue, AtomState] = {}

    rule = rewrite.Walk(place2move.InsertMoves(move_scheduler, placement_analysis))

    test_block = ir.Block(
        [
            py.Constant(10),
            place.CZ(state_before, qubits=(0, 1, 2, 3)),
        ]
    )
    expected_block = ir.Block(
        [
            py.Constant(10),
            (current_state := move.Load()),
            (current_state := move.Move(current_state.result, lanes=lane_group)),
            move.Store(current_state.result),
            place.CZ(state_before, qubits=(0, 1, 2, 3)),
        ]
    )
    rule.rewrite(test_block)

    assert_nodes(test_block, expected_block)


def test_insert_palindrom_moves():
    lane_group = (
        layout.SiteLaneAddress(0, 0, 0, layout.Direction.FORWARD),
        layout.SiteLaneAddress(0, 1, 0, layout.Direction.FORWARD),
    )
    reverse_moves = (
        layout.SiteLaneAddress(0, 0, 0, layout.Direction.BACKWARD),
        layout.SiteLaneAddress(0, 1, 0, layout.Direction.BACKWARD),
    )

    state_before = ir.TestValue()

    rule = rewrite.Walk(place2move.InsertPalindromeMoves())

    test_body = ir.Region(
        ir.Block(
            [
                (current_state := move.Load()),
                (current_state := move.Move(current_state.result, lanes=lane_group)),
                move.Store(current_state.result),
                stmt := place.CZ(state_before, qubits=(0, 1, 2, 3)),
                place.Yield(stmt.results[0]),
            ]
        )
    )

    test_block = ir.Block(
        [
            py.Constant(10),
            place.StaticPlacement(
                qubits := (ir.TestValue(), ir.TestValue()), test_body
            ),
        ]
    )

    expected_body = ir.Region(
        ir.Block(
            [
                (current_state := move.Load()),
                (current_state := move.Move(current_state.result, lanes=lane_group)),
                move.Store(current_state.result),
                stmt := place.CZ(state_before, qubits=(0, 1, 2, 3)),
                (current_state := move.Load()),
                (current_state := move.Move(current_state.result, lanes=reverse_moves)),
                move.Store(current_state.result),
                place.Yield(stmt.results[0]),
            ]
        )
    )

    expected_block = ir.Block(
        [py.Constant(10), place.StaticPlacement(qubits, expected_body)]
    )

    rule.rewrite(test_block)
    assert_nodes(test_block, expected_block)


def test_insert_cz_no_op():
    state_before = ir.TestValue()

    placement_analysis: dict[ir.SSAValue, AtomState] = {}

    rule = rewrite.Walk(place2move.RewriteGates(placement_analysis))

    test_block = ir.Block(
        [
            py.Constant(10),
            stmt := place.CZ(state_before, qubits=(0, 1, 2, 3)),
        ]
    )

    placement_analysis[stmt.results[0]] = AtomState.top()

    result = rule.rewrite(test_block)
    assert not result.has_done_something


def test_insert_cz():
    state_before = ir.TestValue()

    placement_analysis: dict[ir.SSAValue, AtomState] = {}

    rule = rewrite.Walk(place2move.RewriteGates(placement_analysis))

    test_block = ir.Block(
        [
            py.Constant(10),
            stmt := place.CZ(state_before, qubits=(0, 1, 2, 3)),
        ]
    )

    placement_analysis[stmt.results[0]] = ExecuteCZ(
        frozenset(), (), (), frozenset([layout.ZoneAddress(0)])
    )

    expected_block = ir.Block(
        [
            py.Constant(10),
            current_state := move.Load(),
            (
                current_state := move.CZ(
                    current_state.result, zone_address=layout.ZoneAddress(0)
                )
            ),
            move.Store(current_state.result),
        ],
    )

    rule.rewrite(test_block)
    assert_nodes(test_block, expected_block)


def test_global_rz():
    state_before = ir.TestValue()

    placement_analysis: dict[ir.SSAValue, AtomState] = {}

    rule = rewrite.Walk(place2move.RewriteGates(placement_analysis))
    test_block = ir.Block(
        [
            rotation_angle := py.Constant(0.5),
            stmt := place.Rz(state_before, rotation_angle.result, qubits=()),
        ]
    )

    placement_analysis[stmt.results[0]] = ConcreteState(frozenset(), (), ())

    expected_block = ir.Block(
        [
            rotation_angle := py.Constant(0.5),
            current_state := move.Load(),
            (
                current_state := move.GlobalRz(
                    current_state.result, rotation_angle.result
                )
            ),
            move.Store(current_state.result),
        ],
    )

    rule.rewrite(test_block)
    assert_nodes(test_block, expected_block)


def test_global_r():
    state_before = ir.TestValue()

    placement_analysis: dict[ir.SSAValue, AtomState] = {}

    rule = rewrite.Walk(place2move.RewriteGates(placement_analysis))
    test_block = ir.Block(
        [
            rotation_angle := py.Constant(0.5),
            stmt := place.R(
                state_before, rotation_angle.result, rotation_angle.result, qubits=()
            ),
        ]
    )

    placement_analysis[stmt.results[0]] = ConcreteState(frozenset(), (), ())

    expected_block = ir.Block(
        [
            rotation_angle := py.Constant(0.5),
            current_state := move.Load(),
            (
                current_state := move.GlobalR(
                    current_state.result, rotation_angle.result, rotation_angle.result
                )
            ),
            move.Store(current_state.result),
        ],
    )

    rule.rewrite(test_block)
    assert_nodes(test_block, expected_block)


def test_local_rz():
    state_before = ir.TestValue()

    placement_analysis: dict[ir.SSAValue, AtomState] = {}

    rule = rewrite.Walk(place2move.RewriteGates(placement_analysis))
    test_block = ir.Block(
        [
            rotation_angle := py.Constant(0.5),
            stmt := place.Rz(state_before, rotation_angle.result, qubits=()),
        ]
    )

    placement_analysis[stmt.results[0]] = ConcreteState(
        frozenset([layout.LocationAddress(0, 0)]), (), ()
    )

    expected_block = ir.Block(
        [
            rotation_angle := py.Constant(0.5),
            current_state := move.Load(),
            (
                current_state := move.LocalRz(
                    current_state.result, rotation_angle.result, location_addresses=()
                )
            ),
            move.Store(current_state.result),
        ],
    )

    rule.rewrite(test_block)
    assert_nodes(test_block, expected_block)


def test_local_r():
    state_before = ir.TestValue()

    placement_analysis: dict[ir.SSAValue, AtomState] = {}

    rule = rewrite.Walk(place2move.RewriteGates(placement_analysis))
    test_block = ir.Block(
        [
            rotation_angle := py.Constant(0.5),
            stmt := place.R(
                state_before, rotation_angle.result, rotation_angle.result, qubits=()
            ),
        ]
    )

    placement_analysis[stmt.results[0]] = ConcreteState(
        frozenset([layout.LocationAddress(0, 0)]), (), ()
    )

    expected_block = ir.Block(
        [
            rotation_angle := py.Constant(0.5),
            current_state := move.Load(),
            (
                current_state := move.LocalR(
                    current_state.result,
                    rotation_angle.result,
                    rotation_angle.result,
                    location_addresses=(),
                )
            ),
            move.Store(current_state.result),
        ],
    )

    rule.rewrite(test_block)
    assert_nodes(test_block, expected_block)


def test_insert_measure_no_op():
    state_before = ir.TestValue()

    placement_analysis: dict[ir.SSAValue, AtomState] = {}

    rule = rewrite.Walk(place2move.InsertMeasure(placement_analysis))
    test_block = ir.Block(
        [
            py.Constant(10),
            stmt := place.EndMeasure(state_before, qubits=(0, 1)),
        ]
    )

    placement_analysis[stmt.results[0]] = AtomState.top()

    result = rule.rewrite(test_block)
    assert not result.has_done_something


def test_insert_measure():
    state_before = ir.TestValue()

    placement_analysis: dict[ir.SSAValue, AtomState] = {}

    rule = rewrite.Walk(place2move.InsertMeasure(placement_analysis))
    test_block = ir.Block(
        [
            py.Constant(10),
            stmt := place.EndMeasure(state_before, qubits=(0, 1)),
            place.Yield(stmt.results[0], *stmt.results[1:]),
        ]
    )
    qubit_layout = (
        layout.LocationAddress(0, 1),
        layout.LocationAddress(0, 0),
    )
    placement_analysis[stmt.results[0]] = ExecuteMeasure(
        frozenset(), qubit_layout, (), (layout.ZoneAddress(0), layout.ZoneAddress(1))
    )

    expected_block = ir.Block(
        [
            py.Constant(10),
            current_state := move.Load(),
            future := move.EndMeasure(
                current_state.result,
                zone_addresses=(layout.ZoneAddress(0), layout.ZoneAddress(1)),
            ),
            zone_result_0 := move.GetFutureResult(
                future.result,
                zone_address=layout.ZoneAddress(0),
                location_address=layout.LocationAddress(0, 1),
            ),
            zone_result_1 := move.GetFutureResult(
                future.result,
                zone_address=layout.ZoneAddress(1),
                location_address=layout.LocationAddress(0, 0),
            ),
            place.Yield(state_before, zone_result_0.result, zone_result_1.result),
        ],
    )

    rule.rewrite(test_block)
    test_block.print()
    expected_block.print()

    assert_nodes(test_block, expected_block)
