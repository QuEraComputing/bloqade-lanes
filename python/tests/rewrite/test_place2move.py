from bloqade.test_utils import assert_nodes
from kirin import ir, rewrite, types
from kirin.dialects import func, py

from bloqade.lanes.analysis.placement.lattice import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    ExecuteMeasure,
)
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode import word
from bloqade.lanes.bytecode._native import (
    Grid as RustGrid,
    LocationAddress as RustLocAddr,
    Mode as RustMode,
    Zone as RustZone,
)
from bloqade.lanes.bytecode.encoding import (
    Direction,
    LocationAddress,
    SiteLaneAddress,
    ZoneAddress,
)
from bloqade.lanes.dialects import move, place
from bloqade.lanes.rewrite import place2move

_word = word.Word(sites=((0, 0), (0, 1)))
_rust_grid = RustGrid.from_positions([0.0], [0.0, 1.0])
_rust_zone = RustZone(
    name="test",
    grid=_rust_grid,
    site_buses=[],
    word_buses=[],
    words_with_site_buses=[],
    sites_with_word_buses=[],
)
_rust_mode = RustMode(
    name="all",
    zones=[0],
    bitstring_order=[RustLocAddr(0, 0, 0), RustLocAddr(0, 0, 1)],
)
ARCH_SPEC = ArchSpec.from_components(
    words=(_word,),
    zones=(_rust_zone,),
    modes=[_rust_mode],
)


def test_insert_move_no_op():

    placement_analysis: dict[ir.SSAValue, AtomState] = {}

    rule = rewrite.Walk(place2move.InsertMoves(placement_analysis))

    test_block = ir.Block(
        [
            place.CZ(ir.TestValue(), qubits=(0, 1, 2, 3)),
        ]
    )
    result = rule.rewrite(test_block)

    assert not result.has_done_something


def test_insert_move():

    state_before = ir.TestValue()

    test_block = ir.Block(
        [
            py.Constant(10),
            cz_stmt := place.CZ(state_before, qubits=(0, 1, 2, 3)),
        ]
    )

    lane_group = (
        SiteLaneAddress(0, 0, 0, Direction.FORWARD),
        SiteLaneAddress(1, 0, 0, Direction.FORWARD),
    )

    placement_analysis: dict[ir.SSAValue, AtomState] = {
        cz_stmt.state_after: ExecuteCZ(
            frozenset(),
            (),
            (),
            frozenset([ZoneAddress(0)]),
            move_layers=(lane_group,),
        )
    }

    rule = rewrite.Walk(place2move.InsertMoves(placement_analysis))

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
        SiteLaneAddress(0, 0, 0, Direction.FORWARD),
        SiteLaneAddress(1, 0, 0, Direction.FORWARD),
    )
    reverse_moves = (
        SiteLaneAddress(0, 0, 0, Direction.BACKWARD),
        SiteLaneAddress(1, 0, 0, Direction.BACKWARD),
    )

    state_before = ir.TestValue()

    rule = rewrite.Walk(place2move.InsertReturnMoves())

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
        frozenset(), (), (), frozenset([ZoneAddress(0)])
    )

    expected_block = ir.Block(
        [
            py.Constant(10),
            current_state := move.Load(),
            (
                current_state := move.CZ(
                    current_state.result, zone_address=ZoneAddress(0)
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
        frozenset([LocationAddress(0, 0)]), (), ()
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
        frozenset([LocationAddress(0, 0)]), (), ()
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
        LocationAddress(0, 1),
        LocationAddress(0, 0),
    )
    placement_analysis[stmt.results[0]] = ExecuteMeasure(
        frozenset(), qubit_layout, (), (ZoneAddress(0), ZoneAddress(1))
    )

    expected_block = ir.Block(
        [
            py.Constant(10),
            current_state := move.Load(),
            future := move.EndMeasure(
                current_state.result,
                zone_addresses=(ZoneAddress(0), ZoneAddress(1)),
            ),
            zone_result_0 := move.GetFutureResult(
                future.result,
                zone_address=ZoneAddress(0),
                location_address=LocationAddress(0, 1),
            ),
            zone_result_1 := move.GetFutureResult(
                future.result,
                zone_address=ZoneAddress(1),
                location_address=LocationAddress(0, 0),
            ),
            place.Yield(state_before, zone_result_0.result, zone_result_1.result),
        ],
    )

    rule.rewrite(test_block)
    test_block.print()
    expected_block.print()

    assert_nodes(test_block, expected_block)


# ---------------------------------------------------------------------------
# InsertFill tests
# ---------------------------------------------------------------------------


def _make_function_with_qubits(
    addresses: tuple[LocationAddress, ...],
) -> func.Function:
    """Build a minimal func.Function whose body starts with NewLogicalQubit
    statements, each pre-stamped with the given location_address values."""
    angle = ir.TestValue()
    stmts: list[ir.Statement] = []
    for addr in addresses:
        nlq = place.NewLogicalQubit(angle, angle, angle)
        nlq.location_address = addr
        stmts.append(nlq)

    block = ir.Block(stmts)
    region = ir.Region(block)
    return func.Function(
        sym_name="test_fn",
        signature=func.Signature((), types.NoneType),
        slots=(),
        body=region,
    )


def test_insert_fill_emits_fill_with_correct_addresses():
    """InsertFill collects location_address from NewLogicalQubit statements
    and emits a move.Fill with those addresses in IR-walk order."""
    addr0 = LocationAddress(0, 0)
    addr1 = LocationAddress(0, 1)

    fn = _make_function_with_qubits((addr0, addr1))
    rule = rewrite.Walk(place2move.InsertFill())
    result = rule.rewrite(fn)

    assert result.has_done_something

    first_stmt = fn.body.blocks[0].first_stmt
    assert isinstance(first_stmt, move.Load)

    fill_stmt = first_stmt.next_stmt
    assert isinstance(fill_stmt, move.Fill)
    assert fill_stmt.location_addresses == (addr0, addr1)

    store_stmt = fill_stmt.next_stmt
    assert isinstance(store_stmt, move.Store)


def test_insert_fill_no_qubits_is_noop():
    """InsertFill is a no-op when the function body has no NewLogicalQubit
    statements (i.e. no location_addresses to collect)."""
    block = ir.Block([])
    region = ir.Region(block)
    fn = func.Function(
        sym_name="empty_fn",
        signature=func.Signature((), types.NoneType),
        slots=(),
        body=region,
    )
    result = rewrite.Walk(place2move.InsertFill()).rewrite(fn)
    assert not result.has_done_something


def test_insert_fill_already_filled_is_noop():
    """InsertFill is a no-op when the function body already begins with
    a move.Fill statement (i.e. the fill was already emitted in a prior pass)."""
    addr0 = LocationAddress(0, 0)
    angle = ir.TestValue()
    nlq = place.NewLogicalQubit(angle, angle, angle)
    nlq.location_address = addr0

    # Construct the block so move.Fill is genuinely the first statement,
    # which is the condition InsertFill checks to detect an already-filled func.
    load_outer = move.Load()
    fill = move.Fill(load_outer.result, location_addresses=(addr0,))
    store_outer = move.Store(fill.result)

    block = ir.Block([fill, store_outer, nlq])
    region = ir.Region(block)
    fn = func.Function(
        sym_name="already_filled",
        signature=func.Signature((), types.NoneType),
        slots=(),
        body=region,
    )
    result = rewrite.Walk(place2move.InsertFill()).rewrite(fn)
    assert not result.has_done_something


def test_insert_fill_none_location_address_is_noop():
    """InsertFill gives up (returns no-op RewriteResult) when any NewLogicalQubit
    has location_address=None — indicating the post-ResolvePinnedAddresses
    invariant has not been met."""
    angle = ir.TestValue()
    nlq = place.NewLogicalQubit(angle, angle, angle)
    # deliberately leave location_address=None (the default)

    block = ir.Block([nlq])
    region = ir.Region(block)
    fn = func.Function(
        sym_name="unresolved_fn",
        signature=func.Signature((), types.NoneType),
        slots=(),
        body=region,
    )
    result = rewrite.Walk(place2move.InsertFill()).rewrite(fn)
    assert not result.has_done_something
    # No move.Fill should have been inserted
    assert not any(isinstance(s, move.Fill) for s in block.walk())


# ---------------------------------------------------------------------------
# InsertInitialize tests
# ---------------------------------------------------------------------------


def test_insert_initialize_emits_logical_initialize():
    """InsertInitialize collects location_address, theta, phi, lam from two
    NewLogicalQubit statements and emits move.LogicalInitialize before the
    first non-NewLogicalQubit statement."""
    addr0 = LocationAddress(0, 0)
    addr1 = LocationAddress(0, 1)
    angle = ir.TestValue()

    nlq0 = place.NewLogicalQubit(angle, angle, angle)
    nlq0.location_address = addr0
    nlq1 = place.NewLogicalQubit(angle, angle, angle)
    nlq1.location_address = addr1
    terminator = py.Constant(42)

    test_block = ir.Block([nlq0, nlq1, terminator])
    rule = rewrite.Walk(place2move.InsertInitialize())
    result = rule.rewrite(test_block)

    assert result.has_done_something

    # The block should now have: NLQ0, NLQ1, Load, LogicalInitialize, Store, Constant
    stmt = test_block.first_stmt
    assert isinstance(stmt, place.NewLogicalQubit)
    stmt = stmt.next_stmt
    assert isinstance(stmt, place.NewLogicalQubit)
    stmt = stmt.next_stmt
    assert isinstance(stmt, move.Load)
    stmt = stmt.next_stmt
    assert isinstance(stmt, move.LogicalInitialize)
    assert stmt.location_addresses == (addr0, addr1)
    assert stmt.thetas == (angle, angle)
    assert stmt.phis == (angle, angle)
    assert stmt.lams == (angle, angle)
    stmt = stmt.next_stmt
    assert isinstance(stmt, move.Store)
    stmt = stmt.next_stmt
    assert isinstance(stmt, py.Constant)


def test_insert_initialize_empty_block_is_noop():
    """InsertInitialize is a no-op when the block's first statement is not a
    NewLogicalQubit (len(location_addresses) == 0 after the loop)."""
    test_block = ir.Block([py.Constant(99)])
    result = rewrite.Walk(place2move.InsertInitialize()).rewrite(test_block)
    assert not result.has_done_something


def test_insert_initialize_all_nlq_no_terminator_is_noop():
    """InsertInitialize is a no-op when the block contains only NewLogicalQubit
    statements with no following non-NewLogicalQubit insertion point (stmt is
    None after the loop)."""
    angle = ir.TestValue()
    nlq0 = place.NewLogicalQubit(angle, angle, angle)
    nlq0.location_address = LocationAddress(0, 0)
    nlq1 = place.NewLogicalQubit(angle, angle, angle)
    nlq1.location_address = LocationAddress(0, 1)

    test_block = ir.Block([nlq0, nlq1])
    result = rewrite.Walk(place2move.InsertInitialize()).rewrite(test_block)
    assert not result.has_done_something


def test_insert_initialize_none_location_address_is_noop():
    """InsertInitialize gives up (returns no-op RewriteResult) when any
    NewLogicalQubit has location_address=None — indicating the post-
    ResolvePinnedAddresses invariant has not been met."""
    angle = ir.TestValue()
    nlq = place.NewLogicalQubit(angle, angle, angle)
    # deliberately leave location_address=None (the default)
    terminator = py.Constant(0)

    test_block = ir.Block([nlq, terminator])
    result = rewrite.Walk(place2move.InsertInitialize()).rewrite(test_block)
    assert not result.has_done_something
    # No move.LogicalInitialize should have been inserted
    assert not any(isinstance(s, move.LogicalInitialize) for s in test_block.walk())
