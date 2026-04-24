"""End-to-end: bytecode Program -> decode -> lower -> measure_lower."""

from bloqade.test_utils import assert_nodes
from kirin import ir, types
from kirin.dialects import func, ilist, py
from kirin.rewrite import Walk
from kirin.rewrite.dce import DeadCodeElimination

from bloqade.lanes._prelude import kernel
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.bytecode import Instruction, Program
from bloqade.lanes.bytecode._native import Direction, MoveType
from bloqade.lanes.bytecode.decode import load_program
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import LaneAddress, LocationAddress, ZoneAddress
from bloqade.lanes.rewrite.measure_lower import MeasureLower
from bloqade.lanes.rewrite.stack_move2move import RewriteStackMoveToMove


def test_minimal_program_runs_end_to_end():
    """Bytecode Program -> stack_move IR -> multi-dialect IR with EndMeasure.

    Asserts the full expected block structure after both rewrites:
    move.Load, move.Fill (InitialFill lowers to Fill), move.EndMeasure,
    move.GetFutureResult*N (one per location yielded by the ArchSpec
    for zone 0), ilist.New (AwaitMeasure bundle), move.Store, func.Return.
    """
    prog = Program(
        version=(1, 0),
        instructions=[
            Instruction.const_loc(0, 0, 0),
            Instruction.initial_fill(1),
            Instruction.const_zone(0),
            Instruction.measure(1),
            Instruction.await_measure(),
            Instruction.return_(),
        ],
    )

    arch_spec = get_arch_spec()

    # Stage 1: decode bytecode -> stack_move IR.
    method = load_program(prog)

    # Stage 2: lower stack_move -> multi-dialect IR. Walk can operate
    # directly on the method's top-level Function statement — no need to
    # dig out the block.
    Walk(RewriteStackMoveToMove(arch_spec=arch_spec)).rewrite(method.code)

    # Stage 2b: DCE sweeps away stack_move.Const{Loc,Lane,Zone} stmts
    # left behind by stack_move2move. The rewrite lifts their attribute
    # values into downstream move.* attributes via ``_lift_attrs`` but
    # intentionally doesn't touch the defining statements — DCE removes
    # them once the consumer edges are gone.
    Walk(DeadCodeElimination()).rewrite(method.code)

    # The method's dialect group still points at the decoder's
    # stack_move+func group; AtomInterpreter needs the full move-pipeline
    # dialect group to dispatch against the lowered statements.
    method.dialects = kernel

    # Stage 3: measure_lower.
    rule = MeasureLower.from_method(method, arch_spec=arch_spec)
    Walk(rule).rewrite(method.code)

    # Assert the full block structure (types, order, and key attribute
    # values for the stateful-op sites).
    block = method.callable_region.blocks[0]
    stmts = list(block.stmts)

    expected_zone_locs = list(arch_spec.yield_zone_locations(ZoneAddress(0)))
    expected_types = (
        [move.Load, move.Fill, move.EndMeasure]
        + [move.GetFutureResult] * len(expected_zone_locs)
        + [ilist.New, move.Store, func.Return]
    )
    assert [
        type(s) for s in stmts
    ] == expected_types, (
        f"block structure mismatch; got {[type(s).__name__ for s in stmts]}"
    )

    fill = next(s for s in stmts if isinstance(s, move.Fill))
    assert fill.location_addresses == (LocationAddress(0, 0, 0),)

    end_measure = next(s for s in stmts if isinstance(s, move.EndMeasure))
    assert end_measure.zone_addresses == (ZoneAddress(0),)

    # GetFutureResults are ordered by ArchSpec iteration and all share
    # the same zone_address (zone 0).
    gfrs = [s for s in stmts if isinstance(s, move.GetFutureResult)]
    assert [g.location_address for g in gfrs] == expected_zone_locs
    assert all(g.zone_address == ZoneAddress(0) for g in gfrs)


def test_realistic_cz_sandwich_runs_end_to_end():
    """Realistic logical-arch kernel: fill 4 atoms, entangle pairs with
    CZ-sandwich patterns, measure, extract results at the original
    filled word_ids, return the extracted 4-element array.

    Program layout:

    1. Initial fill: 4 atoms at zone 0, site 0, words 0/2/4/6 (atoms
       0/1/2/3 respectively).
    2. Move atom 0 word 0 -> word 3 (two lanes in one ``move``
       instruction: bus 0 fwd 0->1, then bus 9 fwd 1->3).
    3. CZ on zone 0 (atom 0 at word 3 now pairs with atom 1 at word 2
       through entangling pair (2,3)).
    4. Move atom 0 back word 3 -> word 0 (two lanes: bus 9 bwd 3->1,
       bus 0 bwd 1->0).
    5. Move atoms 2 and 3 from words 4/6 -> words 1/3 (bus 5 fwd; pairs
       atom 2 with atom 1 at word 2 via entangling pair (1,2) fails —
       pairs are (0,1)/(2,3), so atom 2 at word 1 partners with an
       empty word 0; actually with atoms 2->word 1 and 3->word 3, the
       CZ pair (2,3) now entangles atoms 1 and 3). The structural
       assertions below don't depend on the interpretation of CZ; the
       point is to exercise a second realistic move+CZ+move-back
       pattern.
    6. CZ on zone 0.
    7. Move atoms 2 and 3 back from words 1/3 -> words 4/6.
    8. Measure zone 0.
    9. Await future; extract results at indices 0/2/4/6 (the original
       fill word_ids) and bundle into a new 1-D integer array.
    10. Return that array.
    """

    # ── Build the bytecode program ────────────────────────────────────
    instrs: list[Instruction] = [
        # 1. Fill 4 atoms at zone 0, site 0, words 0/2/4/6.
        Instruction.const_loc(0, 0, 0),
        Instruction.const_loc(0, 2, 0),
        Instruction.const_loc(0, 4, 0),
        Instruction.const_loc(0, 6, 0),
        Instruction.initial_fill(4),
        # 2. Move atom 0: word 0 -> word 1 (bus 0 fwd), word 1 -> word 3
        #    (bus 9 fwd). Two lanes in one move instruction — the Rust
        #    apply_moves runs lanes sequentially, so atom 0 hops
        #    0->1->3.
        Instruction.const_lane(MoveType.WORD, 0, 0, 0, 0, Direction.FORWARD),
        Instruction.const_lane(MoveType.WORD, 0, 1, 0, 9, Direction.FORWARD),
        Instruction.move_(2),
        # 3. CZ on zone 0.
        Instruction.const_zone(0),
        Instruction.cz(),
        # 4. Move atom 0 back: 3 -> 1 (bus 9 bwd with word_id=1),
        #    1 -> 0 (bus 0 bwd with word_id=0).
        Instruction.const_lane(MoveType.WORD, 0, 1, 0, 9, Direction.BACKWARD),
        Instruction.const_lane(MoveType.WORD, 0, 0, 0, 0, Direction.BACKWARD),
        Instruction.move_(2),
        # 5. Move atoms 2 and 3: word 4 -> word 1 (bus 5 fwd),
        #    word 6 -> word 3 (bus 5 fwd). One move instruction,
        #    two lanes.
        Instruction.const_lane(MoveType.WORD, 0, 4, 0, 5, Direction.FORWARD),
        Instruction.const_lane(MoveType.WORD, 0, 6, 0, 5, Direction.FORWARD),
        Instruction.move_(2),
        # 6. CZ on zone 0.
        Instruction.const_zone(0),
        Instruction.cz(),
        # 7. Move atoms 2 and 3 back: 1 -> 4, 3 -> 6 (bus 5 backward).
        Instruction.const_lane(MoveType.WORD, 0, 4, 0, 5, Direction.BACKWARD),
        Instruction.const_lane(MoveType.WORD, 0, 6, 0, 5, Direction.BACKWARD),
        Instruction.move_(2),
        # 8. Measure zone 0.
        Instruction.const_zone(0),
        Instruction.measure(1),
        # 9. Await the future (consumes it, pushes the 1-D result array).
        Instruction.await_measure(),
        # 10. Extract results at the original fill word_ids (0, 2, 4, 6)
        #     and bundle into a new 1-D Int array. Stack discipline:
        #     dup the array each time, then leave the last get_item to
        #     consume the final array copy without an extra dup.
        Instruction.dup(),
        Instruction.const_int(0),
        Instruction.get_item(1),
        Instruction.swap(),
        Instruction.dup(),
        Instruction.const_int(2),
        Instruction.get_item(1),
        Instruction.swap(),
        Instruction.dup(),
        Instruction.const_int(4),
        Instruction.get_item(1),
        Instruction.swap(),
        Instruction.const_int(6),
        Instruction.get_item(1),
        # new_array(type_tag=1 Int, dim0=4, dim1=0) — placeholder: no
        # dedicated MeasurementResult tag yet (see follow-up #547).
        Instruction.new_array(1, 4, 0),
        # 11. Return the 4-element result array.
        Instruction.return_(),
    ]

    arch_spec = get_arch_spec()

    # ── Run the full pipeline ─────────────────────────────────────────
    method = load_program(Program(version=(1, 0), instructions=instrs))
    Walk(RewriteStackMoveToMove(arch_spec=arch_spec)).rewrite(method.code)
    # DCE sweeps away stack_move.Const{Loc,Lane,Zone} stmts that were
    # left in place by stack_move2move — their attribute values have
    # been lifted into downstream move.* attributes and the SSAs have
    # no remaining uses.
    Walk(DeadCodeElimination()).rewrite(method.code)
    method.dialects = kernel
    Walk(MeasureLower.from_method(method, arch_spec=arch_spec)).rewrite(method.code)

    # ── Build the expected block by hand and compare structurally ────
    zone0 = ZoneAddress(0)

    def lane(word_id: int, bus_id: int, direction: Direction) -> LaneAddress:
        return LaneAddress(
            move_type=MoveType.WORD,
            word_id=word_id,
            site_id=0,
            bus_id=bus_id,
            direction=direction,
            zone_id=0,
        )

    atom_0_forward = (
        lane(word_id=0, bus_id=0, direction=Direction.FORWARD),
        lane(word_id=1, bus_id=9, direction=Direction.FORWARD),
    )
    atom_0_backward = (
        lane(word_id=1, bus_id=9, direction=Direction.BACKWARD),
        lane(word_id=0, bus_id=0, direction=Direction.BACKWARD),
    )
    atoms_23_forward = (
        lane(word_id=4, bus_id=5, direction=Direction.FORWARD),
        lane(word_id=6, bus_id=5, direction=Direction.FORWARD),
    )
    atoms_23_backward = (
        lane(word_id=4, bus_id=5, direction=Direction.BACKWARD),
        lane(word_id=6, bus_id=5, direction=Direction.BACKWARD),
    )

    # State threading: each stateful op consumes the previous state and
    # produces the next one. ``current_state`` tracks the most recent
    # StateType SSA value as we build the expected block. Address
    # constants (ConstLoc/ConstLane/ConstZone) are *not* emitted in the
    # target IR — stack_move2move reads their values off the defining
    # statement via ``_lift_attrs`` and then a DCE pass removes the
    # orphaned stack_move.Const* stmts before we build the expected
    # block. ``measure_lower`` forwards the state from ``move4`` to the
    # trailing ``move.Store``.
    load = move.Load()
    fill = move.Fill(
        load.result,
        location_addresses=(
            LocationAddress(0, 0, 0),
            LocationAddress(2, 0, 0),
            LocationAddress(4, 0, 0),
            LocationAddress(6, 0, 0),
        ),
    )
    move1 = move.Move(fill.result, lanes=atom_0_forward)
    cz1 = move.CZ(move1.result, zone_address=zone0)
    move2 = move.Move(cz1.result, lanes=atom_0_backward)
    move3 = move.Move(move2.result, lanes=atoms_23_forward)
    cz2 = move.CZ(move3.result, zone_address=zone0)
    move4 = move.Move(cz2.result, lanes=atoms_23_backward)
    end_measure = move.EndMeasure(current_state=move4.result, zone_addresses=(zone0,))

    # AwaitMeasure expands to one move.GetFutureResult per location
    # yielded by the ArchSpec for zone 0, bundled into an ilist.New.
    zone_locs = list(arch_spec.yield_zone_locations(zone0))
    gfrs = [
        move.GetFutureResult(
            measurement_future=end_measure.result,
            zone_address=zone0,
            location_address=loc,
        )
        for loc in zone_locs
    ]
    await_bundle = ilist.New(values=tuple(g.result for g in gfrs))

    # Four get_item(1) instructions lower to py.Constant(idx) +
    # py.indexing.GetItem into the await bundle.
    idx0 = py.Constant(0)
    get0 = py.indexing.GetItem(obj=await_bundle.result, index=idx0.result)
    idx2 = py.Constant(2)
    get2 = py.indexing.GetItem(obj=await_bundle.result, index=idx2.result)
    idx4 = py.Constant(4)
    get4 = py.indexing.GetItem(obj=await_bundle.result, index=idx4.result)
    idx6 = py.Constant(6)
    get6 = py.indexing.GetItem(obj=await_bundle.result, index=idx6.result)

    # new_array(type_tag=1 Int, dim0=4, dim1=0) -> flat ilist.New of
    # the four extracted values.
    result_bundle = ilist.New(
        values=(get0.result, get2.result, get4.result, get6.result)
    )

    # rewrite_Block inserts the final move.Store before the terminator;
    # measure_lower forwarded move.Measure's state result to move4's
    # state (i.e. the state just before the measurement), so Store
    # consumes move4.result.
    store = move.Store(move4.result)
    ret = func.Return(result_bundle.result)

    # The decoded method's entry block carries a single argument of
    # MethodType (the usual func.Function prelude); mirror it here so
    # structural equality matches the block args as well as the
    # statements.
    expected_block = ir.Block(
        [
            load,
            fill,
            move1,
            cz1,
            move2,
            move3,
            cz2,
            move4,
            end_measure,
            *gfrs,
            await_bundle,
            idx0,
            get0,
            idx2,
            get2,
            idx4,
            get4,
            idx6,
            get6,
            result_bundle,
            store,
            ret,
        ],
        argtypes=(types.MethodType,),
    )

    actual_block = method.callable_region.blocks[0]
    assert_nodes(actual_block, expected_block)
