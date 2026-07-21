"""Regression tests for ExecuteCZReturn palindrome return-lane ordering.

The bug (fixed in #733): return_move_layers reversed each lane's direction but
preserved the original within-layer iteration order.  For multi-hop moves where
atom A vacates position X and atom B immediately occupies X in the same forward
layer, the return must process B first (so B vacates X) before A tries to
re-enter X.  Without the fix, A collides with B on the first return step.
"""

from kirin.dialects import ilist
from tests._squin_to_move_helper import squin_to_move

from bloqade import squin
from bloqade.lanes.analysis import atom
from bloqade.lanes.analysis.atom.lattice import AtomState
from bloqade.lanes.analysis.placement import PalindromePlacementStrategy
from bloqade.lanes.analysis.placement.lattice import ExecuteCZReturn
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.bytecode.encoding import (
    Direction,
    SiteLaneAddress,
    ZoneAddress,
)
from bloqade.lanes.dialects.move import EmitsState
from bloqade.lanes.heuristics.logical import (
    LogicalLayoutHeuristic,
    LogicalPlacementStrategyNoHome,
)

# ---------------------------------------------------------------------------
# Unit tests: ExecuteCZReturn.return_move_layers ordering
# ---------------------------------------------------------------------------


def _fwd(word: int, site: int, bus: int = 0) -> SiteLaneAddress:
    return SiteLaneAddress(word, site, bus, Direction.FORWARD)


def _make_return(move_layers):
    return ExecuteCZReturn(
        frozenset(),
        (),
        (),
        frozenset([ZoneAddress(0)]),
        move_layers=move_layers,
        initial_layout=(),
    )


def test_return_layers_single_independent_pair():
    """Two independent lanes in one layer: return has each reversed, in reversed order."""
    lane_a = _fwd(0, 0)
    lane_b = _fwd(1, 0)
    state = _make_return(((lane_a, lane_b),))

    assert state.return_move_layers == ((lane_b.reverse(), lane_a.reverse()),)


def test_return_layers_dependent_single_layer():
    """Atom A vacates X, then atom B enters X within the same forward layer.

    Forward:  (A: X→Y,  B: Z→X)   — A must go first
    Return:   (B: X→Z,  A: Y→X)   — B must go first (it occupies X)

    The buggy code produced (A: Y→X, B: X→Z), causing A to collide with B.
    """
    lane_a = _fwd(0, 0)  # A moves first, vacating position X
    lane_b = _fwd(1, 0)  # B moves second, entering X after A left
    state = _make_return(((lane_a, lane_b),))

    # B reversed must come before A reversed in the return
    assert state.return_move_layers == ((lane_b.reverse(), lane_a.reverse()),)


def test_return_layers_multi_layer_order_reversal():
    """Three forward layers → return layers appear in reversed order."""
    layer0 = (_fwd(0, 0),)
    layer1 = (_fwd(1, 0),)
    layer2 = (_fwd(2, 0),)
    state = _make_return((layer0, layer1, layer2))

    ret = state.return_move_layers
    assert len(ret) == 3
    # Layer order reversed: return_layer0 comes from forward layer2, etc.
    assert ret[0] == tuple(lane.reverse() for lane in reversed(layer2))
    assert ret[1] == tuple(lane.reverse() for lane in reversed(layer1))
    assert ret[2] == tuple(lane.reverse() for lane in reversed(layer0))


def test_return_layers_multihop_three_layer_dependent():
    """Reproduce the CZ-layer-2 steane7 multi-hop scenario structurally.

    Forward hop1: (A: p→q,  B: r→s,  C: t→u)   — independent
    Forward hop2: (A: q→v,  B: s→q,  C: u→w)   — B enters q after A vacated it
    Forward hop3: (C: w→x,)                     — single lane

    Correct return:
      ret_layer0 (from hop3): (C: x→w,)
      ret_layer1 (from hop2 reversed): (C: w→u,  B: q→s,  A: v→q)
      ret_layer2 (from hop1 reversed): (C: u→t,  B: s→r,  A: q→p)
    """
    a1, b1, c1 = _fwd(0, 0), _fwd(2, 0), _fwd(4, 0)  # hop1 — independent
    a2, b2, c2 = _fwd(1, 0), _fwd(3, 0), _fwd(5, 0)  # hop2 — B enters pos vacated by A
    c3 = _fwd(6, 0)  # hop3 — single lane

    state = _make_return(
        (
            (a1, b1, c1),  # layer0 / hop1
            (a2, b2, c2),  # layer1 / hop2
            (c3,),  # layer2 / hop3
        )
    )

    ret = state.return_move_layers
    assert len(ret) == 3
    assert ret[0] == (c3.reverse(),)
    assert ret[1] == (c2.reverse(), b2.reverse(), a2.reverse())
    assert ret[2] == (c1.reverse(), b1.reverse(), a1.reverse())


def test_return_layers_empty_move_layers():
    """No forward moves → no return moves."""
    state = _make_return(())
    assert state.return_move_layers == ()


def test_return_layers_single_lane_single_layer():
    """Trivial single-lane case: reversed(tuple-of-one) is the same tuple."""
    lane = _fwd(3, 1)
    state = _make_return(((lane,),))
    assert state.return_move_layers == ((lane.reverse(),),)


# ---------------------------------------------------------------------------
# Integration test: no collisions after multi-hop CZ return moves
# ---------------------------------------------------------------------------


def test_no_collisions_in_multihop_cz_return():
    """Verify the atom interpreter sees no collisions after compiling a 7-qubit
    circuit whose CZ layer requires multi-hop movement on the Gemini logical arch.

    CZ pattern: (q0↔q3, q2↔q5, q4↔q6) — qubits at non-adjacent words (home
    at even words 0,2,4,6,8,10,12) force 3-hop forward paths with within-layer
    dependencies.  The bug caused q3/q5 to collide on the palindrome return,
    removing both atoms from the state and corrupting subsequent operations.
    """
    arch_spec = get_arch_spec()

    @squin.kernel(typeinfer=True, fold=True)
    def circuit():
        reg = squin.qalloc(7)
        # CZ layer 2 of steane7: controls at even words, targets at non-adjacent words
        squin.broadcast.cz(
            ilist.IList([reg[0], reg[2], reg[4]]),
            ilist.IList([reg[3], reg[5], reg[6]]),
        )

    move_mt = squin_to_move(
        circuit,
        layout_heuristic=LogicalLayoutHeuristic(arch_spec=arch_spec),
        placement_strategy=PalindromePlacementStrategy(
            inner=LogicalPlacementStrategyNoHome(arch_spec=arch_spec)
        ),
        logical_initialize=False,
        no_raise=True,
    )

    interp = atom.AtomInterpreter(move_mt.dialects, arch_spec=arch_spec)
    frame, _ = interp.run(move_mt)

    collisions_found = []
    for stmt in move_mt.callable_region.walk():
        trait = stmt.get_trait(EmitsState)
        if trait is None:
            continue
        state_val = frame.get(trait.get_state_result(stmt))
        if not isinstance(state_val, AtomState):
            continue
        if state_val.data.collision:
            collisions_found.append(
                (type(stmt).__name__, dict(state_val.data.collision))
            )

    assert (
        collisions_found == []
    ), f"Atom collisions detected in multi-hop CZ return: {collisions_found}"
