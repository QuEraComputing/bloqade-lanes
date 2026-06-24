"""Tests for the movement-dialect cz_partner(loc) feature."""

import typing

from kirin import ir, lowering, rewrite
from kirin.analysis import const
from kirin.dialects import ilist, py

from bloqade import squin
from bloqade.gemini import physical
from bloqade.gemini.common.dialects import movement, qubit
from bloqade.gemini.common.dialects.movement.rewrite import ResolveCzPartner
from bloqade.gemini.common.dialects.movement.stmts import CzPartner, Loc
from bloqade.lanes.arch.gemini.physical import get_physical_layout_arch_spec
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import move as move_dialect
from bloqade.lanes.heuristics.physical import make_physical_placement_strategy
from bloqade.lanes.pipeline import PhysicalPipeline

N = typing.TypeVar("N")


# ---------------------------------------------------------------------------
# Statement / wrapper shape
# ---------------------------------------------------------------------------


def test_cz_partner_statement_shape():
    assert issubclass(CzPartner, ir.Statement)
    assert CzPartner.name == "cz_partner"
    assert any(isinstance(t, lowering.FromPythonCall) for t in CzPartner.traits)
    # CzPartner is now Pure: materialization rides on the standard
    # const-prop / fold machinery once arch_spec is bound by the pipeline.
    assert any(isinstance(t, ir.Pure) for t in CzPartner.traits)
    assert CzPartner in movement.dialect.stmts

    # arch_spec is an attribute (default None) — populated by the pipeline's
    # BindCzPartnerArchSpec pass before unrolling.
    addr = ir.TestValue()
    stmt = CzPartner(addr)
    assert stmt.arch_spec is None


def test_cz_partner_wrapper_callable():
    assert callable(movement.cz_partner)


# ---------------------------------------------------------------------------
# ResolveCzPartner rewrite (unit)
# ---------------------------------------------------------------------------


def _const_loc_value(w, s, z=0):
    return const.Value(LocationAddress(zone_id=z, word_id=w, site_id=s))


def test_resolve_replaces_const_input_with_partner_loc():
    arch = get_physical_layout_arch_spec()
    loc0 = LocationAddress(zone_id=0, word_id=0, site_id=0)
    partner = arch.get_cz_partner(loc0)
    assert partner is not None

    # A CzPartner whose address operand carries a const LocationAddress hint.
    addr = ir.TestValue()
    addr.hints["const"] = const.Value(loc0)
    stmt = CzPartner(addr)
    block = ir.Block([stmt])
    region = ir.Region([block])

    rewrite.Walk(ResolveCzPartner(arch)).rewrite(region)

    # CzPartner is gone; a Loc on the partner's (zone, word, site) replaced it.
    assert not any(isinstance(s, CzPartner) for s in region.walk())
    locs = [s for s in region.walk() if isinstance(s, Loc)]
    assert len(locs) == 1
    consts = {
        getattr(s.value, "data", None)
        for s in region.walk()
        if isinstance(s, py.Constant)
    }
    assert {partner.zone_id, partner.word_id, partner.site_id} <= consts


def test_resolve_leaves_non_const_input_untouched():
    arch = get_physical_layout_arch_spec()
    addr = ir.TestValue()  # no const hint
    stmt = CzPartner(addr)
    region = ir.Region([ir.Block([stmt])])

    rewrite.Walk(ResolveCzPartner(arch)).rewrite(region)

    # Unresolvable: left in place (existing location validation reports it).
    assert any(isinstance(s, CzPartner) for s in region.walk())


# ---------------------------------------------------------------------------
# End-to-end through the physical pipeline
# ---------------------------------------------------------------------------


def _build_kernel():
    krn = physical.kernel

    @krn(verify=False)
    def locs(words: ilist.IList[int, N], sites: ilist.IList[int, N]):
        def _inner(i: int):
            return movement.loc(0, words[i], sites[i])

        return ilist.map(_inner, ilist.range(len(words)))

    @krn(verify=False)
    def alloc(addresses: ilist.IList[LocationAddress, N]):
        def _inner(addr: LocationAddress):
            return qubit.new_at(0, addr.word_id, addr.site_id)

        return ilist.map(_inner, addresses)

    @krn(aggressive_unroll=True, verify=False)
    def main():
        static_addrs = locs(ilist.IList([0, 4]), ilist.IList([0, 0]))
        mobile_addrs = locs(ilist.IList([2, 6]), ilist.IList([0, 0]))
        static = alloc(static_addrs)
        mobile = alloc(mobile_addrs)

        # Stage mobile onto static's CZ partner sites via cz_partner — no
        # hardcoded partner words. Already-paired short-circuit then makes the
        # CZ itself a no-op, so all moves are user-directed and predictable.
        def _partner(i: int):
            return movement.cz_partner(static_addrs[i])

        partners = ilist.map(_partner, ilist.range(2))
        movement.move_to(mobile, partners)
        squin.broadcast.cx(mobile, static)
        movement.move_to(mobile, mobile_addrs)

    return main


def test_cz_partner_end_to_end_compiles_and_no_residual_nodes():
    strat = make_physical_placement_strategy(return_moves=False)
    pipeline = PhysicalPipeline(placement_strategy=strat)
    out = pipeline.emit(_build_kernel())

    stmts = list(out.callable_region.walk())
    # No residual cz_partner survives compilation.
    assert not any(isinstance(s, CzPartner) for s in stmts)
    # The staging move_to produced real move-dialect moves.
    assert any(isinstance(s, move_dialect.Move) for s in stmts)


def test_cz_partner_matches_hardcoded_partner_words():
    """A kernel using cz_partner(static) must compile to the same move IR as
    one that hardcodes static's partner words (1, 5)."""

    krn = physical.kernel

    @krn(verify=False)
    def locs(words: ilist.IList[int, N], sites: ilist.IList[int, N]):
        def _inner(i: int):
            return movement.loc(0, words[i], sites[i])

        return ilist.map(_inner, ilist.range(len(words)))

    @krn(verify=False)
    def alloc(addresses: ilist.IList[LocationAddress, N]):
        def _inner(addr: LocationAddress):
            return qubit.new_at(0, addr.word_id, addr.site_id)

        return ilist.map(_inner, addresses)

    @krn(aggressive_unroll=True, verify=False)
    def with_partner():
        static_addrs = locs(ilist.IList([0, 4]), ilist.IList([0, 0]))
        mobile_addrs = locs(ilist.IList([2, 6]), ilist.IList([0, 0]))
        static = alloc(static_addrs)
        mobile = alloc(mobile_addrs)

        def _partner(i: int):
            return movement.cz_partner(static_addrs[i])

        movement.move_to(mobile, ilist.map(_partner, ilist.range(2)))
        squin.broadcast.cx(mobile, static)

    @krn(aggressive_unroll=True, verify=False)
    def hardcoded():
        static_addrs = locs(ilist.IList([0, 4]), ilist.IList([0, 0]))
        mobile_addrs = locs(ilist.IList([2, 6]), ilist.IList([0, 0]))
        partner_addrs = locs(ilist.IList([1, 5]), ilist.IList([0, 0]))
        static = alloc(static_addrs)
        mobile = alloc(mobile_addrs)
        movement.move_to(mobile, partner_addrs)
        squin.broadcast.cx(mobile, static)

    def _move_lanes(kernel):
        strat = make_physical_placement_strategy(return_moves=False)
        out = PhysicalPipeline(placement_strategy=strat).emit(kernel)
        return [
            s.lanes
            for s in out.callable_region.walk()
            if isinstance(s, move_dialect.Move)
        ]

    assert _move_lanes(with_partner) == _move_lanes(hardcoded)
