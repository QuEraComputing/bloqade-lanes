"""Tests for the arch-dialect cz_partner(loc) feature."""

import typing

from kirin import ir, lowering, rewrite
from kirin.dialects import ilist

from bloqade import squin
from bloqade.gemini import physical
from bloqade.gemini.common.dialects import arrange, qubit
from bloqade.lanes.arch.gemini.physical import get_physical_layout_arch_spec
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import arch, move as move_dialect
from bloqade.lanes.dialects.arch import CzPartner
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
    assert CzPartner in arch.dialect.stmts

    # arch_spec is an attribute (default None) — populated by the pipeline's
    # BindCzPartnerArchSpec pass before unrolling.
    addr = ir.TestValue()
    stmt = CzPartner(addr)
    assert stmt.arch_spec is None


def test_cz_partner_wrapper_callable():
    assert callable(arch.cz_partner)


# ---------------------------------------------------------------------------
# End-to-end through the physical pipeline
# ---------------------------------------------------------------------------


def test_bind_arch_spec_sets_attribute_on_unbound_statement():
    from bloqade.lanes.dialects.arch import BindCzPartnerArchSpec

    arch = get_physical_layout_arch_spec()
    addr = ir.TestValue()
    stmt = CzPartner(addr)
    assert stmt.arch_spec is None
    region = ir.Region([ir.Block([stmt])])

    rewrite.Walk(BindCzPartnerArchSpec(arch)).rewrite(region)

    assert stmt.arch_spec is arch


def test_bind_arch_spec_leaves_already_bound_statement_alone():
    from bloqade.lanes.dialects.arch import BindCzPartnerArchSpec

    arch_a = get_physical_layout_arch_spec()
    arch_b = get_physical_layout_arch_spec()
    assert arch_a is not arch_b

    addr = ir.TestValue()
    stmt = CzPartner(addr, arch_spec=arch_a)
    region = ir.Region([ir.Block([stmt])])

    rewrite.Walk(BindCzPartnerArchSpec(arch_b)).rewrite(region)

    assert stmt.arch_spec is arch_a


# NOTE on ``verify=False`` below: the helper kernels apply an ``ilist.map`` over
# a closure that *captures* outer SSA values (``locs``'s ``_inner`` closes over
# ``words``/``sites``; ``main``/``with_partner``'s ``_partner`` closes over
# ``static_addrs``). Kirin's verify pipeline (the No-Cloning forward analysis)
# mis-invokes such captured closures ("Method called with N arguments, expected
# M") — the same interprocedural ilist.map gap tracked in QuEraComputing/kirin#679
# and QuEraComputing/bloqade-circuit#830. This is unrelated to cz_partner (a plain
# capturing-closure ``ilist.map`` kernel trips it too), so verify stays off until
# that limitation is fixed. ``alloc`` uses a non-capturing closure and verifies
# cleanly, so it keeps the default ``verify=True``.
def _build_kernel():
    krn = physical.kernel

    @krn(verify=False)
    def locs(words: ilist.IList[int, N], sites: ilist.IList[int, N]):
        def _inner(i: int):
            return arch.loc(0, words[i], sites[i])

        return ilist.map(_inner, ilist.range(len(words)))

    @krn()
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
            return arch.cz_partner(static_addrs[i])

        partners = ilist.map(_partner, ilist.range(2))
        arrange.move_to(mobile, partners)
        squin.broadcast.cx(mobile, static)
        arrange.move_to(mobile, mobile_addrs)

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
            return arch.loc(0, words[i], sites[i])

        return ilist.map(_inner, ilist.range(len(words)))

    @krn()
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
            return arch.cz_partner(static_addrs[i])

        arrange.move_to(mobile, ilist.map(_partner, ilist.range(2)))
        squin.broadcast.cx(mobile, static)

    @krn(aggressive_unroll=True, verify=False)
    def hardcoded():
        static_addrs = locs(ilist.IList([0, 4]), ilist.IList([0, 0]))
        mobile_addrs = locs(ilist.IList([2, 6]), ilist.IList([0, 0]))
        partner_addrs = locs(ilist.IList([1, 5]), ilist.IList([0, 0]))
        static = alloc(static_addrs)
        mobile = alloc(mobile_addrs)
        arrange.move_to(mobile, partner_addrs)
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


def test_cz_partner_partnerless_location_does_not_const_resolve():
    """Regression for the no-CZ-partner case.

    A ``cz_partner`` on a location *with* a partner const-folds to a concrete
    ``LocationAddress`` value; a ``cz_partner`` on a location with *no* partner
    must NOT fold — it stays top/Unknown so the unresolved partner propagates
    as a non-const ``move_to`` target (and ultimately a compile error) instead
    of silently resolving to a wrong location.

    We assert the const-prop behavior directly (the mechanism that makes an
    unresolved cz_partner surface downstream). An end-to-end "pipeline.emit
    raises" assertion was deliberately avoided: with no_raise=False the
    Physical Terminal Measurement validation fires first on this kernel shape
    regardless of the partner, so it cannot isolate the partnerless behavior.
    """
    from kirin.analysis import const

    from bloqade.lanes.dialects.arch import BindCzPartnerArchSpec

    arch_spec = get_physical_layout_arch_spec()

    def cz_partner_lattice(loc: LocationAddress):
        krn = physical.kernel

        @krn(verify=False)
        def k():
            return arch.cz_partner(arch.loc(loc.zone_id, loc.word_id, loc.site_id))

        # Bind the arch spec exactly as the pipeline does, then run const-prop.
        rewrite.Walk(BindCzPartnerArchSpec(arch_spec)).rewrite(k.code)
        frame, _ = const.Propagate(k.dialects).run(k)
        cz = next(s for s in k.callable_region.walk() if isinstance(s, CzPartner))
        return frame.entries.get(cz.result)

    # A partnered location (zone 0 is fully partnered in the physical arch).
    partnered = LocationAddress(zone_id=0, word_id=0, site_id=0)
    assert arch_spec.get_cz_partner(partnered) is not None

    # Find a partnerless location to drive the regression.
    partnerless = next(
        (
            candidate
            for zone_id in range(1, 3)
            for candidate in [LocationAddress(zone_id=zone_id, word_id=0, site_id=0)]
            if arch_spec.get_cz_partner(candidate) is None
        ),
        None,
    )
    assert partnerless is not None, "no partnerless location found in physical arch"

    # Partnered → const-folds to a concrete LocationAddress value.
    assert isinstance(cz_partner_lattice(partnered), const.Value)
    # Partnerless → stays unresolved (not a const Value).
    assert not isinstance(cz_partner_lattice(partnerless), const.Value)
