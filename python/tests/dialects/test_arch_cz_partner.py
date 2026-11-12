"""Tests for the arch-dialect cz_partner(loc) feature."""

from kirin import ir, lowering, rewrite

from bloqade.lanes.arch.gemini.physical import get_physical_layout_arch_spec
from bloqade.lanes.dialects import arch
from bloqade.lanes.dialects.arch import CzPartner

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
    # BindArchSpec pass before unrolling.
    addr = ir.TestValue()
    stmt = CzPartner(addr)
    assert stmt.arch_spec is None


def test_cz_partner_wrapper_callable():
    assert callable(arch.cz_partner)


# ---------------------------------------------------------------------------
# Binding architecture specifications
# ---------------------------------------------------------------------------


def test_bind_arch_spec_sets_attribute_on_unbound_statement():
    from bloqade.lanes.dialects.arch import BindArchSpec

    arch = get_physical_layout_arch_spec()
    addr = ir.TestValue()
    stmt = CzPartner(addr)
    assert stmt.arch_spec is None
    region = ir.Region([ir.Block([stmt])])

    rewrite.Walk(BindArchSpec(arch)).rewrite(region)

    assert stmt.arch_spec is arch


def test_bind_arch_spec_leaves_already_bound_statement_alone():
    from bloqade.lanes.dialects.arch import BindArchSpec

    arch_a = get_physical_layout_arch_spec()
    arch_b = get_physical_layout_arch_spec()
    assert arch_a is not arch_b

    addr = ir.TestValue()
    stmt = CzPartner(addr, arch_spec=arch_a)
    region = ir.Region([ir.Block([stmt])])

    rewrite.Walk(BindArchSpec(arch_b)).rewrite(region)

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
