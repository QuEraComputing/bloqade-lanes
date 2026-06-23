"""Tests for LocationAddress attribute statements + GetAttr rewrite."""

from kirin import ir, lowering, types
from kirin.dialects import func, py

from bloqade.gemini.common.dialects.movement.rewrite import RewriteLocationAttr
from bloqade.gemini.common.dialects.movement.stmts import (
    LocationAddressType,
    SiteId,
    WordId,
    ZoneId,
)
from bloqade.gemini.logical import loc
from bloqade.gemini.physical import kernel as movement_kernel
from bloqade.lanes.bytecode.encoding import LocationAddress


def test_statements_are_pure_and_named():
    for stmt_cls, name in [
        (WordId, "word_id"),
        (SiteId, "site_id"),
        (ZoneId, "zone_id"),
    ]:
        assert issubclass(stmt_cls, ir.Statement)
        assert stmt_cls.name == name
        # Pure so the fold pass can evaluate them.
        assert any(isinstance(t, ir.Pure) for t in stmt_cls.traits)
        # Not lowered from a Python call (constructed only by the rewrite).
        assert not any(isinstance(t, lowering.FromPythonCall) for t in stmt_cls.traits)


# -- rule unit tests: the guard cases must NOT rewrite --


def test_rewrite_skips_unsupported_attrname():
    addr = ir.TestValue(type=LocationAddressType)
    node = py.GetAttr(obj=addr, attrname="nonexistent")
    result = RewriteLocationAttr().rewrite_Statement(node)
    assert not result.has_done_something


def test_rewrite_skips_non_location_type():
    obj = ir.TestValue(type=types.Int)
    node = py.GetAttr(obj=obj, attrname="word_id")
    result = RewriteLocationAttr().rewrite_Statement(node)
    assert not result.has_done_something


def test_rewrite_skips_bottom_type():
    # Bottom is a subtype of every type, so without the explicit guard this
    # would wrongly match. It must be skipped.
    obj = ir.TestValue(type=types.Bottom)
    node = py.GetAttr(obj=obj, attrname="word_id")
    result = RewriteLocationAttr().rewrite_Statement(node)
    assert not result.has_done_something


# -- integration: a symbolic (non-constant) address rewrites to the statement --


def test_param_word_id_rewrites_to_statement():
    @movement_kernel(verify=False)
    def k(a: LocationAddress):
        return a.word_id

    stmts = list(k.callable_region.walk())
    assert not any(isinstance(s, py.GetAttr) for s in stmts)
    ret = next(s for s in stmts if isinstance(s, func.Return))
    assert isinstance(ret.value.owner, WordId)


# -- integration: a constant address folds the attribute read to a constant --


def test_constant_word_id_folds():
    @movement_kernel(verify=False)
    def k():
        return loc(zone_id=2, word_id=3, site_id=5).word_id

    stmts = list(k.callable_region.walk())
    assert not any(isinstance(s, py.GetAttr) for s in stmts)
    ret = next(s for s in stmts if isinstance(s, func.Return))
    assert isinstance(ret.value.owner, py.Constant)
    assert ret.value.owner.value.unwrap() == 3


def test_constant_site_id_folds():
    @movement_kernel(verify=False)
    def k():
        return loc(zone_id=2, word_id=3, site_id=5).site_id

    stmts = list(k.callable_region.walk())
    assert not any(isinstance(s, py.GetAttr) for s in stmts)
    ret = next(s for s in stmts if isinstance(s, func.Return))
    assert isinstance(ret.value.owner, py.Constant)
    assert ret.value.owner.value.unwrap() == 5


def test_constant_zone_id_folds():
    @movement_kernel(verify=False)
    def k():
        return loc(zone_id=2, word_id=3, site_id=5).zone_id

    stmts = list(k.callable_region.walk())
    assert not any(isinstance(s, py.GetAttr) for s in stmts)
    ret = next(s for s in stmts if isinstance(s, func.Return))
    assert isinstance(ret.value.owner, py.Constant)
    assert ret.value.owner.value.unwrap() == 2
