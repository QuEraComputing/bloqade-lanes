"""Tests for LocationAddress attribute statements + GetAttr rewrite."""

from kirin import ir, lowering, types
from kirin.dialects import py

from bloqade.lanes.dialects.arch import (
    LocationAddressType,
    RewriteLocationAttr,
    SiteId,
    WordId,
    ZoneId,
)


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
