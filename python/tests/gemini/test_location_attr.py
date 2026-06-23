"""Tests for LocationAddress attribute statements + GetAttr rewrite."""

from kirin import ir, lowering

from bloqade.gemini.common.dialects.movement.stmts import SiteId, WordId, ZoneId


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
