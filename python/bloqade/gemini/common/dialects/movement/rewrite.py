from dataclasses import dataclass

from kirin import ir, types
from kirin.dialects import py
from kirin.rewrite import Walk
from kirin.rewrite.abc import RewriteResult, RewriteRule

from ._dialect import dialect
from .stmts import LocationAddressType, SiteId, WordId, ZoneId

_ATTR_TO_STMT = {"word_id": WordId, "site_id": SiteId, "zone_id": ZoneId}


@dataclass
class RewriteLocationAttr(RewriteRule):
    """Lower ``py.GetAttr`` on a ``LocationAddress`` (``word_id`` / ``site_id`` /
    ``zone_id``) into the corresponding movement-dialect statement."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, py.GetAttr):
            return RewriteResult()
        stmt_cls = _ATTR_TO_STMT.get(node.attrname)
        if stmt_cls is None:
            return RewriteResult()
        obj_type = node.obj.type
        # Bottom is a subtype of every type; exclude it explicitly so that
        # never-typed values are not rewritten.
        if obj_type is types.Bottom or not obj_type.is_subseteq(LocationAddressType):
            return RewriteResult()
        node.replace_by(stmt_cls(node.obj))
        return RewriteResult(has_done_something=True)


# Registered wrapped in Walk: PostInference applies registered rules as
# `rule.rewrite(mt.code)` on the top function node only (no recursion), so a
# bare rule would never reach nested GetAttr statements. See QuEraComputing/
# kirin#676.
dialect.rules.inference.append(Walk(RewriteLocationAttr()))
