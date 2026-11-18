from dataclasses import dataclass

from kirin import ir, types
from kirin.dialects import py
from kirin.ir.nodes.base import IRNode
from kirin.rewrite import Walk
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.arch.spec import ArchSpec

from ._dialect import dialect
from .stmts import ArchResolvedStmt, LocationAddressType, SiteId, WordId, ZoneId

_ATTR_TO_STMT = {"word_id": WordId, "site_id": SiteId, "zone_id": ZoneId}


@dataclass
class RewriteLocationAttr(RewriteRule):
    """Lower ``py.GetAttr`` on a ``LocationAddress`` (``word_id`` / ``site_id`` /
    ``zone_id``) into the corresponding arch-dialect statement."""

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


class _RewriteLocationAttrs(RewriteRule):
    def rewrite(self, node: IRNode) -> RewriteResult:
        # PostInference invokes a shared registered rule on the function root.
        # Walk owns a mutable worklist, so each invocation needs its own walker
        # when different kernels are compiled concurrently (e.g. simulator tasks).
        return Walk(RewriteLocationAttr()).rewrite(node)


dialect.rules.inference.append(_RewriteLocationAttrs())


@dataclass
class BindArchSpec(RewriteRule):
    """Populate ``arch_spec`` on every unbound arch-resolved statement
    (``Loc``, ``CzPartner``, and any future ``ArchResolvedStmt``).

    Run once as a ``CallGraphPass`` before any folding so the const-prop impls
    can resolve them during ``AggressiveUnroll``. Only binds statements whose
    attribute is still ``None`` — a statement that was constructed with an
    explicit arch spec is left alone.
    """

    arch_spec: ArchSpec

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, ArchResolvedStmt) or node.arch_spec is not None:
            return RewriteResult()
        node.arch_spec = self.arch_spec
        return RewriteResult(has_done_something=True)
