from dataclasses import dataclass

from kirin import ir, types
from kirin.analysis import const
from kirin.dialects import py
from kirin.rewrite import Walk
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LocationAddress

from ._dialect import dialect
from .stmts import CzPartner, Loc, LocationAddressType, SiteId, WordId, ZoneId

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


@dataclass
class ResolveCzPartner(RewriteRule):
    """Resolve ``movement.cz_partner(loc)`` into a constant ``Loc``.

    ``CzPartner`` cannot const-fold on its own — the blockade-partner relation
    lives in the architecture spec, not in the kernel. This arch-aware rewrite
    reads the const-folded input location, looks up its partner, and replaces
    the statement with a ``Loc`` built from the partner's ``(zone, word, site)``
    so the standard const-folding machinery propagates it (e.g. into a
    ``move_to`` locations list).

    Cases this rewrite cannot resolve are left untouched:

    * the input ``address`` did not const-fold — the unfolded source location
      statement is itself flagged by the existing location validation;
    * the location has no CZ partner — the surviving ``CzPartner`` keeps the
      downstream ``move_to`` locations non-const, which the move_to validation
      reports.

    Must run after const-folding (so ``address`` carries a ``const`` hint) and
    be followed by another fold pass so the new ``Loc`` propagates.
    """

    arch_spec: ArchSpec

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, CzPartner):
            return RewriteResult()

        hint = node.address.hints.get("const")
        if not isinstance(hint, const.Value) or not isinstance(
            hint.data, LocationAddress
        ):
            return RewriteResult()

        partner = self.arch_spec.get_cz_partner(hint.data)
        if partner is None:
            return RewriteResult()

        zone = py.Constant(partner.zone_id)
        word = py.Constant(partner.word_id)
        site = py.Constant(partner.site_id)
        for stmt in (zone, word, site):
            stmt.insert_before(node)
        node.replace_by(Loc(zone.result, word.result, site.result))
        return RewriteResult(has_done_something=True)
