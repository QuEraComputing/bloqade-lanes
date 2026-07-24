"""The ``bloqade.lanes.arch`` dialect: architecture-level location primitives.

Statements for constructing and inspecting ``LocationAddress`` values —
``loc`` (build one), ``cz_partner`` (resolve the CZ blockade partner), and the
``word_id`` / ``site_id`` / ``zone_id`` attribute reads — plus the const-prop,
rewrites, and interpreter impls around them. These are machine-agnostic
(``LocationAddress`` + ``ArchSpec`` only) and independent of the movement
dialect, which consumes ``LocationAddress`` values these produce.
"""

from . import (
    constprop as constprop,
    impl as impl,
    rewrite as rewrite,
    stmts as stmts,
)
from ._dialect import dialect as dialect
from ._interface import cz_partner as cz_partner, loc as loc
from .rewrite import (
    BindArchSpec as BindArchSpec,
    RewriteLocationAttr as RewriteLocationAttr,
)
from .stmts import (
    ArchResolvedStmt as ArchResolvedStmt,
    CzPartner as CzPartner,
    Loc as Loc,
    LocationAddressType as LocationAddressType,
    SiteId as SiteId,
    WordId as WordId,
    ZoneId as ZoneId,
)
