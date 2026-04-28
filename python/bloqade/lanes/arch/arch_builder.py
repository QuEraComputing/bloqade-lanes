"""TRANSITIONAL SHIM — see ``.superpowers/plans/2026-04-27-archspec-package-merge.md``
(Stage 8) for the rationale. Removed in the final cleanup stage once all
in-flight branches have rebased onto ``bloqade.lanes.arch.build.imperative``.
"""

from bloqade.lanes.arch.build.imperative import (  # noqa: F401
    ArchBuilder as ArchBuilder,
    ZoneBuilder as ZoneBuilder,
    _normalize_index as _normalize_index,
    _SiteGridQuery as _SiteGridQuery,
    _to_nm as _to_nm,
    _validate_aod_rectangle as _validate_aod_rectangle,
    _WordGridQuery as _WordGridQuery,
)
