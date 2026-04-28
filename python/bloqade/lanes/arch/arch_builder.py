"""TRANSITIONAL SHIM for the move to ``bloqade.lanes.arch.build.imperative``
(issue #569, stage 8). Removed in the final cleanup stage once all
in-flight branches have rebased onto the canonical path.
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
