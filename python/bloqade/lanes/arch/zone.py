"""TRANSITIONAL SHIM — see ``.superpowers/plans/2026-04-27-archspec-package-merge.md``
(Stage 8) for the rationale. Removed in the final cleanup stage once all
in-flight branches have rebased onto ``bloqade.lanes.arch.build.blueprint``.
"""

from bloqade.lanes.arch.build.blueprint import (
    ArchBlueprint as ArchBlueprint,
    DeviceLayout as DeviceLayout,
    ZoneSpec as ZoneSpec,
)
