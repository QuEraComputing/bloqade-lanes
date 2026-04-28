"""TRANSITIONAL SHIM — see ``.superpowers/plans/2026-04-27-archspec-package-merge.md``
(Stage 5) for the rationale. Removed in the final cleanup stage once all
in-flight branches have rebased onto ``bloqade.lanes.arch.geometry``.
"""

from bloqade.lanes.arch.geometry import (
    ArchSpecGeometry as ArchSpecGeometry,
    BusDescriptor as BusDescriptor,
)
