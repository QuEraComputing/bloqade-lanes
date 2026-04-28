"""TRANSITIONAL SHIM for the move to ``bloqade.lanes.arch.build.blueprint``
(issue #569, stage 8). Removed in the final cleanup stage once all
in-flight branches have rebased onto the canonical path.
"""

from bloqade.lanes.arch.build.blueprint import (
    ArchResult as ArchResult,
    build_arch as build_arch,
)
