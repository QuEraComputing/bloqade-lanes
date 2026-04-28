"""TRANSITIONAL SHIM for the move to ``bloqade.lanes.arch.geometry`` (issue
#569, stage 5). Removed in the final cleanup stage once all in-flight
branches have rebased onto the canonical path.
"""

from bloqade.lanes.arch.geometry import (
    ArchSpecGeometry as ArchSpecGeometry,
    BusDescriptor as BusDescriptor,
)
