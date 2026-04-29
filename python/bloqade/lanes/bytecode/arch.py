"""Backward-compatible re-exports of ``grid_to_rust`` and ``grid_from_rust``.

The file was renamed from ``bytecode/arch.py`` to ``bytecode/grid.py`` to
reflect its responsibility and free the ``arch`` name for the higher-level
architecture surface. The canonical import path is ``bloqade.lanes.bytecode.grid``.
"""

from bloqade.lanes.bytecode.grid import (
    grid_from_rust as grid_from_rust,
    grid_to_rust as grid_to_rust,
)
