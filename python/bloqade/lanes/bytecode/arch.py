"""TRANSITIONAL SHIM — see ``.superpowers/plans/2026-04-27-archspec-package-merge.md``
(Stage 4) for the rationale. The original ``bytecode/arch.py`` only held
two ``Grid`` conversion utilities; the file was renamed to ``grid.py``
(matching its responsibility) to free the ``arch`` slot for the higher-tier
arch surface.

Removed in the final cleanup stage.
"""

from bloqade.lanes.bytecode.grid import (
    grid_from_rust as grid_from_rust,
    grid_to_rust as grid_to_rust,
)
