"""bloqade.lanes — machine-agnostic movement compilation.

The top level is intentionally free of eager re-exports. ``bloqade`` layering
runs ``bloqade.gemini`` on top of ``bloqade.lanes``, and some lanes submodules
import ``bloqade.gemini`` at module load (e.g. ``rewrite.circuit2place`` ->
``bloqade.gemini.logical``), so an eager re-export here can re-enter an
in-progress ``bloqade.gemini`` import and break initialization.

Import concrete symbols from their submodules instead, e.g.::

    from bloqade.lanes.transform import LogicalPipeline
    from bloqade.lanes.metrics import Metrics
    from bloqade.lanes.noise_model import (
        generate_logical_noise_model,
        generate_simple_noise_model,
    )
    from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC

Gemini-machine specifics (device classes, Steane defaults, compile entry points)
live in ``bloqade.gemini``.
"""
