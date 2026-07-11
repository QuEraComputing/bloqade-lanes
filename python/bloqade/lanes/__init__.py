"""Bloqade Lanes — neutral-atom movement compilation.

This package's top level is intentionally free of eager re-exports. ``bloqade``
layering runs ``bloqade.gemini`` on top of ``bloqade.lanes``, and lanes
submodules (pipelines, rewrites, dialects) import ``bloqade.gemini`` at module
load (e.g. ``rewrite.circuit2place`` -> ``bloqade.gemini.logical``). Any eager
re-export here re-enters an in-progress ``bloqade.gemini`` import and breaks
initialization — most visibly once a dialect imports ``LocationAddress`` from
``bloqade.lanes.bytecode.encoding``.

Import concrete symbols from their submodules instead, e.g.::

    from bloqade.lanes.metrics import Metrics
    from bloqade.lanes.noise_model import (
        generate_logical_noise_model,
        generate_simple_noise_model,
    )
    from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC
    from bloqade.lanes.steane_defaults import steane7_m2dets, steane7_m2obs

Device classes (``Result``, ``GeminiLogicalSimulator``, ...) live in
``bloqade.gemini``.
"""
