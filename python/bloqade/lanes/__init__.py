"""Bloqade Lanes — neutral-atom movement compilation.

This package's top level is intentionally kept free of eager re-exports. Lanes
submodules (pipelines, rewrites, dialects) import ``bloqade.gemini`` at module
load, while ``bloqade.gemini`` imports ``bloqade.lanes`` — so any eager
re-export added here re-enters an in-progress ``bloqade.gemini`` import and
breaks initialization. Import concrete symbols from their submodules instead,
e.g.::

    from bloqade.lanes.metrics import Metrics
    from bloqade.lanes.steane_defaults import steane7_m2dets, steane7_m2obs
    from bloqade.lanes.noise_model import generate_logical_noise_model
"""
