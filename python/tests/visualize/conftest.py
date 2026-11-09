"""Shared fixtures for the visualization tests.

Plotly is optional -- it arrives with the ``visualization`` extra, and a plain
``uv sync`` leaves it out. Without this, ``just test-python`` showed 40 red
tests in this directory, every one a ``ModuleNotFoundError: No module named
'plotly'``, which reads as a regression rather than as a missing extra.

A module-level ``pytest.importorskip`` -- the shape used in
``heuristics/test_physical_initial_layout.py`` for pymetis -- would be wrong
here. These files are mixed: 25 of their tests need no Plotly at all and cover
matplotlib bounds, circuit-column layout, argument validation and renderer
selection. Skipping a whole file to spare 40 tests would stop running those 25
for everyone without the extra, which is how coverage quietly disappears. So
the skip is per test, via the ``plotly`` fixture.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def plotly():
    """Skip the requesting test unless the ``visualization`` extra is installed.

    Request this from any test that builds a figure. Tests that only exercise
    pure-Python helpers should not, so they keep running without the extra.
    """
    return pytest.importorskip("plotly")
