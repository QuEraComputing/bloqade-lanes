"""Architecture specs for tests, loaded from checked-in JSON snapshots.

Lanes compiles against an :class:`ArchSpec` and does not build one: the
architecture builder lives in ``bloqade.gemini.arch``, in the ``bloqade``
distribution, which lanes cannot depend on. Tests that need a small synthetic
architecture therefore load one of the snapshots in ``fixtures/arch/``.

Each snapshot is produced by ``fixtures/arch/generate.py``, which records how it
was built. To add one, add a recipe there (or a parameter set to
:data:`GATE_ZONE_FIXTURES`) and run that script with ``bloqade`` installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode._native import ArchSpec as _RustArchSpec

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "arch"

WordTopologyName = Literal["hypercube", "diagonal"]
SiteTopologyName = Literal["hypercube", "all_to_all"]

GateZoneParameters = tuple[int, int, int, WordTopologyName, SiteTopologyName]
"""``(num_rows, num_cols, sites_per_word, word_topology, site_topology)``."""

GATE_ZONE_FIXTURES: tuple[GateZoneParameters, ...] = (
    (1, 4, 1, "hypercube", "all_to_all"),
    (1, 8, 1, "hypercube", "all_to_all"),
    (1, 4, 3, "hypercube", "all_to_all"),
    (1, 4, 5, "hypercube", "all_to_all"),
    (1, 4, 7, "hypercube", "all_to_all"),
    (1, 4, 10, "hypercube", "all_to_all"),
    (1, 8, 10, "hypercube", "all_to_all"),
    (2, 2, 4, "hypercube", "hypercube"),
    (4, 2, 2, "hypercube", "hypercube"),
    (4, 2, 4, "hypercube", "hypercube"),
    (1, 2, 4, "diagonal", "hypercube"),
    (4, 2, 4, "diagonal", "hypercube"),
    (1, 2, 16, "diagonal", "hypercube"),
    (4, 2, 16, "diagonal", "hypercube"),
    (5, 2, 16, "diagonal", "hypercube"),
    (8, 2, 16, "diagonal", "hypercube"),
)
"""Every single-zone entangling architecture available to :func:`gate_zone_arch`."""


def gate_zone_name(
    num_rows: int,
    num_cols: int,
    sites_per_word: int,
    word_topology: WordTopologyName,
    site_topology: SiteTopologyName,
) -> str:
    """Snapshot name of a single-zone entangling architecture."""
    return (
        f"gate_{num_rows}x{num_cols}_s{sites_per_word}_"
        f"{word_topology}_{site_topology}"
    )


def load_arch(name: str) -> ArchSpec:
    """Load the architecture snapshot ``fixtures/arch/<name>.json``."""
    path = FIXTURE_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no architecture snapshot {path.name!r} in {FIXTURE_DIR}; add a "
            "recipe to fixtures/arch/generate.py and run it with bloqade installed"
        )
    return ArchSpec(_RustArchSpec.from_json_validated(path.read_text(encoding="utf-8")))


def gate_zone_arch(
    *,
    num_rows: int,
    num_cols: int,
    sites_per_word: int,
    word_topology: WordTopologyName,
    site_topology: SiteTopologyName,
) -> ArchSpec:
    """A single entangling zone named ``"gate"``, with the default device layout.

    The words form a ``num_rows`` x ``num_cols`` grid. Words ``0, 2, 4, ...``
    are home, and word ``2k`` is the CZ partner of word ``2k + 1``.
    """
    return load_arch(
        gate_zone_name(num_rows, num_cols, sites_per_word, word_topology, site_topology)
    )
