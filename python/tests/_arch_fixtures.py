"""Architecture specs for tests, loaded from checked-in JSON snapshots.

Lanes compiles against an :class:`ArchSpec` and does not build one: the
architecture builder lives in ``bloqade.gemini.arch``, in the ``bloqade``
distribution, which lanes cannot depend on. Tests that need a small synthetic
architecture therefore load one of the snapshots in ``fixtures/arch/``.

The snapshots are plain test data. Each one is validated when it is loaded
(``ArchSpec.from_json_validated``). To add one, write the JSON by hand or build
it with any tool, then commit it here. Single-zone entangling snapshots are
named by :func:`gate_zone_name`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode._native import ArchSpec as _RustArchSpec

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "arch"

WordTopologyName = Literal["hypercube", "diagonal"]
SiteTopologyName = Literal["hypercube", "all_to_all"]


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
            f"no architecture snapshot {path.name!r} in {FIXTURE_DIR}; add the "
            "ArchSpec JSON there"
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
