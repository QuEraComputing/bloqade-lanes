"""Tests for the architecture builder."""

import pytest

from bloqade.lanes.arch.builder import ArchResult, build_arch
from bloqade.lanes.arch.topology import (
    HypercubeSiteTopology,
    HypercubeWordTopology,
    MatchingTopology,
)
from bloqade.lanes.arch.zone import ArchBlueprint, DeviceLayout, ZoneSpec


def _two_zone_blueprint(
    entangling_word_topo: bool = True,
    entangling_site_topo: bool = True,
) -> ArchBlueprint:
    """Helper: 2-zone blueprint (proc + mem), 2x2 grid, 4 sites per word."""
    return ArchBlueprint(
        zones={
            "proc": ZoneSpec(
                num_rows=2, num_cols=2, entangling=True,
                word_topology=HypercubeWordTopology() if entangling_word_topo else None,
                site_topology=HypercubeSiteTopology() if entangling_site_topo else None,
            ),
            "mem": ZoneSpec(num_rows=2, num_cols=2),
        },
        layout=DeviceLayout(sites_per_word=4, site_spacing=10.0),
    )


class TestBuildArchSingleZone:
    def test_minimal_no_topology(self) -> None:
        bp = ArchBlueprint(
            zones={"proc": ZoneSpec(num_rows=1, num_cols=2, entangling=True)},
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        assert isinstance(result, ArchResult)
        assert len(result.arch.words) == 2
        assert "proc" in result.zone_grids
        assert result.zone_indices == {"proc": 1}

    def test_with_word_topology(self) -> None:
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=2, num_cols=2, entangling=True,
                    word_topology=HypercubeWordTopology(),
                ),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        # 2x2 grid → 1 row dim + 1 col dim = 2 word buses
        assert len(result.arch.word_buses) == 2

    def test_with_site_topology(self) -> None:
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=1, num_cols=2, entangling=True,
                    site_topology=HypercubeSiteTopology(),
                ),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        # 4 sites → log2(4) = 2 site buses
        assert len(result.arch.site_buses) == 2
        assert result.arch.has_site_buses == frozenset({0, 1})


class TestBuildArchTwoZones:
    def test_two_zones_no_connection(self) -> None:
        bp = _two_zone_blueprint()
        result = build_arch(bp)
        assert len(result.arch.words) == 8  # 4 + 4
        assert result.zone_indices == {"proc": 1, "mem": 2}
        # Only proc has word topology → 2 word buses
        assert len(result.arch.word_buses) == 2

    def test_two_zones_with_matching(self) -> None:
        bp = _two_zone_blueprint()
        result = build_arch(bp, connections={
            ("proc", "mem"): MatchingTopology(),
        })
        # 2 intra-zone (proc hypercube) + 1 inter-zone (matching) = 3
        assert len(result.arch.word_buses) == 3

    def test_zone_grids_correct(self) -> None:
        bp = _two_zone_blueprint()
        result = build_arch(bp)
        proc_grid = result.zone_grids["proc"]
        mem_grid = result.zone_grids["mem"]
        assert proc_grid.word_id_offset == 0
        assert mem_grid.word_id_offset == 4
        assert proc_grid.num_rows == 2
        assert mem_grid.num_rows == 2

    def test_entangling_zones(self) -> None:
        bp = _two_zone_blueprint()
        result = build_arch(bp)
        # Only proc (zone index 1) is entangling
        assert result.arch.entangling_zones == frozenset({1})

    def test_measurement_zones(self) -> None:
        bp = _two_zone_blueprint()
        result = build_arch(bp)
        # Both zones have measurement=True (default)
        # measurement_mode_zones = (0, 1, 2) — zone 0 (all) + proc + mem
        assert result.arch.measurement_mode_zones == (0, 1, 2)

    def test_has_word_buses_all_sites(self) -> None:
        bp = _two_zone_blueprint()
        result = build_arch(bp)
        assert result.arch.has_word_buses == frozenset({0, 1, 2, 3})

    def test_has_site_buses_union(self) -> None:
        bp = _two_zone_blueprint(entangling_site_topo=True)
        result = build_arch(bp)
        # Only proc has site_topology → only proc word IDs
        assert result.arch.has_site_buses == frozenset({0, 1, 2, 3})


class TestBuildArchThreeZones:
    def test_three_zones(self) -> None:
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=2, num_cols=2, entangling=True,
                    word_topology=HypercubeWordTopology(),
                ),
                "buffer": ZoneSpec(num_rows=2, num_cols=2),
                "mem": ZoneSpec(num_rows=2, num_cols=2),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp, connections={
            ("proc", "buffer"): MatchingTopology(),
            ("buffer", "mem"): MatchingTopology(),
        })
        assert len(result.arch.words) == 12
        assert result.zone_indices == {"proc": 1, "buffer": 2, "mem": 3}
        # 2 intra (proc) + 2 inter = 4 word buses
        assert len(result.arch.word_buses) == 4


class TestBuildArchValidation:
    def test_unknown_zone_in_connection_raises(self) -> None:
        bp = ArchBlueprint(
            zones={"proc": ZoneSpec(num_rows=1, num_cols=2)},
            layout=DeviceLayout(sites_per_word=4),
        )
        with pytest.raises(ValueError, match="Unknown zone 'mem'"):
            build_arch(bp, connections={("proc", "mem"): MatchingTopology()})

    def test_rust_validation_passes(self) -> None:
        """Ensure the generated ArchSpec passes Rust validation."""
        bp = _two_zone_blueprint()
        result = build_arch(bp, connections={
            ("proc", "mem"): MatchingTopology(),
        })
        # from_components calls _inner.validate() — no exception = pass
        assert result.arch is not None
