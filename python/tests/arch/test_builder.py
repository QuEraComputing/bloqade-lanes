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
                num_rows=2,
                num_cols=2,
                entangling=True,
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
        assert result.zone_indices == {"proc": 0}

    def test_with_word_topology(self) -> None:
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=2,
                    num_cols=2,
                    entangling=True,
                    word_topology=HypercubeWordTopology(),
                ),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        # 2x2 grid → 1 row dim + 1 col dim = 2 word buses
        assert len(result.arch.word_buses) == 4  # 2 per zone * 2 zones (entangling split)

    def test_with_site_topology(self) -> None:
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=1,
                    num_cols=2,
                    entangling=True,
                    site_topology=HypercubeSiteTopology(),
                ),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        # 4 sites → log2(4) = 2 site buses
        assert len(result.arch.site_buses) == 4  # 2 per zone * 2 zones (entangling split)
        assert result.arch.has_site_buses == frozenset({0, 1})


class TestBuildArchTwoZones:
    def test_two_zones_no_connection(self) -> None:
        bp = _two_zone_blueprint()
        result = build_arch(bp)
        assert len(result.arch.words) == 8  # 4 + 4
        assert result.zone_indices == {"proc": 0, "mem": 2}
        # Only proc has word topology → 2 word buses
        assert len(result.arch.word_buses) == 4  # 2 per zone * 2 zones (entangling split)

    def test_two_zones_with_matching(self) -> None:
        bp = _two_zone_blueprint()
        result = build_arch(
            bp,
            connections={
                ("proc", "mem"): MatchingTopology(),
            },
        )
        # Entangling proc zone splits into 2 sub-zones, each gets:
        #   2 hypercube word buses + 1 matching bus = 3 per sub-zone = 6
        # Plus mem zone gets 1 matching bus = 7 total
        assert len(result.arch.word_buses) == 7

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
        # Proc zone splits into 2 sub-zones: zone 0 (even) and zone 1 (odd)
        assert len(result.arch.entangling_zone_pairs) == 1
        assert result.arch.entangling_zone_pairs[0] == (0, 1)

    def test_measurement_zones(self) -> None:
        bp = _two_zone_blueprint()
        result = build_arch(bp)
        # Both blueprint zones have measurement=True (default).
        # Proc splits into sub-zones 0 and 1; mem is zone 2.
        # Modes: "all" (zones [0,1,2]), "proc" (zones [0,1]), "mem" (zones [2])
        modes = result.arch.modes
        assert len(modes) == 3
        assert modes[0].name == "all"
        assert modes[0].zones == [0, 1, 2]
        assert modes[1].name == "proc"
        assert modes[1].zones == [0, 1]
        assert modes[2].name == "mem"
        assert modes[2].zones == [2]

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
                    num_rows=2,
                    num_cols=2,
                    entangling=True,
                    word_topology=HypercubeWordTopology(),
                ),
                "buffer": ZoneSpec(num_rows=2, num_cols=2),
                "mem": ZoneSpec(num_rows=2, num_cols=2),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(
            bp,
            connections={
                ("proc", "buffer"): MatchingTopology(),
                ("buffer", "mem"): MatchingTopology(),
            },
        )
        assert len(result.arch.words) == 12
        # Proc is entangling → splits into zones 0 (even) and 1 (odd)
        # buffer → zone 2, mem → zone 3
        assert result.zone_indices == {"proc": 0, "buffer": 2, "mem": 3}
        # Each proc sub-zone: 2 hypercube + 1 matching = 3 each = 6
        # buffer: 2 matching (proc + mem) = 2
        # mem: 1 matching = 1  → 9 total
        assert len(result.arch.word_buses) == 9


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
        result = build_arch(
            bp,
            connections={
                ("proc", "mem"): MatchingTopology(),
            },
        )
        # from_components calls _inner.validate() — no exception = pass
        assert result.arch is not None

    def test_self_connection_raises(self) -> None:
        bp = _two_zone_blueprint()
        with pytest.raises(ValueError, match="Self-connection not allowed"):
            build_arch(bp, connections={("proc", "proc"): MatchingTopology()})

    def test_no_site_topology_anywhere(self) -> None:
        """All zones without site_topology → zero site buses, still valid."""
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(num_rows=1, num_cols=2, entangling=True),
                "mem": ZoneSpec(num_rows=1, num_cols=2),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        assert len(result.arch.site_buses) == 0
        assert result.arch.has_site_buses == frozenset()

    def test_non_measurement_zone(self) -> None:
        """Zone with measurement=False is excluded from per-zone modes."""
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(num_rows=1, num_cols=2, entangling=True),
                "buffer": ZoneSpec(num_rows=1, num_cols=2, measurement=False),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        # Modes: "all" (zones [0,1,2]) + "proc" (zones [0,1]), buffer excluded
        modes = result.arch.modes
        assert len(modes) == 2
        assert modes[0].name == "all"
        assert modes[1].name == "proc"


class TestBuildArchPerBusWords:
    def test_mixed_site_topologies(self) -> None:
        """Two zones with different site topologies get separate buses."""
        from bloqade.lanes.arch.topology import AllToAllSiteTopology

        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=1,
                    num_cols=2,
                    entangling=True,
                    site_topology=HypercubeSiteTopology(),
                ),
                "mem": ZoneSpec(
                    num_rows=1,
                    num_cols=2,
                    site_topology=AllToAllSiteTopology(),
                ),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        arch = result.arch

        # Hypercube(4 sites) = 2 buses per sub-zone × 2 sub-zones = 4
        # AllToAll(4 sites) = 6 buses for mem → 10 total
        assert len(arch.site_buses) == 10

        # Proc zone 0 has word 0, zone 1 has word 1
        assert arch.zones[0].words_with_site_buses == [0]
        assert arch.zones[1].words_with_site_buses == [1]
        # Mem zone 2 has words [2, 3]
        assert arch.zones[2].words_with_site_buses == [2, 3]

        # has_site_buses = union of all
        assert arch.has_site_buses == frozenset({0, 1, 2, 3})

    def test_single_zone_site_buses_have_words(self) -> None:
        """Even single-zone site buses get words set via zone."""
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=1,
                    num_cols=2,
                    entangling=True,
                    site_topology=HypercubeSiteTopology(),
                ),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        # Entangling splits into 2 sub-zones: zone 0 has word 0, zone 1 has word 1
        assert result.arch.zones[0].words_with_site_buses == [0]
        assert result.arch.zones[1].words_with_site_buses == [1]

    def test_zone_without_site_topology_excluded(self) -> None:
        """Zones without site_topology don't contribute to site buses."""
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=1,
                    num_cols=2,
                    entangling=True,
                    site_topology=HypercubeSiteTopology(),
                ),
                "mem": ZoneSpec(num_rows=1, num_cols=2),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        # Only proc buses, mem has no site topology
        # Proc splits into 2 sub-zones with 2 site buses each = 4
        assert len(result.arch.site_buses) == 4
        # Zone 0 has word 0, zone 1 has word 1
        assert result.arch.zones[0].words_with_site_buses == [0]
        assert result.arch.zones[1].words_with_site_buses == [1]
        # has_site_buses = only proc words
        assert result.arch.has_site_buses == frozenset({0, 1})


class TestPathFinderIntegration:
    def test_per_zone_site_bus_scoping(self) -> None:
        """PathFinder graph only connects site bus edges for the bus's words."""
        from bloqade.lanes.layout.encoding import LocationAddress
        from bloqade.lanes.layout.path import PathFinder

        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=1,
                    num_cols=2,
                    entangling=True,
                    site_topology=HypercubeSiteTopology(),
                ),
                "mem": ZoneSpec(num_rows=1, num_cols=2),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        # No connections — zones are isolated
        result = build_arch(bp)
        pf = PathFinder(result.arch)

        # proc word 0, site 0 → site 1: reachable (proc has site buses)
        assert pf.find_path(LocationAddress(0, 0, 0), LocationAddress(0, 0, 1)) is not None

        # mem word 2, site 0 → site 1: NOT reachable (no site buses, no connections)
        assert pf.find_path(LocationAddress(0, 2, 0), LocationAddress(0, 2, 1)) is None

    def test_cross_zone_reachable_via_word_bus(self) -> None:
        """PathFinder can route across zones via inter-zone word buses."""
        from bloqade.lanes.layout.encoding import LocationAddress
        from bloqade.lanes.layout.path import PathFinder

        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=1,
                    num_cols=2,
                    entangling=True,
                    site_topology=HypercubeSiteTopology(),
                ),
                "mem": ZoneSpec(num_rows=1, num_cols=2),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(
            bp,
            connections={
                ("proc", "mem"): MatchingTopology(),
            },
        )
        pf = PathFinder(result.arch)

        # proc word 0 → mem word 2 (same site): reachable via matching word bus
        assert pf.find_path(LocationAddress(0, 0, 0), LocationAddress(0, 2, 0)) is not None
