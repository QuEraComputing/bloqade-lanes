"""Tests for zone-addressed ArchSpec APIs (#419/#420)."""

import pytest
from bloqade.geometry.dialects.grid import Grid as GeoGrid

from bloqade.lanes.arch import (
    ArchBlueprint,
    DeviceLayout,
    HypercubeSiteTopology,
    HypercubeWordTopology,
    MatchingTopology,
    ZoneSpec,
    build_arch,
)
from bloqade.lanes.layout import BusDescriptor, Direction, MoveType


def _single_zone_arch():
    bp = ArchBlueprint(
        zones={
            "gate": ZoneSpec(
                num_rows=2,
                num_cols=2,
                entangling=True,
                word_topology=HypercubeWordTopology(),
                site_topology=HypercubeSiteTopology(),
            ),
        },
        layout=DeviceLayout(sites_per_word=4),
    )
    return build_arch(bp).arch


def _two_zone_arch():
    bp = ArchBlueprint(
        zones={
            "proc": ZoneSpec(
                num_rows=2,
                num_cols=2,
                entangling=True,
                word_topology=HypercubeWordTopology(),
                site_topology=HypercubeSiteTopology(),
            ),
            "mem": ZoneSpec(num_rows=2, num_cols=2),
        },
        layout=DeviceLayout(sites_per_word=4),
    )
    return build_arch(bp, connections={("proc", "mem"): MatchingTopology()}).arch


# ── get_zone_grid ──


class TestGetZoneGrid:
    def test_returns_geo_grid(self):
        arch = _single_zone_arch()
        grid = arch.get_zone_grid(0)
        assert isinstance(grid, GeoGrid)

    def test_grid_has_correct_shape(self):
        arch = _single_zone_arch()
        grid = arch.get_zone_grid(0)
        # 2 cols × 4 sites/word interleaved → multiple x positions
        # 2 rows → multiple y positions
        assert grid.shape[0] > 0
        assert grid.shape[1] > 0

    def test_invalid_zone_id_raises(self):
        arch = _single_zone_arch()
        with pytest.raises(IndexError):
            arch.get_zone_grid(99)

    def test_negative_zone_id_raises(self):
        arch = _single_zone_arch()
        with pytest.raises(IndexError):
            arch.get_zone_grid(-1)

    def test_two_zone_different_grids(self):
        arch = _two_zone_arch()
        grid0 = arch.get_zone_grid(0)
        grid1 = arch.get_zone_grid(1)
        # Both zones have the same grid dimensions (same ZoneSpec shape)
        assert grid0.shape == grid1.shape


# ── get_all_sites ──


class TestGetAllSites:
    def test_returns_list_of_tuples(self):
        arch = _single_zone_arch()
        sites = arch.get_all_sites()
        assert isinstance(sites, list)
        assert all(isinstance(s, tuple) and len(s) == 2 for s in sites)

    def test_single_zone_site_count(self):
        arch = _single_zone_arch()
        sites = arch.get_all_sites()
        # 4 words × 4 sites = 16 positions (single zone)
        assert len(sites) == 16

    def test_positions_are_unique(self):
        arch = _single_zone_arch()
        sites = arch.get_all_sites()
        assert len(sites) == len(set(sites))

    def test_two_zone_includes_both(self):
        arch = _two_zone_arch()
        sites = arch.get_all_sites()
        # Each zone resolves all 8 words × 4 sites = 32, 2 zones = 64
        assert len(sites) == 64


# ── get_available_buses ──


class TestGetAvailableBuses:
    def test_returns_bus_descriptors(self):
        arch = _single_zone_arch()
        buses = arch.get_available_buses(0)
        assert isinstance(buses, list)
        assert all(isinstance(b, BusDescriptor) for b in buses)

    def test_site_and_word_buses_present(self):
        arch = _single_zone_arch()
        buses = arch.get_available_buses(0)
        move_types = {b.move_type for b in buses}
        assert MoveType.SITE in move_types
        assert MoveType.WORD in move_types

    def test_forward_and_backward(self):
        arch = _single_zone_arch()
        buses = arch.get_available_buses(0)
        directions = {b.direction for b in buses}
        assert Direction.FORWARD in directions
        assert Direction.BACKWARD in directions

    def test_num_lanes_positive(self):
        arch = _single_zone_arch()
        buses = arch.get_available_buses(0)
        assert all(b.num_lanes > 0 for b in buses)

    def test_invalid_zone_raises(self):
        arch = _single_zone_arch()
        with pytest.raises(IndexError):
            arch.get_available_buses(99)

    def test_zone_without_buses(self):
        arch = _two_zone_arch()
        # mem zone has no site or word topology
        buses = arch.get_available_buses(1)
        assert len(buses) == 0


# ── get_grid_endpoints ──


class TestGetGridEndpoints:
    def test_returns_two_grids(self):
        arch = _single_zone_arch()
        src_grid, dst_grid = arch.get_grid_endpoints(
            0, 0, MoveType.SITE, Direction.FORWARD
        )
        assert isinstance(src_grid, GeoGrid)
        assert isinstance(dst_grid, GeoGrid)

    def test_src_dst_different(self):
        arch = _single_zone_arch()
        src_grid, dst_grid = arch.get_grid_endpoints(
            0, 0, MoveType.SITE, Direction.FORWARD
        )
        assert src_grid != dst_grid

    def test_word_bus_endpoints(self):
        arch = _single_zone_arch()
        src_grid, dst_grid = arch.get_grid_endpoints(
            0, 0, MoveType.WORD, Direction.FORWARD
        )
        assert isinstance(src_grid, GeoGrid)
        assert isinstance(dst_grid, GeoGrid)

    def test_invalid_zone_raises(self):
        arch = _single_zone_arch()
        with pytest.raises(IndexError):
            arch.get_grid_endpoints(99, 0, MoveType.SITE, Direction.FORWARD)

    def test_invalid_bus_id_raises(self):
        arch = _single_zone_arch()
        with pytest.raises(IndexError):
            arch.get_grid_endpoints(0, 99, MoveType.SITE, Direction.FORWARD)

    def test_backward_swaps_endpoints(self):
        arch = _single_zone_arch()
        fwd_src, fwd_dst = arch.get_grid_endpoints(
            0, 0, MoveType.SITE, Direction.FORWARD
        )
        bwd_src, bwd_dst = arch.get_grid_endpoints(
            0, 0, MoveType.SITE, Direction.BACKWARD
        )
        # Backward swaps src and dst
        assert fwd_src == bwd_dst
        assert fwd_dst == bwd_src
