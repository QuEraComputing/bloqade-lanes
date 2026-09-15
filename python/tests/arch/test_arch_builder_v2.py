"""Tests for the spec-shaped ArchBuilder in ``arch.build.v2``."""

import warnings

import pytest

from bloqade.lanes.arch.build.v2 import ArchBuilder
from bloqade.lanes.arch.gemini.logical.spec import get_arch_spec as logical_spec
from bloqade.lanes.arch.gemini.physical.spec import get_arch_spec as physical_spec

_CL = 0.25


def interleaved(rows: int = 2, cols: int = 4) -> ArchBuilder:
    """A template whose word IDs are not monotone in the grid.

    Mirrors the shipped physical architecture: each word's sites are
    interleaved with its neighbours' along x, so scanning x visits words
    0, 1, ..., cols-1, 0, 1, ... and a partial x-selection reaches them out
    of ID order.
    """
    b = ArchBuilder(grid_shape=(cols * 2, rows), word_shape=(2, 1))
    for row in range(rows):
        for col in range(cols):
            b.add_word(x=[col, col + cols], y=[row])
    return b


def with_zone(b: ArchBuilder, name: str = "z", **kwargs) -> ArchBuilder:
    """Attach a unit-spaced zone covering the builder's index space."""
    nx, ny = b.grid_shape
    b.add_zone(
        name,
        x=[float(i) for i in range(nx)],
        y=[10.0 * j for j in range(ny)],
        x_clearance=_CL,
        y_clearance=3.0,
        **kwargs,
    )
    return b


# ── The word template is spec-wide ──


class TestTemplate:
    def test_words_are_frozen_by_the_first_zone(self):
        b = with_zone(interleaved())
        with pytest.raises(ValueError, match="word template is frozen"):
            b.add_word(x=[0, 1], y=[0])

    def test_zone_needs_a_template_first(self):
        b = ArchBuilder(grid_shape=(4, 1), word_shape=(2, 1))
        with pytest.raises(ValueError, match="add_word before adding a zone"):
            with_zone(b)

    def test_overlapping_words_are_rejected(self):
        b = ArchBuilder(grid_shape=(4, 1), word_shape=(2, 1))
        b.add_word(x=[0, 1], y=[0])
        with pytest.raises(ValueError, match="already belongs to word 0"):
            b.add_word(x=[1, 2], y=[0])

    def test_word_shape_is_enforced(self):
        b = ArchBuilder(grid_shape=(4, 1), word_shape=(2, 1))
        with pytest.raises(ValueError, match="word_shape requires 2"):
            b.add_word(x=[0], y=[0])

    def test_every_zone_indexes_the_same_space(self):
        b = interleaved()
        with pytest.raises(ValueError, match="grid_shape is 8x2"):
            b.add_zone("z", x=[0.0, 1.0], y=[0.0], x_clearance=_CL, y_clearance=_CL)


class TestTemplateQueries:
    def test_partial_selection_follows_the_grid_not_word_ids(self):
        # x index 3 is a site of word 3; x index 4 is a site of word 0.
        assert interleaved().words[[3, 4], [0, 1]] == [3, 0, 7, 4]

    def test_axis_order_is_the_callers(self):
        b = interleaved()
        assert b.words[[4, 3], [0, 1]] == [0, 3, 4, 7]
        assert b.words[[3, 4], [1, 0]] == [7, 4, 3, 0]

    def test_rows_are_the_outer_loop(self):
        assert interleaved().words[:, :] == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_word_reached_twice_is_appended_once(self):
        # Both x=0 and x=4 are sites of word 0.
        assert interleaved().words[[0, 4], 0] == [0]

    def test_out_of_range_raises(self):
        with pytest.raises(IndexError, match=r"x grid index \[99\]"):
            interleaved().words[[0, 99], 0]

    def test_negative_raises(self):
        with pytest.raises(IndexError, match=r"x grid index \[-1\]"):
            interleaved().words[[-1], 0]

    def test_repeated_index_raises(self):
        with pytest.raises(ValueError, match=r"x grid index \[0\] repeated"):
            interleaved().words[[0, 0, 1], 0]

    def test_slices_are_clamped_not_rejected(self):
        assert interleaved().words[0:99, 0] == [0, 1, 2, 3]

    def test_reverse_slice_expands_descending(self):
        assert interleaved().words[::-1, 0] == [3, 2, 1, 0]

    def test_site_query_validates_too(self):
        with pytest.raises(IndexError, match=r"x site index \[9\]"):
            interleaved().sites[[9], 0]


# ── Bus realizability ──


class TestBusRealizability:
    """A bus is realizable when it is separable and order-preserving.

    An AOD sweeps whole x- and y-tones independently, so an atom may not
    leave its own row/column and tones may not cross.  Displacement need
    *not* be uniform: compressing or expanding a rectangle is one AOD
    operation.
    """

    def _rows(self, cols: int = 4) -> ArchBuilder:
        b = ArchBuilder(grid_shape=(cols, 2), word_shape=(1, 1))
        for y in range(2):
            for x in range(cols):
                b.add_word(x=[x], y=[y])
        return with_zone(b)

    def test_uniform_translation_is_accepted(self):
        b = self._rows()
        b.add_word_bus("z", src=[0, 1, 2, 3], dst=[4, 5, 6, 7])

    def test_reversal_is_rejected(self):
        b = self._rows()
        with pytest.raises(ValueError, match="stay on its own"):
            b.add_word_bus("z", src=[0, 1, 2, 3], dst=[7, 6, 5, 4])

    def test_swapping_two_columns_is_rejected(self):
        b = self._rows()
        with pytest.raises(ValueError, match="stay on its own"):
            b.add_word_bus("z", src=[0, 1], dst=[5, 4])

    def test_shear_is_rejected(self):
        b = ArchBuilder(grid_shape=(2, 2), word_shape=(1, 1))
        for y in range(2):
            for x in range(2):
                b.add_word(x=[x], y=[y])
        with_zone(b)
        # row 0 holds still; row 1 swaps columns.
        with pytest.raises(ValueError, match="stay on its own"):
            b.add_word_bus("z", src=[0, 1, 2, 3], dst=[0, 1, 3, 2])

    def test_tone_count_must_match(self):
        b = self._rows(cols=3)
        with pytest.raises(ValueError, match="add or drop a tone"):
            b.add_word_bus("z", src=[0, 1], dst=[3, 3])

    def test_non_uniform_but_separable_is_accepted(self):
        """Compressing a rectangle gives each column its own displacement."""
        b = ArchBuilder(grid_shape=(4, 2), word_shape=(1, 1))
        for x in (0, 2, 3):  # x = 0, 10, 20
            b.add_word(x=[x], y=[0])
        for x in (0, 1, 2):  # x = 0, 5, 10
            b.add_word(x=[x], y=[1])
        b.add_zone(
            "z",
            x=[0.0, 5.0, 10.0, 20.0],
            y=[0.0, 30.0],
            x_clearance=_CL,
            y_clearance=3.0,
        )
        b.add_word_bus("z", src=[0, 1, 2], dst=[3, 4, 5])

    def test_empty_bus_is_rejected(self):
        b = self._rows()
        with pytest.raises(ValueError, match="at least one word"):
            b.add_word_bus("z", src=[], dst=[])
        with pytest.raises(ValueError, match="at least one site"):
            b.add_site_bus("z", src=[], dst=[])

    def test_mismatched_lengths_are_rejected(self):
        b = self._rows()
        with pytest.raises(ValueError, match="src has 2 entries but dst has 1"):
            b.add_word_bus("z", src=[0, 1], dst=[4])

    def test_out_of_range_word_is_rejected(self):
        b = self._rows()
        with pytest.raises(IndexError, match=r"out of range \[0, 8\)"):
            b.add_word_bus("z", src=[0], dst=[99])

    def test_negative_word_index_is_rejected(self):
        """A negative index would otherwise wrap to a different word."""
        b = self._rows()
        with pytest.raises(IndexError, match=r"\[-1\] out of range"):
            b.add_word_bus("z", src=[0], dst=[-1])


class TestStowaways:
    """The AOD traps at every intersection of its tones.

    An intersection occupied by an atom the bus does not carry would be
    picked up too; an intersection no atom occupies is harmless.
    """

    def _l_shape(self, **zone_kwargs) -> ArchBuilder:
        b = ArchBuilder(grid_shape=(4, 2), word_shape=(2, 1))
        b.add_word(x=[0, 1], y=[0])
        b.add_word(x=[2, 3], y=[0])
        b.add_word(x=[0, 1], y=[1])
        return with_zone(b, **zone_kwargs)

    def test_empty_intersection_is_allowed(self):
        """No word sits at the open crossing, so nothing rides along."""
        b = self._l_shape()
        b.add_site_bus("z", src=[0], dst=[1])

    def test_occupied_intersection_is_rejected(self):
        """A fourth word fills the crossing but sits out the bus."""
        b = ArchBuilder(grid_shape=(4, 2), word_shape=(2, 1))
        for y in range(2):
            for x0 in (0, 2):
                b.add_word(x=[x0, x0 + 1], y=[y])
        with_zone(b, words_with_site_buses=[0, 1, 2])  # word 3 opts out
        with pytest.raises(ValueError, match="would be carried along"):
            b.add_site_bus("z", src=[0], dst=[1])

    def test_including_the_intruder_makes_it_legal(self):
        b = ArchBuilder(grid_shape=(4, 2), word_shape=(2, 1))
        for y in range(2):
            for x0 in (0, 2):
                b.add_word(x=[x0, x0 + 1], y=[y])
        with_zone(b)  # every word participates
        b.add_site_bus("z", src=[0], dst=[1])


class TestParticipation:
    def test_subset_reaches_the_spec(self):
        b = with_zone(interleaved(), words_with_site_buses=[1, 3])
        b.add_site_bus("z", src=[0], dst=[1])
        b.add_mode("all", ["z"])
        spec = b.build()
        assert list(spec.zones[0].words_with_site_buses) == [1, 3]

    def test_sites_with_word_buses_is_not_forced_to_every_site(self):
        """The legacy builder could not express a proper subset here."""
        b = ArchBuilder(grid_shape=(4, 2), word_shape=(2, 1))
        for y in range(2):
            for x0 in (0, 2):
                b.add_word(x=[x0, x0 + 1], y=[y])
        with_zone(b, sites_with_word_buses=[0])
        b.add_word_bus("z", src=[0, 1], dst=[2, 3])
        b.add_mode("all", ["z"])
        assert list(b.build().zones[0].sites_with_word_buses) == [0]

    def test_out_of_range_participant_is_rejected(self):
        with pytest.raises(IndexError, match="words_with_site_buses"):
            with_zone(interleaved(), words_with_site_buses=[99])

    def test_repeated_participant_is_rejected(self):
        with pytest.raises(ValueError, match="sites_with_word_buses"):
            with_zone(interleaved(), sites_with_word_buses=[0, 0])


# ── Assembly ──


class TestBuild:
    def test_selected_endpoints_pair_into_a_routable_bus(self):
        b = interleaved()
        with_zone(b)
        src, dst = b.words[:, 0], b.words[:, 1]
        assert (src, dst) == ([0, 1, 2, 3], [4, 5, 6, 7])
        b.add_word_bus("z", src, dst)
        b.add_mode("all", ["z"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            spec = b.build()
        assert len(spec.paths) == 16

    def test_build_requires_words_and_zones(self):
        b = ArchBuilder(grid_shape=(2, 1), word_shape=(1, 1))
        with pytest.raises(ValueError, match="no words defined"):
            b.build()
        b.add_word(x=[0], y=[0])
        with pytest.raises(ValueError, match="no zones defined"):
            b.build()

    def test_blockade_radius_derives_pairs_per_zone(self):
        b = ArchBuilder(grid_shape=(4, 1), word_shape=(1, 1))
        for x in range(4):
            b.add_word(x=[x], y=[0])
        b.add_zone(
            "z",
            x=[0.0, 1.0, 10.0, 11.0],
            y=[0.0],
            x_clearance=_CL,
            y_clearance=_CL,
        )
        b.set_blockade_radius(2.0)
        b.add_mode("all", ["z"])
        spec = b.build()
        assert list(spec.zones[0].entangling_pairs) == [(0, 1), (2, 3)]
        assert spec.blockade_radius == 2.0

    def test_unknown_zone_is_rejected(self):
        b = with_zone(interleaved())
        with pytest.raises(ValueError, match="unknown zone: 'nope'"):
            b.add_word_bus("nope", src=[0], dst=[4])


# ── Round-trip ──


@pytest.mark.parametrize(
    "getter", [physical_spec, logical_spec], ids=["physical", "logical"]
)
class TestFromSpec:
    def test_round_trip_is_lossless(self, getter):
        """Rebuilding a shipped spec reproduces it exactly, paths included."""
        spec = getter()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rebuilt = ArchBuilder.from_spec(spec).build(
                feed_forward=spec.feed_forward,
                atom_reloading=spec.atom_reloading,
            )
        assert rebuilt._inner.words == spec._inner.words
        assert list(rebuilt._inner.zones) == list(spec._inner.zones)
        assert list(rebuilt._inner.zone_buses) == list(spec._inner.zone_buses)
        assert rebuilt.paths == spec.paths

    def test_inherited_paths_survive_because_the_search_cannot_recreate_them(
        self, getter
    ):
        """Bundled lane geometry is hardware-derived, not search output.

        No clearance reproduces it, and move durations are measured off
        these segment lengths, so a round-trip must not quietly swap them
        for search results.
        """
        spec = getter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            searched = ArchBuilder.from_spec(
                spec, x_clearance=0.5, y_clearance=1.0
            ).build(recompute_paths=True)
        assert searched.paths != spec.paths
        assert set(searched.paths) == set(spec.paths)

    def test_editing_a_bus_without_clearances_raises(self, getter):
        spec = getter()
        b = ArchBuilder.from_spec(spec)
        name = spec._inner.zones[0].name
        b.add_word_bus(name, src=b.words[0, 0], dst=b.words[0, 0])
        with pytest.raises(ValueError, match="no clearances"):
            b.build()
