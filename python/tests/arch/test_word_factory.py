"""Tests for word creation helpers."""

from bloqade.lanes.arch.word_factory import WordGrid, create_zone_words
from bloqade.lanes.arch.zone import DeviceLayout, ZoneSpec
from bloqade.lanes.layout.encoding import LocationAddress


class TestWordGrid:
    def _make_grid(self) -> WordGrid:
        """Create a 2x4 entangling grid with offset 10."""
        spec = ZoneSpec(num_rows=2, num_cols=4, entangling=True)
        layout = DeviceLayout(sites_per_word=3)
        return create_zone_words(spec, layout, word_id_offset=10)

    def test_word_count(self) -> None:
        grid = self._make_grid()
        assert len(grid.words) == 8
        assert grid.num_rows == 2
        assert grid.num_cols == 4
        assert grid.word_id_offset == 10

    def test_word_id_at(self) -> None:
        grid = self._make_grid()
        assert grid.word_id_at(0, 0) == 10
        assert grid.word_id_at(0, 3) == 13
        assert grid.word_id_at(1, 0) == 14
        assert grid.word_id_at(1, 3) == 17

    def test_word_at_has_correct_sites(self) -> None:
        grid = self._make_grid()
        word = grid.word_at(0, 0)
        assert len(word.site_indices) == 3
        assert word.site_indices == ((0, 0), (0, 1), (0, 2))

    def test_cz_pairs(self) -> None:
        grid = self._make_grid()
        pairs = list(grid.cz_pairs())
        assert pairs == [(10, 11), (12, 13), (14, 15), (16, 17)]


class TestCreateZoneWords:
    def test_entangling_zone_cz_pairing(self) -> None:
        spec = ZoneSpec(num_rows=1, num_cols=2, entangling=True)
        layout = DeviceLayout(sites_per_word=3)
        grid = create_zone_words(spec, layout, word_id_offset=0)

        assert len(grid.words) == 2
        # Word 0 → partner word 1
        assert grid.words[0].has_cz is not None
        assert grid.words[0].has_cz[0] == LocationAddress(1, 0)
        # Word 1 → partner word 0
        assert grid.words[1].has_cz is not None
        assert grid.words[1].has_cz[0] == LocationAddress(0, 0)

    def test_non_entangling_zone_no_cz(self) -> None:
        spec = ZoneSpec(num_rows=1, num_cols=2, entangling=False)
        layout = DeviceLayout(sites_per_word=3)
        grid = create_zone_words(spec, layout)

        for word in grid.words:
            assert word.has_cz is None

    def test_word_grid_shape(self) -> None:
        spec = ZoneSpec(num_rows=1, num_cols=2, entangling=False)
        layout = DeviceLayout(sites_per_word=5)
        grid = create_zone_words(spec, layout)

        for word in grid.words:
            assert word.positions.shape == (1, 5)

    def test_pair_spacing(self) -> None:
        spec = ZoneSpec(num_rows=1, num_cols=4, entangling=True)
        layout = DeviceLayout(
            sites_per_word=2, site_spacing=10.0,
            word_spacing=2.0, pair_spacing=10.0,
        )
        grid = create_zone_words(spec, layout)

        # Pair 0: col 0 at x=0, col 1 at x=2
        assert grid.words[0].site_position(0)[0] == 0.0
        assert grid.words[1].site_position(0)[0] == 2.0
        # Pair 1: col 2 at x=12, col 3 at x=14
        assert grid.words[2].site_position(0)[0] == 12.0
        assert grid.words[3].site_position(0)[0] == 14.0

    def test_row_spacing(self) -> None:
        spec = ZoneSpec(num_rows=2, num_cols=2, entangling=True)
        layout = DeviceLayout(
            sites_per_word=3, site_spacing=10.0, row_spacing=20.0,
        )
        grid = create_zone_words(spec, layout)

        # Row 0 at y=0
        assert grid.words[0].site_position(0)[1] == 0.0
        # Row 1 at y = (3-1)*10 + 20 = 40
        assert grid.words[2].site_position(0)[1] == 40.0

    def test_word_id_offset_in_cz(self) -> None:
        spec = ZoneSpec(num_rows=1, num_cols=2, entangling=True)
        layout = DeviceLayout(sites_per_word=3)
        grid = create_zone_words(spec, layout, word_id_offset=10)

        assert grid.words[0].has_cz is not None
        assert grid.words[0].has_cz[0] == LocationAddress(11, 0)
        assert grid.words[1].has_cz is not None
        assert grid.words[1].has_cz[0] == LocationAddress(10, 0)

    def test_xy_offset(self) -> None:
        spec = ZoneSpec(num_rows=1, num_cols=2, entangling=False)
        layout = DeviceLayout(sites_per_word=2, site_spacing=10.0, word_spacing=2.0)
        grid = create_zone_words(spec, layout, x_offset=100.0, y_offset=200.0)

        assert grid.words[0].site_position(0) == (100.0, 200.0)
        assert grid.words[1].site_position(0) == (102.0, 200.0)
