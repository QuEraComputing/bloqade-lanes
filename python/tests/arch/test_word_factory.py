"""Tests for word creation helpers."""

from bloqade.lanes.arch.word_factory import WordGrid, create_zone_words
from bloqade.lanes.arch.zone import DeviceLayout, ZoneSpec


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
        # Row word: sites along x-axis → (i, 0)
        assert word.site_indices == ((0, 0), (1, 0), (2, 0))

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
        # CZ pairing is now at the architecture level via cz_pairs()
        pairs = list(grid.cz_pairs())
        assert pairs == [(0, 1)]

    def test_non_entangling_zone_words_created(self) -> None:
        spec = ZoneSpec(num_rows=1, num_cols=2, entangling=False)
        layout = DeviceLayout(sites_per_word=3)
        grid = create_zone_words(spec, layout)

        # Words are created; CZ pairing is decided at the architecture level
        assert len(grid.words) == 2

    def test_word_grid_shape(self) -> None:
        """Words are rows: N x-positions, 1 y-position → shape (N, 1)."""
        spec = ZoneSpec(num_rows=1, num_cols=2, entangling=False)
        layout = DeviceLayout(sites_per_word=5)
        grid = create_zone_words(spec, layout)

        for word in grid.words:
            assert word.positions.shape == (5, 1)

    def test_interleaved_positions(self) -> None:
        """Even/odd words have interleaved x-positions."""
        spec = ZoneSpec(num_rows=1, num_cols=2, entangling=True)
        layout = DeviceLayout(sites_per_word=3, site_spacing=10.0)
        grid = create_zone_words(spec, layout)

        # Even word (col 0): sites at x = 0, 20, 40
        w0 = grid.words[0]
        assert w0.site_position(0) == (0.0, 0.0)
        assert w0.site_position(1) == (20.0, 0.0)
        assert w0.site_position(2) == (40.0, 0.0)

        # Odd word (col 1): sites at x = 10, 30, 50
        w1 = grid.words[1]
        assert w1.site_position(0) == (10.0, 0.0)
        assert w1.site_position(1) == (30.0, 0.0)
        assert w1.site_position(2) == (50.0, 0.0)

    def test_pair_spacing(self) -> None:
        """Gap between CZ pairs."""
        spec = ZoneSpec(num_rows=1, num_cols=4, entangling=True)
        layout = DeviceLayout(sites_per_word=2, site_spacing=10.0, pair_spacing=20.0)
        grid = create_zone_words(spec, layout)

        # Pair 0: even at x=0,20; odd at x=10,30. Pair width = (2*2-1)*10 = 30
        assert grid.words[0].site_position(0)[0] == 0.0
        assert grid.words[1].site_position(0)[0] == 10.0
        # Pair 1 starts at x = 30 + 20 (pair_spacing) = 50
        assert grid.words[2].site_position(0)[0] == 50.0
        assert grid.words[3].site_position(0)[0] == 60.0

    def test_row_spacing(self) -> None:
        spec = ZoneSpec(num_rows=2, num_cols=2, entangling=True)
        layout = DeviceLayout(sites_per_word=3, site_spacing=10.0, row_spacing=50.0)
        grid = create_zone_words(spec, layout)

        # Row 0 at y=0
        assert grid.words[0].site_position(0)[1] == 0.0
        # Row 1 at y=50
        assert grid.words[2].site_position(0)[1] == 50.0

    def test_word_id_offset_in_cz(self) -> None:
        spec = ZoneSpec(num_rows=1, num_cols=2, entangling=True)
        layout = DeviceLayout(sites_per_word=3)
        grid = create_zone_words(spec, layout, word_id_offset=10)

        # CZ pairing uses word IDs with offset
        pairs = list(grid.cz_pairs())
        assert pairs == [(10, 11)]

    def test_xy_offset(self) -> None:
        spec = ZoneSpec(num_rows=1, num_cols=2, entangling=False)
        layout = DeviceLayout(sites_per_word=2, site_spacing=10.0)
        grid = create_zone_words(spec, layout, x_offset=100.0, y_offset=200.0)

        assert grid.words[0].site_position(0) == (100.0, 200.0)
        # Odd word site 0 at x = 100 + 10 (site_spacing)
        assert grid.words[1].site_position(0) == (110.0, 200.0)
