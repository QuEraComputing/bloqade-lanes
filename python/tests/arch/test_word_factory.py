"""Tests for word creation helpers."""

from bloqade.lanes.arch.build.blueprint import DeviceLayout, ZoneSpec
from bloqade.lanes.arch.build.word_factory import WordGrid, create_zone_words


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
        # Even-column word: sites at even x indices (interleaved CZ pairing)
        assert word.site_indices == ((0, 0), (2, 0), (4, 0))

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

    def test_word_has_correct_site_count(self) -> None:
        """Words have the correct number of sites."""
        spec = ZoneSpec(num_rows=1, num_cols=2, entangling=False)
        layout = DeviceLayout(sites_per_word=5)
        grid = create_zone_words(spec, layout)

        for word in grid.words:
            assert len(word.site_indices) == 5

    def test_word_id_offset_in_cz(self) -> None:
        spec = ZoneSpec(num_rows=1, num_cols=2, entangling=True)
        layout = DeviceLayout(sites_per_word=3)
        grid = create_zone_words(spec, layout, word_id_offset=10)

        # CZ pairing uses word IDs with offset
        pairs = list(grid.cz_pairs())
        assert pairs == [(10, 11)]
