"""Tests for the Gemini architecture adapter."""

from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
from bloqade.lanes.arch.gemini_adapter import build_gemini_arch, gemini_blueprint
from bloqade.lanes.layout.encoding import LocationAddress
from bloqade.lanes.layout.path import PathFinder


class TestGeminiBlueprint:
    def test_default_params(self) -> None:
        bp = gemini_blueprint()
        assert len(bp.zones) == 1
        spec = bp.zones["gate"]
        assert spec.num_rows == 1
        assert spec.num_cols == 32  # 2 * 2^4
        assert spec.entangling is True
        assert spec.num_words == 32

    def test_custom_dims(self) -> None:
        bp = gemini_blueprint(hypercube_dims=2, word_size_y=4)
        spec = bp.zones["gate"]
        assert spec.num_cols == 8  # 2 * 2^2
        assert bp.layout.sites_per_word == 4


class TestBuildGeminiArch:
    def test_builds_valid_arch(self) -> None:
        result = build_gemini_arch(hypercube_dims=2, word_size_y=4)
        arch = result.arch
        assert len(arch.words) == 8  # 2 * 2^2
        # entangling_zones is now a tuple of zones, each with word pairs
        assert len(arch.entangling_zones) == 1
        # The gate zone has 4 CZ pairs for 8 words
        assert len(arch.entangling_zones[0]) == 4

    def test_word_buses_include_hypercube_dims(self) -> None:
        result = build_gemini_arch(hypercube_dims=2, word_size_y=4)
        # 2^2 = 4 old words → 8 new words → log2(8) = 3 col dims
        # dim 0: CZ pair, dim 1-2: old hypercube
        assert len(result.arch.word_buses) == 3

    def test_site_buses_present(self) -> None:
        result = build_gemini_arch(hypercube_dims=2, word_size_y=4)
        # AllToAll(4 sites) = 6 buses, applied to all 8 words
        assert len(result.arch.site_buses) == 6
        for bus in result.arch.site_buses:
            assert bus.words is not None
            assert len(bus.words) == 8

    def test_cz_pairing(self) -> None:
        result = build_gemini_arch(hypercube_dims=2, word_size_y=4)
        grid = result.zone_grids["gate"]
        pairs = list(grid.cz_pairs())
        # 4 CZ pairs: (0,1), (2,3), (4,5), (6,7)
        assert pairs == [(0, 1), (2, 3), (4, 5), (6, 7)]


class TestGeminiConnectivityEquivalence:
    """Verify that the new Gemini arch has equivalent connectivity to the old."""

    def test_old_hypercube_pairs_reachable(self) -> None:
        """All word pairs connected in old hypercube are reachable in new arch."""
        dims = 2
        result = build_gemini_arch(hypercube_dims=dims, word_size_y=4)
        pf = PathFinder(result.arch)

        # Old word k maps to new words 2k (even) and 2k+1 (odd)
        # Old hypercube connects word pairs differing by one bit
        num_old_words = 2**dims
        for old_src in range(num_old_words):
            for d in range(dims):
                old_dst = old_src ^ (1 << d)
                if old_dst <= old_src:
                    continue
                # Check new words can reach each other (via same site)
                new_src = 2 * old_src  # even col of old src word
                new_dst = 2 * old_dst  # even col of old dst word
                path = pf.find_path(
                    LocationAddress(new_src, 0),
                    LocationAddress(new_dst, 0),
                )
                assert path is not None, (
                    f"Old word {old_src} → {old_dst} not reachable "
                    f"(new word {new_src} → {new_dst})"
                )

    def test_intra_pair_reachable(self) -> None:
        """CZ pair members can reach each other (was old site bus)."""
        result = build_gemini_arch(hypercube_dims=2, word_size_y=4)
        pf = PathFinder(result.arch)

        # Word 0 site 0 → Word 1 site 0 (CZ pair)
        path = pf.find_path(LocationAddress(0, 0), LocationAddress(1, 0))
        assert path is not None

    def test_intra_word_site_reachable(self) -> None:
        """Sites within a single word can reach each other (site buses)."""
        result = build_gemini_arch(hypercube_dims=2, word_size_y=4)
        pf = PathFinder(result.arch)

        # Site 0 → site 3 within word 0
        path = pf.find_path(LocationAddress(0, 0), LocationAddress(0, 3))
        assert path is not None

    def test_total_sites_match(self) -> None:
        """Same total number of atom sites as old architecture."""
        dims = 4
        word_size_y = 5
        old_arch = generate_arch_hypercube(dims, word_size_y)
        new_result = build_gemini_arch(dims, word_size_y)

        old_total = len(old_arch.words) * len(old_arch.words[0].site_indices)
        new_total = len(new_result.arch.words) * len(
            new_result.arch.words[0].site_indices
        )
        assert old_total == new_total  # 16*10 = 32*5 = 160
