"""Tests for SearchParams validation and defaults."""

import pytest

from bloqade.lanes.search.search_params import SearchParams


class TestSearchParams:
    def test_defaults(self):
        p = SearchParams()
        assert p.w_d == 1.0
        assert p.w_m == 0.1
        assert p.alpha == 1.0
        assert p.beta == 2.0
        assert p.gamma == 0.5
        assert p.top_c == 3
        assert p.max_candidates == 2
        assert p.reversion_steps == 1
        assert p.delta_e == 1
        assert p.e_max == 4

    def test_custom_values(self):
        p = SearchParams(w_d=2.0, e_max=8, delta_e=2)
        assert p.w_d == 2.0
        assert p.e_max == 8
        assert p.delta_e == 2

    def test_delta_e_minimum_enforced(self):
        with pytest.raises(ValueError, match="delta_e"):
            SearchParams(delta_e=0)

    def test_e_max_minimum_enforced(self):
        with pytest.raises(ValueError, match="e_max"):
            SearchParams(e_max=1)

    def test_top_c_minimum_enforced(self):
        with pytest.raises(ValueError, match="top_c"):
            SearchParams(top_c=0)

    def test_max_candidates_minimum_enforced(self):
        with pytest.raises(ValueError, match="max_candidates"):
            SearchParams(max_candidates=0)

    def test_reversion_steps_minimum_enforced(self):
        with pytest.raises(ValueError, match="reversion_steps"):
            SearchParams(reversion_steps=0)
