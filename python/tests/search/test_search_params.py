"""Tests for SearchParams validation and defaults."""

import pytest

from bloqade.lanes.search.search_params import SearchParams


class TestSearchParams:
    def test_e_max_minimum_enforced(self):
        with pytest.raises(ValueError, match="e_max"):
            SearchParams(e_max=SearchParams.MIN_E_MAX - 1)

    def test_max_candidates_minimum_enforced(self):
        with pytest.raises(ValueError, match="max_candidates"):
            SearchParams(max_candidates=SearchParams.MIN_MAX_CANDIDATES - 1)

    def test_w_t_minimum_enforced(self):
        with pytest.raises(ValueError, match="w_t"):
            SearchParams(w_t=SearchParams.MIN_W_T - 0.01)

    def test_w_t_maximum_enforced(self):
        with pytest.raises(ValueError, match="w_t"):
            SearchParams(w_t=SearchParams.MAX_W_T + 0.01)
