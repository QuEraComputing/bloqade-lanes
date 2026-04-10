"""Tests for SearchParams validation and defaults."""

import pytest

from bloqade.lanes.search.search_params import SearchParams


class TestSearchParams:
    def test_delta_e_minimum_enforced(self):
        with pytest.raises(ValueError, match="delta_e"):
            SearchParams(delta_e=SearchParams.MIN_DELTA_E - 1)

    def test_e_max_minimum_enforced(self):
        with pytest.raises(ValueError, match="e_max"):
            SearchParams(e_max=SearchParams.MIN_E_MAX - 1)

    def test_top_c_minimum_enforced(self):
        with pytest.raises(ValueError, match="top_c"):
            SearchParams(top_c=SearchParams.MIN_TOP_C - 1)

    def test_max_candidates_minimum_enforced(self):
        with pytest.raises(ValueError, match="max_candidates"):
            SearchParams(max_candidates=SearchParams.MIN_MAX_CANDIDATES - 1)

    def test_reversion_steps_minimum_enforced(self):
        with pytest.raises(ValueError, match="reversion_steps"):
            SearchParams(reversion_steps=SearchParams.MIN_REVERSION_STEPS - 1)
