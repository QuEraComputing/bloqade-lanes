import pytest
from benchmarks.kernels import BENCHMARK_CASES, SIZE_BUCKETS, select_benchmark_cases


def test_benchmark_registry_has_unique_callable_cases():
    assert BENCHMARK_CASES
    case_ids = [case.case_id for case in BENCHMARK_CASES]
    assert len(case_ids) == len(set(case_ids))
    assert all(callable(case.kernel) for case in BENCHMARK_CASES)


def test_case_selector_without_filter_returns_all_cases():
    assert select_benchmark_cases(None) == BENCHMARK_CASES


def test_case_selector_returns_requested_case_id():
    case_id = BENCHMARK_CASES[0].case_id
    selected = select_benchmark_cases({case_id})
    assert [case.case_id for case in selected] == [case_id]


def test_case_selector_supports_bucket_selectors():
    bucket = SIZE_BUCKETS[0]
    selected = select_benchmark_cases({bucket})
    all_cases = select_benchmark_cases(None)
    assert selected
    assert len(selected) < len(all_cases)


def test_case_selector_rejects_unknown_selector():
    with pytest.raises(ValueError, match="Unknown case selector"):
        select_benchmark_cases({"definitely_not_a_kernel"})
