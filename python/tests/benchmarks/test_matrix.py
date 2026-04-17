from benchmarks.harness.matrix import default_strategy_configs, expand_benchmark_jobs
from benchmarks.kernels import BENCHMARK_CASES


def test_matrix_expansion_builds_case_strategy_cross_product():
    strategies = default_strategy_configs()
    jobs = expand_benchmark_jobs(BENCHMARK_CASES, strategies)
    assert len(jobs) == len(BENCHMARK_CASES) * len(strategies)


def test_matrix_expansion_applies_filters():
    strategies = default_strategy_configs()
    jobs = expand_benchmark_jobs(
        BENCHMARK_CASES,
        strategies,
        case_filter={"ghz_4"},
        strategy_filter={"rust_astar"},
    )
    assert len(jobs) == 1
    assert jobs[0].case.case_id == "ghz_4"
    assert jobs[0].strategy.strategy_id == "rust_astar"


def test_default_strategy_configs_are_importable_and_buildable():
    strategies = default_strategy_configs()
    assert strategies
    for strategy in strategies:
        placement_strategy = strategy.build_placement_strategy()
        assert placement_strategy is not None
