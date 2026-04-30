from typing import cast

from benchmarks.harness.matrix import default_strategy_configs, expand_benchmark_jobs
from benchmarks.harness.models import StrategyConfig
from benchmarks.kernels import BENCHMARK_CASES

from bloqade.lanes.analysis.placement import PlacementStrategyABC
from bloqade.lanes.arch import ArchSpec


def test_matrix_expansion_builds_case_strategy_cross_product():
    strategies = default_strategy_configs()
    jobs = expand_benchmark_jobs(BENCHMARK_CASES, strategies)
    assert len(jobs) == len(BENCHMARK_CASES) * len(strategies)


def test_matrix_expansion_applies_strategy_filter():
    strategies = default_strategy_configs()
    jobs = expand_benchmark_jobs(
        BENCHMARK_CASES,
        strategies,
        strategy_filter={"rust_astar"},
    )
    assert len(jobs) == len(BENCHMARK_CASES)
    assert all(job.strategy.strategy_id == "rust_astar" for job in jobs)


def test_default_strategy_configs_are_importable_and_buildable():
    strategies = default_strategy_configs()
    assert strategies
    for strategy in strategies:
        placement_strategy = strategy.build_placement_strategy()
        assert placement_strategy is not None


def test_strategy_config_arch_spec_id_defaults_to_builtin():
    strategy = StrategyConfig(
        strategy_id="s",
        backend="python",
        generator_id="heuristic",
        build_placement_strategy=lambda: cast(PlacementStrategyABC, object()),
    )
    assert strategy.arch_spec_id == "builtin"


def test_default_strategy_configs_stamp_builtin_arch_spec_id():
    strategies = default_strategy_configs()
    assert all(s.arch_spec_id == "builtin" for s in strategies)


def test_default_strategy_configs_accepts_arch_spec_pair():
    sentinel = cast(ArchSpec, object())
    factory_calls: list[int] = []

    def factory() -> ArchSpec:
        factory_calls.append(1)
        return sentinel

    strategies = default_strategy_configs(arch_spec=("custom", factory))

    assert strategies
    assert all(s.arch_spec_id == "custom" for s in strategies)
    # Factory should be lazy — not invoked until build_placement_strategy is called.
    assert factory_calls == []


def test_default_strategy_configs_factory_is_called_per_build():
    calls: list[int] = []

    def factory() -> ArchSpec:
        calls.append(1)
        return cast(ArchSpec, object())

    strategies = default_strategy_configs(arch_spec=("x", factory))
    # One build per strategy → one factory call per strategy.
    for strategy in strategies:
        try:
            strategy.build_placement_strategy()
        except Exception:
            pass  # PlacementStrategy construction may fail on object() arch, ok.
    assert len(calls) == len(strategies)
