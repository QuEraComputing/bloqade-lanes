from typing import cast

from benchmarks.harness.matrix import default_strategy_configs, expand_benchmark_jobs
from benchmarks.harness.models import StrategyConfig
from benchmarks.kernels import BENCHMARK_CASES

from bloqade.lanes.analysis.placement import PlacementStrategyABC
from bloqade.lanes.layout.arch import ArchSpec


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


def test_default_strategy_configs_accepts_factory_and_arch_spec_id():
    sentinel = cast(ArchSpec, object())
    factory_calls: list[int] = []

    def factory() -> ArchSpec:
        factory_calls.append(1)
        return sentinel

    strategies = default_strategy_configs(
        arch_spec_factory=factory, arch_spec_id="custom"
    )

    assert strategies
    assert all(s.arch_spec_id == "custom" for s in strategies)
    # Factory should be lazy — not invoked until build_placement_strategy is called.
    assert factory_calls == []


def test_default_strategy_configs_factory_is_called_per_build():
    calls: list[int] = []

    def factory() -> ArchSpec:
        calls.append(1)
        return cast(ArchSpec, object())

    strategies = default_strategy_configs(arch_spec_factory=factory, arch_spec_id="x")
    # One build per strategy → one factory call per strategy.
    for strategy in strategies:
        try:
            strategy.build_placement_strategy()
        except Exception:
            pass  # PlacementStrategy construction may fail on object() arch, ok.
    assert len(calls) == len(strategies)
