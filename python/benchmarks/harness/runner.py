"""Benchmark runner and metric collection."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

from benchmarks.harness.models import BenchmarkJob, BenchmarkRow
from bloqade.analysis.fidelity import FidelityAnalysis

from bloqade.lanes.compile import (
    compile_to_physical_squin_noise_model as compile_physical_noise_model,
)
from bloqade.lanes.dialects import move, place
from bloqade.lanes.heuristics import logical_layout
from bloqade.lanes.heuristics.physical_layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical_placement import (
    PhysicalPlacementStrategy,
    PlacementTraversalABC,
    RustPlacementTraversal,
)
from bloqade.lanes.metrics import Metrics
from bloqade.lanes.upstream import squin_to_move


@dataclass
class SearchStatsCollector(PlacementTraversalABC):
    """Wrap traversal to collect search result metadata."""

    inner: PlacementTraversalABC
    nodes_expanded_total: int = 0
    max_depth_reached: int = 0
    runs: int = 0

    def path_to_target_config(self, *, tree, target):
        result = self.inner.path_to_target_config(tree=tree, target=target)
        self.nodes_expanded_total += result.nodes_expanded
        self.max_depth_reached = max(self.max_depth_reached, result.max_depth_reached)
        self.runs += 1
        return result


@dataclass(frozen=True)
class _RunArtifacts:
    move_mt: object
    nodes_explored: int | None
    max_depth_reached: int | None
    notes: str = ""


@dataclass
class BenchmarkRunner:
    """Executes expanded benchmark jobs and returns output rows."""

    repeats: int = 1
    warmup: int = 0
    insert_return_moves: bool = True

    def run_jobs(
        self,
        jobs: list[BenchmarkJob],
        on_job_start: Callable[[BenchmarkJob, bool], None] | None = None,
    ) -> list[BenchmarkRow]:
        rows: list[BenchmarkRow] = []
        previous_case_id: str | None = None
        for job in jobs:
            if on_job_start is not None:
                new_case = job.case.case_id != previous_case_id
                on_job_start(job, new_case)
            rows.append(self._run_one(job))
            previous_case_id = job.case.case_id
        return rows

    def _run_one(self, job: BenchmarkJob) -> BenchmarkRow:
        elapsed_samples: list[float] = []
        move_mt = None
        nodes_explored = None
        max_depth_reached = None
        notes = [job.strategy.notes] if job.strategy.notes else []

        try:
            for _ in range(self.warmup):
                _ = self._compile(job)

            for idx in range(self.repeats):
                start = time.perf_counter()
                artifacts = self._compile(job)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                elapsed_samples.append(elapsed_ms)
                if idx == 0:
                    move_mt = artifacts.move_mt
                    nodes_explored = artifacts.nodes_explored
                    max_depth_reached = artifacts.max_depth_reached
                    if artifacts.notes:
                        notes.append(artifacts.notes)

            if move_mt is None:
                raise RuntimeError("No benchmark run completed.")

            move_count_events, move_count_lanes = _count_moves(move_mt)
            fidelity = self._estimate_fidelity(job)
            if fidelity is None:
                notes.append("fidelity skipped for physical-only compilation mode")
            wall_time_ms = statistics.fmean(elapsed_samples)

            return BenchmarkRow(
                case_id=job.case.case_id,
                strategy_id=job.strategy.strategy_id,
                backend=job.strategy.backend,
                generator_id=job.strategy.generator_id,
                success=True,
                wall_time_ms=wall_time_ms,
                move_count_events=move_count_events,
                move_count_lanes=move_count_lanes,
                estimated_fidelity=fidelity,
                nodes_explored=nodes_explored,
                max_depth_reached=max_depth_reached,
                notes="; ".join([n for n in notes if n]),
            )
        except Exception as exc:  # pragma: no cover - defensive output row
            return BenchmarkRow(
                case_id=job.case.case_id,
                strategy_id=job.strategy.strategy_id,
                backend=job.strategy.backend,
                generator_id=job.strategy.generator_id,
                success=False,
                wall_time_ms=None,
                move_count_events=None,
                move_count_lanes=None,
                estimated_fidelity=None,
                nodes_explored=None,
                max_depth_reached=None,
                notes=f"{'; '.join([n for n in notes if n])}; error={exc}".strip("; "),
            )

    def _compile(self, job: BenchmarkJob) -> _RunArtifacts:
        placement_strategy = job.strategy.build_placement_strategy()
        stats_collector: SearchStatsCollector | None = None
        layout_heuristic = self._build_layout_heuristic(job)

        if isinstance(placement_strategy, PhysicalPlacementStrategy):
            traversal = placement_strategy.traversal
            if isinstance(traversal, PlacementTraversalABC):
                stats_collector = SearchStatsCollector(inner=traversal)
                placement_strategy.traversal = stats_collector

        move_mt = squin_to_move(
            job.case.kernel,
            layout_heuristic=layout_heuristic,
            placement_strategy=placement_strategy,
            insert_return_moves=self.insert_return_moves,
            logical_initialize=job.case.logical_initialize,
        )
        _assert_move_lowering_complete(move_mt)

        if stats_collector is not None:
            return _RunArtifacts(
                move_mt=move_mt,
                nodes_explored=stats_collector.nodes_expanded_total,
                max_depth_reached=stats_collector.max_depth_reached,
            )

        notes = ""
        nodes: int | None = None
        depth: int | None = None
        if isinstance(placement_strategy, PhysicalPlacementStrategy) and isinstance(
            placement_strategy.traversal, RustPlacementTraversal
        ):
            nodes = placement_strategy.rust_nodes_expanded_total
        return _RunArtifacts(
            move_mt=move_mt,
            nodes_explored=nodes,
            max_depth_reached=depth,
            notes=notes,
        )

    def _estimate_fidelity(self, job: BenchmarkJob) -> float | None:
        if job.case.logical_initialize:
            placement_strategy = job.strategy.build_placement_strategy()
            metrics = Metrics(arch_spec=placement_strategy.arch_spec)
            fidelity = metrics.analyze_fidelity(
                job.case.kernel,
                placement_strategy=placement_strategy,
                insert_return_moves=self.insert_return_moves,
            )
            return fidelity.gate_fidelity_product

        placement_strategy = job.strategy.build_placement_strategy()
        layout_heuristic = self._build_layout_heuristic(job)
        physical_squin = compile_physical_noise_model(
            job.case.kernel,
            placement_strategy=placement_strategy,
            layout_heuristic=layout_heuristic,
            insert_return_moves=self.insert_return_moves,
        )
        analysis = FidelityAnalysis(physical_squin.dialects)
        analysis.run(physical_squin)
        fidelity_product = 1.0
        for gate_fid in analysis.gate_fidelities:
            fidelity_product *= gate_fid.min
        return fidelity_product

    def _build_layout_heuristic(self, job: BenchmarkJob):
        if job.case.logical_initialize:
            return logical_layout.LogicalLayoutHeuristic()
        return PhysicalLayoutHeuristicGraphPartitionCenterOut()


def _count_moves(move_mt) -> tuple[int, int]:
    move_count_events = 0
    move_count_lanes = 0
    for stmt in move_mt.callable_region.walk():
        if isinstance(stmt, move.Move):
            move_count_events += 1
            move_count_lanes += len(stmt.lanes)
    return move_count_events, move_count_lanes


def _assert_move_lowering_complete(move_mt) -> None:
    if any(isinstance(stmt, place.CZ) for stmt in move_mt.callable_region.walk()):
        raise RuntimeError(
            "Compilation did not fully lower to Move IR; place.CZ statements remain."
        )
