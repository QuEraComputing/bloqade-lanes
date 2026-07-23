"""Benchmark runner and metric collection."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

from benchmarks.harness.models import BenchmarkJob, BenchmarkRow
from bloqade.analysis.fidelity import FidelityAnalysis
from kirin import ir

from bloqade.lanes.analysis import atom
from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.analysis.placement.strategy import PlacementStrategyABC
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects import move, place
from bloqade.lanes.heuristics.logical import layout as logical_layout
from bloqade.lanes.heuristics.physical.layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical.placement import (
    PhysicalPlacementStrategy,
    RustPlacementTraversal,
)
from bloqade.lanes.noise_model import (
    generate_logical_noise_model,
    generate_simple_noise_model,
)
from bloqade.lanes.passes import SequentialPlacePass
from bloqade.lanes.transform import (
    LogicalNativeToPlace,
    MoveToSquinLogical,
    MoveToSquinPhysical,
    NativeToPlace,
    PhysicalPipeline,
    PlaceToMove,
    transversal_rewrites,
)


def _squin_to_move(
    mt: ir.Method,
    *,
    layout_heuristic: LayoutHeuristicABC,
    placement_strategy: PlacementStrategyABC,
    logical_initialize: bool,
) -> ir.Method:
    NativeStage = LogicalNativeToPlace if logical_initialize else NativeToPlace
    place_mt = NativeStage(arch_spec=getattr(layout_heuristic, "arch_spec", None)).emit(
        mt
    )
    SequentialPlacePass(place_mt.dialects)(place_mt)
    return PlaceToMove(
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        insert_initialize=logical_initialize,
    ).emit(place_mt)


@dataclass(frozen=True)
class _RunArtifacts:
    move_mt: object
    arch_spec: ArchSpec
    nodes_explored: int | None
    max_depth_reached: int | None
    notes: str = ""


@dataclass
class BenchmarkRunner:
    """Executes expanded benchmark jobs and returns output rows."""

    repeats: int = 1
    warmup: int = 0

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
        artifacts: _RunArtifacts | None = None
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

            if move_mt is None or artifacts is None:
                raise RuntimeError("No benchmark run completed.")

            move_count_events, move_count_lanes = _count_moves(
                move_mt,
                artifacts.arch_spec,
            )
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
                arch_spec_id=job.strategy.arch_spec_id,
                notes="; ".join([n for n in notes if n]),
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
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
                arch_spec_id=job.strategy.arch_spec_id,
                notes=f"{'; '.join([n for n in notes if n])}; error={exc}".strip("; "),
            )

    def _compile(self, job: BenchmarkJob) -> _RunArtifacts:
        placement_strategy = self._build_placement_strategy(job)
        layout_heuristic = self._build_layout_heuristic(job)

        move_mt = _squin_to_move(
            job.case.kernel,
            layout_heuristic=layout_heuristic,
            placement_strategy=placement_strategy,
            logical_initialize=job.case.logical_initialize,
        )
        _assert_move_lowering_complete(move_mt)

        nodes: int | None = None
        inner = getattr(placement_strategy, "inner", placement_strategy)
        if isinstance(inner, PhysicalPlacementStrategy) and isinstance(
            inner.traversal, RustPlacementTraversal
        ):
            nodes = inner.rust_nodes_expanded_total
        return _RunArtifacts(
            move_mt=move_mt,
            arch_spec=placement_strategy.arch_spec,
            nodes_explored=nodes,
            max_depth_reached=None,
        )

    def _estimate_fidelity(self, job: BenchmarkJob) -> float | None:
        if job.case.logical_initialize:
            placement_strategy = self._build_placement_strategy(job)
            move_mt = _squin_to_move(
                job.case.kernel,
                layout_heuristic=logical_layout.LogicalLayoutHeuristic(),
                placement_strategy=placement_strategy,
                logical_initialize=True,
            )
            move_mt = transversal_rewrites(move_mt)
            # aggressive_unroll is required so the broadcasted state-prep loops
            # (ilist.map/foldl in the init kernels) are fully unrolled; otherwise
            # FidelityAnalysis cannot resolve the per-qubit noise channels behind
            # the rolled loops and reports an inflated fidelity.
            physical_squin = MoveToSquinLogical(
                arch_spec=placement_strategy.arch_spec,
                noise_model=generate_logical_noise_model(),
                add_noise=True,
                aggressive_unroll=True,
            ).emit(move_mt)
            analysis = FidelityAnalysis(physical_squin.dialects)
            analysis.run(physical_squin)
            fidelity_product = 1.0
            for gate_fid in analysis.gate_fidelities:
                fidelity_product *= gate_fid.min
            return fidelity_product

        placement_strategy = self._build_placement_strategy(job)
        layout_heuristic = self._build_layout_heuristic(job)
        # Construct one ArchSpec and reuse it for both the move compilation and
        # the noise-insertion step so they cannot disagree on the target arch.
        arch_spec = get_physical_arch_spec()
        move_mt = PhysicalPipeline(
            arch_spec=arch_spec,
            layout_heuristic=layout_heuristic,
            placement_strategy=placement_strategy,
        ).emit(job.case.kernel)
        physical_squin = MoveToSquinPhysical(
            arch_spec=arch_spec,
            noise_model=generate_simple_noise_model(),
            aggressive_unroll=False,
        ).emit(move_mt)
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

    def _build_placement_strategy(self, job: BenchmarkJob):
        return job.strategy.build_placement_strategy()


def _count_moves(move_mt, arch_spec: ArchSpec) -> tuple[int, int]:
    atom_interp = atom.AtomInterpreter(move_mt.dialects, arch_spec=arch_spec)
    frame, _ = atom_interp.run(move_mt)
    move_count_events = 0
    move_count_lanes = 0
    for stmt in move_mt.callable_region.walk():
        if isinstance(stmt, move.Move):
            move_count_events += 1
            state = frame.get(stmt.current_state)
            assert isinstance(
                state, atom.AtomState
            ), "Expected concrete atom state before move.Move while counting lanes"
            move_count_lanes += sum(
                1
                for lane in stmt.lanes
                if (endpoints := arch_spec.try_get_endpoints(lane)) is not None
                and endpoints[0] in state.data.locations_to_qubit
            )
    return move_count_events, move_count_lanes


def _assert_move_lowering_complete(move_mt) -> None:
    if any(isinstance(stmt, place.CZ) for stmt in move_mt.callable_region.walk()):
        raise RuntimeError(
            "Compilation did not fully lower to Move IR; place.CZ statements remain."
        )
