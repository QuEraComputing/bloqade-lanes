"""Epic 4, phase 1: does ranking CZ-stage candidates by a Push-and-Rotate plan
beat today's pick?

For every CZ stage the physical benchmark kernels produce along today's own
compile trajectory, route each candidate target placement twice -- with the
pipeline's own router, to cost it, and with Push and Rotate, to rank it -- and
compare today's pick with the Push-and-Rotate-ranked pick. It is a per-stage
counterfactual: the stages are the ones today's pipeline reaches, and each
candidate is costed from the same starting placement.

Two candidate spaces, reported separately:

``nohome``
    The default pipeline's (NoHome's) per-pair mover choice: for each CZ pair,
    which atom moves to the other's partner site. Today's pick is NoHome's
    fixed rule (``resolve_cz_targets``): the control if both atoms share a
    word, else the target if it sits on a home site, else the control. All
    assignments are tried when there are at most ``--max-assignments`` of them;
    otherwise the rule, every single-pair flip, and seeded random assignments.
``generator``
    The Python target generators' candidate lists under
    ``PhysicalPlacementStrategy``, where today's pick is the first candidate
    that routes.

**Go / no-go, fixed before running (2026-09-24):** summed over the kernels,
the ranked pick routes in at least 5% fewer operations than today's pick, with
no success regressions (a stage today's pick routes but the ranked pick does
not). Per candidate space.

Run: ``uv run --no-sync python -m benchmarks.ranking [--cases a,b] [--output f.json]``
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from benchmarks.harness import BenchmarkRunner
from benchmarks.harness.models import BenchmarkJob, StrategyConfig
from benchmarks.kernels import select_benchmark_cases

from bloqade.lanes.analysis.placement import PalindromePlacementStrategy
from bloqade.lanes.arch import ArchSpec
from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.bytecode import _native
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.physical.movement import (
    PhysicalPlacementStrategy,
    RustPlacementTraversal,
    _move_search_from_traversal,
    make_physical_placement_strategy,
)
from bloqade.lanes.heuristics.physical.nohome import NoHomePlacementStrategy
from bloqade.lanes.heuristics.physical.target_generator import (
    AODClusterTargetGenerator,
    CongestionAwareTargetGenerator,
    LookaheadCongestionAwareTargetGenerator,
    TargetContext,
    TargetGeneratorABC,
)

GO_THRESHOLD = 0.05

Native = dict[int, _native.LocationAddress]


@dataclass
class Stage:
    """One CZ stage as the router saw it, with its candidate targets."""

    case: str
    index: int
    initial: Native
    blocked: list[_native.LocationAddress]
    candidates: list[Native]
    today: int  # index into `candidates`
    max_expansions: int | None
    engine: _native.SearchEngine
    router: _native.MoveSearch
    # NoHome only: whether NoHome's own placement equals the rule candidate,
    # the check that this module's re-derivation of the rule is faithful.
    rule_matches: bool | None = None
    # Filled by `evaluate`.
    route_ops: list[int | None] = field(default_factory=list)
    pr_ops: list[int | None] = field(default_factory=list)


def _to_py(loc: _native.LocationAddress) -> LocationAddress:
    return LocationAddress(loc.word_id, loc.site_id, loc.zone_id)


def _compile(case, build: Callable[[], Any]) -> None:
    config = StrategyConfig(
        strategy_id="ranking_probe",
        backend="rust",
        generator_id="rust_solver",
        build_placement_strategy=build,
    )
    BenchmarkRunner()._compile(BenchmarkJob(case=case, strategy=config))


# ── Space A: NoHome's per-pair mover choice ──


def _mover_assignments(k: int, rule: tuple[int, ...], cap: int, seed: int):
    """The rule first, then either every assignment or a bounded sample."""
    if 2**k <= cap:
        rest = [a for a in itertools.product((0, 1), repeat=k) if a != rule]
        return [rule, *rest]
    flips = [
        tuple(1 - b if j == i else b for j, b in enumerate(rule)) for i in range(k)
    ]
    seen = {rule, *flips}
    rng = random.Random(seed)
    sample: list[tuple[int, ...]] = []
    while len(sample) < cap - len(seen) and len(seen) + len(sample) < 2**k:
        a = tuple(rng.randint(0, 1) for _ in range(k))
        if a not in seen:
            seen.add(a)
            sample.append(a)
    return [rule, *flips, *sample]


def _mover_candidates(
    arch: ArchSpec, initial: Native, pairs: list[tuple[int, int]], cap: int, seed: int
) -> tuple[list[Native], int] | None:
    """Candidate targets for every mover assignment, the rule's first.

    ``None`` when some pair has no per-pair move (NoHome places those in free
    entangling slots instead, which this space does not model).
    """
    options = []
    rule = []
    for c, t in pairs:
        c_addr, t_addr = _to_py(initial[c]), _to_py(initial[t])
        c_dst, t_dst = arch.get_cz_partner(t_addr), arch.get_cz_partner(c_addr)
        if c_dst is None or t_dst is None:
            return None
        moves_target = c_addr.word_id != t_addr.word_id and arch.is_home_position(
            t_addr
        )
        options.append(((c, c_dst), (t, t_dst)))
        rule.append(1 if moves_target else 0)
    candidates: list[Native] = []
    for n, assignment in enumerate(
        _mover_assignments(len(pairs), tuple(rule), cap, seed)
    ):
        target = dict(initial)
        for choice, (move_c, move_t) in zip(assignment, options):
            qid, dst = move_t if choice else move_c
            target[qid] = dst._inner
        if len(
            {(loc.zone_id, loc.word_id, loc.site_id) for loc in target.values()}
        ) < len(target):
            if n == 0:
                return None  # the rule's own assignment collides: not modelled
            continue  # two movers onto one site
        candidates.append(target)
    return candidates, 0


def record_nohome(case, arch: ArchSpec, cap: int) -> list[Stage]:
    strategy = make_physical_placement_strategy(arch_spec=arch)
    inner = getattr(strategy, "inner", strategy)
    assert isinstance(inner, NoHomePlacementStrategy), type(inner)
    stages: list[Stage] = []
    original = inner._invoke_placement

    def spy(engine, move_search, initial, cz_pairs, blocked, future):
        result = original(engine, move_search, initial, cz_pairs, blocked, future)
        built = _mover_candidates(arch, dict(initial), list(cz_pairs), cap, len(stages))
        if built is not None:
            candidates, today = built
            key = lambda d: {q: (v.zone_id, v.word_id, v.site_id) for q, v in d.items()}
            matches = (
                key(dict(result.goal_config)) == key(candidates[today])
                if result.status == _native.SolveStatus.SOLVED
                else None
            )
            stages.append(
                Stage(
                    case=case.case_id,
                    index=len(stages),
                    initial=dict(initial),
                    blocked=list(blocked),
                    candidates=candidates,
                    today=today,
                    max_expansions=inner.max_expansions,
                    engine=engine,
                    router=move_search,
                    rule_matches=matches,
                )
            )
        return result

    object.__setattr__(inner, "_invoke_placement", spy)
    _compile(case, lambda: strategy)
    return stages


# ── Space B: the Python target generators ──


def record_generator(
    case, arch: ArchSpec, generator: TargetGeneratorABC
) -> list[Stage]:
    inner = PhysicalPlacementStrategy(
        arch_spec=arch, traversal=RustPlacementTraversal(), target_generator=generator
    )
    strategy = PalindromePlacementStrategy(inner=inner)
    stages: list[Stage] = []
    original = inner._build_candidates
    router = _move_search_from_traversal(inner.traversal)

    def spy(ctx: TargetContext):
        candidates = original(ctx)
        participants = set(ctx.controls) | set(ctx.targets)
        placement = ctx.placement
        initial = {q: loc._inner for q, loc in placement.items() if q in participants}
        blocked = [loc._inner for loc in ctx.state.occupied] + [
            loc._inner for q, loc in placement.items() if q not in participants
        ]
        stages.append(
            Stage(
                case=case.case_id,
                index=len(stages),
                initial=initial,
                blocked=blocked,
                candidates=[
                    {q: loc._inner for q, loc in c.items() if q in participants}
                    for c in candidates
                ],
                today=-1,  # the first candidate that routes; set in `evaluate`
                max_expansions=inner.traversal.max_expansions,
                engine=inner._get_engine(),
                router=router,
            )
        )
        return candidates

    object.__setattr__(inner, "_build_candidates", spy)
    _compile(case, lambda: strategy)
    return stages


# ── Evaluation ──

_PUSH_ROTATE = _native.MoveSearch.ids().with_options(
    _native.SolveOptions(strategy=_native.SearchStrategy.PUSH_ROTATE)
)


def _ops(result: _native.SolveResult) -> int | None:
    return (
        len(result.move_layers) if result.status == _native.SolveStatus.SOLVED else None
    )


def evaluate(stage: Stage) -> None:
    route = _native.TargetSolver(stage.engine, stage.router)
    rank = _native.TargetSolver(stage.engine, _PUSH_ROTATE)
    for target in stage.candidates:
        stage.route_ops.append(
            _ops(
                route.solve(stage.initial, target, stage.blocked, stage.max_expansions)
            )
        )
        stage.pr_ops.append(
            _ops(rank.solve(stage.initial, target, stage.blocked, None))
        )
    if stage.today < 0:
        routed = [i for i, ops in enumerate(stage.route_ops) if ops is not None]
        stage.today = routed[0] if routed else 0


@dataclass
class Summary:
    space: str
    stages: int = 0
    stages_with_choice: int = 0
    today_ops: int = 0
    ranked_ops: int = 0
    best_ops: int = 0
    regressions: int = 0
    rescues: int = 0
    unsolved_today: int = 0
    rule_mismatches: int = 0

    @property
    def improvement(self) -> float:
        return 1.0 - self.ranked_ops / self.today_ops if self.today_ops else 0.0

    @property
    def captured(self) -> float:
        gap = self.today_ops - self.best_ops
        return (self.today_ops - self.ranked_ops) / gap if gap else 0.0

    @property
    def go(self) -> bool:
        return self.improvement >= GO_THRESHOLD and self.regressions == 0


def summarise(space: str, stages: list[Stage]) -> Summary:
    s = Summary(space)
    for st in stages:
        s.stages += 1
        if st.rule_matches is False:
            s.rule_mismatches += 1
        today = st.route_ops[st.today]
        ranked_i = min(
            (i for i, ops in enumerate(st.pr_ops) if ops is not None),
            key=lambda i: (st.pr_ops[i], i),
            default=st.today,
        )
        ranked = st.route_ops[ranked_i]
        routed = [ops for ops in st.route_ops if ops is not None]
        if len(routed) >= 2:
            s.stages_with_choice += 1
        if today is None:
            s.unsolved_today += 1
            if ranked is not None:
                s.rescues += 1
            continue
        if ranked is None:
            s.regressions += 1
            continue
        s.today_ops += today
        s.ranked_ops += ranked
        s.best_ops += min(routed)
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--cases", default=None, help="comma-separated case names")
    parser.add_argument("--max-assignments", type=int, default=64)
    parser.add_argument("--output", default=None, help="JSON summary path")
    args = parser.parse_args()
    # The physical architecture compiles without logical initialization, as
    # `benchmarks.cli` does for `--architecture physical`.
    cases = tuple(
        replace(case, logical_initialize=False)
        for case in select_benchmark_cases(
            set(args.cases.split(",")) if args.cases else None
        )
    )
    arch = physical.get_arch_spec()

    spaces: dict[str, Callable[[Any], list[Stage]]] = {
        "nohome": lambda case: record_nohome(case, arch, args.max_assignments),
        "generator:congestion_aware": lambda case: record_generator(
            case, arch, CongestionAwareTargetGenerator()
        ),
        "generator:aod_cluster": lambda case: record_generator(
            case, arch, AODClusterTargetGenerator()
        ),
        "generator:lookahead_congestion_aware": lambda case: record_generator(
            case, arch, LookaheadCongestionAwareTargetGenerator()
        ),
    }
    report: dict[str, Any] = {"threshold": GO_THRESHOLD, "spaces": {}}
    for name, record in spaces.items():
        per_case: dict[str, Any] = {}
        all_stages: list[Stage] = []
        for case in cases:
            start = time.perf_counter()
            try:
                stages = record(case)
            # A kernel this space cannot compile.
            except Exception as exc:  # noqa: BLE001
                per_case[case.case_id] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            for st in stages:
                evaluate(st)
            summary = summarise(name, stages)
            per_case[case.case_id] = {
                **asdict(summary),
                "improvement": summary.improvement,
                "seconds": round(time.perf_counter() - start, 1),
            }
            all_stages.extend(stages)
            print(f"{name:40} {case.case_id:20} {per_case[case.case_id]}", flush=True)
        total = summarise(name, all_stages)
        report["spaces"][name] = {
            "total": {
                **asdict(total),
                "improvement": total.improvement,
                "captured": total.captured,
                "go": total.go,
            },
            "cases": per_case,
        }
        print(
            f"== {name}: today {total.today_ops} ops, ranked {total.ranked_ops}, "
            f"best {total.best_ops}; improvement {100 * total.improvement:.1f}%, "
            f"captured {100 * total.captured:.0f}%, regressions {total.regressions}, "
            f"rescues {total.rescues} -> {'GO' if total.go else 'NO-GO'}",
            flush=True,
        )
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
