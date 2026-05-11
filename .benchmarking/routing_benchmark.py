#!/usr/bin/env python
"""Benchmark routing algorithms on random CZ placement problems.

Generates random initial placements and CZ gate sets on scalable
architectures, then times the Rust solver across multiple strategies
and problem sizes.  Results are written to a single CSV per architecture
that can be visualised with ``.benchmarking/plot_all.py``.

Strategies are defined once in a shared pool and experiments are views
into that pool, so each strategy is only benchmarked once even if it
appears in multiple experiments.

Usage:
    uv run python .benchmarking/routing_benchmark.py                     # all experiments
    uv run python .benchmarking/routing_benchmark.py exp1a exp3          # specific experiments
    uv run python .benchmarking/routing_benchmark.py --trials 10 exp1a   # quick smoke test
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from bloqade.lanes.arch.builder import build_arch
from bloqade.lanes.arch.topology import (
    DiagonalWordTopology,
    HypercubeSiteTopology,
    HypercubeWordTopology,
)
from bloqade.lanes.arch.zone import ArchBlueprint, DeviceLayout, ZoneSpec
from bloqade.lanes.bytecode._native import MoveSolver, SearchStrategy, SolveOptions
from bloqade.lanes.layout import LocationAddress
from bloqade.lanes.layout.arch import ArchSpec

# ── Architecture ────────────────────────────────────────────────────


def make_arch(num_rows: int, num_cols: int = 2, sites_per_word: int = 16) -> ArchSpec:
    """Build a scalable physical architecture."""
    if num_cols == 2:
        word_topo = DiagonalWordTopology()
    else:
        word_topo = HypercubeWordTopology()

    bp = ArchBlueprint(
        zones={
            "gate": ZoneSpec(
                num_rows=num_rows,
                num_cols=num_cols,
                entangling=True,
                word_topology=word_topo,
                site_topology=HypercubeSiteTopology(),
            )
        },
        layout=DeviceLayout(sites_per_word=sites_per_word),
    )
    return build_arch(bp).arch


# ── Problem generation ──────────────────────────────────────────────


@dataclass
class RoutingProblem:
    """A single CZ routing problem instance."""

    arch: ArchSpec
    initial: dict[int, LocationAddress]
    target: dict[int, LocationAddress]
    blocked: frozenset[LocationAddress]
    n_qubits: int
    n_cz_pairs: int


def generate_problem(
    arch: ArchSpec,
    n_qubits: int,
    n_cz_pairs: int,
    rng: random.Random,
) -> RoutingProblem:
    """Generate a random CZ routing problem with realistic home positions."""
    n_words = len(arch.words)
    home_sites = list(arch._inner.zone_by_id(0).site_buses[0].src)

    home_locs = [LocationAddress(w, s) for w in range(n_words) for s in home_sites]
    assert n_qubits <= len(
        home_locs
    ), f"n_qubits ({n_qubits}) exceeds available home sites ({len(home_locs)})"

    chosen = rng.sample(home_locs, n_qubits)
    placement = dict(enumerate(chosen))
    occupied = {(loc.word_id, loc.site_id) for loc in chosen}

    qubit_ids = list(range(n_qubits))
    rng.shuffle(qubit_ids)

    cz_controls: list[int] = []
    cz_targets: list[int] = []
    used: set[int] = set()
    vacated: set[tuple[int, int]] = set()

    for i in range(0, len(qubit_ids) - 1, 2):
        if len(cz_controls) >= n_cz_pairs:
            break
        c, t = qubit_ids[i], qubit_ids[i + 1]
        partner = arch.get_blockaded_location(placement[t])
        if partner is None:
            continue
        partner_key = (partner.word_id, partner.site_id)
        if partner_key in occupied and partner_key not in vacated:
            occupant = next(
                (
                    q
                    for q, loc in placement.items()
                    if (loc.word_id, loc.site_id) == partner_key
                ),
                None,
            )
            if occupant is not None and occupant not in used:
                continue

        cz_controls.append(c)
        cz_targets.append(t)
        used.add(c)
        used.add(t)
        c_loc = placement[c]
        vacated.add((c_loc.word_id, c_loc.site_id))

    target_config = dict(placement)
    for c_qid, t_qid in zip(cz_controls, cz_targets):
        partner = arch.get_blockaded_location(placement[t_qid])
        if partner is not None:
            target_config[c_qid] = partner

    blocked = frozenset(placement[qid] for qid in range(n_qubits) if qid not in used)

    return RoutingProblem(
        arch=arch,
        initial=placement,
        target=target_config,
        blocked=blocked,
        n_qubits=n_qubits,
        n_cz_pairs=len(cz_controls),
    )


# ── Solver wrappers ─────────────────────────────────────────────────


@dataclass
class SolveResult:
    """Uniform result across solvers."""

    solved: bool
    time_s: float
    n_move_layers: int = 0
    n_atom_moves: int = 0
    status: str = ""


_BENCH_STRATEGY_MAP: dict[str, SearchStrategy] = {
    "astar": SearchStrategy.ASTAR,
    "dfs": SearchStrategy.DFS,
    "bfs": SearchStrategy.BFS,
    "greedy": SearchStrategy.GREEDY,
    "ids": SearchStrategy.IDS,
    "cascade": SearchStrategy.CASCADE_IDS,
    "cascade-ids": SearchStrategy.CASCADE_IDS,
    "cascade-dfs": SearchStrategy.CASCADE_DFS,
    "cascade-entropy": SearchStrategy.CASCADE_ENTROPY,
    "entropy": SearchStrategy.ENTROPY,
}


def solve_rust(
    problem: RoutingProblem,
    solver: MoveSolver,
    strat: StrategyConfig,
) -> SolveResult:
    """Run the Rust MoveSolver with the given strategy config."""
    opts = SolveOptions(
        strategy=_BENCH_STRATEGY_MAP[strat.strategy],
        max_movesets_per_group=strat.max_movesets_per_group,
        weight=strat.weight,
        restarts=strat.restarts,
        lookahead=strat.lookahead,
    )

    t0 = time.perf_counter()
    result = solver.solve(
        problem.initial,
        problem.target,
        list(problem.blocked),
        strat.max_expansions,
        options=opts,
    )
    dt = time.perf_counter() - t0

    if result.status == "solved":
        n_atom_moves = sum(len(layer) for layer in result.move_layers)
        return SolveResult(
            solved=True,
            time_s=dt,
            n_move_layers=len(result.move_layers),
            n_atom_moves=n_atom_moves,
            status="solved",
        )
    return SolveResult(solved=False, time_s=dt, status=result.status)


# ── Strategy & Experiment definitions ───────────────────────────────


@dataclass
class StrategyConfig:
    """Configuration for a single benchmark strategy."""

    name: str
    strategy: str = "ids"
    weight: float = 1.0
    max_expansions: int | None = 500
    restarts: int = 1
    lookahead: bool = False
    deadlock_policy: str = "skip"
    max_movesets_per_group: int = 3


@dataclass
class ArchConfig:
    """Architecture configuration."""

    name: str
    rows: int
    cols: int
    sites_per_word: int = 16


ARCH_SMALL = ArchConfig("small", rows=4, cols=4)
DEFAULT_ARCHS = [ARCH_SMALL]

RESTARTS = 28  # 2x CPU cores (14)


# ── Shared strategy pool ──────────────────────────────────────────────
# Each strategy is defined once.  Experiments reference them by name.

STRATEGIES: dict[str, StrategyConfig] = {
    # IDS variants
    "ids-base": StrategyConfig("ids-base", strategy="ids"),
    "ids-la": StrategyConfig("ids-la", strategy="ids", lookahead=True),
    "ids-r28": StrategyConfig("ids-r28", strategy="ids", restarts=RESTARTS),
    "ids-r28-la": StrategyConfig(
        "ids-r28-la", strategy="ids", restarts=RESTARTS, lookahead=True
    ),
    # Entropy variants
    "entropy-base": StrategyConfig("entropy-base", strategy="entropy"),
    "entropy-la": StrategyConfig("entropy-la", strategy="entropy", lookahead=True),
    "entropy-r28": StrategyConfig("entropy-r28", strategy="entropy", restarts=RESTARTS),
    "entropy-r28-la": StrategyConfig(
        "entropy-r28-la", strategy="entropy", restarts=RESTARTS, lookahead=True
    ),
    "ids-r28-mmg10": StrategyConfig(
        "ids-r28-mmg10", strategy="ids", restarts=RESTARTS, max_movesets_per_group=10
    ),
    # Deadlock policy variants
    "ids-r28-mb": StrategyConfig(
        "ids-r28-mb", strategy="ids", restarts=RESTARTS, deadlock_policy="move_blockers"
    ),
    "ids-r28-la-mb": StrategyConfig(
        "ids-r28-la-mb",
        strategy="ids",
        restarts=RESTARTS,
        lookahead=True,
        deadlock_policy="move_blockers",
    ),
    # Cascade variants (higher budget for the A* refinement phase)
    "cascade-ids-r28": StrategyConfig(
        "cascade-ids-r28",
        strategy="cascade-ids",
        restarts=RESTARTS,
        max_expansions=5000,
    ),
    "cascade-ids-r28-la": StrategyConfig(
        "cascade-ids-r28-la",
        strategy="cascade-ids",
        restarts=RESTARTS,
        lookahead=True,
        max_expansions=5000,
    ),
    # A* variants (10x budget to give A* a fair chance)
    "astar-w1": StrategyConfig(
        "astar-w1", strategy="astar", weight=1.0, max_expansions=5000
    ),
    "astar-w30": StrategyConfig(
        "astar-w30", strategy="astar", weight=30.0, max_expansions=5000
    ),
    "astar-w300": StrategyConfig(
        "astar-w300", strategy="astar", weight=300.0, max_expansions=5000
    ),
}


# ── Experiment definitions (views into the shared strategy pool) ────


@dataclass
class Experiment:
    """A benchmark experiment — a named view over shared strategies."""

    name: str
    description: str
    strategy_names: list[str]
    archs: list[ArchConfig] = field(default_factory=lambda: list(DEFAULT_ARCHS))
    qubit_fractions: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5]
    )


ALL_EXPERIMENTS: dict[str, Experiment] = {
    "exp1a": Experiment(
        name="exp1a_ids_ablation",
        description="IDS Feature Ablation",
        strategy_names=["ids-base", "ids-la", "ids-r28", "ids-r28-la"],
    ),
    "exp1b": Experiment(
        name="exp1b_entropy_ablation",
        description="Entropy Feature Ablation",
        strategy_names=["entropy-base", "entropy-la", "entropy-r28", "entropy-r28-la"],
    ),
    "exp1c": Experiment(
        name="exp1c_ids_vs_entropy",
        description="IDS vs Entropy",
        strategy_names=[
            "ids-base",
            "ids-la",
            "ids-r28",
            "ids-r28-la",
            "entropy-base",
            "entropy-la",
            "entropy-r28",
            "entropy-r28-la",
        ],
    ),
    "exp2": Experiment(
        name="exp2_expander_params",
        description="Expander Parameters (max_movesets_per_group)",
        strategy_names=["ids-r28", "ids-r28-mmg10"],
    ),
    "exp3": Experiment(
        name="exp3_deadlock_policy",
        description="Deadlock Policy Comparison",
        strategy_names=["ids-r28", "ids-r28-mb", "ids-r28-la", "ids-r28-la-mb"],
    ),
    "exp4": Experiment(
        name="exp4_cascade",
        description="Cascade vs No Cascade (IDS)",
        strategy_names=[
            "ids-r28",
            "cascade-ids-r28",
            "ids-r28-la",
            "cascade-ids-r28-la",
        ],
    ),
    "exp5": Experiment(
        name="exp5_weighted_astar",
        description="Weighted A* vs IDS vs Entropy",
        strategy_names=[
            "astar-w1",
            "astar-w30",
            "astar-w300",
            "ids-base",
            "entropy-base",
        ],
    ),
}


# ── Benchmark runner ────────────────────────────────────────────────

CSV_COLUMNS = [
    "arch",
    "strategy",
    "n_qubits",
    "n_cz_pairs",
    "trial",
    "solved",
    "time_s",
    "n_move_layers",
    "n_atom_moves",
    "status",
]


def run_benchmarks(
    experiments: list[Experiment],
    n_trials: int = 200,
    seed: int = 42,
    cz_fraction: float = 0.25,
    output_dir: str = ".benchmarking/results",
) -> list[Path]:
    """Run all requested experiments, deduplicating shared strategies.

    Collects unique (strategy, arch) pairs across all requested experiments
    and runs each once, writing a single CSV per architecture.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect unique strategy names and arch configs across all experiments.
    unique_strat_names: set[str] = set()
    arch_map: dict[str, ArchConfig] = {}
    # Per-arch qubit fractions: use the union (superset) across experiments.
    arch_fractions: dict[str, list[float]] = {}

    for exp in experiments:
        unique_strat_names.update(exp.strategy_names)
        for arch_cfg in exp.archs:
            arch_map[arch_cfg.name] = arch_cfg
            existing = set(arch_fractions.get(arch_cfg.name, []))
            existing.update(exp.qubit_fractions)
            arch_fractions[arch_cfg.name] = sorted(existing)

    strat_list = sorted(unique_strat_names)
    csv_paths: list[Path] = []

    for arch_name, arch_cfg in sorted(arch_map.items()):
        arch = make_arch(arch_cfg.rows, arch_cfg.cols, arch_cfg.sites_per_word)
        total_sites = len(arch.words) * arch_cfg.sites_per_word
        max_q = total_sites // 2
        fractions = arch_fractions[arch_name]
        qubit_counts = sorted(set(max(4, int(max_q * f)) for f in fractions))

        rust_solver = MoveSolver.from_arch_spec(arch._inner)

        csv_path = output_path / f"results_{arch_name}.csv"
        csv_paths.append(csv_path)

        print(f"\n{'='*60}")
        print(
            f"  Architecture: {arch_name} ({len(arch.words)}w x {arch_cfg.sites_per_word}s)"
        )
        print(f"  Strategies: {len(strat_list)}")
        print(f"  Qubit counts: {qubit_counts}")
        print(f"  Trials: {n_trials}")
        print(f"{'='*60}")

        skipped: set[str] = set()
        total_combos = len(qubit_counts) * len(strat_list)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            f.flush()

            pbar = tqdm(
                total=total_combos,
                desc=f"  {arch_name}",
                bar_format="  {desc} |{bar:30}| {n}/{total} [{elapsed}<{remaining}]",
                file=sys.stdout,
            )

            for n_q in qubit_counts:
                n_pairs = max(1, int(n_q * cz_fraction / 2))
                rng = random.Random(seed)
                problems = [
                    generate_problem(arch, n_q, n_pairs, rng) for _ in range(n_trials)
                ]

                for strat_name in strat_list:
                    strat = STRATEGIES[strat_name]

                    if strat_name in skipped:
                        pbar.set_postfix_str(f"{strat_name} {n_q}q skip")
                        pbar.update(1)
                        continue

                    pbar.set_postfix_str(f"{strat_name} {n_q}q")

                    solved_count = 0
                    for trial_idx, p in enumerate(problems):
                        r = solve_rust(p, rust_solver, strat)

                        writer.writerow(
                            {
                                "arch": arch_name,
                                "strategy": strat_name,
                                "n_qubits": n_q,
                                "n_cz_pairs": p.n_cz_pairs,
                                "trial": trial_idx,
                                "solved": int(r.solved),
                                "time_s": f"{r.time_s:.6f}",
                                "n_move_layers": r.n_move_layers,
                                "n_atom_moves": r.n_atom_moves,
                                "status": r.status,
                            }
                        )
                        if r.solved:
                            solved_count += 1

                    f.flush()

                    rate = solved_count / n_trials
                    pbar.update(1)
                    if rate < 0.9:
                        skipped.add(strat_name)

            pbar.close()

        print(f"  -> {csv_path}")

    return csv_paths


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run routing benchmark experiments")
    parser.add_argument(
        "experiments",
        nargs="*",
        default=list(ALL_EXPERIMENTS.keys()),
        help=f"Experiments to run (default: all). Choices: {', '.join(ALL_EXPERIMENTS.keys())}",
    )
    parser.add_argument(
        "--trials", type=int, default=1000, help="Trials per (strategy, size)"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--output-dir",
        default=".benchmarking/results",
        help="Output directory for CSVs",
    )
    args = parser.parse_args()

    selected: list[Experiment] = []
    for exp_name in args.experiments:
        if exp_name not in ALL_EXPERIMENTS:
            print(f"Unknown experiment: {exp_name}")
            print(f"Available: {', '.join(ALL_EXPERIMENTS.keys())}")
            return
        selected.append(ALL_EXPERIMENTS[exp_name])

    t_start = time.perf_counter()
    paths = run_benchmarks(
        selected,
        n_trials=args.trials,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    elapsed = time.perf_counter() - t_start
    print(f"\nAll done in {elapsed:.1f}s. CSVs written:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
