from __future__ import annotations

import argparse
import cProfile
import csv
import math
import pstats
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

Ret = TypeVar("Ret")


def _add_repo_paths() -> None:
    """Make the demo package and sibling decoder checkout importable."""

    project_root_candidates = [Path.cwd(), Path.cwd().parent]
    for candidate in project_root_candidates:
        candidate = candidate.resolve()
        if (candidate / "demo" / "msd_utils").exists():
            sys.path.insert(0, str(candidate))
            break
    else:
        raise FileNotFoundError("Could not locate repo root containing demo/msd_utils.")

    decoder_src_candidates = [
        Path.cwd() / ".." / "bloqade-decoders" / "src",
        Path.cwd() / "bloqade-decoders" / "src",
        Path.cwd().parent / "bloqade-decoders" / "src",
        Path.cwd().parent.parent / "bloqade-decoders" / "src",
    ]
    for candidate in decoder_src_candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            return


_add_repo_paths()

from bloqade.decoders import GurobiDecoder, TableDecoder  # noqa: E402
from demo.msd_extras.qet import (  # noqa: E402
    QET_VALID_POSTSELECTION_PATTERNS,
    bloch_vector_from_state,
    build_qet_primitives,
    build_qet_target_state,
)
from demo.msd_utils import (  # noqa: E402
    DecoderCurveOptions,
    MSDDecoderWorkflowConfig,
    SparseTableDecoder,
    SyndromeLayout,
    build_mle_decoder_suite,
    build_msd_tomography_kernels,
    build_msd_tomography_tasks,
    evaluate_decoder_curves,
    sample_actual_data,
    train_mld_decoder_suite,
)

from bloqade.lanes import GeminiLogicalSimulator  # noqa: E402


def _parse_chunk_size(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("chunk size must be positive or 'none'.")
    return parsed


def _profile_call(
    label: str,
    fn: Callable[[], Ret],
    *,
    output_dir: Path,
    sort_by: str,
    limit: int,
) -> tuple[Ret, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / f"{label}.prof"
    text_path = output_dir / f"{label}.txt"

    profiler = cProfile.Profile()
    start = time.perf_counter()
    result = profiler.runcall(fn)
    elapsed = time.perf_counter() - start
    profiler.dump_stats(profile_path)

    with text_path.open("w") as f:
        stats = pstats.Stats(profiler, stream=f).strip_dirs().sort_stats(sort_by)
        stats.print_stats(limit)

    print(f"[profile] {label}: wall={elapsed:.3f}s")
    print(f"[profile] wrote {profile_path}")
    print(f"[profile] wrote {text_path}")
    return result, elapsed


def _build_config(
    args: argparse.Namespace, train_shots: int, eval_shots: int
) -> MSDDecoderWorkflowConfig:
    theta = 0.25 * math.pi
    phi0 = 0.4 * math.pi
    phi1 = 0.6 * math.pi
    phi2 = 0.9 * math.pi
    psi_in = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

    target_state = build_qet_target_state(
        theta=theta,
        phi0=phi0,
        phi1=phi1,
        phi2=phi2,
        psi_in=psi_in,
    )
    target_bloch_vector = bloch_vector_from_state(target_state)
    decoder_primitive_set = build_qet_primitives(theta, phi0, phi1, phi2)

    rank_train_shots = (
        train_shots if args.mld_rank_train_shots is None else args.mld_rank_train_shots
    )
    return MSDDecoderWorkflowConfig(
        mld_train_shots=train_shots,
        mld_rank_train_shots=rank_train_shots,
        eval_shots=eval_shots,
        target_bloch_vector=target_bloch_vector,
        theta=theta,
        phi=phi0,
        lam=0.0,
        decoder_primitive_set=decoder_primitive_set,
        valid_factory_targets=QET_VALID_POSTSELECTION_PATTERNS,
        num_logical_qubits=9,
        output_qubit=0,
        special_kernel_strategy=args.special_kernel_strategy,
        sim_type=args.sim_type,
        chunk_size=args.chunk_size,
        binary_precision=args.binary_precision,
        uncertainty_backend=args.uncertainty_backend,
        max_grid_points=args.max_grid_points,
        layout=SyndromeLayout(output_detector_count=3, output_observable_count=1),
        sign_vector=(1.0, -1.0, -1.0),
        log=args.log,
    )


def _table_decoder_cls(name: str):
    if name == "table":
        return TableDecoder
    if name == "sparse":
        return SparseTableDecoder
    raise ValueError(f"Unknown table decoder: {name}")


def _max_point_fidelity(curve: dict[str, np.ndarray]) -> float:
    values = curve.get("point_fidelity", curve["fidelity"])
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    return float(np.nanmax(values))


def _plot_summary(rows: list[dict[str, float]], output_dir: Path) -> Path:
    eval_groups: dict[float, list[dict[str, float]]] = {}
    for row in rows:
        eval_groups.setdefault(row["eval_shots"], []).append(row)

    fig, ax_time = plt.subplots(figsize=(8.5, 5.2))
    ax_fidelity = ax_time.twinx()

    all_train_shots: list[float] = []
    for eval_shots in sorted(eval_groups):
        group = sorted(eval_groups[eval_shots], key=lambda r: r["train_shots"])
        shot_values = np.array([r["train_shots"] for r in group], dtype=np.float64)
        all_train_shots.extend(shot_values.tolist())
        mld_times = np.array([r["mld_total_seconds"] for r in group], dtype=np.float64)
        mle_times = np.array([r["mle_total_seconds"] for r in group], dtype=np.float64)
        mld_fidelities = np.array(
            [r["mld_max_point_fidelity"] for r in group], dtype=np.float64
        )
        mle_fidelities = np.array(
            [r["mle_max_point_fidelity"] for r in group], dtype=np.float64
        )

        suffix = "" if len(eval_groups) == 1 else f" (eval={int(eval_shots):,})"
        ax_time.plot(
            shot_values, mld_times, marker="o", label=f"MLD time{suffix}", color="C0"
        )
        ax_time.plot(
            shot_values, mle_times, marker="o", label=f"MLE time{suffix}", color="C1"
        )
        ax_fidelity.plot(
            shot_values,
            mld_fidelities,
            marker="s",
            linestyle="--",
            label=f"MLD max point fidelity{suffix}",
            color="C0",
        )
        ax_fidelity.plot(
            shot_values,
            mle_fidelities,
            marker="s",
            linestyle="--",
            label=f"MLE max point fidelity{suffix}",
            color="C1",
        )

    ax_time.set_xlabel("MLD table-training shots")
    ax_time.set_ylabel("Wall time (s)")
    ax_fidelity.set_ylabel("Max point fidelity")
    ax_time.set_title("QET Decoder Scaling")
    if len(np.unique(np.array(all_train_shots))) > 1:
        ax_time.set_xscale("log")
    ax_time.grid(True, alpha=0.25)

    lines_time, labels_time = ax_time.get_legend_handles_labels()
    lines_fid, labels_fid = ax_fidelity.get_legend_handles_labels()
    ax_time.legend(lines_time + lines_fid, labels_time + labels_fid, loc="best")

    plot_path = output_dir / "qet_decoder_scaling.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _plot_eval_shots_summary(rows: list[dict[str, float]], output_dir: Path) -> Path:
    train_groups: dict[float, list[dict[str, float]]] = {}
    for row in rows:
        train_groups.setdefault(row["train_shots"], []).append(row)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    all_eval_shots: list[float] = []
    multi_train = len(train_groups) > 1
    for train_shots in sorted(train_groups):
        group = sorted(train_groups[train_shots], key=lambda r: r["eval_shots"])
        eval_values = np.array([r["eval_shots"] for r in group], dtype=np.float64)
        all_eval_shots.extend(eval_values.tolist())
        sample_times = np.array(
            [r["sample_actual_data_seconds"] for r in group], dtype=np.float64
        )
        mld_curve_times = np.array(
            [r["mld_curve_seconds"] for r in group], dtype=np.float64
        )
        mle_curve_times = np.array(
            [r["mle_curve_seconds"] for r in group], dtype=np.float64
        )

        suffix = "" if not multi_train else f" (train={int(train_shots):,})"
        ax.plot(
            eval_values,
            sample_times,
            marker="^",
            label=f"sample_actual_data{suffix}",
            color="C2",
        )
        ax.plot(
            eval_values,
            mld_curve_times,
            marker="o",
            label=f"MLD curve{suffix}",
            color="C0",
        )
        ax.plot(
            eval_values,
            mle_curve_times,
            marker="o",
            label=f"MLE curve{suffix}",
            color="C1",
        )

    ax.set_xlabel("Evaluation shots")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("QET Decoder Eval-Shots Scaling")
    if len(np.unique(np.array(all_eval_shots))) > 1:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    plot_path = output_dir / "qet_decoder_eval_shots_scaling.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _write_summary_csv(rows: list[dict[str, float]], output_dir: Path) -> Path:
    csv_path = output_dir / "qet_decoder_scaling_summary.csv"
    fieldnames = [
        "train_shots",
        "rank_train_shots",
        "eval_shots",
        "sample_actual_data_seconds",
        "mld_train_seconds",
        "mld_curve_seconds",
        "mld_total_seconds",
        "mld_max_point_fidelity",
        "mle_build_seconds",
        "mle_curve_seconds",
        "mle_total_seconds",
        "mle_max_point_fidelity",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile QET MLD/MLE decoder scaling versus MLD table-training shots."
        )
    )
    parser.add_argument(
        "--train-shots",
        type=int,
        nargs="+",
        required=True,
        help="One or more MLD table-training shot counts to benchmark.",
    )
    parser.add_argument(
        "--mld-rank-train-shots",
        type=int,
        default=None,
        help="MLD ranking shots. Defaults to the current --train-shots value.",
    )
    parser.add_argument(
        "--eval-shots",
        type=int,
        nargs="+",
        default=[1_000_000],
        help="One or more evaluation shot counts to benchmark.",
    )
    parser.add_argument(
        "--chunk-size",
        type=_parse_chunk_size,
        default=None,
        help="Maximum shots per simulator call, or 'none' for no chunking.",
    )
    parser.add_argument("--sim-type", choices=["tsim", "clifft"], default="clifft")
    parser.add_argument(
        "--table-decoder",
        choices=["table", "sparse"],
        default="table",
        help="MLD table decoder implementation.",
    )
    parser.add_argument(
        "--special-kernel-strategy",
        choices=["compiled_inverse_prefix", "prefix_prepare"],
        default="compiled_inverse_prefix",
    )
    parser.add_argument("--binary-precision", type=int, default=4)
    parser.add_argument("--uncertainty-backend", default="wilson")
    parser.add_argument("--max-grid-points", type=int, default=1_500_000)
    parser.add_argument("--threshold-points", type=int, default=24)
    parser.add_argument("--min-accepted-per-basis", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo/results/qet_decoder_scaling"),
    )
    parser.add_argument("--sort-by", default="cumtime")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--log", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_shots_values = sorted(set(args.train_shots))
    eval_shots_values = sorted(set(args.eval_shots))
    table_decoder_cls = _table_decoder_cls(args.table_decoder)
    simulator = GeminiLogicalSimulator()

    curve_options = DecoderCurveOptions(
        threshold_points=args.threshold_points,
        threshold_policy="quantile",
        selection_mode="threshold",
        min_accepted_per_basis=args.min_accepted_per_basis,
    )

    rows: list[dict[str, float]] = []
    for eval_shots in eval_shots_values:
        base_config = _build_config(args, train_shots_values[0], eval_shots)
        kernels = build_msd_tomography_kernels(base_config)
        tomography_tasks = build_msd_tomography_tasks(simulator, base_config, kernels)

        eval_dir = args.output_dir / f"eval_shots_{eval_shots}"
        eval_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n### eval_shots={eval_shots:,} — sampling shared actual data...",
            flush=True,
        )

        def run_sample_actual_data():
            return sample_actual_data(tomography_tasks, base_config, log=args.log)

        actual_data, sample_seconds = _profile_call(
            "sample_actual_data",
            run_sample_actual_data,
            output_dir=eval_dir,
            sort_by=args.sort_by,
            limit=args.limit,
        )

        for train_shots in train_shots_values:
            config = _build_config(args, train_shots, eval_shots)
            run_dir = eval_dir / f"train_shots_{train_shots}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"\n=== eval_shots={eval_shots:,} train_shots={train_shots:,} ===",
                flush=True,
            )

            def run_train_mld():
                return train_mld_decoder_suite(
                    tomography_tasks,
                    config,
                    table_decoder_cls=table_decoder_cls,
                    log=args.log,
                )

            mld_decoders, mld_train_seconds = _profile_call(
                "mld_train_decoder_suite",
                run_train_mld,
                output_dir=run_dir,
                sort_by=args.sort_by,
                limit=args.limit,
            )

            def run_mld_curve():
                return evaluate_decoder_curves(
                    actual_data,
                    {"MLD": mld_decoders},
                    config,
                    curve_options=curve_options,
                    log=args.log,
                )["MLD"]

            mld_curve, mld_curve_seconds = _profile_call(
                "mld_evaluate_curve",
                run_mld_curve,
                output_dir=run_dir,
                sort_by=args.sort_by,
                limit=args.limit,
            )

            def run_build_mle():
                return build_mle_decoder_suite(
                    tomography_tasks,
                    gurobi_decoder_cls=GurobiDecoder,
                    log=args.log,
                )

            mle_decoders, mle_build_seconds = _profile_call(
                "mle_build_decoder_suite",
                run_build_mle,
                output_dir=run_dir,
                sort_by=args.sort_by,
                limit=args.limit,
            )

            def run_mle_curve():
                return evaluate_decoder_curves(
                    actual_data,
                    {"MLE": mle_decoders},
                    config,
                    curve_options=curve_options,
                    log=args.log,
                )["MLE"]

            mle_curve, mle_curve_seconds = _profile_call(
                "mle_evaluate_curve",
                run_mle_curve,
                output_dir=run_dir,
                sort_by=args.sort_by,
                limit=args.limit,
            )

            rows.append(
                {
                    "train_shots": float(train_shots),
                    "rank_train_shots": float(config.resolved_mld_rank_train_shots),
                    "eval_shots": float(config.eval_shots),
                    "sample_actual_data_seconds": sample_seconds,
                    "mld_train_seconds": mld_train_seconds,
                    "mld_curve_seconds": mld_curve_seconds,
                    "mld_total_seconds": mld_train_seconds + mld_curve_seconds,
                    "mld_max_point_fidelity": _max_point_fidelity(mld_curve),
                    "mle_build_seconds": mle_build_seconds,
                    "mle_curve_seconds": mle_curve_seconds,
                    "mle_total_seconds": mle_build_seconds + mle_curve_seconds,
                    "mle_max_point_fidelity": _max_point_fidelity(mle_curve),
                }
            )

    csv_path = _write_summary_csv(rows, args.output_dir)
    plot_path = _plot_summary(rows, args.output_dir)
    eval_plot_path = _plot_eval_shots_summary(rows, args.output_dir)
    print(f"\n[summary] wrote {csv_path}")
    print(f"[summary] wrote {plot_path}")
    print(f"[summary] wrote {eval_plot_path}")


if __name__ == "__main__":
    main()
