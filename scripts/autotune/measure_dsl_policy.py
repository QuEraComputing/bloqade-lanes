"""Measure command for autotune: run the DSL policy through the benchmark harness.

Invokes `python -m benchmarks.cli` with the `dsl_autotune` strategy against a
curated kernel subset, then prints the resulting CSV to stdout for the
companion adaptor script to parse.

The CSV format (one row per case+strategy) is defined by
`python/benchmarks/harness/output.py`.

Environment:
  BLOQADE_DSL_POLICY      — relative path to the .star policy under evaluation.
                            Defaults to `policies/autotune/candidate.star`.
  BLOQADE_DSL_KERNELS     — comma-separated list of kernel case_ids to run.
                            Defaults to `ghz_4,ghz_6,adder_4,steane_logical_5`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_KERNELS = (
    "ghz_4,ghz_6,adder_4,steane_logical_5,"
    "qpe_9,"
    "adder_64,bv_70,steane_physical_35,trotter_rand_35"
)


def main() -> int:
    kernels = os.environ.get("BLOQADE_DSL_KERNELS", DEFAULT_KERNELS)
    policy = os.environ.get("BLOQADE_DSL_POLICY", "policies/autotune/candidate.star")

    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "dsl_run.csv"
        env = os.environ.copy()
        env["BLOQADE_DSL_POLICY"] = policy

        cmd = [
            sys.executable,
            "-m",
            "benchmarks.cli",
            "--cases",
            kernels,
            "--strategies",
            "dsl_autotune",
            "--architecture",
            "physical",
            "--repeats",
            "1",
            "--output",
            str(csv_path),
        ]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            sys.stderr.write(f"benchmarks.cli exited with {result.returncode}\n")
            return result.returncode

        if not csv_path.exists():
            sys.stderr.write(f"expected CSV at {csv_path}, not produced\n")
            return 1

        sys.stdout.write(csv_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
