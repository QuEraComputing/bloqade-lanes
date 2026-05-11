from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from bloqade.lanes.visualize.entropy_tree.cli import run  # noqa: E402


def test_run_no_interactive_with_default_kernel_succeeds():
    exit_code = run(["--no-interactive", "--max-expansions", "50"])
    assert exit_code == 0
