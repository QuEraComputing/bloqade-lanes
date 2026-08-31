import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "bloqade.lanes.dialects.move",
        "bloqade.lanes.prelude",
        "bloqade.lanes.validation.address",
        "bloqade.gemini.device.logical.utils",
    ],
)
def test_lanes_module_imports_in_fresh_interpreter(module: str):
    """Public Lanes modules import without relying on prior import order."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib; importlib.import_module({module!r})",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
