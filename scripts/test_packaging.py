#!/usr/bin/env python3
"""Integration tests for Python wheel packaging (CLI + C FFI artifacts).

Usage: python scripts/test_packaging.py
"""

import platform
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

PASSED = 0
FAILED = 0


def passed(msg: str):
    global PASSED
    print(f"  PASS: {msg}")
    PASSED += 1


def failed(msg: str):
    global FAILED
    print(f"  FAIL: {msg}")
    FAILED += 1


def check(condition: bool, pass_msg: str, fail_msg: str):
    if condition:
        passed(pass_msg)
    else:
        failed(fail_msg)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT_DIR, check=True, text=True, **kwargs)


def shared_lib_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libbloqade_lanes_bytecode.dylib"
    elif system == "Linux":
        return "libbloqade_lanes_bytecode.so"
    else:
        return "bloqade_lanes_bytecode.dll"


def step1_stage_artifacts():
    print("=== Step 1: Stage CLI + C library artifacts ===")
    run(["just", "stage-clib"])

    dist = ROOT_DIR / "dist-data"
    cli_bin = dist / "scripts" / "bloqade-bytecode"
    check(
        cli_bin.exists() and bool(cli_bin.stat().st_mode & 0o111),
        "CLI binary staged",
        "CLI binary missing",
    )
    check(
        (dist / "data/lib/libbloqade_lanes_bytecode.a").exists(),
        "static lib staged",
        "static lib missing",
    )
    check(
        (dist / "data/include/bloqade_lanes_bytecode.h").exists(),
        "header staged",
        "header missing",
    )

    lib = shared_lib_name()
    check(
        (dist / "data/lib" / lib).exists(),
        f"shared lib staged ({lib})",
        f"shared lib missing ({lib})",
    )


def step2_build_and_inspect_wheel() -> Path:
    print("\n=== Step 2: Build wheel and inspect contents ===")
    run(["just", "build-wheel"])

    wheels_dir = ROOT_DIR / "target" / "wheels"
    wheels = sorted(
        wheels_dir.glob("bloqade_lanes-*.whl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not wheels:
        failed("no wheel found")
        sys.exit(1)

    wheel = wheels[0]
    passed(f"wheel built: {wheel.name}")

    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()

    def check_wheel(pattern: str, desc: str):
        if any(re.search(pattern, n) for n in names):
            passed(f"wheel contains {desc}")
        else:
            failed(f"wheel missing {desc}")

    check_wheel(r"\.data/scripts/bloqade-bytecode", "CLI binary")
    check_wheel(r"\.data/data/lib/.*bloqade_lanes_bytecode", "C library")
    check_wheel(r"\.data/data/include/bloqade_lanes_bytecode\.h", "C header")
    check_wheel(r"bytecode/_clib_path\.py", "_clib_path.py helper")

    return wheel


def step3_install_and_verify(wheel: Path):
    print("\n=== Step 3: Install wheel into temp venv and verify Python helpers ===")

    # Extract Python version from wheel filename (e.g. cp312 -> 3.12)
    match = re.search(r"-cp(\d)(\d+)-", wheel.name)
    if not match:
        failed("could not parse Python version from wheel filename")
        return
    py_ver = f"{match.group(1)}.{match.group(2)}"

    with tempfile.TemporaryDirectory() as tmp:
        venv_dir = Path(tmp) / "venv"
        run(["uv", "venv", str(venv_dir), "--python", py_ver, "--quiet"])
        python = str(venv_dir / "bin" / "python")
        run(["uv", "pip", "install", str(wheel), "--python", python, "--quiet"])

        def py_check(code: str) -> str:
            result = subprocess.run(
                [python, "-c", code], capture_output=True, text=True
            )
            return result.stdout.strip()

        has_clib = py_check(
            "from bloqade.lanes.bytecode import has_clib; print(has_clib())"
        )
        check(
            has_clib == "True",
            "has_clib() returns True",
            f"has_clib() returned {has_clib}",
        )

        lib_exists = py_check(
            "from bloqade.lanes.bytecode import lib_path; print(lib_path().exists())"
        )
        check(
            lib_exists == "True",
            "lib_path() file exists",
            "lib_path() file does not exist",
        )

        header_exists = py_check(
            "from bloqade.lanes.bytecode import include_dir; print((include_dir() / 'bloqade_lanes_bytecode.h').exists())"
        )
        check(
            header_exists == "True",
            "header file exists at include_dir()",
            "header file not found at include_dir()",
        )

        print("\n=== Step 4: Verify CLI binary on PATH ===")
        cli = venv_dir / "bin" / "bloqade-bytecode"
        result = subprocess.run([str(cli), "--help"], capture_output=True, text=True)
        check(
            "Lane-move bytecode" in result.stdout,
            "bloqade-bytecode --help works",
            "bloqade-bytecode --help failed",
        )


def main():
    step1_stage_artifacts()
    wheel = step2_build_and_inspect_wheel()
    step3_install_and_verify(wheel)

    print(f"\n=== Results: {PASSED} passed, {FAILED} failed ===")
    if FAILED > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
