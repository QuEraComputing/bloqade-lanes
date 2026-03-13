#!/usr/bin/env python3
"""Merge Python and Rust Cobertura XML coverage reports into one.

Usage:
    python scripts/merge_coverage.py python-coverage.xml rust-coverage.xml -o combined-coverage.xml
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def merge_cobertura(files: list[Path], output: Path):
    """Merge multiple Cobertura XML files into a single report."""
    if not files:
        raise ValueError("No input files provided")

    base_tree = ET.parse(files[0])
    base_root = base_tree.getroot()
    base_packages = base_root.find("packages")

    if base_packages is None:
        raise RuntimeError(f"No <packages> element found in {files[0]}")

    # Accumulate line/branch stats for the merged summary
    total_lines_valid = 0
    total_lines_covered = 0
    total_branches_valid = 0
    total_branches_covered = 0

    for f in files:
        tree = ET.parse(f)
        root = tree.getroot()

        total_lines_valid += int(root.get("lines-valid", "0"))
        total_lines_covered += int(root.get("lines-covered", "0"))
        total_branches_valid += int(root.get("branches-valid", "0"))
        total_branches_covered += int(root.get("branches-covered", "0"))

        # Merge packages from non-base files
        if f != files[0]:
            packages = root.find("packages")
            if packages is not None:
                for package in packages:
                    base_packages.append(package)

    # Update summary attributes
    base_root.set("lines-valid", str(total_lines_valid))
    base_root.set("lines-covered", str(total_lines_covered))
    base_root.set("branches-valid", str(total_branches_valid))
    base_root.set("branches-covered", str(total_branches_covered))

    if total_lines_valid > 0:
        base_root.set("line-rate", f"{total_lines_covered / total_lines_valid:.4f}")
    if total_branches_valid > 0:
        base_root.set(
            "branch-rate",
            f"{total_branches_covered / total_branches_valid:.4f}",
        )

    base_tree.write(output, xml_declaration=True, encoding="unicode")
    print(
        f"Merged {len(files)} reports -> {output}"
        f" ({total_lines_covered}/{total_lines_valid} lines covered)"
    )


def main():
    parser = argparse.ArgumentParser(description="Merge Cobertura XML coverage reports")
    parser.add_argument("files", nargs="+", type=Path, help="Input XML files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("combined-coverage.xml"),
        help="Output merged XML file",
    )
    args = parser.parse_args()

    for f in args.files:
        if not f.exists():
            raise FileNotFoundError(f"Input file not found: {f}")

    merge_cobertura(args.files, args.output)


if __name__ == "__main__":
    main()
