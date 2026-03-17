#!/usr/bin/env python3
"""Deploy versioned documentation.

Builds the docs and arranges them into a versioned directory layout:
    target/site/<version>/     — this version's docs
    target/site/versions.json  — manifest of all versions
    target/site/index.html     — redirect to latest version

The CI workflow calls this, then deploys target/site/ to GitHub Pages.

Examples:
    python scripts/deploy_docs.py dev               # main branch
    python scripts/deploy_docs.py v0.5.0            # tagged release
    python scripts/deploy_docs.py v0.5.0 --base-url /bloqade-lanes
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def build_docs() -> None:
    """Build the mdBook site and Rust API docs via `just doc-all`."""
    subprocess.run(["just", "doc-all"], check=True)


def update_versions(
    versions_file: Path, version: str, base_url: str
) -> list[dict[str, str]]:
    """Load or create versions.json and add/update the given version."""
    if versions_file.exists():
        versions = json.loads(versions_file.read_text())
    else:
        versions = []

    # Remove existing entry for this version
    versions = [v for v in versions if v["version"] != version]

    # Add new entry
    versions.insert(0, {"version": version, "url": f"{base_url}/{version}/"})

    # Sort: releases in reverse order, then dev last
    releases = sorted(
        [v for v in versions if v["version"] != "dev"],
        key=lambda v: v["version"],
        reverse=True,
    )
    dev = [v for v in versions if v["version"] == "dev"]
    versions = releases + dev

    versions_file.write_text(json.dumps(versions, indent=2) + "\n")
    return versions


def write_redirect(index_path: Path, target: str) -> None:
    """Write a root index.html that redirects to the given version."""
    index_path.write_text(f"""<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="refresh" content="0; url={target}/" />
  <title>Redirecting...</title>
</head>
<body>
  <p>Redirecting to <a href="{target}/">{target}</a>...</p>
</body>
</html>
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy versioned documentation to target/site/."
    )
    parser.add_argument(
        "version",
        help='Version label (e.g. "dev", "v0.5.0"). '
        'Use "dev" for main branch, tag name for releases.',
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Base URL prefix for version links (e.g. /bloqade-lanes). "
        "Defaults to empty string.",
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=Path("target/site"),
        help="Output directory for the versioned site. Default: target/site",
    )
    parser.add_argument(
        "--book-dir",
        type=Path,
        default=Path("target/book"),
        help="mdBook build output directory. Default: target/book",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building docs (assume target/book already exists).",
    )
    args = parser.parse_args()

    print(f"=== Deploying documentation for version: {args.version} ===")

    # Build docs
    if not args.skip_build:
        build_docs()

    # Copy built docs into versioned directory
    version_dir = args.site_dir / args.version
    if version_dir.exists():
        shutil.rmtree(version_dir)
    shutil.copytree(args.book_dir, version_dir)

    # Update versions.json
    versions_file = args.site_dir / "versions.json"
    versions = update_versions(versions_file, args.version, args.base_url)

    # Determine redirect target (latest release, or dev if no releases)
    releases = [v for v in versions if v["version"] != "dev"]
    latest = releases[0]["version"] if releases else "dev"

    # Write root redirect
    write_redirect(args.site_dir / "index.html", latest)

    # Copy versions.json into each version directory for the switcher
    for child in args.site_dir.iterdir():
        if child.is_dir():
            shutil.copy2(versions_file, child / "versions.json")

    print(f"=== Site built at {args.site_dir}/ ===")
    print(f"  Version: {args.version}")
    print(f"  Redirect → {latest}")
    for item in sorted(args.site_dir.iterdir()):
        print(f"  {item.name}")


if __name__ == "__main__":
    main()
