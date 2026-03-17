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
import logging
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("deploy_docs")


def build_docs() -> None:
    """Build the mdBook site and Rust API docs via `just doc-all`."""
    log.info("Building documentation with `just doc-all`...")
    result = subprocess.run(["just", "doc-all"], capture_output=True, text=True)
    if result.stdout:
        log.info(result.stdout.rstrip())
    if result.returncode != 0:
        log.error("Documentation build failed (exit code %d)", result.returncode)
        if result.stderr:
            log.error(result.stderr.rstrip())
        sys.exit(1)
    log.info("Documentation build succeeded.")


def update_versions(
    versions_file: Path,
    version: str,
    base_url: str,
) -> list[dict[str, str]]:
    """Load or create versions.json and add/update the given version."""
    if versions_file.exists():
        try:
            versions = json.loads(versions_file.read_text())
            if not isinstance(versions, list):
                log.warning("versions.json is not a list, resetting: %s", versions_file)
                versions = []
        except json.JSONDecodeError as e:
            log.warning("Corrupt versions.json, resetting: %s", e)
            versions = []
    else:
        log.info("No existing versions.json found, creating new manifest.")
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
    log.info("Updated versions.json with %d version(s).", len(versions))
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
    log.info("Root redirect → %s/", target)


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
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    log.info("Deploying documentation for version: %s", args.version)

    # Build docs
    if not args.skip_build:
        build_docs()
    else:
        log.info("Skipping build (--skip-build).")

    # Verify book output exists
    if not args.book_dir.exists():
        log.error(
            "Book output directory does not exist: %s. "
            "Run `just doc-all` first or remove --skip-build.",
            args.book_dir,
        )
        sys.exit(1)

    # Create site directory
    args.site_dir.mkdir(parents=True, exist_ok=True)

    # Copy built docs into versioned directory
    version_dir = args.site_dir / args.version
    if version_dir.exists():
        log.info("Removing existing version directory: %s", version_dir)
        shutil.rmtree(version_dir)
    shutil.copytree(args.book_dir, version_dir)
    log.info("Copied docs to %s", version_dir)

    # Update versions.json
    versions_file = args.site_dir / "versions.json"
    versions = update_versions(versions_file, args.version, args.base_url)

    # Determine redirect target (latest release, or dev if no releases)
    releases = [v for v in versions if v["version"] != "dev"]
    latest = releases[0]["version"] if releases else "dev"

    # Write root redirect
    write_redirect(args.site_dir / "index.html", latest)

    # Copy versions.json into each version directory for the switcher
    copied = 0
    for child in args.site_dir.iterdir():
        if child.is_dir():
            shutil.copy2(versions_file, child / "versions.json")
            copied += 1
    log.info("Copied versions.json into %d version directories.", copied)

    # Summary
    log.info("Site built at %s/", args.site_dir)
    log.info("  Version: %s", args.version)
    log.info("  Redirect → %s", latest)
    for item in sorted(args.site_dir.iterdir()):
        log.info("  %s", item.name)


if __name__ == "__main__":
    main()
