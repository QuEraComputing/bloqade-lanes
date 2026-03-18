#!/usr/bin/env python3
"""Deploy versioned documentation.

Builds the docs and arranges them into a versioned directory layout:
    target/site/<version>/     — this version's docs
    target/site/versions.json  — manifest of all versions
    target/site/index.html     — redirect to latest version

The CI workflow calls this, then deploys target/site/ to GitHub Pages.
Local builds (just doc / just doc-all) do not use this script — they
build plain docs without versioning.

Prerequisites:
    The gh-pages branch must exist with a valid versions.json (empty array
    is fine). To initialize it for the first time:

        git checkout --orphan gh-pages
        echo '[]' > versions.json
        git add versions.json
        git commit -m "chore: initialize gh-pages with empty versions.json"
        git push origin gh-pages
        git checkout -

Examples:
    uv run python docs/scripts/deploy_docs.py dev               # main branch
    uv run python docs/scripts/deploy_docs.py v0.5.0            # tagged release
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.version import Version

log = logging.getLogger("deploy_docs")

BOOK_TOML = Path("book.toml")
VERSION_SWITCHER_JS = "docs/theme/version-switcher.js"


def patch_book_toml() -> str:
    """Add version-switcher.js to book.toml and return the original content."""
    original = BOOK_TOML.read_text()
    patched = original.rstrip("\n") + "\n"

    # Append additional-js if not already present
    if "additional-js" not in original:
        patched += f'additional-js = ["{VERSION_SWITCHER_JS}"]\n'
    else:
        log.warning("book.toml already contains additional-js, skipping patch.")
        return original

    BOOK_TOML.write_text(patched)
    log.info("Patched book.toml to include version-switcher.js")
    return original


def restore_book_toml(original: str) -> None:
    """Restore book.toml to its original content."""
    BOOK_TOML.write_text(original)
    log.info("Restored book.toml to original state.")


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
) -> list[dict[str, str]]:
    """Load or create versions.json and add/update the given version.

    Raises SystemExit if versions.json exists but is corrupt or malformed.
    A missing versions.json is only expected on the very first deploy when
    the gh-pages branch has been initialized but no versions exist yet.
    """
    if versions_file.exists():
        try:
            versions = json.loads(versions_file.read_text())
        except json.JSONDecodeError as e:
            log.error(
                "Corrupt versions.json at %s: %s. "
                "This requires manual inspection of the gh-pages branch.",
                versions_file,
                e,
            )
            sys.exit(1)
        if not isinstance(versions, list):
            log.error(
                "versions.json at %s is not a JSON array (got %s). "
                "This requires manual inspection of the gh-pages branch.",
                versions_file,
                type(versions).__name__,
            )
            sys.exit(1)
    else:
        log.info("No existing versions.json found, creating new manifest.")
        versions = []

    # Remove existing entry for this version
    versions = [v for v in versions if v["version"] != version]

    # Add new entry (just the version label — the JS constructs URLs)
    versions.insert(0, {"version": version})

    # Sort: releases in descending semver order, then dev last
    releases = sorted(
        [v for v in versions if v["version"] != "dev"],
        key=lambda v: Version(v["version"].lstrip("v")),
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

    # Build docs with version switcher injected
    if not args.skip_build:
        original_toml = patch_book_toml()
        try:
            build_docs()
        finally:
            restore_book_toml(original_toml)
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
    versions = update_versions(versions_file, args.version)

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
