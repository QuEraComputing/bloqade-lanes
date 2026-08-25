"""Stage prose documentation and curated notebooks for one MkDocs build."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCS_SOURCE = REPOSITORY_ROOT / "docs" / "src"
NOTEBOOK_SOURCE = REPOSITORY_ROOT / "demo"
NOTEBOOK_INDEX = REPOSITORY_ROOT / "docs" / "notebooks" / "index.md"
NOTEBOOK_MANIFEST = REPOSITORY_ROOT / "docs" / "notebooks" / "notebooks.json"
THEME_ASSETS = REPOSITORY_ROOT / "docs" / "theme"
STAGING_DIRECTORY = REPOSITORY_ROOT / "target" / "site-docs"


def _read_notebook_manifest() -> list[str]:
    notebook_paths = json.loads(NOTEBOOK_MANIFEST.read_text())
    if not isinstance(notebook_paths, list) or not all(
        isinstance(path, str) for path in notebook_paths
    ):
        raise ValueError(
            f"Notebook manifest must be a list of paths: {NOTEBOOK_MANIFEST}"
        )
    return notebook_paths


def main() -> None:
    """Build the combined MkDocs source tree under ``target/site-docs``."""
    if STAGING_DIRECTORY.exists():
        shutil.rmtree(STAGING_DIRECTORY)
    shutil.copytree(
        DOCS_SOURCE,
        STAGING_DIRECTORY,
        ignore=shutil.ignore_patterns("README.md", "SUMMARY.md", "demos.md"),
    )

    # MkDocs uses index.md for a directory's landing page.
    shutil.copy2(DOCS_SOURCE / "README.md", STAGING_DIRECTORY / "index.md")

    demos_directory = STAGING_DIRECTORY / "demos"
    demos_directory.mkdir()
    shutil.copy2(NOTEBOOK_INDEX, demos_directory / "index.md")

    static_assets = NOTEBOOK_SOURCE / "star_demo_imgs"
    for notebook_path in _read_notebook_manifest():
        source = NOTEBOOK_SOURCE / notebook_path
        if not source.is_file():
            raise FileNotFoundError(f"Notebook source does not exist: {source}")
        source_text = source.read_text()
        if source.suffix == ".py" and "# %%" not in source_text:
            raise ValueError(f"Notebook source has no Jupytext cells: {source}")
        destination = demos_directory / notebook_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

        # mkdocs-jupyter gives each notebook a directory URL. A relative image
        # such as ``./star_demo_imgs/example.png`` is therefore resolved below
        # that page directory, not beside the staged source file.
        if static_assets.is_dir() and "star_demo_imgs/" in source_text:
            page_assets = destination.with_suffix("") / static_assets.name
            shutil.copytree(static_assets, page_assets)

    assets_directory = STAGING_DIRECTORY / "assets"
    assets_directory.mkdir()
    for asset_name in ("version-switcher.js", "version-switcher.css"):
        shutil.copy2(THEME_ASSETS / asset_name, assets_directory / asset_name)


if __name__ == "__main__":
    main()
