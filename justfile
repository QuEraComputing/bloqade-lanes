# Default recipe
default:
    @just --list

# ── Python ──────────────────────────────────────────────────────────

coverage-run:
    coverage run -m pytest python/tests

coverage-xml: coverage-run
    coverage xml

coverage-html: coverage-run
    coverage html

coverage-report: coverage-run
    coverage report

coverage-open: coverage-html
    open htmlcov/index.html

coverage: coverage-run coverage-xml coverage-report

demo-msd:
    python demo/msd.py

demo-pipeline:
    python demo/pipeline_demo.py

pipeline-details:
    python demo/pipeline_details.py

simulator-device-demo:
    python demo/simulator_device_demo.py

demo: demo-msd demo-pipeline pipeline-details simulator-device-demo

doc:
    mkdocs serve

doc-build:
    mkdocs build

sync:
    uv sync --dev --all-extras --index-strategy=unsafe-best-match

# ── Rust ────────────────────────────────────────────────────────────

# Build the CLI crate in release mode
build-cli:
    cargo build --release -p bloqade-lanes-bytecode-cli

# Stage CLI binary, C library, and headers for Python wheel packaging.
# All artifacts go into dist-data/ which maturin includes in the wheel:
#   dist-data/scripts/          -> installed to bin/ (on PATH)
#   dist-data/data/lib/         -> installed to sys.prefix/lib/
#   dist-data/data/include/     -> installed to sys.prefix/include/
#
# Usage:
#   just stage-clib                          # uses target/release
#   just stage-clib target/aarch64-.../release  # cross-compilation
stage-clib release_dir="target/release":
    #!/usr/bin/env bash
    set -euo pipefail

    # Detect platform-specific binary and library names
    case "$(uname -s)" in
        Darwin)
            CLI_BIN="bloqade-bytecode"
            SHARED_LIB="libbloqade_lanes_bytecode.dylib"
            STATIC_LIB="libbloqade_lanes_bytecode.a"
            ;;
        Linux)
            CLI_BIN="bloqade-bytecode"
            SHARED_LIB="libbloqade_lanes_bytecode.so"
            STATIC_LIB="libbloqade_lanes_bytecode.a"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            CLI_BIN="bloqade-bytecode.exe"
            SHARED_LIB="bloqade_lanes_bytecode.dll"
            STATIC_LIB="bloqade_lanes_bytecode.lib"
            ;;
        *)
            echo "Unsupported platform: $(uname -s)"
            exit 1
            ;;
    esac

    RELEASE_DIR="{{ release_dir }}"
    SCRIPTS_DIR="dist-data/scripts"
    LIB_DIR="dist-data/data/lib"
    INCLUDE_DIR="dist-data/data/include"

    # Clean previous staging
    rm -rf dist-data
    mkdir -p "$SCRIPTS_DIR" "$LIB_DIR" "$INCLUDE_DIR"

    # Stage CLI binary
    cp "$RELEASE_DIR/$CLI_BIN" "$SCRIPTS_DIR/$CLI_BIN"
    chmod +x "$SCRIPTS_DIR/$CLI_BIN"

    # Stage C shared library (required)
    if [ ! -f "$RELEASE_DIR/$SHARED_LIB" ]; then
        echo "Error: Shared library '$SHARED_LIB' not found in '$RELEASE_DIR'."
        echo "       Please ensure the C library is built before running 'stage-clib'."
        exit 1
    fi
    cp "$RELEASE_DIR/$SHARED_LIB" "$LIB_DIR/$SHARED_LIB"

    # Stage C static library (required)
    if [ ! -f "$RELEASE_DIR/$STATIC_LIB" ]; then
        echo "Error: Static library '$STATIC_LIB' not found in '$RELEASE_DIR'."
        echo "       Please ensure the C library is built before running 'stage-clib'."
        exit 1
    fi
    cp "$RELEASE_DIR/$STATIC_LIB" "$LIB_DIR/$STATIC_LIB"

    # Stage header file
    cp crates/bloqade-lanes-bytecode-cli/bloqade_lanes_bytecode.h "$INCLUDE_DIR/"

    echo "Staged artifacts:"
    echo "  Scripts:  $SCRIPTS_DIR/"
    ls -la "$SCRIPTS_DIR/"
    echo "  C lib:    $LIB_DIR/"
    ls -la "$LIB_DIR/"
    echo "  Headers:  $INCLUDE_DIR/"
    ls -la "$INCLUDE_DIR/"

# Run a maturin command with the data = "dist-data" directive temporarily
# patched into pyproject.toml.
[private]
maturin-with-data args:
    #!/usr/bin/env bash
    set -euo pipefail
    scripts/patch_pyproject_data.sh patch
    maturin {{ args }} || { scripts/patch_pyproject_data.sh restore; exit 1; }
    scripts/patch_pyproject_data.sh restore

# Build the Python wheel with bundled CLI + C library
build-wheel: build-cli stage-clib
    just maturin-with-data "build --release"

# Development install with bundled CLI + C library
develop: build-cli stage-clib
    just maturin-with-data "develop --release"

# Build only the Python extension (no CLI/C artifacts)
develop-python:
    uv run maturin develop

# Type-check Rust (fast, no linking)
check:
    cargo check

# Format all Rust code
format:
    cargo fmt --all

# Check Rust formatting (CI mode, no changes)
format-check:
    cargo fmt --all --check

# Run clippy lints on core + cli crates
lint:
    cargo clippy -p bloqade-lanes-bytecode-core -p bloqade-lanes-bytecode-cli --all-targets -- -D warnings

# Verify the committed C header matches what cbindgen generates
check-header:
    #!/usr/bin/env bash
    set -euo pipefail
    HEADER="crates/bloqade-lanes-bytecode-cli/bloqade_lanes_bytecode.h"

    # Rebuild (cbindgen runs in build.rs)
    cargo build -p bloqade-lanes-bytecode-cli

    # Freshness: committed header must match regenerated output
    if ! git diff --exit-code "$HEADER" >/dev/null 2>&1; then
        echo "ERROR: $HEADER is out of date. Run 'cargo build -p bloqade-lanes-bytecode-cli' and commit the updated header."
        git diff "$HEADER"
        exit 1
    fi

    # Syntax: header must be valid C
    cc -fsyntax-only -x c "$HEADER"
    echo "Header OK: up-to-date and valid C syntax."

# Build and run the C-FFI smoke test via CMake
test-c-ffi:
    #!/usr/bin/env bash
    set -euo pipefail
    ROOT_DIR="$(pwd)"
    BUILD_DIR="target/c_smoke_build"

    # Build the Rust library (debug mode for testing)
    cargo build -p bloqade-lanes-bytecode-cli

    LIB_DIR="$ROOT_DIR/target/debug"
    INCLUDE_DIR="$ROOT_DIR/crates/bloqade-lanes-bytecode-cli"

    # Configure, build, and test
    cmake -S tests/c_smoke -B "$BUILD_DIR" \
        -DBLOQADE_LIB_DIR="$LIB_DIR" \
        -DBLOQADE_INCLUDE_DIR="$INCLUDE_DIR"
    cmake --build "$BUILD_DIR"
    ctest --test-dir "$BUILD_DIR" --output-on-failure

# Run CLI smoke tests (bytecode validation against example programs)
cli-smoke-test:
    ./scripts/test_smoke.sh

# Run Rust tests (excludes Python-binding crate which needs PyO3)
test-rust:
    cargo test -p bloqade-lanes-bytecode-core -p bloqade-lanes-bytecode-cli

# Run Python tests
test-python:
    uv run --locked pytest python/tests/ -v

# Run all tests
test: test-rust test-python

# Clean staged artifacts
clean-staged:
    rm -rf dist-data

# Bump patch version on a release branch, commit, and tag (e.g. 0.2.2 -> 0.2.3)
release-patch:
    #!/usr/bin/env bash
    set -euo pipefail

    # Verify we're on a release branch
    BRANCH="$(git branch --show-current)"
    if [[ ! "$BRANCH" =~ ^release-[0-9]+-[0-9]+$ ]]; then
        echo "Error: must be on a release-X-Y branch (currently on '$BRANCH')" >&2
        exit 1
    fi

    # Extract major.minor from branch name (release-0-2 -> 0.2)
    BRANCH_VERSION="${BRANCH#release-}"
    BRANCH_VERSION="${BRANCH_VERSION//-/.}"

    # Read current version from pyproject.toml
    CURRENT="$(grep '^version = ' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')"
    CURRENT_MAJOR_MINOR="${CURRENT%.*}"
    CURRENT_PATCH="${CURRENT##*.}"

    # Verify branch and version match
    if [ "$CURRENT_MAJOR_MINOR" != "$BRANCH_VERSION" ]; then
        echo "Error: branch is release for $BRANCH_VERSION but version in pyproject.toml is $CURRENT" >&2
        exit 1
    fi

    # Bump patch
    NEW_PATCH=$((CURRENT_PATCH + 1))
    NEW_VERSION="${BRANCH_VERSION}.${NEW_PATCH}"
    TAG="v${NEW_VERSION}"

    # Verify tag doesn't already exist
    if git rev-parse "$TAG" >/dev/null 2>&1; then
        echo "Error: tag $TAG already exists" >&2
        exit 1
    fi

    echo "Bumping version: $CURRENT -> $NEW_VERSION"

    # Update pyproject.toml
    sed -i.bak "s/^version = \"$CURRENT\"/version = \"$NEW_VERSION\"/" pyproject.toml
    rm -f pyproject.toml.bak

    # Update all Cargo.toml files
    for cargo_toml in \
        crates/bloqade-lanes-bytecode-core/Cargo.toml \
        crates/bloqade-lanes-bytecode-python/Cargo.toml \
        crates/bloqade-lanes-bytecode-cli/Cargo.toml; do
        sed -i.bak "s/^version = \"$CURRENT\"/version = \"$NEW_VERSION\"/" "$cargo_toml"
        rm -f "${cargo_toml}.bak"
    done

    # Regenerate uv.lock
    uv lock --upgrade

    # Commit and tag
    git add pyproject.toml uv.lock \
        crates/bloqade-lanes-bytecode-core/Cargo.toml \
        crates/bloqade-lanes-bytecode-python/Cargo.toml \
        crates/bloqade-lanes-bytecode-cli/Cargo.toml
    git commit -m "Bump version to $NEW_VERSION"
    git tag "$TAG"

    echo ""
    echo "Version bumped to $NEW_VERSION and tagged $TAG"
    echo "Push with:"
    echo "  git push origin $BRANCH --tags"

# Full clean
clean: clean-staged
    cargo clean
    rm -rf dist/
