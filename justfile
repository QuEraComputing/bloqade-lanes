# Pinned tool versions (single source of truth for CI and local builds)
mdbook_version := "0.4.36"

# Default recipe
default:
    @just --list

# ── Python ──────────────────────────────────────────────────────────

coverage-run:
    uv run pytest python/tests --cov --cov-report= -n auto

coverage-xml: coverage-run
    uv run coverage xml

coverage-html: coverage-run
    uv run coverage html

coverage-report: coverage-run
    uv run coverage report

coverage-open: coverage-html
    open htmlcov/index.html

coverage: coverage-run coverage-xml coverage-report

# ── Rust Coverage ───────────────────────────────────────────────────

# Run Rust tests with coverage and generate Cobertura XML
coverage-rust:
    cargo llvm-cov --cobertura --output-path rust-coverage.xml -p bloqade-lanes-bytecode-core -p bloqade-lanes-bytecode-cli

# ── Combined Coverage ──────────────────────────────────────────────

# Run all tests with coverage and generate merged HTML report (Python + Rust)
coverage-all: coverage-run coverage-xml coverage-rust
    uv run python scripts/merge_coverage.py coverage.xml rust-coverage.xml -o combined-coverage.xml
    mkdir -p htmlcov-all
    uv run pycobertura show --format html --output htmlcov-all/index.html combined-coverage.xml
    @echo "Combined coverage report: htmlcov-all/index.html"

# Open combined coverage HTML report
coverage-all-open: coverage-all
    open htmlcov-all/index.html

demo-msd:
    python demo/msd.py

demo-pipeline:
    python demo/pipeline_demo.py

pipeline-details:
    python demo/pipeline_details.py

simulator-device-demo:
    python demo/simulator_device_demo.py

demo-explicit-allocation:
    python demo/explicit_allocation.py

demo: demo-msd demo-pipeline pipeline-details simulator-device-demo demo-explicit-allocation

# Install mdBook at the pinned version
install-mdbook:
    cargo install mdbook@{{ mdbook_version }}

# Build Rust API documentation
doc-rust:
    cargo doc --no-deps -p bloqade-lanes-bytecode-core -p bloqade-lanes-bytecode-cli

# Build the mdBook documentation site
doc-book: install-mdbook
    mdbook build

# Build complete documentation site (book + Rust API at /api/)
doc-all: doc-book doc-rust
    rm -rf target/book/api
    cp -r target/doc target/book/api
    @echo "Site: target/book/ (open target/book/index.html)"

# Build and open documentation site in browser
doc: doc-all
    open target/book/index.html

# Deploy versioned documentation to target/site/
doc-deploy version:
    uv run --dev python docs/scripts/deploy_docs.py {{ version }}

sync:
    uv sync --dev --all-extras --index-strategy=unsafe-best-match

# ── Rust ────────────────────────────────────────────────────────────

# Build the CLI crate in release mode
build-cli:
    cargo build --release -p bloqade-lanes-bytecode-cli

# Stage CLI binary, C library, and headers for Python wheel packaging
stage-clib release_dir="target/release":
    ./scripts/stage_clib.sh {{ release_dir }}

# Build the Python wheel with bundled CLI + C library
build-wheel: build-cli stage-clib
    ./scripts/maturin_with_data.sh build --release

# Development install with bundled CLI + C library
develop: build-cli stage-clib
    ./scripts/maturin_with_data.sh develop --release

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

# Regenerate policies/primer.md from registration-site source.
generate-primer:
    cargo run -p bloqade-lanes-search --bin policies-primer

# Verify policies/primer.md is up to date; exits non-zero with a diff
# if stale. Used by CI alongside `just check-header`.
check-primer:
    cargo run -p bloqade-lanes-search --bin policies-primer -- --check

# Run clippy lints on core + cli crates
lint: format-check check-header check-primer
    cargo clippy -p bloqade-lanes-bytecode-core -p bloqade-lanes-bytecode-cli --all-targets -- -D warnings

# Verify the committed C header matches what cbindgen generates
check-header:
    ./scripts/check_header.sh

# Build and run the C-FFI smoke test via CMake
test-c-ffi:
    ./scripts/test_c_ffi.sh

# Run CLI smoke tests (bytecode validation against example programs)
cli-smoke-test:
    ./scripts/test_smoke.sh

# Run Rust tests (excludes Python-binding crate which needs PyO3)
test-rust:
    cargo test -p bloqade-lanes-bytecode-core -p bloqade-lanes-bytecode-cli

# Run Python tests
test-python:
    uv run --locked pytest python/tests/ -v

# Run benchmark harness in physical architecture mode
benchmark-physical:
    uv run --locked python -m benchmarks.cli --architecture physical --compare python/benchmarks/harness/latest_physical.csv

# Run benchmark harness in logical architecture mode
benchmark-logical:
    uv run --locked python -m benchmarks.cli --architecture logical --compare python/benchmarks/harness/latest_logical.csv

# Run fast Python tests only (skip slow integration tests)
test-python-fast:
    uv run --locked pytest python/tests/ -v -m "not slow"

# Run all Python tests in parallel
test-python-parallel:
    uv run --locked pytest python/tests/ -v -n auto

# Run all tests
test: test-rust test-python

# Clean staged artifacts
clean-staged:
    rm -rf dist-data

# Full clean
clean: clean-staged
    cargo clean
    rm -rf dist/

# Regenerate every policies/fixtures/<kind>/<size>/expected.*.json by
# running its matching policy via `eval-policy --json` and stripping
# fields that are excluded from structural comparison.
regenerate-fixtures:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build -p bloqade-lanes-bytecode-cli --release
    BIN="target/release/bloqade-bytecode"
    for kind_dir in policies/fixtures/move policies/fixtures/target; do
      [ -d "$kind_dir" ] || continue
      for size_dir in "$kind_dir"/*/; do
        problem="$size_dir/problem.json"
        [ -f "$problem" ] || continue
        kind=$(basename "$kind_dir")
        for policy_path in $(ls "$size_dir"expected.*.json 2>/dev/null || true); do
          name=$(basename "$policy_path" .json | sed 's/^expected\.//')
          policy_file="policies/reference/${name}.star"
          if [ ! -f "$policy_file" ]; then
            echo "skip: no policy $policy_file" >&2
            continue
          fi
          tmp=$(mktemp)
          "$BIN" eval-policy --json --policy "$policy_file" --problem "$problem" > "$tmp" || true
          if [ "$kind" = "move" ]; then
            jq '{status, halt_reason, expansions, max_depth}' "$tmp" > "$policy_path"
          else
            jq '{ok, num_candidates, first_candidate_size}' "$tmp" > "$policy_path"
          fi
          rm "$tmp"
        done
      done
    done
    echo "regenerated. eyeball: git diff policies/fixtures/"
