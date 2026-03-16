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

# Run clippy lints on core + cli crates
lint:
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

# Run fast Python tests only (skip slow integration tests)
test-python-fast:
    uv run --locked pytest python/tests/ -v -m "not slow"

# Run all tests
test: test-rust test-python

# Clean staged artifacts
clean-staged:
    rm -rf dist-data

# Full clean
clean: clean-staged
    cargo clean
    rm -rf dist/
