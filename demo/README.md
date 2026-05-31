# Demo Folder for MSD Reproduction/Benchmarking Decoders

## Ad Hoc Setup
1. Clone this branch: https://github.com/QuEraComputing/bloqade-lanes/tree/jasonh/stdlib_firstit
2. Run `cd bloqade-lanes`; then run `uv sync --all-extras --dev`.
3. Run `cd ..`, then clone this branch: https://github.com/QuEraComputing/bloqade-decoders/tree/jasonh/mle-aug
4. To the "demo" folder, add the notebooks you want to run.
5. Run `uv pip install gurobipy` (to run the MLE decoder). You should be able to use the `gurobipy` package on a restricted, non-production license.
6. You should be able to execute the notebooks.
