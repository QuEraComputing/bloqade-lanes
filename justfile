coverage-run:
    coverage run -m pytest test

coverage-xml: coverage-run
    coverage xml

coverage-html:  coverage-run
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
