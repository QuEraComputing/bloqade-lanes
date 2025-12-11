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

demo: demo-msd demo-pipeline

doc:
    mkdocs serve

doc-build:
    mkdocs build
