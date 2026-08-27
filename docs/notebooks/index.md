# Notebook demos

These notebook-style pages demonstrate Bloqade Lanes logical programming,
physical compilation, and simulation workflows. They are generated directly
from the Jupytext programs in the repository's `demo/` directory, so the
documentation and executable examples share one source.

The tutorials are organized by task:

- **Defining custom architectures:** create and visualize an architecture.
- **Other language conversions:** convert a CUDA-Q kernel to SQuIN.
- **Layout and move control:** constrain allocation with `qalloc_at`/`new_at`,
  or explicitly use `move_to` and qubit permutations.
- **Compilation pipeline details:** inspect the place and move dialects.
- **Simulators:** construct simulator devices, select backends, and consume task
  results.
- **Specific use cases:** study a physical `[[8,3,2]]` code example and compile
  and simulate the logical STAR Rz gadget.

For a local preview, run `just doc`. It serves the generated site at
`http://localhost:8000` and opens it in your browser. Do not open
`target/book/index.html` with a `file://` URL: directory-style notebook links
require an HTTP server to resolve each page's `index.html`.

The documentation build renders cells without executing them by default. To
execute every documented notebook before rendering, run:

```console
just doc-notebooks-execute
```

Execution requires the optional dependencies used by the selected demos.

## Adding a demo

1. Add an `.ipynb` file or a percent-format Jupytext `.py` file under `demo/`.
2. Add its relative path to `docs/notebooks/notebooks.json`.
3. Add the staged filename to the navigation and `include` lists in
   `mkdocs.yml`.
4. Run `just doc-notebooks` and inspect the generated page under
   `target/book/demos/<demo-name>/`.
