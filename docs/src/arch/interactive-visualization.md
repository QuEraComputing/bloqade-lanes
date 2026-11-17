# Interactive Visualization and HTML Export

The architecture visualizer, `ArchVisualizer`, has moved to the `bloqade`
distribution. Import it from `bloqade.gemini.visualize.arch`:

```python
from bloqade.lanes.arch.gemini.physical import get_arch_spec

from bloqade.gemini.visualize.arch import ArchVisualizer

figure = ArchVisualizer(get_arch_spec()).plot_interactive()
figure.write_html("architecture.html")
```

Its documentation, including the interactive controls, HTML export and the
JavaScript checks, is in bloqade's docs under "Architecture examples".
