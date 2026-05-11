import importlib.resources

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode._native import ArchSpec as _RustArchSpec

# Physical arch spec: 20 words x 8 sites, 1 zone, 32x5 grid.
#
# Same word/row structure as logical (4 words per row, 5 rows).
# Each word has 8 sites interleaved with its CZ partner along x.
# Grid: x=[0,2,10,12,20,22,...] (alternating 2/8 spacing), y=[0,10,20,30,40]
#
# Site buses: 3D hypercube on 8 sites (3 buses)
# Word buses: 9 merged column-pair shifts + 1 cross-gap (10 buses)
# Entangling pairs: [[0,1],[2,3],...,[18,19]]


def _load_spec_json() -> str:
    ref = importlib.resources.files(__package__) / "_physical_spec.json"
    return ref.read_text(encoding="utf-8")


def get_arch_spec() -> ArchSpec:
    """Physical arch spec for logical compilation (transversal Steane code)."""
    rust_spec = _RustArchSpec.from_json(_load_spec_json())
    return ArchSpec(rust_spec)


def get_physical_layout_arch_spec() -> ArchSpec:
    """Physical arch spec for direct physical qubit placement.

    Same as get_arch_spec() — in the zone-centric model, the site topology
    is encoded in the site buses directly.
    """
    return get_arch_spec()
