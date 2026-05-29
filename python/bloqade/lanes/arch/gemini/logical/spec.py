import importlib.resources

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode._native import ArchSpec as _RustArchSpec

# Logical arch spec: 20 words x 1 site, 1 gate zone.
# 4 words per row, 5 rows; entangling pairs are adjacent x-column pairs.


def _load_spec_json() -> str:
    ref = importlib.resources.files(__package__) / "_logical_spec.json"
    return ref.read_text(encoding="utf-8")


def get_arch_spec() -> ArchSpec:
    rust_spec = _RustArchSpec.from_json(_load_spec_json())
    return ArchSpec(rust_spec)
