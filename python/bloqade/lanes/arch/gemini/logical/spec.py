from bloqade.lanes.arch.builder import build_arch
from bloqade.lanes.arch.topology import DiagonalWordTopology, HypercubeSiteTopology
from bloqade.lanes.arch.zone import ArchBlueprint, DeviceLayout, ZoneSpec
from bloqade.lanes.layout.arch import ArchSpec


def get_arch_spec() -> ArchSpec:
    bp = ArchBlueprint(
        zones={
            "gate": ZoneSpec(
                num_rows=5,
                num_cols=2,
                entangling=True,
                word_topology=DiagonalWordTopology(),
                site_topology=HypercubeSiteTopology(),
            )
        },
        layout=DeviceLayout(sites_per_word=2),
    )
    return build_arch(bp).arch
