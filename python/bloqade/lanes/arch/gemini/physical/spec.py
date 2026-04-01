from bloqade.lanes.arch.builder import build_arch
from bloqade.lanes.arch.topology import (
    AllToAllSiteTopology,
    DiagonalWordTopology,
    HypercubeSiteTopology,
    TransversalSiteTopology,
)
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
                site_topology=TransversalSiteTopology(
                    logical_topology=HypercubeSiteTopology(),
                    code_distance=7,
                    intra_group_topology=AllToAllSiteTopology(),
                ),
            )
        },
        layout=DeviceLayout(sites_per_word=16),
    )
    return build_arch(bp).arch
