import numpy as np

from ..types import ArchSpec, Block, Grid, InterGroup, IntraGroup, as_tuple_int


def logical_arch(physical: bool = False):
    block_size_y = 5

    if physical:
        all_x = sum(((10.0 * i, 10.0 * i + 2) for i in range(0, 14)), ())
        x_00 = all_x[:14:2]
        x_01 = all_x[1:14:2]
        x_10 = all_x[14:28:2]
        x_11 = all_x[15:28:2]
        sites_0 = tuple(
            Grid(x, (10.0 * i,)) for i in range(block_size_y) for x in (x_00, x_01)
        )
        sites_1 = tuple(
            Grid(x, (10.0 * i,)) for i in range(block_size_y) for x in (x_10, x_11)
        )

    else:
        sites_0 = tuple((i, j) for j in range(5) for i in (0, 1))
        sites_1 = tuple((i, j) for j in range(5) for i in (2, 3))

    block_0 = Block(sites_0)
    block_1 = Block(sites_1)

    intra_ids = np.arange(block_size_y * 2).reshape((block_size_y, 2))

    intra_lanes = []
    for shift in range(block_size_y):
        intra_lanes.append(
            IntraGroup(
                src=as_tuple_int(intra_ids[: block_size_y - shift, 0]),
                dst=as_tuple_int(intra_ids[shift:, 1]),
            )
        )

    inter_lanes = [
        InterGroup((0,), (1,), as_tuple_int(intra_ids[:, 1])),
    ]

    return ArchSpec(
        blocks=(block_0, block_1),
        has_intra_lanes=frozenset({block_0, block_1}),
        intra_lanes=tuple(intra_lanes),
        inter_lanes=tuple(inter_lanes),
    )
