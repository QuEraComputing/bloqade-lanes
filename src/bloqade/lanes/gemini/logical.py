from ..types import Block, Grid, ArchSpec, IntraLane, InterLane
from kirin.dialects.ilist import IList


def logical_arch(physical: bool = False):
    block_size_y = 5
    
    if physical:
        all_x = sum(((10.0*i, 10.0*i+2) for i in range(0, 14)), ())
        x_00 = all_x[:14:2]
        x_01 = all_x[1:14:2]
        x_10 = all_x[14:28:2]
        x_11 = all_x[15:28:2]
        sites_0 = tuple(
            Grid(x, (10.0 * i,))
            for i in range(block_size_y)
            for x in (x_00, x_01)
        )
        sites_1 = tuple(
            Grid(x, (10.0 * i,))
            for i in range(block_size_y)
            for x in (x_10, x_11)
        )

    else:
        sites_0 = tuple(
            (i, j)for j in range(5) for i in (0, 1)
        )
        sites_1 = tuple(
            (i, j) for j in range(5) for i in (2, 3)
        )

    block_0 = Block(sites_0)
    block_1 = Block(sites_1)
    
    inter_lanes = [
        InterLane(
            block_1=block_0,
            block_2=block_1,
            site=2*y+1,
        )
        for y in range(block_size_y)
        
    ]
    intra_lanes = [
        IntraLane(
            site_1=2*y,
            site_2=2*(y+i)+1,
        )
        for i in range(block_size_y)
        for y in range(block_size_y - i)
    ]

    return ArchSpec(
        (block_0, block_1),
        tuple(intra_lanes),
        tuple(inter_lanes),
        frozenset({block_0, block_1}),
    )

