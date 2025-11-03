from itertools import product
from typing import TypeVar

from .types import ArchSpec, Block, Grid, InterLane, IntraLane


def intra_hypercube(dim: int) -> tuple[IntraLane, ...]:
    num_sites = 2**dim
    intra_lanes = []
    for i in range(num_sites):
        for d in range(dim):
            if ((i >> d) & 1) == 0:
                neighbor = i | (1 << d)
                lane = IntraLane(site_1=i, site_2=neighbor)
                intra_lanes.append(lane)
    return tuple(intra_lanes)


def partition_grid(sites: Grid, num_x: int, num_y: int, row_major: bool = True):
    if sites.shape[0] % num_x != 0:
        raise ValueError("Number of x sites must be divisible by num_x")
    if sites.shape[1] % num_y != 0:
        raise ValueError("Number of y sites must be divisible by num_y")

    block_size_x = sites.shape[0] // num_x
    block_size_y = sites.shape[1] // num_y

    blocks: list[Block[tuple[float, float]]] = []

    if row_major:
        for by, bx in product(range(num_y), range(num_x)):
            x_start = bx * block_size_x
            x_end = x_start + block_size_x
            y_start = by * block_size_y
            y_end = y_start + block_size_y
            blocks.append(Block(sites=sites[x_start:x_end, y_start:y_end].positions))
    else:
        for bx, by in product(range(num_x), range(num_y)):
            x_start = bx * block_size_x
            x_end = x_start + block_size_x
            y_start = by * block_size_y
            y_end = y_start + block_size_y
            blocks.append(Block(sites=sites[x_start:x_end, y_start:y_end].positions))

    return blocks


Nx = TypeVar("Nx")
Ny = TypeVar("Ny")


def holobyte_geometry(
    compute_sites: Grid[Nx, Ny], cache_sites: Grid[Nx, Ny], *memory_sites: Grid[Nx, Ny]
):
    num_compute_bytes_x = compute_sites.shape[0] // 16
    num_compute_bytes_y = compute_sites.shape[1]

    if compute_sites.shape != cache_sites.shape:
        raise ValueError("Compute sites and cache sites must have the same shape")

    for mem_sites in memory_sites:
        # make sure each memory block matches compute block dimensions
        if mem_sites.shape[0] % 16 != num_compute_bytes_x * 16:
            raise ValueError(
                "Memory sites x dimension must be an integer multiple of compute sites x dimension"
            )
        if mem_sites.shape[1] != num_compute_bytes_y:
            raise ValueError(
                "Memory block sites y dimension must match compute sites y dimension"
            )

    compute_sites = compute_sites[: num_compute_bytes_x * 16, :]
    cache_sites = cache_sites[: num_compute_bytes_x * 16, :]

    compute_blocks = partition_grid(compute_sites, 16, 1)
    cache_blocks = partition_grid(cache_sites, 16, 1)

    blocks = compute_blocks + cache_blocks
    has_intra_blocks = frozenset(compute_blocks)

    inter_lanes = []
    for mem_sites in memory_sites:
        memory_blocks = partition_grid(mem_sites, 16, 1)
        blocks.extend(memory_blocks)
        for mem_block, cache_block in zip(memory_blocks, cache_blocks):
            for site_id in range(len(cache_block.sites)):
                inter_lane = InterLane(
                    block_1=mem_block,
                    block_2=cache_block,
                    site=site_id,
                )
                inter_lanes.append(inter_lane)

    return ArchSpec(
        blocks=tuple(blocks),
        intra_lanes=intra_hypercube(4),
        allowed_inter_lanes=tuple(inter_lanes),
        has_intra_lanes=has_intra_blocks,
    )
