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
    """Split a grid into blocks of equal size.

    This function takes a grid of sites and partitions it into smaller blocks
    of the specified size. The blocks can be arranged in either row-major or
    column-major order.

    Args:
        sites: The grid of sites to partition.
        num_x: The number of blocks in the x direction.
        num_y: The number of blocks in the y direction.
        row_major: Whether to arrange blocks in row-major order. Default is True.

    Returns:
        A list of Block objects representing the partitioned blocks. Each block
        contains the sites within that block. if row_major is True, blocks are
        arranged in row-major order; otherwise, they are arranged in column-major
        order.

    Raises:
        ValueError: If the number of sites is not divisible by the number of
        blocks in either direction.
    """
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
    """Generate a holobyte architecture specification.

    This function takes the compute, cache, and memory sites and generates a
    holobyte architecture specification. Note that the holobytes are stored
    in 16x1 blocks to make inter block connections easier.

    Args:
        compute_sites: The grid of compute sites.
        cache_sites: The grid of cache sites.
        *memory_sites: One or more grids of memory sites.

    Returns:
        An ArchSpec object representing the holobyte architecture.

    Raises:
        ValueError: If the compute and cache sites do not have the same shape, or if
        the memory sites do not match the required dimensions.
    """
    if compute_sites.shape[0] % 4 != 0 or compute_sites.shape[1] % 4 != 0:
        raise ValueError("Compute sites x/y dimension must be multiple of 4")

    if compute_sites.shape != cache_sites.shape:
        raise ValueError("Compute sites and cache sites must have the same shape")

    for mem_sites in memory_sites:
        # make sure each memory block matches compute block dimensions
        if mem_sites.shape != compute_sites.shape:
            raise ValueError("Memory sites must match compute sites dimensions")

    compute_blocks = partition_grid(compute_sites, 4, 4)
    cache_blocks = partition_grid(cache_sites, 4, 4)

    blocks = compute_blocks + cache_blocks
    has_intra_blocks = frozenset(compute_blocks)

    inter_lanes = []
    for compute_block, cache_block in zip(compute_blocks, cache_blocks):
        for site_id in range(len(cache_block.sites)):
            inter_lane = InterLane(
                block_1=compute_block,
                block_2=cache_block,
                site=site_id,
            )
            inter_lanes.append(inter_lane)

    for mem_sites in memory_sites:
        memory_blocks = partition_grid(mem_sites, 4, 4)
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
