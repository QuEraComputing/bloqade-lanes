from itertools import product
from typing import TypeVar

from .types import ArchSpec, Block, Grid, InterLane, IntraLane


def tesseract_intra():
    def iter_x(src_y, dst_y):
        for sx in range(4):
            for sy, dy in zip(src_y, dst_y):
                site_1 = 16 * sy + sx
                site_2 = 16 * dy + sx
                yield IntraLane(
                    site_1=site_1,
                    site_2=site_2,
                )

    def iter_y(src_x, dst_x):
        for sy in range(4):
            for sx, dx in zip(src_x, dst_x):
                site_1 = 16 * sy + sx
                site_2 = 16 * sy + dx
                yield IntraLane(
                    site_1=site_1,
                    site_2=site_2,
                )

    ix = range(4)
    iy = range(4)

    yield from iter_x(iy[0::2], iy[1::2])
    yield from iter_y(ix[0::2], ix[1::2])
    yield from iter_x(iy[:2], iy[2:])
    yield from iter_y(ix[:2], ix[2:])


def partition_grid(sites: Grid, num_x: int, num_y: int):
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

    for by, bx in product(range(num_y), range(num_x)):
        x_start = bx * block_size_x
        x_end = x_start + block_size_x
        y_start = by * block_size_y
        y_end = y_start + block_size_y
        blocks.append(Block(sites=sites[x_start:x_end, y_start:y_end].positions))

    return blocks


Nx = TypeVar("Nx")
Ny = TypeVar("Ny")


def holobyte_geometry(shuttle_sites: Grid[Nx, Ny], cache_sites: Grid[Nx, Ny]):
    """Generate a holobyte architecture specification from geometric information.

    This function takes the compute, cache, and memory sites and generates a
    holobyte architecture specification. Note that the holobytes are stored
    in 4x4 blocks to make inter block connections easier.

    The idea is that each shuttle and cache site are paired in the gate zone. The cache
    doesn't have intra zone moves.



    Args:
        shuttle_sites: The grid of sites that is used for shuttling throughout the gate zone
        cache_sites: The sites for storing stationary atoms

    Returns:
        An ArchSpec object representing the holobyte architecture.

    Raises:
        ValueError: If the compute and cache sites do not have the same shape, or if
        the memory sites do not match the required dimensions.
    """
    if shuttle_sites.shape[0] % 4 != 0 or shuttle_sites.shape[1] % 4 != 0:
        raise ValueError("Compute sites x/y dimension must be multiple of 4")

    if shuttle_sites.shape != cache_sites.shape:
        raise ValueError("Compute sites and cache sites must have the same shape")

    compute_blocks = partition_grid(shuttle_sites, 4, 4)
    cache_blocks = partition_grid(cache_sites, 4, 4)

    blocks = compute_blocks + cache_blocks
    has_intra_blocks = frozenset(compute_blocks)

    inter_lanes = []
    for compute_block, cache_block in zip(compute_blocks, cache_blocks):
        for site_id in range(len(cache_block.sites)):
            inter_lane = InterLane(
                (compute_block, cache_block),
                site_id,
            )
            inter_lanes.append(inter_lane)

    return ArchSpec(
        blocks=tuple(blocks),
        intra_lanes=tuple(tesseract_intra()),
        allowed_inter_lanes=tuple(inter_lanes),
        has_intra_lanes=has_intra_blocks,
    )
