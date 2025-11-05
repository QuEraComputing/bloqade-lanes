from itertools import chain
from typing import TypeVar

import numpy as np

from .types.arch import ArchSpec, Lane
from .types.block import Block
from .types.grid import Grid
from .types.numpy_compat import as_tuple_int


def tesseract_intra():

    sites = np.arange(16).reshape((4, 4))

    return (
        Lane(as_tuple_int(sites[::2, :]), as_tuple_int(sites[1::2, :])),
        Lane(as_tuple_int(sites[:, ::2]), as_tuple_int(sites[:, 1::2])),
        Lane(as_tuple_int(sites[:2, :]), as_tuple_int(sites[2:, :])),
        Lane(as_tuple_int(sites[:, :2]), as_tuple_int(sites[:, 2:])),
    )


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

    blocks: list[list[Block[tuple[float, float]]]] = []

    for by in range(num_y):
        row: list[Block[tuple[float, float]]] = []
        for bx in range(num_x):
            x_start = bx * block_size_x
            x_end = x_start + block_size_x
            y_start = by * block_size_y
            y_end = y_start + block_size_y
            row.append(Block(sites=sites[x_start:x_end, y_start:y_end].positions))
        blocks.append(row)

    return np.asarray(blocks)


T = TypeVar("T")


def flatten(blocks: list[list[T]]) -> list[T]:
    return list(chain.from_iterable(blocks))


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

    shuttle_blocks = partition_grid(shuttle_sites, 4, 4)
    cache_blocks = partition_grid(cache_sites, 4, 4)

    block_ids = np.arange(shuttle_blocks.size).reshape(shuttle_blocks.shape)

    inter_lanes = []

    num_rows = shuttle_blocks.shape[0]
    num_cols = shuttle_blocks.shape[1]

    for row_shift in range(num_rows):
        src_blocks = as_tuple_int(block_ids[: num_rows - row_shift, :].flatten())
        dst_blocks = as_tuple_int(block_ids[row_shift:, :].flatten())
        inter_lanes.append(Lane(src_blocks, dst_blocks))

    for col_shift in range(num_cols):
        src_blocks = as_tuple_int(block_ids[:, : num_cols - col_shift].flatten())
        dst_blocks = as_tuple_int(block_ids[:, col_shift:].flatten())
        inter_lanes.append(Lane(src_blocks, dst_blocks))

    blocks = tuple(shuttle_blocks.flatten().tolist() + cache_blocks.flatten().tolist())

    block_ids = {i: block for i, block in enumerate(blocks)}
    has_intra_lanes = frozenset(range(shuttle_blocks.size))

    return ArchSpec(
        blocks=blocks,
        intra_lanes=tuple(tesseract_intra()),
        inter_lanes=tuple(inter_lanes),
        has_intra_lanes=has_intra_lanes,
        has_inter_lanes=frozenset(range(shuttle_blocks.size)),
    )
