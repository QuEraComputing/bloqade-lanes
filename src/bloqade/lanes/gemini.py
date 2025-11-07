import numpy as np

from bloqade.lanes.types.arch import ArchSpec, Lane
from bloqade.lanes.types.block import Block
from bloqade.lanes.types.grid import Grid
from bloqade.lanes.types.numpy_compat import as_flat_tuple_int


def intra_moves(block_size_y: int):
    intra_ids = np.arange(block_size_y * 2).reshape((block_size_y, 2))

    intra_lanes: list[Lane] = []
    for shift in range(block_size_y):
        intra_lanes.append(
            Lane(
                src=as_flat_tuple_int(intra_ids[: block_size_y - shift, 0]),
                dst=as_flat_tuple_int(intra_ids[shift:, 1]),
            )
        )

    return tuple(intra_lanes)


def generate_arch():

    block_size_y = 5
    block_size_x = 2
    num_block_x = 16

    x_positions = (0.0, 2.0)
    y_positions = tuple(10.0 * i for i in range(block_size_y))

    grid = Grid(x_positions, y_positions)

    blocks = tuple(
        Block(grid.shift(10.0 * ix, 0.0).positions) for ix in range(num_block_x)
    )

    inter_lanes: list[Lane] = []
    for shift in range(4):
        m = 1 << shift

        srcs = []
        dsts = []
        for src in range(num_block_x):
            if src & m != 0:
                continue

            dst = src | m
            srcs.append(src)
            dsts.append(dst)

        inter_lanes.append(Lane(tuple(srcs), tuple(dsts)))

    block_ids = np.arange(block_size_x * block_size_y).reshape(
        block_size_y, block_size_x
    )

    return ArchSpec(
        blocks=blocks,
        has_intra_lanes=frozenset(range(num_block_x)),
        has_inter_lanes=frozenset(as_flat_tuple_int(block_ids[:, 1])),
        intra_lanes=intra_moves(block_size_y),
        inter_lanes=tuple(inter_lanes),
    )
