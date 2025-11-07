import numpy as np

from bloqade.lanes.types.arch import ArchSpec, Bus
from bloqade.lanes.types.grid import Grid
from bloqade.lanes.types.numpy_compat import as_flat_tuple_int
from bloqade.lanes.types.word import Word


def site_buses(word_size_y: int):
    site_addresses = np.arange(word_size_y * 2).reshape((word_size_y, 2))

    site_buses: list[Bus] = []
    for shift in range(word_size_y):
        site_buses.append(
            Bus(
                src=as_flat_tuple_int(site_addresses[: word_size_y - shift, 0]),
                dst=as_flat_tuple_int(site_addresses[shift:, 1]),
            )
        )

    return tuple(site_buses)


def generate_arch():

    word_size_y = 5
    word_size_x = 2
    num_word_x = 16

    x_positions = (0.0, 2.0)
    y_positions = tuple(10.0 * i for i in range(word_size_y))

    grid = Grid(x_positions, y_positions)

    words = tuple(
        Word(grid.shift(10.0 * ix, 0.0).positions) for ix in range(num_word_x)
    )

    word_buses: list[Bus] = []
    for shift in range(4):
        m = 1 << shift

        srcs = []
        dsts = []
        for src in range(num_word_x):
            if src & m != 0:
                continue

            dst = src | m
            srcs.append(src)
            dsts.append(dst)

        word_buses.append(Bus(tuple(srcs), tuple(dsts)))

    word_ids = np.arange(word_size_x * word_size_y).reshape(word_size_y, word_size_x)

    site_bus_compatibility = tuple(
        frozenset(j for j in range(num_word_x) if j != i) for i in range(num_word_x)
    )
    return ArchSpec(
        words,
        frozenset(range(num_word_x)),
        frozenset(as_flat_tuple_int(word_ids[:, 1])),
        site_buses(word_size_y),
        tuple(word_buses),
        site_bus_compatibility,
    )
