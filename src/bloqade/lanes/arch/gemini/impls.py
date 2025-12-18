import numpy as np
from bloqade.geometry.dialects.grid import Grid

from bloqade.lanes.layout.arch import ArchSpec, Bus
from bloqade.lanes.layout.numpy_compat import as_flat_tuple_int
from bloqade.lanes.layout.word import Word


def site_buses(site_addresses: np.ndarray):
    word_size_y = site_addresses.shape[0]

    site_buses: list[Bus] = []
    for shift in range(word_size_y):
        site_buses.append(
            Bus(
                src=as_flat_tuple_int(site_addresses[: word_size_y - shift, 0]),
                dst=as_flat_tuple_int(site_addresses[shift:, 1]),
            )
        )

    for diff in range(1, word_size_y):
        shift = word_size_y - diff
        site_buses.append(
            Bus(
                dst=as_flat_tuple_int(site_addresses[: word_size_y - shift, 1]),
                src=as_flat_tuple_int(site_addresses[shift:, 0]),
            )
        )
    return tuple(site_buses)


def hypercube_busses(hypercube_dims: int):
    word_buses: list[Bus] = []
    for shift in range(hypercube_dims):
        m = 1 << (hypercube_dims - shift - 1)

        srcs = []
        dsts = []
        for src in range(2**hypercube_dims):
            if src & m != 0:
                continue

            dst = src | m
            srcs.append(src)
            dsts.append(dst)

        word_buses.append(Bus(tuple(srcs), tuple(dsts)))

    return tuple(word_buses)


def generate_arch(hypercube_dims: int = 4, word_size_y: int = 5) -> ArchSpec:
    word_size_x = 2
    num_word_x = 2**hypercube_dims

    x_positions = (0.0, 2.0)
    y_positions = tuple(10.0 * i for i in range(word_size_y))

    grid = Grid.from_positions(x_positions, y_positions)

    has_cz = tuple(
        (i + word_size_y) % (2 * word_size_y) for i in range(2 * word_size_y)
    )
    words = tuple(
        Word(tuple(grid.shift(10.0 * ix, 0.0).positions), has_cz)
        for ix in range(num_word_x)
    )

    site_ids = (
        np.arange(word_size_x * word_size_y)
        .reshape(word_size_x, word_size_y)
        .transpose()
    )
    word_buses = hypercube_busses(hypercube_dims)
    site_bus_compatibility = tuple(
        frozenset(range(num_word_x)) for _ in range(num_word_x)
    )

    gate_zone = tuple(range(len(words)))
    cz_gate_zones = frozenset([0])
    measurement_zones = (0,)

    return ArchSpec(
        words,
        (gate_zone,),
        measurement_zones,
        cz_gate_zones,
        frozenset(range(num_word_x)),
        frozenset(as_flat_tuple_int(site_ids[:, 1])),
        site_buses(site_ids),
        word_buses,
        site_bus_compatibility,
    )
