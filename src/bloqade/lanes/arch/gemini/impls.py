from dataclasses import replace
from itertools import product
from typing import Sequence

import numpy as np
from bloqade.geometry.dialects.grid import Grid
from deprecated import deprecated

from bloqade.lanes import layout
from bloqade.lanes.layout.arch import ArchSpec, Bus
from bloqade.lanes.layout.encoding import LaneAddress, SiteLaneAddress, WordLaneAddress
from bloqade.lanes.layout.numpy_compat import as_flat_tuple_int
from bloqade.lanes.layout.word import Word


def get_path_from_shift(
    shifts: Sequence[tuple[float, float]], pos: tuple[float, float]
):
    return (pos,) + tuple(
        pos := (pos[0] + shift_x, pos[1] + shift_y) for shift_x, shift_y in shifts
    )


def site_bus_shifts(shift: int):
    if shift == 0:
        return [(2.0, 0.0)]
    elif shift == 1:
        return [(0.0, 5.0), (2.0, 0.0), (0.0, 5.0)]
    elif shift == -1:
        return [(0.0, -5.0), (2.0, 0.0), (0.0, -5.0)]
    elif shift > 1:
        return [
            (0.0, 5.0),
            (-4.0, 0.0),
            (0.0, (shift - 1) * 10.0),
            (6.0, 0.0),
            (0.0, 5.0),
        ]
    else:
        return [
            (0.0, -5.0),
            (-4.0, 0.0),
            (0.0, (1 + shift) * 10.0),
            (6.0, 0.0),
            (0.0, -5.0),
        ]


def get_word_bus_shifts(shift: int):
    return ((0.0, 5.0), (shift * 10.0, 0.0), (0.0, -5.0))


def calc_site_path_dict(words: tuple[Word, ...]):
    word_size_y = words[0].positions.shape[1]

    path_dict: dict[LaneAddress, tuple[tuple[float, float], ...]] = {}

    site_ids = range(word_size_y)
    for site_shift in range(word_size_y):
        site_shifts = site_bus_shifts(site_shift)

        bus_id = site_shift
        for site_id in site_ids[: word_size_y - site_shift]:
            for word_id, word in enumerate(words):
                lane_addr = SiteLaneAddress(word_id, site_id, bus_id)
                path = get_path_from_shift(site_shifts, word.site_position(site_id))
                rev_path = tuple(reversed(path))

                path_dict[lane_addr] = path
                path_dict[lane_addr.reverse()] = rev_path

    for diff in range(1, word_size_y):
        site_shift = word_size_y - diff
        site_shifts = site_bus_shifts(-site_shift)
        bus_id = 2 * word_size_y - site_shift - 1

        for site_id in site_ids[site_shift:]:
            for word_id, word in enumerate(words):
                lane_addr = SiteLaneAddress(word_id, site_id, bus_id)
                path = get_path_from_shift(site_shifts, word.site_position(site_id))
                rev_path = tuple(reversed(path))

                path_dict[lane_addr] = path
                path_dict[lane_addr.reverse()] = rev_path

    return path_dict


def calc_linear_word_path_dict(words: tuple[Word, ...]):
    word_size_y = words[0].positions.shape[1]
    path_dict: dict[LaneAddress, tuple[tuple[float, float], ...]] = {}

    for word_shift in range(1, len(words)):
        word_shifts = get_word_bus_shifts(word_shift)

        bus_id = word_shift
        for word_id, word in enumerate(words):
            for site_id in range(word_size_y, 2 * word_size_y):
                lane_addr = WordLaneAddress(word_id, site_id, bus_id)
                path = get_path_from_shift(word_shifts, word.site_position(site_id))
                rev_path = tuple(reversed(path))

                path_dict[lane_addr] = path
                path_dict[lane_addr.reverse()] = rev_path

    return path_dict


def calc_hypercube_word_path_dict(words: tuple[Word, ...]):
    word_size_y = words[0].positions.shape[1]
    num_words = len(words)
    hypercube_dims = num_words.bit_length() - 1

    path_dict: dict[LaneAddress, tuple[tuple[float, float], ...]] = {}

    for bus_id in range(hypercube_dims):
        shift = 2 ** (hypercube_dims - bus_id - 1)
        word_shifts = get_word_bus_shifts(shift)

        for word_id, word in enumerate(words[:shift]):
            for site_id in range(word_size_y, 2 * word_size_y):
                lane_addr = WordLaneAddress(word_id, site_id, bus_id)
                path = get_path_from_shift(word_shifts, word.site_position(site_id))
                rev_path = tuple(reversed(path))

                path_dict[lane_addr] = path
                path_dict[lane_addr.reverse()] = rev_path

    return path_dict


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


def generate_linear_busses(num_words: int):
    buses = []

    for shift in range(1, num_words):
        buses.append(
            Bus(src=tuple(range(num_words - shift)), dst=tuple(range(shift, num_words)))
        )

    return tuple(buses)


def generate_base_arch(num_words_x: int, word_size_y: int) -> ArchSpec:
    word_size_x = 2

    x_positions = (0.0, 2.0)
    y_positions = tuple(10.0 * i for i in range(word_size_y))

    grid = Grid.from_positions(x_positions, y_positions)

    def get_cz_pair(word_id: int):
        return tuple(
            layout.LocationAddress(word_id, (i + word_size_y) % (2 * word_size_y))
            for i in range(2 * word_size_y)
        )

    site_indices = tuple(product(range(word_size_x), range(word_size_y)))

    words = tuple(
        Word(grid.shift(10.0 * ix, 0.0), site_indices, get_cz_pair(ix))
        for ix in range(num_words_x)
    )

    site_ids = (
        np.arange(word_size_x * word_size_y)
        .reshape(word_size_x, word_size_y)
        .transpose()
    )

    gate_zone = tuple(range(len(words)))
    cz_gate_zones = frozenset([0])
    measurement_zones = (0,)

    return ArchSpec(
        words,
        (gate_zone,),
        measurement_zones,
        cz_gate_zones,
        frozenset(range(num_words_x)),
        frozenset(as_flat_tuple_int(site_ids[:, 1])),
        site_buses(site_ids),
        (),
        calc_site_path_dict(words),
    )


def generate_arch_hypercube(hypercube_dims: int = 4, word_size_y: int = 5) -> ArchSpec:
    num_words_x = 2**hypercube_dims
    base_arch = generate_base_arch(num_words_x, word_size_y)
    word_buses = hypercube_busses(hypercube_dims)
    base_arch.paths.update(calc_hypercube_word_path_dict(base_arch.words))
    return replace(base_arch, word_buses=word_buses)


def generate_arch_linear(num_words: int = 16, word_size_y: int = 5) -> ArchSpec:
    base_arch = generate_base_arch(num_words, word_size_y)
    word_buses = generate_linear_busses(num_words)
    base_arch.paths.update(calc_linear_word_path_dict(base_arch.words))
    return replace(base_arch, word_buses=word_buses)


@deprecated(
    version="0.2.1",
    reason="Multiple physical arch may be needed; use generate_arch_hypercube instead.",
)
def generate_arch(hypercube_dims: int = 4, word_size_y: int = 5) -> ArchSpec:
    return generate_arch_hypercube(hypercube_dims, word_size_y)
