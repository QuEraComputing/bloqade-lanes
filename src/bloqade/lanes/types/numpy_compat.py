from numpy import typing as npt


def as_tuple_int(arr: npt.NDArray) -> tuple[int, ...]:
    return tuple(map(int, arr.flatten()))
