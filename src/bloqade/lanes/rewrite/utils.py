from typing import Sequence, TypeGuard, TypeVar

T = TypeVar("T")


def no_none_elements(xs: Sequence[T | None]) -> TypeGuard[Sequence[T]]:
    """Check that there are no None elements in the sequence.

    Args:
        xs: A sequence that may contain None elements.

    Returns:
        A TypeGuard indicating that all elements are not None.

    """
    return all(x is not None for x in xs)


def no_none_elements_tuple(xs: tuple[T | None, ...]) -> TypeGuard[tuple[T, ...]]:
    """Check that there are no None elements in the tuple.

    Args:
        xs: A tuple that may contain None elements.

    Returns:
        A TypeGuard indicating that all elements are not None.

    """
    return all(x is not None for x in xs)
