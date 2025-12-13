from typing import Sequence, TypeVar, TypeGuard


T = TypeVar("T")


def no_none_elements(xs: Sequence[T | None]) -> TypeGuard[Sequence[T]]:
    """Check that there are no None elements in the sequence.

    Args:
        xs: A sequence that may contain None elements.

    Returns:
        A TypeGuard indicating that all elements are not None.

    """
    return all(x is not None for x in xs)
