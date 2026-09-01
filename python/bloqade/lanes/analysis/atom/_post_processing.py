from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from kirin.dialects import ilist

from ...utils import no_none_elements_tuple
from .lattice import (
    DetectorResult,
    IListResult,
    MeasureResult,
    MoveExecution,
    ObservableResult,
    TupleResult,
    Value,
)


def constructor_function(
    elem: MoveExecution,
    *,
    use_qubit_id: bool = False,
) -> Callable[[Sequence[bool]], Any] | None:
    if isinstance(elem, MeasureResult):

        def _get_measurement(measurements: Sequence[bool]):
            measurement_index = elem.qubit_id if use_qubit_id else elem.measurement_id
            # ``bool`` so a numpy row and a list of bools produce the same
            # Python type, matching ``generate_post_processing``.
            return bool(measurements[measurement_index])

        return _get_measurement
    elif isinstance(elem, (DetectorResult, ObservableResult)):
        inner_func = constructor_function(elem.data, use_qubit_id=use_qubit_id)
        if inner_func is None:
            return None

        def _get_detector(measurements: Sequence[bool]):
            # ``np.logical_xor.reduce`` rather than ``functools.reduce`` to
            # match ``generate_post_processing``: an annotation covering no
            # measurements reduces to False here, where ``reduce`` with no
            # initial value raises TypeError on the empty sequence.
            return bool(np.logical_xor.reduce(inner_func(measurements), axis=0))

        return _get_detector

    elif isinstance(elem, IListResult):
        inner_funcs = tuple(
            constructor_function(sub_elem, use_qubit_id=use_qubit_id)
            for sub_elem in elem.data
        )
        if not no_none_elements_tuple(inner_funcs):
            return None

        def _get_ilist(measurements: Sequence[bool]):
            return ilist.IList([func(measurements) for func in inner_funcs])

        return _get_ilist
    elif isinstance(elem, TupleResult):
        inner_funcs = tuple(
            constructor_function(sub_elem, use_qubit_id=use_qubit_id)
            for sub_elem in elem.data
        )
        if not no_none_elements_tuple(inner_funcs):
            return None

        def _get_tuple(measurements: Sequence[bool]):
            return tuple(func(measurements) for func in inner_funcs)

        return _get_tuple
    elif isinstance(elem, Value):

        def _return_value(measurements: Sequence[bool]):
            return elem.value

        return _return_value
    else:
        return None
