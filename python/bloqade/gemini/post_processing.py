from __future__ import annotations

import typing

import numpy as np
from bloqade.analysis.measure_id import MeasurementIDAnalysis, lattice
from kirin import ir, types
from kirin.passes import HintConst

if typing.TYPE_CHECKING:
    from bloqade.lanes.analysis.atom import PostProcessing

T = typing.TypeVar("T")


def _has_no_none(value: tuple[T | None, ...]) -> typing.TypeGuard[tuple[T, ...]]:
    return all(v is not None for v in value)


def _post_processing_function(
    value: lattice.MeasureId,
) -> typing.Callable[[typing.Sequence[bool]], typing.Any] | None:
    if isinstance(value, lattice.RawMeasureId):

        def _measure_func(measurements: typing.Sequence[bool]):
            # Measurement IDs are one-based record identifiers; simulator
            # result rows are ordinary zero-based Python sequences.
            return bool(measurements[value.idx - 1])

        return _measure_func
    elif isinstance(value, (lattice.DetectorId, lattice.ObservableId)):
        measurement_func = _post_processing_function(value.data)
        if measurement_func is None:
            return None

        def _xor_func(measurements: typing.Sequence[bool]):
            measurements = measurement_func(measurements)
            return bool(np.logical_xor.reduce(measurements, axis=0))

        return _xor_func
    elif isinstance(value, lattice.MeasureIdTuple):
        funcs = tuple(_post_processing_function(v) for v in value.data)
        if not _has_no_none(funcs):
            return None

        def _tuple_func(measurements: typing.Sequence[bool]):
            return value.obj_type([f(measurements) for f in funcs])

        return _tuple_func
    elif isinstance(value, lattice.ConstantCarrier):

        def _constant_func(measurements: typing.Sequence[bool]):
            return value.data

        return _constant_func
    else:
        return None


Params = typing.ParamSpec("Params")
ReturnType = typing.TypeVar("ReturnType")


def generate_post_processing(
    mt: ir.Method[Params, ReturnType],
) -> PostProcessing[ReturnType]:
    """Generate post-processing emitters for raw measurement results.

    Args:
        mt: The entry point of the program.

    Returns:
        Emitters for user return values, detectors, and observables. Each
        emitter accepts a two-dimensional array-like object with shape
        ``(n_shots, n_measurements)`` and yields one value per shot.

    Raises:
        ValueError: If any required post-processing value cannot be inferred.
    """

    # Work on an owned copy: physical SQuIN kernels may still contain generic
    # helper calls whose concrete measurement-list lengths are only exposed by
    # unrolling, while decoded kernels need their constant hints rebuilt.
    # NOTE: physical squin kernels need to be AggressiveUnroll'd. Only works for logical kernels
    analysis_kernel = mt.similar()
    HintConst(analysis_kernel.dialects, no_raise=False)(analysis_kernel)

    analysis = MeasurementIDAnalysis(analysis_kernel.dialects)
    _, user_output = analysis.run(analysis_kernel)

    return_func: typing.Callable[[typing.Sequence[bool]], ReturnType] | None
    if isinstance(
        user_output, lattice.NotMeasureId
    ) and analysis_kernel.return_type.is_subseteq(types.NoneType):

        def _return_none(measurements: typing.Sequence[bool]) -> ReturnType:
            return typing.cast(ReturnType, None)

        return_func = _return_none

    else:
        return_func = typing.cast(
            typing.Callable[[typing.Sequence[bool]], ReturnType] | None,
            _post_processing_function(user_output),
        )
    if return_func is None:
        raise ValueError("Unable to infer return result value from method output")

    detector_funcs = tuple(
        typing.cast(
            typing.Callable[[typing.Sequence[bool]], bool] | None,
            _post_processing_function(value),
        )
        for value in analysis.detectors
    )
    if not _has_no_none(detector_funcs):
        raise ValueError("Unable to infer detector measurement values")

    observable_funcs = tuple(
        typing.cast(
            typing.Callable[[typing.Sequence[bool]], bool] | None,
            _post_processing_function(value),
        )
        for value in analysis.observables
    )
    if not _has_no_none(observable_funcs):
        raise ValueError("Unable to infer observable measurement values")

    def emit_return(measurements: typing.Sequence[typing.Sequence[bool]]):
        yield from map(return_func, measurements)

    def emit_detectors(measurements: typing.Sequence[typing.Sequence[bool]]):
        yield from (
            [func(measurement_shot) for func in detector_funcs]
            for measurement_shot in measurements
        )

    def emit_observables(measurements: typing.Sequence[typing.Sequence[bool]]):
        yield from (
            [func(measurement_shot) for func in observable_funcs]
            for measurement_shot in measurements
        )

    # Imported here, not at module scope: ``bloqade.lanes.dialects.move``
    # imports ``bloqade.gemini``, so an atom-first import reaches this module
    # while ``bloqade.lanes.analysis.atom`` is still initialising. Same
    # lazy-import treatment as #985.
    from bloqade.lanes.analysis.atom import PostProcessing

    return PostProcessing(emit_return, emit_detectors, emit_observables)
