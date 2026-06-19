from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable, cast

import numpy as np
import stim
from bloqade.decoders import BaseDecoder

from bloqade.gemini.device import GeminiLogicalSimulator

from .confidence import ConfidenceDecoder
from .constants import DEFAULT_BASIS_LABELS
from .dem import _sub_detector_error_model
from .kernels import _build_tomography_primitives, _DecoderPrimitiveSet
from .layout import DEFAULT_SYNDROME_LAYOUT
from .msd import _build_decoder_kernel_bundle, _build_msd_primitives
from .postselection import (
    DecodedPostselectionResult,
    DecoderPair,
    _build_generic_threshold_tables,
    _evaluate_cached_threshold_curve,
    _shots_at_accepted_fraction,
)
from .sampling import BasisDataset
from .special_tasks import _apply_special_tsim_circuit_strategy
from .tasks import DemoTask
from .tomography import DEFAULT_TARGET_BLOCH, TomographyResult
from .types import KirinKernel, SquinKernel
from .workflow import plot_decoder_curves

DecoderConstructor = Callable[..., BaseDecoder]


# TODO: fix the type-checks on this file; the type-checks aren't working for some reason
def magic_state_dist_steane(
    *,
    theta_offset: float = 0.0,
    phi_offset: float = 0.0,
    lam_offset: float = 0.0,
) -> _DecoderPrimitiveSet:
    ideal_theta = math.acos(1 / math.sqrt(3))
    ideal_phi = 0.25 * math.pi
    ideal_lam = 0.0

    theta = ideal_theta + theta_offset
    phi = ideal_phi + phi_offset
    lam = ideal_lam + lam_offset

    return _build_msd_primitives(theta, phi, lam)


def single_qubit_state_tomography() -> dict[str, SquinKernel]:
    # return a list?
    # should return (X, Y, Z) in order, but can check this.
    return _build_tomography_primitives(output_qubit=0)


def empty_logical_circuit() -> SquinKernel:
    """Return a no-op logical Squin kernel for injected-state tomography."""

    from bloqade import squin

    @squin.kernel
    def empty_logical(reg):
        return

    return empty_logical


# TODO: make this "cache" class abstract as well?
class _PostSelectionExperimentCache:
    # Going to add the kernels here for consistency.
    # NOTE: basis-labeled dictionaries are slightly inconsistent with tuple
    # based representations, but mappings are more flexible and match the
    # lower-level code.
    dem_kernels: dict[str, KirinKernel] | None
    dem_circuits: Mapping[str, object] | None
    dems: Mapping[str, stim.DetectorErrorModel] | None
    decoders_with_confidence: Mapping[str, DecoderPair] | None
    raw_results: Mapping[str, BasisDataset] | None
    # decoded_results maps each basis to decoded output observable shots plus
    # one confidence score per accepted shot.
    decoded_results: Mapping[str, DecodedPostselectionResult] | None
    thresholded_data: Mapping[str, np.ndarray] | None
    hardware_tasks: Mapping[str, object] | None

    def __init__(self):
        self.dem_kernels = None
        self.dem_circuits = None
        self.dems = None
        self.decoders_with_confidence = None
        self.raw_results = None
        self.decoded_results = None
        self.thresholded_data = None
        self.hardware_tasks = None


def _task_impl(task: object) -> object:
    return task.task if isinstance(task, DemoTask) else task


def _basis_dataset_from_task_result(result: object) -> BasisDataset:
    if isinstance(result, BasisDataset):
        return result
    return BasisDataset(
        detectors=np.asarray(result.detectors, dtype=np.uint8),  # type: ignore[attr-defined]
        observables=np.asarray(result.observables, dtype=np.uint8),  # type: ignore[attr-defined]
    )


# TODO: should inherit from some "abstract" experiment workflow class?
# ^^ what methods should this "abstract" experiment workflow class have???
class PostSelectionExperiment:
    # TODO: have to specify the number of logical qubits and number of output qubits here? We can't call put the number of qubits
    # in a kernel because then it makes it hard to compose kernels (e.g., for tomography)
    def __init__(
        self,
        noncliff_prefix: SquinKernel,
        main_cliff_circ: SquinKernel,
        # NOTE: again, I'm kind of cheating here because we can't really specify
        # the shape of the numpy array in the dtype. But this should be a 2D
        # numpy array, where I have a list/array of possible valid
        # postselection conditions.
        postselection_condition: np.ndarray,
        decoder: DecoderConstructor,
        tomography_kernels: Mapping[str, SquinKernel],
        # specifying these as a dictionary is reasonable? Use a mapping instead?
        decoder_init_args: dict[str, Any] | None = None,
    ):
        self.noncliff_prefix = noncliff_prefix
        self.main_cliff_circ = main_cliff_circ
        self.postselection_condition = postselection_condition
        self.decoder = decoder
        self.tomography_kernels = dict(tomography_kernels)
        self.decoder_init_args = {} if decoder_init_args is None else decoder_init_args

        self._postselection_exp_cache = _PostSelectionExperimentCache()
        # NOTE: hardcoding this for now (I guess) to support having some
        # interface for adding noise to the circuit and compiling it down.
        self._simulator = GeminiLogicalSimulator()

    # TODO: implement a pass to infer the number of qubits and the output qubit from a kernel?
    # for the device.
    def kernels(
        self,
        num_logical_qubits: int = 5,
    ) -> dict[str, KirinKernel]:
        # TODO: change the name of _DecoderPrimitiveSet --> whole_circuit
        decoder_primitive_set = _DecoderPrimitiveSet(
            state_injection_circuit=self.noncliff_prefix,
            logical_circuit=self.main_cliff_circ,
        )
        dem_kernels = _build_decoder_kernel_bundle(
            decoder_primitive_set,
            num_logical_qubits,
            tomography_kernels=self.tomography_kernels,
        )
        self._postselection_exp_cache.dem_kernels = dem_kernels
        # Assumes that there is some way of getting the num_logical_qubits.
        return dem_kernels

    # TODO: split up the kernels and DEM kernels functions
    def dem_circuits(self) -> dict[str, object]:
        tomography_kernels = self._postselection_exp_cache.dem_kernels
        if tomography_kernels is None:
            raise RuntimeError("kernels must be called before dem_circuits.")
        dem_tasks = {
            basis: self._simulator.task(kernel.similar(), None, None)
            for basis, kernel in tomography_kernels.items()
        }
        dem_tasks = _apply_special_tsim_circuit_strategy(dem_tasks)
        special_tsim_circuits = {
            basis: _task_impl(task).tsim_circuit  # type: ignore[attr-defined]
            for basis, task in dem_tasks.items()
        }
        self._postselection_exp_cache.dem_circuits = special_tsim_circuits
        return special_tsim_circuits

    def dems(self) -> dict[str, stim.DetectorErrorModel]:
        # Note that this depends on the state and so arguably it's unclear to
        # the user the exact inputs to this function.
        dem_circuits = self._postselection_exp_cache.dem_circuits
        if dem_circuits is None:
            raise RuntimeError("dem_circuits must be called before dems.")
        dems = {
            basis: circ.detector_error_model(approximate_disjoint_errors=True)  # type: ignore[attr-defined]
            for basis, circ in dem_circuits.items()
        }
        self._postselection_exp_cache.dems = dems
        return dems

    # TODO: where do we pass in shot counts, etc. for training the decoder?
    def initialize_decoders(self) -> dict[str, DecoderPair]:
        dems_bases = self._postselection_exp_cache.dems
        if dems_bases is None:
            raise RuntimeError("dems must be called before initialize_decoders.")
        decoders_bases: dict[str, DecoderPair] = {}
        # NOTE: there is a question of if we want to support multi-qubit
        # tomography in this experiment. For now, probably not; if we did, we
        # would have to specify the number of output qubits and their locations
        # and use that information to construct a custom SyndromeLayout.
        layout = DEFAULT_SYNDROME_LAYOUT
        for basis_label, dem_base in dems_bases.items():
            full_dem = _sub_detector_error_model(
                dem_base,
                detector_indices=range(dem_base.num_detectors),
                observable_indices=range(layout.output_observable_count),
            )
            factory_dem = _sub_detector_error_model(
                dem_base,
                detector_indices=range(
                    layout.output_detector_count, dem_base.num_detectors
                ),
                observable_indices=range(
                    layout.output_observable_count,
                    dem_base.num_observables,
                ),
            )
            # NOTE: this is important. We are sampling 2x to build the 2 decoders
            # (not once); I think it is OK (so long as they approximate the same
            # table? but it IS more expensive)
            full_decoder = self.decoder(full_dem, **self.decoder_init_args)
            factory_decoder = cast(
                ConfidenceDecoder,
                self.decoder(factory_dem, **self.decoder_init_args),
            )
            decoders_bases[basis_label] = (factory_decoder, full_decoder)
        self._postselection_exp_cache.decoders_with_confidence = decoders_bases
        return decoders_bases

    def make_tasks(self, device: GeminiLogicalSimulator) -> dict[str, object]:
        tomography_kernels = self._postselection_exp_cache.dem_kernels
        if tomography_kernels is None:
            raise RuntimeError("kernels must be called before make_tasks.")
        actual_tasks: dict[str, object] = {
            basis: device.task(kernel.similar())
            for basis, kernel in tomography_kernels.items()
        }
        self._postselection_exp_cache.hardware_tasks = actual_tasks
        return actual_tasks

    # NOTE: this is NOT idempotent. calling it multiple times WILL give you DIFFERENT sample data
    def get_samples(
        self,
        num_shots: int,
    ) -> dict[str, BasisDataset]:
        actual_tasks = self._postselection_exp_cache.hardware_tasks
        if actual_tasks is None:
            raise RuntimeError("make_tasks must be called before get_samples.")
        actual_data = {
            basis: _basis_dataset_from_task_result(task.run(num_shots))  # type: ignore[attr-defined]
            for basis, task in actual_tasks.items()
        }
        self._postselection_exp_cache.raw_results = actual_data
        return actual_data

    def decode_and_postselect(
        self,
        decoder_name: str | None = "decoder",
    ) -> dict[str, DecodedPostselectionResult]:
        actual_data = self._postselection_exp_cache.raw_results
        decoder_map = self._postselection_exp_cache.decoders_with_confidence
        if actual_data is None:
            raise RuntimeError("get_samples must be called before decoding.")
        if decoder_map is None:
            raise RuntimeError("initialize_decoders must be called before decoding.")
        targets = np.asarray(self.postselection_condition, dtype=np.uint8)
        if targets.ndim != 2:
            raise ValueError("postselection_condition must be a 2D array.")
        basis_labels = list(decoder_map.keys())
        decoded_results = _build_generic_threshold_tables(
            actual_data,
            decoder_map,
            targets=targets,
            basis_labels=basis_labels,
            progress_label=decoder_name,
        )
        self._postselection_exp_cache.decoded_results = decoded_results
        return decoded_results

    def analysis_f_vs_fraction(
        self,
        *,
        target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
        basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
        threshold_points: int = 64,
        min_accepted_per_basis: int = 50,
    ) -> dict[str, np.ndarray]:
        decoded_results = self._postselection_exp_cache.decoded_results
        actual_data = self._postselection_exp_cache.raw_results
        if decoded_results is None:
            raise RuntimeError("decode_and_postselect must be called before analysis.")
        if actual_data is None:
            raise RuntimeError("get_samples must be called before analysis.")
        total_shots = sum(len(dataset.observables) for dataset in actual_data.values())
        threshold_curve = _evaluate_cached_threshold_curve(
            decoded_results,
            threshold_points=threshold_points,
            target_bloch=target_bloch,
            basis_labels=basis_labels,
            min_accepted_per_basis=min_accepted_per_basis,
            total_shots=total_shots,
        )
        self._postselection_exp_cache.thresholded_data = threshold_curve
        return threshold_curve

    # NOTE: the maximum number of shots that you can "accept" is equal to the
    # number of shots that pass decoding of the ancilla after all observables on ancilla
    # are 0. If you wanted to truly have the ability to accept shots including those that
    # DON'T pass ancilla postselection threshold, then you'd have to run your
    # decoder on the output observable for ALL shots which is more expensive.
    def tomography_result(
        self,
        accepted_fraction: float,
        *,
        basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    ) -> TomographyResult:
        """Return tomography shots after confidence-ranked postselection.

        ``accepted_fraction`` is interpreted relative to the shots that already
        passed factory postselection.
        """

        decoded_results = self._postselection_exp_cache.decoded_results
        if decoded_results is None:
            decoded_results = self.decode_and_postselect()

        shots_by_basis = _shots_at_accepted_fraction(
            decoded_results,
            accepted_fraction,
            basis_labels=basis_labels,
        )
        return TomographyResult(shots_by_basis)

    def analysis_visualization(
        self, min_accepted_fraction: float = 0.04, title: str | None = None
    ):
        if self._postselection_exp_cache.thresholded_data is None:
            raise RuntimeError(
                "analysis_f_vs_fraction must be called before visualization."
            )
        return plot_decoder_curves(
            {"decoder": self._postselection_exp_cache.thresholded_data},
            min_accepted_fraction=min_accepted_fraction,
            title=title,
        )
