import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence, cast

import numpy as np
import stim
import tsim
from bloqade.analysis.tomography import (
    DEFAULT_TARGET_BLOCH,
    FidelitySummary,
    expectation_with_error_bar,
    posterior_fidelity_summary,
)
from bloqade.decoders import BaseDecoder
from bloqade.decoders.dem import detector_error_model_matrices, matrix_to_dem
from demo.msd_utils.domain.kernels import _build_tomography_primitives
from demo.msd_utils.domain.layout import _normalize_valid_factory_targets

from bloqade.gemini.decoding.constants import DEFAULT_BASIS_LABELS
from bloqade.gemini.decoding.kernels import DecoderPrimitiveSet
from bloqade.gemini.decoding.layout import DEFAULT_SYNDROME_LAYOUT
from bloqade.gemini.decoding.msd import (
    TomographyKernels,
    build_decoder_kernel_bundle,
    build_msd_primitives,
)
from bloqade.gemini.decoding.postselection import (
    DecoderAdapter,
    _build_generic_threshold_tables,
    _evaluate_cached_threshold_curve,
)
from bloqade.gemini.decoding.sampling import BasisDataset, run_task
from bloqade.gemini.decoding.special_tasks import (
    apply_special_tsim_circuit_strategy,
    build_task_map,
)
from bloqade.gemini.decoding.tasks import DemoTask
from bloqade.gemini.decoding.types import SquinKernel
from bloqade.gemini.decoding.workflow import plot_decoder_curves
from bloqade.gemini.device import GeminiLogicalSimulator


class BaseDecoderFactory(Protocol):
    def __call__(self, dem: stim.DetectorErrorModel, **kwargs: Any) -> BaseDecoder: ...


# TODO: should this be ConfidenceDecoderFactory or DecoderAdaptersFactory?
class DecoderAdaptersFactory(Protocol):
    # TODO: specify the EXACT **kwargs to have?
    def __call__(
        self,
        circuits_per_basis: Mapping[str, DemoTask],
        full_factory_decoders_per_basis: Mapping[str, tuple[BaseDecoder, BaseDecoder]],
        valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
        target_bloch: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, DecoderAdapter]: ...


@dataclass(frozen=True)
class TomographyResult:
    """Single-qubit tomography counts and reconstruction settings.

    The result intentionally stores tomography information rather than a
    target-specific fidelity. ``fidelity_bloch`` projects the reconstructed
    Bloch estimate onto a requested target Bloch vector.
    """

    basis_labels: tuple[str, ...]
    zero_counts: np.ndarray
    one_counts: np.ndarray
    method: Literal["wilson", "bayesian_bloch_ball"]
    sign_vector: np.ndarray
    binary_precision: int | None = None
    max_grid_points: int = 1_500_000

    @property
    def total_counts(self) -> np.ndarray:
        """Total accepted zero/one counts for each tomography basis."""

        return self.zero_counts + self.one_counts

    @property
    def bloch(self) -> np.ndarray:
        """Point-estimate Bloch vector in this result's sign convention."""

        totals = np.maximum(self.total_counts, 1)
        expectations = (self.zero_counts - self.one_counts) / totals
        return expectations.astype(np.float64) * self.sign_vector

    def fidelity_bloch(
        self,
        target_bloch_or_theta: Sequence[float] | np.ndarray | float,
        phi: float | None = None,
        z: float | None = None,
    ) -> FidelitySummary:
        """Project this tomography result onto a target Bloch vector.

        Args:
            target_bloch_or_theta: Either a length-3 target Bloch vector, the
                target ``x`` coordinate when ``phi`` and ``z`` are also given,
                or the Bloch-sphere polar angle ``theta`` when ``phi`` is given
                and ``z`` is omitted.
            phi: Either the target ``y`` coordinate, or the Bloch-sphere
                azimuthal angle paired with ``theta``.
            z: Optional target ``z`` coordinate.

        Returns:
            Fidelity summary for the requested target Bloch vector.
        """

        target = _resolve_target_bloch(target_bloch_or_theta, phi, z)
        bloch = self.bloch
        point = float(0.5 + 0.5 * np.dot(bloch, target))

        if self.method == "wilson":
            errors = []
            for zero, one in zip(
                self.zero_counts.tolist(),
                self.one_counts.tolist(),
                strict=True,
            ):
                _, err = expectation_with_error_bar(int(zero), int(one))
                errors.append(err)
            fidelity_err = 0.5 * float(
                np.sqrt(np.sum((target * np.asarray(errors, dtype=np.float64)) ** 2))
            )
            low = point - fidelity_err
            high = point + fidelity_err
            median = point
        elif self.method == "bayesian_bloch_ball":
            posterior = posterior_fidelity_summary(
                self.total_counts.astype(np.int64),
                self.zero_counts.astype(np.int64),
                target,
                sign=self.sign_vector,
                binary_precision=self.binary_precision,
                max_grid_points=self.max_grid_points,
            )
            low = float(posterior["low"])
            high = float(posterior["high"])
            fidelity_err = float(posterior["error"])
            median = float(posterior["median"])
            point = float(posterior["point"])
        else:
            raise ValueError("method must be 'wilson' or 'bayesian_bloch_ball'.")

        return {
            "point": float(point),
            "median": float(median),
            "low": float(low),
            "high": float(high),
            "error": float(fidelity_err),
            "bloch": (float(bloch[0]), float(bloch[1]), float(bloch[2])),
        }


def _resolve_target_bloch(
    target_bloch_or_theta: Sequence[float] | np.ndarray | float,
    phi: float | None,
    z: float | None,
) -> np.ndarray:
    """Normalize target Bloch-vector inputs accepted by ``fidelity_bloch``."""

    if phi is None and z is None:
        target = np.asarray(target_bloch_or_theta, dtype=np.float64)
        if target.shape != (3,):
            raise ValueError("target_bloch must be a length-3 vector.")
        return target

    if phi is not None and z is not None:
        x = np.asarray(target_bloch_or_theta, dtype=np.float64)
        if x.shape != ():
            raise ValueError("x coordinate must be scalar when y and z are given.")
        return np.array(
            [float(x), float(phi), float(z)],
            dtype=np.float64,
        )

    if phi is None:
        raise ValueError("phi must be provided when using Bloch-sphere angles.")
    theta_array = np.asarray(target_bloch_or_theta, dtype=np.float64)
    if theta_array.shape != ():
        raise ValueError("theta must be scalar when using Bloch-sphere angles.")
    theta = float(theta_array)
    azimuth = float(phi)
    return np.array(
        [
            math.sin(theta) * math.cos(azimuth),
            math.sin(theta) * math.sin(azimuth),
            math.cos(theta),
        ],
        dtype=np.float64,
    )


def _counts_at_accepted_fraction(
    per_basis_tables: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    global_weights: np.ndarray,
    accepted_fraction: float,
    *,
    basis_labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate score-sorted counts until the requested fraction is reached."""

    if not 0.0 <= accepted_fraction <= 1.0:
        raise ValueError("accepted_fraction must be between 0 and 1.")

    zero_counts = np.zeros(len(basis_labels), dtype=np.int64)
    one_counts = np.zeros(len(basis_labels), dtype=np.int64)
    total_accepted = int(np.sum(global_weights))
    if total_accepted <= 0 or accepted_fraction == 0.0:
        return zero_counts, one_counts

    target_count = max(1, int(math.ceil(float(accepted_fraction) * total_accepted)))
    table_state = []
    for basis in basis_labels:
        scores, zeros, ones = per_basis_tables[basis]
        idx = len(scores) - 1
        table_state.append([idx, scores, zeros, ones])

    kept = 0
    while kept < target_count:
        best_basis = -1
        best_score = -np.inf
        for basis_index, (idx, scores, _zeros, _ones) in enumerate(table_state):
            if idx >= 0 and float(scores[idx]) > best_score:
                best_basis = basis_index
                best_score = float(scores[idx])

        if best_basis < 0:
            break

        idx, _scores, zeros, ones = table_state[best_basis]
        zero = int(zeros[idx])
        one = int(ones[idx])
        zero_counts[best_basis] += zero
        one_counts[best_basis] += one
        kept += zero + one
        table_state[best_basis][0] = idx - 1

    return zero_counts, one_counts


# TODO: fix the type-checks on this file; the type-checks aren't working for some reason
def magic_state_dist_steane() -> DecoderPrimitiveSet:
    ideal_theta = 0.3041 * math.pi
    ideal_phi = 0.25 * math.pi
    ideal_lam = 0.0

    theta_offset = 0.30
    phi_offset = 0.0
    lam_offset = 0.0

    theta = ideal_theta + theta_offset
    phi = ideal_phi + phi_offset
    lam = ideal_lam + lam_offset

    msd_kernels = build_msd_primitives(theta, phi, lam)
    return msd_kernels


def single_qubit_state_tomography() -> dict[str, SquinKernel]:
    # return a list?
    # should return (X, Y, Z) in order, but can check this.
    return _build_tomography_primitives(output_qubit=0)


# TODO: make this "cache" class abstract as well?
class PostSelectionExperimentCache:
    # Going to add the kernels here for consistency.
    # NOTE: I know the TomographyKernels are dictionaries keyed by basis so is slightly inconsistent with the rest of the code.
    # Can think about either converting this to a datastructure containing two tuples, OR convert all other instances into dict's.
    # ^ so this comes to whether we use tuples of the things OR we do mappings. Let's just do mappings; it's more flexible.
    # Maybe it doesn't need to be this flexible (we can use tuples of a certain length), but right now the lower-level code uses mappings
    # so it's easier to implement for now.
    dem_kernels: TomographyKernels

    dem_circuits: Mapping[str, tsim.Circuit] | None
    dems: Mapping[str, stim.DetectorErrorModel] | None
    # decoders --> did we want this field? I think it is covered by the two fields below
    decoders: Mapping[str, tuple[BaseDecoder, BaseDecoder]] | None
    decoders_with_confidence: Mapping[str, DecoderAdapter] | None
    # NOTE: not using fields initialized_decoders_postselection and initialized_decoders_final directly for this implementation
    # Q: should the decoders enforce the shape of the input?
    # initialized_decoders_postselection: Mapping[str, ConfidenceDecoder] | None
    # initialized_decoders_final: Mapping[str, BaseDecoder] | None
    # Can think more carefully about the datatype and the shape of the following arrays.
    raw_results: Mapping[str, BasisDataset] | None
    # In this workflow, decoding is kind of coupled to postselection. The workflow is decoding ancilla -> check ancillae match postselection condition
    # -> decode output qubit. In other words, we don't always decode the output qubit (this is for speed). I guess we can return the decoded observables on the
    # ancillae qubits only..? OR, we can separate out the ancilla qubits decoded results and the observable qubit observable results. Might opt
    # for the latter, for now -- BUT, decoding is NOT coupled to confidence score I don't think
    # decoded_results: Mapping[str, tuple[np.ndarray, np.ndarray]] | None
    # the specific type of decoded_results is specified by the return type of _build_generic_threshold_tables.
    decoded_results: (
        tuple[
            Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
            np.ndarray,
            np.ndarray,
            int,
        ]
        | None
    )

    thresholded_data: Mapping[str, np.ndarray] | None

    hardware_tasks: Mapping[str, DemoTask] | None

    def __init__(self):
        self.dem_circuits = None
        self.dems = None
        self.decoders = None
        self.decoders_with_confidence = None
        self.raw_results = None
        self.decoded_results = None
        self.thresholded_data = None
        self.hardware_tasks = None


# TODO: should inherit from some "abstract" experiment workflow class?
# ^^ what methods should this "abstract" experiment workflow class have???
class PostSelectionExperiment:

    # TODO: have to specify the number of logical qubits and number of output qubits here? We can't call put the number of qubits
    # in a kernel because then it makes it hard to compose kernels (e.g., for tomography)
    def __init__(
        self,
        noncliff_prefix: SquinKernel,
        main_cliff_circ: SquinKernel,
        tomo_circs: Mapping[str, SquinKernel],
        # NOTE: again, I'm kind of cheating here because we can't really specify the shape of the numpy array in the dtype.
        # But this should be a 2D numpy array, where I have a list/array of possible valid postselection conditions.
        postselection_condition: np.ndarray,
        # this implies that the table construction AND the ranking logic will ALL live in ConfidenceDecoder
        # NOTE: now this is "decoders_postselection" instead of just "decoder_postselection"
        # TODO: in the actual code, the user has to add code to specify the implementations of these guys
        decoder_postselection: DecoderAdaptersFactory,
        decoder_final: BaseDecoderFactory,
        # specifying these as a dictionary is reasonable? Use a mapping instead?
        decoder_init_args: dict[str, Any],
        # NOTE: unfortunately, we need to supply "target_bloch" here to compute the ancilla scores
        target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    ):
        self.noncliff_prefix = noncliff_prefix
        self.main_cliff_circ = main_cliff_circ
        # NOTE: tomo_circs are currently not used; can supply these to build_decoder_kernel_bundle
        self.tomo_circs = tomo_circs
        self.postselection_condition = postselection_condition
        self.decoder_postselection = decoder_postselection
        self.decoder_final = decoder_final
        self.decoder_init_args = decoder_init_args
        self.target_bloch = target_bloch

        self.postselection_exp_cache = PostSelectionExperimentCache()
        # NOTE: hardcoding this for now (I guess) to support having some interface for adding noise to the circuit and compiling it down?
        # ^ maybe in the future, a user could specify their own simulator to use; specifically, what specific compilation pass to apply?
        # NOTE: there are two uses of a simulator object. One is the case where we actually need a simulator object to sample shots to do the decoding
        # (TableDecoder). The other is, we need to define some kind of compilation pipeline for our kernels down to tasks. This definition of the
        # simulator object is for the latter.
        # ^ Actually, for the former, the simulation is kind of hard-coded to be tsim in the current pipeline, which generates tsim circuits. However,
        # it might be nice to allow the user to specify the simulator backend (to use a different backend than tsim, for example), in decoder_init_args.
        self.simulator = GeminiLogicalSimulator()

    # TODO: implement a pass to infer the number of qubits and the output qubit from a kernel?
    def kernels(
        self,
        num_logical_qubits: int = 5,
        output_qubit: int = 0,
        special_kernel_strategy: Literal[
            "prefix_prepare", "compiled_inverse_prefix"
        ] = "prefix_prepare",
    ) -> TomographyKernels:
        decoder_primitive_set = DecoderPrimitiveSet(
            state_injection_circuit=self.noncliff_prefix,
            logical_circuit=self.main_cliff_circ,
        )
        tomography_kernels = build_decoder_kernel_bundle(
            decoder_primitive_set,
            num_logical_qubits,
            output_qubit,
            special_kernel_strategy,
        )
        self.postselection_exp_cache.dem_kernels = tomography_kernels
        # Assumes that there is some way of getting the num_logical_qubits.
        return tomography_kernels

    # NOTE: both to construct the kernels, AND to actually get the tasks, we need to call SPECIAL_KERNEL_STRATEGY.
    def dem_circuits(
        self,
        special_kernel_strategy: Literal[
            "prefix_prepare", "compiled_inverse_prefix"
        ] = "prefix_prepare",
    ) -> dict[str, tsim.Circuit]:
        tomography_kernels = self.postselection_exp_cache.dem_kernels
        special_tasks = build_task_map(
            self.simulator,
            tomography_kernels._special,
            m2dets=None,
            m2obs=None,
            append_measurements=False,
        )
        special_tasks = apply_special_tsim_circuit_strategy(
            special_tasks,
            special_kernel_strategy,
        )
        special_tsim_circuits = cast(
            dict[str, tsim.Circuit],
            {basis: task.tsim_circuit for basis, task in special_tasks.items()},
        )
        self.postselection_exp_cache.dem_circuits = special_tsim_circuits
        # again, check that values are returned in-order
        return special_tsim_circuits

    def dems(
        self,
    ) -> dict[str, stim.DetectorErrorModel]:
        # Note that this depends on the state and so arguably it's unclear to the user the exact inputs to this function.
        dem_circuits = self.postselection_exp_cache.dem_circuits
        if dem_circuits is None:
            raise RuntimeError("dem_circuits must be called before dems.")
        dems = {
            basis: circ.detector_error_model(approximate_disjoint_errors=True)
            for basis, circ in dem_circuits.items()
        }
        self.postselection_exp_cache.dems = dems
        return dems

    # TODO: where do we pass in shot counts, etc. for training the decoder?
    # NOTE: could probably split this up into separate functions; one that outputs just the decoders, and another
    # that outputs the decoders with confidences
    def initialize_decoders(
        self,
    ) -> dict[str, tuple[BaseDecoder, BaseDecoder]]:
        dems_bases = self.postselection_exp_cache.dems
        if dems_bases is None:
            raise RuntimeError("dems must be called before initialize_decoders.")
        decoders_bases = {}
        # NOTE: there is a question of if we want to support multi-qubit tomography in this experiment. For now, probably not; if we did, we
        # would have to specify the number of output qubits and their locations and use that information to construct a custom SyndromeLayout.
        # Not sure what else would change if we had to support multi-qubit tomography. But tbh this experiment is pretty coupled to single qubit tomography atm
        # anyways.
        layout = DEFAULT_SYNDROME_LAYOUT
        for basis_label, dem_base in dems_bases.items():
            dem_data = detector_error_model_matrices(dem_base)
            full_dem = matrix_to_dem(
                dem_data["H"],
                dem_data["O"][: layout.output_observable_count, :],
                dem_data["priors"],
            )
            factory_dem = matrix_to_dem(
                dem_data["H"][layout.output_detector_count :, :],
                dem_data["O"][layout.output_observable_count :, :],
                dem_data["priors"],
            )
            # NOTE: this is important. We are sampling 2x to build the 2 decoders (not once); I think it is OK (so long as they approximate
            # the same table? but it IS more expensive)
            full_decoder = self.decoder_final(full_dem, **self.decoder_init_args)
            factory_decoder = self.decoder_final(factory_dem, **self.decoder_init_args)
            decoders_bases[basis_label] = (full_decoder, factory_decoder)
        self.postselection_exp_cache.decoders = decoders_bases
        return decoders_bases

    def prep_decoders(self) -> dict[str, DecoderAdapter]:
        decoders_bases = self.postselection_exp_cache.decoders
        if decoders_bases is None:
            raise RuntimeError(
                "initialize_decoders must be called before prep_decoders."
            )
        # NOTE: have to construct the MSD circuit due to the way that MSD experiment calculates confidence, here
        tomography_kernels = self.postselection_exp_cache.dem_kernels
        actual_tasks = build_task_map(
            self.simulator,
            tomography_kernels.actual,
            m2dets=None,
            m2obs=None,
            append_measurements=False,
        )
        # self.postselection_exp_cache.compiled_noncliff_tasks = actual_tasks
        # actual_circuits = {
        #     basis_label: act_task.tsim_circuit
        #     for basis_label, act_task in actual_tasks.items()
        # }

        # TODO: deal with None possible type for 'decoders_bases'
        conf_decoders = self.decoder_postselection(
            actual_tasks,
            decoders_bases,
            valid_factory_targets=self.postselection_condition,
            target_bloch=self.target_bloch,
            **self.decoder_init_args,
        )

        self.postselection_exp_cache.decoders_with_confidence = conf_decoders

        return conf_decoders

    def make_tasks(self, device: GeminiLogicalSimulator) -> dict[str, DemoTask]:
        tomography_kernels = self.postselection_exp_cache.dem_kernels
        actual_tasks = build_task_map(
            device,
            tomography_kernels.actual,
            m2dets=None,
            m2obs=None,
            append_measurements=False,
        )
        self.postselection_exp_cache.hardware_tasks = actual_tasks
        return actual_tasks

    # NOTE: this is NOT idempotent. calling it multiple times WILL give you DIFFERENT sample data
    def get_samples(self, num_shots: int) -> dict[str, BasisDataset]:
        # NOTE: repeated code below; can get rid of it by making a field in PostSelectionExperimentCache?
        actual_tasks = self.postselection_exp_cache.hardware_tasks
        if actual_tasks is None:
            raise RuntimeError("make_tasks must be called before get_samples.")
        actual_data = {
            basis: run_task(task, num_shots, with_noise=True)
            for basis, task in actual_tasks.items()
        }
        self.postselection_exp_cache.raw_results = actual_data
        return actual_data

    def decode_and_postselect(self, decoder_name="decoder") -> tuple[
        Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        np.ndarray,
        np.ndarray,
        int,
    ]:
        actual_data = self.postselection_exp_cache.raw_results
        decoder_map = self.postselection_exp_cache.decoders_with_confidence
        if actual_data is None:
            raise RuntimeError("get_samples must be called before decoding.")
        if decoder_map is None:
            raise RuntimeError("prep_decoders must be called before decoding.")
        targets = _normalize_valid_factory_targets(self.postselection_condition)
        basis_labels = list(decoder_map.keys())
        # NOTE: hardcoded for now
        layout = DEFAULT_SYNDROME_LAYOUT
        progress_label = decoder_name
        decoded_results_tuple = _build_generic_threshold_tables(
            actual_data,
            decoder_map,
            targets=targets,
            basis_labels=basis_labels,
            layout=layout,
            progress_label=progress_label,
        )
        self.postselection_exp_cache.decoded_results = decoded_results_tuple
        return decoded_results_tuple

    def analysis_f_vs_fraction(
        self,
        binary_precision: int | None = None,
        *,
        # tomography args
        sign_vector: Sequence[float],
        target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
        basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
        max_grid_points: int = 1_500_000,
        uncertainty_backend: str = "wilson",
        # curve args
        threshold_points: int,
        min_accepted_per_basis: int = 50,
        threshold_policy: str = "quantile",
    ) -> dict[str, np.ndarray]:
        if self.postselection_exp_cache.decoded_results is None:
            raise RuntimeError("decode_and_postselect must be called before analysis.")
        (
            per_basis_tables,
            score_array,
            score_weights,
            total_shots,
        ) = self.postselection_exp_cache.decoded_results
        threshold_curve = _evaluate_cached_threshold_curve(
            per_basis_tables,
            score_array,
            score_weights=score_weights,
            binary_precision=binary_precision,
            threshold_points=threshold_points,
            sign_vector=sign_vector,
            target_bloch=target_bloch,
            basis_labels=basis_labels,
            min_accepted_per_basis=min_accepted_per_basis,
            threshold_policy=threshold_policy,
            total_shots=total_shots,
            uncertainty_backend=uncertainty_backend,
            max_grid_points=max_grid_points,
        )
        self.postselection_exp_cache.thresholded_data = threshold_curve
        return threshold_curve

    def tomography_result(
        self,
        accepted_fraction: float,
        method: Literal["wilson", "bayesian_bloch_ball"],
        *,
        sign_vector: Sequence[float] = (1.0, 1.0, 1.0),
        basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
        binary_precision: int | None = None,
        max_grid_points: int = 1_500_000,
    ) -> TomographyResult:
        """Return tomography counts after confidence-ranked postselection.

        ``accepted_fraction`` is interpreted relative to shots that have already
        passed the factory/ancilla postselection stage. Because counts are
        grouped by confidence score, the returned result may keep slightly more
        than the requested fraction.
        """

        if self.postselection_exp_cache.decoded_results is None:
            decoded_results = self.decode_and_postselect()
        else:
            decoded_results = self.postselection_exp_cache.decoded_results

        (
            per_basis_tables,
            _score_array,
            score_weights,
            _total_shots,
        ) = decoded_results
        zero_counts, one_counts = _counts_at_accepted_fraction(
            per_basis_tables,
            score_weights,
            accepted_fraction,
            basis_labels=basis_labels,
        )
        return TomographyResult(
            basis_labels=tuple(basis_labels),
            zero_counts=zero_counts,
            one_counts=one_counts,
            method=method,
            sign_vector=np.asarray(sign_vector, dtype=np.float64),
            binary_precision=binary_precision,
            max_grid_points=max_grid_points,
        )

    def analysis_visualization(
        self, min_accepted_fraction: float = 0.04, title: str | None = None
    ):
        if self.postselection_exp_cache.thresholded_data is None:
            raise RuntimeError(
                "analysis_f_vs_fraction must be called before visualization."
            )
        return plot_decoder_curves(
            {"decoder": self.postselection_exp_cache.thresholded_data},
            min_accepted_fraction=min_accepted_fraction,
            title=title,
        )


# def plot_decoder_curves(
#     curves: Mapping[str, Mapping[str, np.ndarray]],
#     *,
#     injected_summary: FidelitySummary | None = None,
#     min_accepted_fraction: float = 0.04,
#     ax: "Axes | None" = None,
#     title: str | None = None,
#     log: bool = True,
# ) -> tuple["Figure", "Axes"]:


# Rough plan for initializing decoders:
# 1. Define a "from_dem()" method on TableDecoder
# 2. User defines a closure for a method that takes in a DEM and outputs the trained TableDecoder. This closure is basically used to mirror the "constructor interface"
# to construct a Decoder object. meth(dem) -> Decoder will be our rough interface
# 3. Within that closure, for the TableDecoder, we can call our "from_dem()" method which will use the TableDecoder's __init__ constructor.

# Rough plan for initialize_decoders():
# 1. Construct decoders for each basis using DEM from each basis
# ^ have to be slightly careful because we have to create factory and full.
# can't figure out how to "subset the full table" for shots cleanly, so for now, do
# construct full with subset DEM (assumes sim cost isn't the primary bottleneck) -- NOTE: this is why decoupling sampling data from decoder construction is helpful; so we can
# sample once and take views of the shot data (instead of views of the whole table)
# 2. Use those decoders and feed into "decoders_postselection" argument to construct the ConfidenceDecoders for each basis
# 3. Once you have that, then we can just wrap things into DecoderAdapter's for each basis.

# TODO: implement the workflow for the "injected_baseline" workflow too, and think about all the details.
