import math
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np
import stim
import tsim
from bloqade.decoders import BaseDecoder, ConfidenceDecoder
from bloqade.decoders.dem import detector_error_model_matrices, matrix_to_dem
from demo.msd_utils import (
    DEFAULT_SYNDROME_LAYOUT,
    DEFAULT_TARGET_BLOCH,
    DecoderAdapter,
    DecoderPrimitiveSet,
    DemoTask,
    SquinKernel,
    TomographyKernels,
    apply_special_tsim_circuit_strategy,
    build_decoder_kernel_bundle,
    build_msd_primitives,
    build_task_map,
)
from demo.msd_utils.domain.kernels import _build_tomography_primitives

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
    # NOTE: not using fields initialized_decoders_postselection directly for this implementation
    # Q: should the decoders enforce the shape of the input?
    initialized_decoders_postselection: Mapping[str, ConfidenceDecoder] | None
    initialized_decoders_final: Mapping[str, BaseDecoder] | None
    # Can think more carefully about the datatype and the shape of the following arrays.
    raw_results: Mapping[str, np.ndarray] | None
    # In this workflow, decoding is kind of coupled to postselection. The workflow is decoding ancilla -> check ancillae match postselection condition
    # -> decode output qubit. In other words, we don't always decode the output qubit (this is for speed). I guess we can return the decoded observables on the
    # ancillae qubits only..? OR, we can separate out the ancilla qubits decoded results and the observable qubit observable results. Might opt
    # for the latter, for now -- BUT, decoding is NOT coupled to confidence score I don't think
    decoded_results: Mapping[str, tuple[np.ndarray, np.ndarray]] | None

    def __init__(self):
        return


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
        self.tomo_circs = tomo_circs
        self.postselection_condition = postselection_condition
        self.decoder_postselection = decoder_postselection
        self.decoder_final = decoder_final
        self.decoder_init_args = decoder_init_args
        self.target_bloch = target_bloch

        self.postselection_exp_cache = PostSelectionExperimentCache
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
            # m2dets=MSD_MEASUREMENT_MAPS[0],
            # m2obs=MSD_MEASUREMENT_MAPS[1],
            append_measurements=False,
        )
        special_tasks = apply_special_tsim_circuit_strategy(
            special_tasks,
            special_kernel_strategy,
        )
        special_tsim_circuits = {
            basis: task.tsim_circuit for basis, task in special_tasks.items()
        }
        self.postselection_exp_cache.dem_circuits = special_tsim_circuits
        # again, check that values are returned in-order
        return special_tsim_circuits

    def dems(
        self,
    ) -> dict[str, stim.DetectorErrorModel]:
        # Note that this depends on the state and so arguably it's unclear to the user the exact inputs to this function.
        dem_circuits = self.postselection_exp_cache.dem_circuits
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
            full_decoder = self.decoder_final(full_dem, **self.decoder_init_args)
            factory_decoder = self.decoder_final(factory_dem, **self.decoder_init_args)
            decoders_bases[basis_label] = (full_decoder, factory_decoder)
        self.postselection_exp_cache.decoders = decoders_bases
        return decoders_bases

    def prep_decoders(self) -> dict[str, DecoderAdapter]:
        decoders_bases = self.postselection_exp_cache.decoders
        # have to construct the MSD circuit due to the way that MSD experiment calculates confidence, here
        tomography_kernels = self.postselection_exp_cache.dem_kernels
        actual_tasks = build_task_map(
            self.simulator,
            tomography_kernels.actual,
            # m2dets=MSD_MEASUREMENT_MAPS[0],
            # m2obs=MSD_MEASUREMENT_MAPS[1],
            append_measurements=False,
        )
        actual_circuits = {
            basis_label: act_task.tsim_circuit
            for basis_label, act_task in actual_tasks.items()
        }

        # TODO: deal with None possible type for 'decoders_bases'
        conf_decoders = self.decoder_postselection(
            actual_circuits,
            decoders_bases,
            valid_factory_targets=self.postselection_condition,
            target_bloch=self.target_bloch,
        )

        return conf_decoders

        # def factory_decode_impl(syndrome: np.ndarray) -> tuple[np.ndarray, float]:
        #     correction, confidence = factory_decoder.decode_with_confidence(
        #         syndrome.astype(bool)
        #     )
        #     return np.asarray(correction, dtype=np.uint8), float(np.float64(confidence))

        # num_logical_qubits = self.postselection_exp_cache.num_logical_qubits
        # num_ancilla_qubits = num_logical_qubits - 1
        # NUM_DETECTORS = 3

        # decoder_adapters_list = []

        # for decoders_basis, factory_decoder_basis in zip(decoders_bases, conf_decoders):
        #     full_decoder_basis = decoders_basis[0]

        #     def factory_decode_impl(syndrome: np.ndarray) -> tuple[np.ndarray, float]:
        #         correction, confidence = factory_decoder_basis.decode_with_confidence(
        #             syndrome.astype(bool)
        #         )
        #         return np.asarray(correction, dtype=np.uint8), float(
        #             np.float64(confidence)
        #         )

        #     decoder_adapter = tuple(
        #         _make_decoder_adapter(
        #             full_decoder=full_decoder_basis,
        #             factory_decoder=factory_decoder_basis,
        #             full_syndrome_length=num_logical_qubits * 3,
        #             factory_syndrome_length=num_ancilla_qubits * 3,
        #             factory_decode_impl=factory_decode_impl,
        #             score_mode=self.decoder_init_args["score_mode"],
        #         )
        #     )

        #     decoder_adapters_list.append(decoder_adapter)

        # return tuple(decoder_adapters_list)


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
