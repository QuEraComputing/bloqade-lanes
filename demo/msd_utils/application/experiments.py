import math
from typing import Any, Literal

import numpy as np
import stim
import tsim
from bloqade.decoders import BaseDecoder, ConfidenceDecoder
from demo.msd_utils import (
    DecoderPrimitiveSet,
    TomographyKernels,
    apply_special_tsim_circuit_strategy,
    build_decoder_kernel_bundle,
    build_msd_primitives,
    build_task_map,
)
from demo.msd_utils.domain.kernels import _build_tomography_primitives
from kirin import ir

from bloqade.gemini.device import GeminiLogicalSimulator


# TODO: fix the type-checks on this file; the type-checks aren't working for some reason
def magic_state_dist_steane() -> tuple[ir.Method[..., Any], ir.Method[..., Any]]:
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
    return msd_kernels.state_injection_circuit, msd_kernels.logical_circuit


def single_qubit_state_tomography() -> (
    tuple[ir.Method[..., Any], ir.Method[..., Any], ir.Method[..., Any]]
):
    # return a list?
    # should return (X, Y, Z) in order, but can check this.
    return tuple(_build_tomography_primitives(output_qubit=0).values())


# TODO: make this "cache" class abstract as well?
class PostSelectionExperimentCache:
    # Going to add the kernels here for consistency.
    # NOTE: I know the TomographyKernels are dictionaries keyed by basis so is slightly inconsistent with the rest of the code.
    # Can think about either converting this to a datastructure containing two tuples, OR convert all other instances into dict's.
    dem_kernels: TomographyKernels

    dem_circuits: tuple[tsim.Circuit, tsim.Circuit, tsim.Circuit] | None
    dems: (
        tuple[stim.DetectorErrorModel, stim.DetectorErrorModel, stim.DetectorErrorModel]
        | None
    )
    # decoders --> did we want this field? I think it is covered by the two fields below
    # Q: should the decoders enforce the shape of the input?
    initialized_decoders_postselection: (
        tuple[ConfidenceDecoder, ConfidenceDecoder, ConfidenceDecoder] | None
    )
    initialized_decoders_final: tuple[BaseDecoder, BaseDecoder, BaseDecoder] | None
    # Can think more carefully about the datatype and the shape of the following arrays.
    raw_results: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    # In this workflow, decoding is kind of coupled to postselection. The workflow is decoding ancilla -> check ancillae match postselection condition
    # -> decode output qubit. In other words, we don't always decode the output qubit (this is for speed). I guess we can return the decoded observables on the
    # ancillae qubits only..? OR, we can separate out the ancilla qubits decoded results and the observable qubit observable results. Might opt
    # for the latter, for now -- BUT, decoding is NOT coupled to confidence score I don't think
    decoded_results: (
        tuple[
            tuple[np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray],
        ]
        | None
    )

    def __init__(self):
        return


# TODO: should inherit from some "abstract" experiment workflow class?
# ^^ what methods should this "abstract" experiment workflow class have???
class PostSelectionExperiment:

    # TODO: have to specify the number of logical qubits and number of output qubits here? We can't call put the number of qubits
    # in a kernel because then it makes it hard to compose kernels (e.g., for tomography)
    def __init__(
        self,
        noncliff_prefix: ir.Method[..., Any],
        main_cliff_circ: ir.Method[..., Any],
        tomo_circs: tuple[
            ir.Method[..., Any], ir.Method[..., Any], ir.Method[..., Any]
        ],
        # NOTE: again, I'm kind of cheating here because we can't really specify the shape of the numpy array in the dtype.
        # But this should be a 2D numpy array, where I have a list/array of possible valid postselection conditions.
        postselection_condition: np.ndarray,
        # this implies that the table construction AND the ranking logic will ALL live in ConfidenceDecoder
        decoder_postselection: ConfidenceDecoder,
        decoder_final: BaseDecoder,
        # specifying these as a dictionary is reasonable?
        decoder_init_args: dict[str, Any],
    ):
        self.noncliff_prefix = noncliff_prefix
        self.main_cliff_circ = main_cliff_circ
        self.tomo_circs = tomo_circs
        self.postselection_condition = postselection_condition
        self.decoder_postselection = decoder_postselection
        self.decoder_final = decoder_final
        self.decoder_init_args = decoder_init_args

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
        return tomography_kernels

    # NOTE: both to construct the kernels, AND to actually get the tasks, we need to call SPECIAL_KERNEL_STRATEGY.
    def dem_circuits(
        self,
        special_kernel_strategy: Literal[
            "prefix_prepare", "compiled_inverse_prefix"
        ] = "prefix_prepare",
    ) -> tuple[tsim.Circuit, tsim.Circuit, tsim.Circuit]:
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
        special_tsim_circuits = tuple(
            spec_task.tsim_circuit for spec_task in special_tasks.values()
        )
        self.postselection_exp_cache.dem_circuits = special_tsim_circuits
        # again, check that values are returned in-order
        return special_tsim_circuits

    def dems(
        self,
    ) -> tuple[
        stim.DetectorErrorModel, stim.DetectorErrorModel, stim.DetectorErrorModel
    ]:
        # Note that this depends on the state and so arguably it's unclear to the user the exact inputs to this function.
        dem_circuits = self.postselection_exp_cache.dem_circuits
        dems = tuple(
            dem_circ.detector_error_model(approximate_disjoint_errors=True)
            for dem_circ in dem_circuits
        )
        self.postselection_exp_cache.dems = dems
        return dems

    # def initialize_decoders(self) -> DecoderAdapter:


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
# construct factory with subset DEM --> WITHIN factory for MLD, sample the ranking
