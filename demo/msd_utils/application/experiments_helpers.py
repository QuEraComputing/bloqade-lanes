from typing import Any, Mapping, Sequence

import numpy as np
from bloqade.decoders import BaseDecoder
from demo.msd_utils import (
    DecoderAdapter,
    DemoTask,
    build_mld_decoders_from_pair,
    estimate_mld_ancilla_scores,
    run_task,
    split_factory_bits,
)


# NOTE: the weird thing for this function is that it basically needs like EVERYTHING that you would to run the whole experiment to get the estimated fidelity
def construct_confidence_decoders_mld(
    circuits_per_basis: Mapping[str, DemoTask],
    full_factory_decoders_per_basis: Mapping[str, tuple[BaseDecoder, BaseDecoder]],
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
    target_bloch: np.ndarray,
    **kwargs: Any,
) -> Mapping[str, DecoderAdapter]:

    # basis_labels = ["X", "Y", "Z"]
    mld_ranking_train_shots = kwargs.get(
        "mld_rank_train_shots", kwargs["mld_train_shots"]
    )
    mld_ranking_data = {}
    for basis, task in circuits_per_basis.items():
        print(
            f"Sampling MLD ranking data for {basis} with {mld_ranking_train_shots:,} shots..."
        )
        dataset = run_task(
            task, mld_ranking_train_shots, with_noise=True, sim_type=kwargs["sim_type"]
        )
        mld_ranking_data[basis] = dataset
        print("cached MLD ranking data")

    mld_decoder_pairs = {
        basis: decoder_pair
        for basis, decoder_pair in full_factory_decoders_per_basis.items()
    }

    # TODO: if merged main into my bloqade-lanes branch, this can be changed to (1.0, 1.0, 1.0) or probably just omitted because lanes has the fix now to the
    # sign problem. Alternatively, we can automate the computation of the sign, but this is probably not needed.
    MLD_SIGN_VECTOR = (1.0, -1.0, 1.0)

    mld_ancilla_scores = estimate_mld_ancilla_scores(
        mld_decoder_pairs,
        mld_ranking_data,
        valid_factory_targets=valid_factory_targets,
        basis_labels=list(circuits_per_basis.keys()),
        sign_vector=MLD_SIGN_VECTOR,
        target_bloch=target_bloch,
        # AND you have a BUNCH of fidelity arguments you can supply as well.
    )

    # TODO: continue working here, June 8 (hacks here due to type checks; get rid of it and change it.)
    mld_training_data = {}
    mld_training = {
        basis: build_mld_decoders_from_pair(
            full_decoder=mld_decoder_pairs[basis][0],
            factory_decoder=mld_decoder_pairs[basis][1],
            full_syndrome_length=dataset.detectors.shape[1],
            factory_syndrome_length=split_factory_bits(
                dataset.detectors, dataset.observables
            )[0].shape[1],
            ancilla_scores=mld_ancilla_scores,
        )
        for basis, dataset in mld_training_data.items()
    }
    return mld_training
