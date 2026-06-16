from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import stim
from bloqade.decoders import BaseDecoder, ConfidenceDecoder, GurobiDecoder
from bloqade.decoders.bit_packing import pack_boolean_array
from demo.msd_utils.application.table_decoders import TableDecoder

from bloqade.gemini.decoding.layout import (
    DEFAULT_SYNDROME_LAYOUT,
    SyndromeLayout,
    split_factory_bits,
)
from bloqade.gemini.decoding.mld import (
    build_mld_decoders_from_pair,
    estimate_mld_ancilla_scores,
)
from bloqade.gemini.decoding.postselection import (
    DecoderAdapter,
    _make_decoder_adapter,
)
from bloqade.gemini.decoding.sampling import run_task
from bloqade.gemini.decoding.tasks import DemoTask


def _mld_batch_size(kwargs: Mapping[str, Any]) -> int:
    batch_size = kwargs.get("batch_size", 65536)
    if batch_size is None:
        batch_size = 65536
    resolved_batch_size = int(batch_size)
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return resolved_batch_size


# NOTE: the weird thing for this function is that it basically needs like EVERYTHING that you would to run the whole experiment to get the estimated fidelity
def construct_confidence_decoders_mld(
    circuits_per_basis: Mapping[str, DemoTask],
    full_factory_decoders_per_basis: Mapping[str, tuple[BaseDecoder, BaseDecoder]],
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
    target_bloch: np.ndarray,
    **kwargs: Any,
) -> dict[str, DecoderAdapter]:

    layout = cast(SyndromeLayout, kwargs.get("layout", DEFAULT_SYNDROME_LAYOUT))
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
            task,
            mld_ranking_train_shots,
            with_noise=True,
            sim_type=str(kwargs.get("sim_type", "tsim")),
        )
        mld_ranking_data[basis] = dataset
        print("cached MLD ranking data")

    mld_decoder_pairs = {
        basis: decoder_pair
        for basis, decoder_pair in full_factory_decoders_per_basis.items()
    }

    # TODO: if merged main into my bloqade-lanes branch, this can be changed to (1.0, 1.0, 1.0) or probably just omitted because lanes has the fix now to the
    # sign problem. Alternatively, we can automate the computation of the sign, but this is probably not needed.
    MLD_SIGN_VECTOR = (1.0, 1.0, 1.0)

    mld_ancilla_scores = estimate_mld_ancilla_scores(
        mld_decoder_pairs,
        mld_ranking_data,
        valid_factory_targets=valid_factory_targets,
        basis_labels=list(circuits_per_basis.keys()),
        layout=layout,
        sign_vector=MLD_SIGN_VECTOR,
        target_bloch=target_bloch,
        # AND you have a BUNCH of fidelity arguments you can supply as well.
    )

    # TODO: check that we can get the shapes for full/factory syndrome length from the ranking data too.
    mld_training = {
        basis: build_mld_decoders_from_pair(
            full_decoder=mld_decoder_pairs[basis][0],
            factory_decoder=mld_decoder_pairs[basis][1],
            full_syndrome_length=dataset.detectors.shape[1],
            factory_syndrome_length=split_factory_bits(
                dataset.detectors,
                dataset.observables,
                layout=layout,
            )[0].shape[1],
            ancilla_scores=mld_ancilla_scores,
        )
        for basis, dataset in mld_ranking_data.items()
    }
    return mld_training


# TODO: make sure we are "principled" with the kwargs? The kwargs should strictly be initialization args for the decoders?
# NOTE: if we wanted to implement the "enumerate only errors of a certain weight", we'd just supply a different factory function and implement a "from_dem_low_weight"
# function.
def construct_full_factory_decoders_mld(
    dem: stim.DetectorErrorModel, **kwargs: Any
) -> BaseDecoder:
    return TableDecoder.from_dem(
        dem,
        num_shots=kwargs["mld_train_shots"],
        step_size=_mld_batch_size(kwargs),
    )


def construct_confidence_decoders_mle(
    circuits_per_basis: Mapping[str, DemoTask],
    full_factory_decoders_per_basis: Mapping[str, tuple[BaseDecoder, BaseDecoder]],
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
    target_bloch: np.ndarray,
    **kwargs: Any,
) -> dict[str, DecoderAdapter]:
    # NOTE: this is kind of hacky, but gets the job done (can more 'principledly' construct the DecoderAdapter later)
    # this is a long way of basically getting the decoders and "casting" them into DecoderAdapter's

    mle_decoder_adapters = {}

    for basis, (
        full_decoder,
        factory_decoder,
    ) in full_factory_decoders_per_basis.items():

        # TODO: this attribute should be defined by the MLE decoder class
        score_mode = str(
            getattr(factory_decoder, "confidence_score_mode", "confidence")
        )

        casted_factory = cast(ConfidenceDecoder, factory_decoder)

        def make_factory_decode_impl(
            factory: ConfidenceDecoder,
        ) -> Callable[[np.ndarray], tuple[np.ndarray, float]]:
            def factory_decode_impl(syndrome: np.ndarray) -> tuple[np.ndarray, float]:
                correction, confidence = factory.decode_with_confidence(
                    syndrome.astype(bool)
                )
                return np.asarray(correction, dtype=np.uint8), float(
                    np.float64(confidence)
                )

            return factory_decode_impl

        factory_decode_impl = make_factory_decode_impl(casted_factory)

        adapter = _make_decoder_adapter(
            full_decoder=full_decoder,
            factory_decoder=casted_factory,
            # TODO: think about how else to compute this information?
            full_syndrome_length=int(getattr(full_decoder, "num_detectors")),
            factory_syndrome_length=int(getattr(factory_decoder, "num_detectors")),
            factory_decode_impl=factory_decode_impl,
            factory_score_mode=score_mode,
        )
        sample_syndrome = np.zeros(
            int(getattr(factory_decoder, "num_detectors")),
            dtype=np.uint8,
        )
        adapter.decode_factory(int(pack_boolean_array(sample_syndrome)[0]))

        mle_decoder_adapters[basis] = adapter

    return mle_decoder_adapters


def construct_full_factory_decoders_mle(
    dem: stim.DetectorErrorModel, **kwargs: Any
) -> BaseDecoder:
    return GurobiDecoder(dem)
