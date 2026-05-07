from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .common import DEFAULT_SYNDROME_LAYOUT, SyndromeLayout
from .core import DEFAULT_BASIS_LABELS, DEFAULT_TARGET_BLOCH, BasisDataset


def build_shared_mld_postselection_scores(
    training_data_by_basis: Mapping[str, BasisDataset],
    *,
    table_decoder_cls: type,
    valid_factory_targets: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
    ranking_data_by_basis: Mapping[str, BasisDataset] | None = None,
    basis_labels: Sequence[str] = DEFAULT_BASIS_LABELS,
    sign_vector: Sequence[float] = (1.0, -1.0, 1.0),
    target_bloch: np.ndarray = DEFAULT_TARGET_BLOCH,
    layout: SyndromeLayout = DEFAULT_SYNDROME_LAYOUT,
) -> np.ndarray:
    from .decoders import estimate_mld_ancilla_scores, train_mld_decoder_pair

    if set(training_data_by_basis) != set(basis_labels):
        raise ValueError(
            "Need X/Y/Z training datasets to build shared MLD postselection scores."
        )
    score_data_by_basis = (
        ranking_data_by_basis
        if ranking_data_by_basis is not None
        else training_data_by_basis
    )
    decoder_by_basis = {
        basis: train_mld_decoder_pair(
            training_data_by_basis[basis],
            table_decoder_cls=table_decoder_cls,
            layout=layout,
        )
        for basis in basis_labels
    }
    return estimate_mld_ancilla_scores(
        decoder_by_basis,
        score_data_by_basis,
        valid_factory_targets=valid_factory_targets,
        basis_labels=basis_labels,
        sign_vector=sign_vector,
        target_bloch=target_bloch,
        layout=layout,
    )
