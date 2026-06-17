"""Table decoder classes used by MSD/QET examples."""

from __future__ import annotations

import logging
from typing import TypeAlias

import numpy as np
import stim
from bloqade.decoders import BaseDecoder, TableDecoder
from demo.msd_utils.domain.confidence import (
    ConfidenceDecoder,
    TableDecoderWithConfidence,
)
from demo.msd_utils.standard.bit_packing import (
    pack_boolean_array,
    shots_to_counts,
    unpack_packed_bits,
)

logger = logging.getLogger(__name__)

_COUNT_DTYPE = np.uint32
_COUNT_MAX = np.iinfo(_COUNT_DTYPE).max


def _as_uint32_count_table(counts: np.ndarray) -> np.ndarray:
    arr = np.asarray(counts)
    if np.any(arr < 0):
        raise ValueError("det_obs_counts cannot contain negative counts.")
    if np.any(arr > _COUNT_MAX):
        raise OverflowError(
            f"det_obs_counts contains a value larger than uint32 max ({_COUNT_MAX})."
        )
    return arr.astype(_COUNT_DTYPE, copy=False)


class SparseTableDecoder(BaseDecoder):
    """Sparse lookup-table decoder with dense ``TableDecoder`` argmax semantics."""

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        det_obs_counts: np.ndarray | None = None,
    ) -> None:
        super().__init__(dem)
        self._dem = dem
        self._counts_by_detector: dict[int, dict[int, int]] = {}
        self._maximum_likelihood_correction: dict[int, int] | None = None
        if det_obs_counts is not None:
            self._add_dense_counts(det_obs_counts)

    @property
    def num_detectors(self) -> int:
        return self._dem.num_detectors

    @property
    def num_observables(self) -> int:
        return self._dem.num_observables

    @classmethod
    def from_det_obs_shots(
        cls,
        dem: stim.DetectorErrorModel,
        det_obs_shots: np.ndarray,
    ) -> SparseTableDecoder:
        decoder = cls(dem)
        decoder.update_det_obs_counts(det_obs_shots)
        return decoder

    def _add_dense_counts(self, det_obs_counts: np.ndarray) -> None:
        dense = np.asarray(det_obs_counts)
        expected_len = 1 << (self.num_detectors + self.num_observables)
        if dense.shape != (expected_len,):
            raise ValueError(
                f"det_obs_counts must have shape ({expected_len},), got {dense.shape}"
            )

        det_mask = (1 << self.num_detectors) - 1
        for packed in np.flatnonzero(dense).tolist():
            det = int(packed) & det_mask
            obs = int(packed) >> self.num_detectors
            detector_counts = self._counts_by_detector.setdefault(det, {})
            detector_counts[obs] = detector_counts.get(obs, 0) + int(dense[packed])

    def update_det_obs_counts(self, det_obs_shots: np.ndarray) -> None:
        shots = np.asarray(det_obs_shots, dtype=np.uint8)
        expected_width = self.num_detectors + self.num_observables
        if shots.ndim != 2 or shots.shape[1] != expected_width:
            raise ValueError(
                f"Expected det_obs_shots with shape (N, {expected_width}), "
                f"got {shots.shape}"
            )

        packed_det = pack_boolean_array(shots[:, : self.num_detectors])
        if self.num_observables:
            packed_obs = pack_boolean_array(shots[:, self.num_detectors :])
        else:
            packed_obs = np.zeros(len(shots), dtype=np.uint64)

        pairs = np.column_stack([packed_det, packed_obs])
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        for row, count in zip(unique_pairs, counts, strict=True):
            det = int(row[0])
            obs = int(row[1])
            detector_counts = self._counts_by_detector.setdefault(det, {})
            detector_counts[obs] = detector_counts.get(obs, 0) + int(count)
        self._maximum_likelihood_correction = None

    def cache_correction(self) -> None:
        if self._maximum_likelihood_correction is not None:
            return
        correction: dict[int, int] = {}
        for det, obs_counts in self._counts_by_detector.items():
            best_obs = 0
            best_count = -1
            for obs, count in obs_counts.items():
                if count > best_count or (count == best_count and obs < best_obs):
                    best_obs = int(obs)
                    best_count = int(count)
            correction[det] = best_obs
        self._maximum_likelihood_correction = correction

    def _decode(self, detector_bits: np.ndarray) -> np.ndarray:
        self.cache_correction()
        assert self._maximum_likelihood_correction is not None
        packed = int(pack_boolean_array(np.asarray(detector_bits, dtype=np.uint8))[0])
        obs = self._maximum_likelihood_correction.get(packed, 0)
        return unpack_packed_bits(obs, self.num_observables).astype(np.bool_)


class TableDecoderWithSimplerConfidence(TableDecoder, ConfidenceDecoder):
    """Dense table decoder that stores count tables as ``uint32``.

    When ``det_obs_counts`` is omitted, the table is trained by sampling the
    provided detector error model.
    """

    confidence_score_mode = "correction_frequency"

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        det_obs_counts: np.ndarray | None = None,
        *,
        num_shots: int = 10**8,
        seed: int | None = None,
        step_size: int | None = 65536,
    ) -> None:
        num_observables = dem.num_observables
        num_detectors = dem.num_detectors
        data_len = num_observables + num_detectors
        if data_len > 64:
            raise ValueError(
                f"Total data length {data_len} (detectors + observables) "
                "exceeds 64 bits and cannot be packed into int64."
            )

        if det_obs_counts is None:
            counts_table = np.zeros(2**data_len, dtype=_COUNT_DTYPE)
            should_train_from_dem = True
        else:
            counts_table = _as_uint32_count_table(det_obs_counts)
            should_train_from_dem = False

        super().__init__(dem=dem, det_obs_counts=counts_table)
        self._det_obs_counts = _as_uint32_count_table(self._det_obs_counts)
        self._correction_confidence: np.ndarray | None = None

        if should_train_from_dem:
            self._train_from_dem(
                num_shots=num_shots,
                seed=seed,
                step_size=step_size,
            )

    def _train_from_dem(
        self,
        *,
        num_shots: int,
        seed: int | None,
        step_size: int | None,
    ) -> None:
        if num_shots < 0:
            raise ValueError("num_shots must be non-negative.")
        if num_shots == 0:
            return
        if step_size is None:
            step_size = num_shots
        if step_size <= 0:
            raise ValueError("step_size must be positive.")

        try:
            from tqdm import tqdm
        except ImportError as e:
            raise ImportError(
                "The tqdm package is required for "
                "TableDecoderWithSimplerConfidence DEM training. "
                'Install it via: pip install "tqdm"'
            ) from e

        sampler = self._dem.compile_sampler(seed=seed)
        progress_bar_steps = ((num_shots - 1) // step_size) + 1 if num_shots else 0
        total_sampled = 0

        logger.info("Building decoder from detector error model...")
        for _ in tqdm(range(progress_bar_steps)):
            next_shots = min(step_size, num_shots - total_sampled)
            total_sampled += next_shots
            sample_result = sampler.sample(
                shots=next_shots,
                bit_packed=False,
            )
            if not isinstance(sample_result, tuple) or len(sample_result) < 2:
                raise RuntimeError(
                    "Expected DEM sampler.sample to return detector and observable "
                    f"samples, got {type(sample_result)}"
                )
            det_samples, obs_samples = sample_result[:2]
            if not isinstance(det_samples, np.ndarray):
                raise RuntimeError(
                    "Expected np.ndarray detector samples from sampler.sample, "
                    f"got {type(det_samples)}"
                )
            if not isinstance(obs_samples, np.ndarray):
                raise RuntimeError(
                    "Expected np.ndarray observable samples from sampler.sample, "
                    f"got {type(obs_samples)}"
                )
            det_obs_shots = np.concatenate([det_samples, obs_samples], axis=1)
            self.update_det_obs_counts(det_obs_shots)

    def update_det_obs_counts(self, det_obs_shots: np.ndarray) -> None:
        shots = np.asarray(det_obs_shots, dtype=np.uint8)
        expected_width = self.num_detectors + self.num_observables
        if shots.ndim != 2:
            raise ValueError("det_obs_shots must be a 2D array.")
        if shots.shape[1] != expected_width:
            raise ValueError(
                f"Expected det_obs_shots with {expected_width} columns, "
                f"got {shots.shape[1]}"
            )
        step_counts = shots_to_counts(shots)
        remaining = _COUNT_MAX - self._det_obs_counts
        if np.any(step_counts > remaining):
            raise OverflowError(
                f"TableDecoder count table would exceed uint32 max ({_COUNT_MAX})."
            )
        self._det_obs_counts += step_counts.astype(_COUNT_DTYPE, copy=False)
        self._is_cached_df = False
        self._is_cached_correction = False
        self._correction_confidence = None

    def cache_correction(self) -> None:
        """Build correction and per-detector confidence lookup tables."""

        if self._is_cached_correction and self._correction_confidence is not None:
            return

        obs_counts = self._det_obs_counts.reshape(
            2**self.num_observables,
            2**self.num_detectors,
        )
        self._maximum_likelihood_correction = np.argmax(obs_counts, axis=0).reshape(-1)

        max_counts = np.max(obs_counts, axis=0).astype(np.float64, copy=False)
        total_counts = np.sum(obs_counts, axis=0, dtype=np.uint64)
        confidence = np.full(total_counts.shape, np.nan, dtype=np.float64)
        np.divide(
            max_counts,
            total_counts.astype(np.float64, copy=False),
            out=confidence,
            where=total_counts > 0,
        )
        # NOTE: this current implementation with cache both the correction AND the confidence
        # even if you don't use it (simplifies the implementation a bit). In the future,
        # if you don't want to use the confidence, maybe you can specify as a flag in the
        # decoder constructor to not cache the confidence.
        self._correction_confidence = confidence
        self._is_cached_correction = True

    def decode_with_confidence(
        self,
        detector_bits: np.ndarray,
    ) -> tuple[np.ndarray, np.float64]:
        """Decode one detector syndrome and return its empirical confidence."""

        if detector_bits.ndim != 1:
            raise ValueError(
                "decode_with_confidence expects a single detector shot (1D array)."
            )
        correction = np.asarray(self.decode(detector_bits), dtype=np.bool_)
        self.cache_correction()
        assert self._correction_confidence is not None
        packed = int(
            pack_boolean_array(
                np.asarray(detector_bits, dtype=np.uint8).reshape(1, -1)
            )[0]
        )
        return correction, np.float64(self._correction_confidence[packed])

    # TODO: can implement batch decoding on a batch of det_obs_counts here with confidence (
    # calls the method decode_det_obs_counts in TableDecoder and also returns the batch of confidence scores)


TableDecoderClass: TypeAlias = (
    type[TableDecoder]
    | type[SparseTableDecoder]
    | type[TableDecoderWithSimplerConfidence]
)


__all__ = [
    "SparseTableDecoder",
    "TableDecoder",
    "TableDecoderClass",
    "TableDecoderWithSimplerConfidence",
    "TableDecoderWithConfidence",
]
