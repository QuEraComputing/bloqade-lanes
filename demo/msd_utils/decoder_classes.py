from __future__ import annotations

import numpy as np
import stim

from .core import pack_boolean_array, unpack_packed_bits


# NOTE: this basically uses a dictionary opposed to a table for the decoder; this is the case where classical memory is the bottleneck (at the cost of time).
# ^ To be honest, this probably won't be used in production that much (we will probably stick to a np.array lookup table for fast lookups), so not reviewing it super carefully atm.
# ^ for a couple reasons -- we'd probably want our tabledecoder to NOT be that sparse if it was good (we'd want to have seen a lot of different detector patterns) -- this was more
# just implemented for prototyping reasons.
class SparseTableDecoder:
    """Sparse lookup-table decoder with the same MLD argmax semantics.

    Stores only observed detector/observable pairs, then decodes each detector
    syndrome to the most frequently observed observable correction. Ties are
    broken toward the smallest observable index, matching ``np.argmax`` on a
    dense count table.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        det_obs_counts: np.ndarray | None = None,
    ) -> None:
        self._dem = dem
        self._counts_by_detector: dict[int, dict[int, int]] = {}
        self._maximum_likelihood_correction: dict[int, int] | None = None
        if det_obs_counts is not None:
            dense = np.asarray(det_obs_counts)
            expected_len = 1 << (self.num_detectors + self.num_observables)
            if dense.shape != (expected_len,):
                raise ValueError(
                    f"det_obs_counts must have shape ({expected_len},), got {dense.shape}"
                )
            nonzero = np.flatnonzero(dense)
            if len(nonzero):
                det_mask = (1 << self.num_detectors) - 1
                for packed in nonzero.tolist():
                    det = int(packed) & det_mask
                    obs = int(packed) >> self.num_detectors
                    detector_counts = self._counts_by_detector.setdefault(det, {})
                    detector_counts[obs] = detector_counts.get(obs, 0) + int(
                        dense[packed]
                    )

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
    ) -> "SparseTableDecoder":
        decoder = cls(dem)
        decoder.update_det_obs_counts(det_obs_shots)
        return decoder

    def update_det_obs_counts(self, det_obs_shots: np.ndarray) -> None:
        shots = np.asarray(det_obs_shots, dtype=np.uint8)
        expected_width = self.num_detectors + self.num_observables
        if shots.ndim != 2 or shots.shape[1] != expected_width:
            raise ValueError(
                f"Expected det_obs_shots with shape (N, {expected_width}), got {shots.shape}"
            )

        packed_det = pack_boolean_array(shots[:, : self.num_detectors]).astype(
            np.uint64
        )
        if self.num_observables:
            packed_obs = pack_boolean_array(
                shots[:, self.num_detectors : expected_width]
            ).astype(np.uint64)
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

    def decode(self, detector_bits: np.ndarray) -> np.ndarray:
        arr = np.asarray(detector_bits, dtype=np.uint8)
        self.cache_correction()
        assert self._maximum_likelihood_correction is not None
        if arr.ndim == 1:
            packed = int(pack_boolean_array(arr)[0])
            obs = self._maximum_likelihood_correction.get(packed, 0)
            return unpack_packed_bits(obs, self.num_observables).astype(np.bool_)

        packed = pack_boolean_array(arr).astype(np.uint64)
        obs = np.array(
            [self._maximum_likelihood_correction.get(int(p), 0) for p in packed],
            dtype=np.uint64,
        )
        if self.num_observables == 0:
            return np.zeros((len(obs), 0), dtype=np.bool_)
        shifts = np.arange(self.num_observables, dtype=np.uint64).reshape(1, -1)
        return ((obs.reshape(-1, 1) >> shifts) & 1).astype(np.bool_)
