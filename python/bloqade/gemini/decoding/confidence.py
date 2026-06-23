"""Confidence decoder wrappers used by MSD/QET table decoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, NamedTuple, cast

import numpy as np
import numpy.typing as npt
from bloqade.decoders import GurobiDecoder


# NOTE: The code in this file should be moved t
# TODO: this should inherit from BaseDecoder, but pyright fails on bloqade-decoders
# main ver. because GurobiDecoder decode() method has return type that doesn't
# match decode() signature on BaseDecoder (this is a problem with bloqade-decoders
# main branch, not the code here)
class ConfidenceDecoder(ABC):
    """Decoder interface for a correction plus a scalar confidence score."""

    @abstractmethod
    def decode_with_confidence(
        self,
        detector_bits: npt.NDArray[np.bool_],
    ) -> tuple[npt.NDArray[np.bool_], np.float64]:
        """Decode one detector syndrome and return a confidence score."""


class GurobiDecoderWithConfidence(GurobiDecoder, ConfidenceDecoder):
    """Gurobi MLE decoder with logical-gap confidence."""

    _env: ClassVar[object | None] = None

    class _ConfidenceSolveResult(NamedTuple):
        error: np.ndarray
        logical: np.ndarray
        objective: float

    @classmethod
    def _get_env(cls) -> object:
        import gurobipy as gp

        if cls._env is None:
            cls._env = gp.Env()
        return cls._env

    def _solve_single_shot_for_confidence(
        self,
        detector_shot: np.ndarray,
        *,
        verbose: bool = False,
        forbidden_logical: np.ndarray | None = None,
    ) -> _ConfidenceSolveResult | None:
        import gurobipy as gp
        from gurobipy import GRB

        env = cast(Any, self._get_env())
        env.setParam("OutputFlag", 1 if verbose else 0)  # type: ignore[union-attr]

        m = gp.Model("mip1", env=env)
        weights = self._weights
        detector_vertices = self._detector_vertices
        observable_indices = self._observable_indices

        error_variables: list[gp.Var] = []
        detector_variables: list[gp.Var] = []
        logical_variables: list[gp.Var] = []
        objective: gp.LinExpr = gp.LinExpr(0)

        for i, weight in enumerate(weights):
            error_variables.append(m.addVar(vtype=GRB.BINARY, name="e" + str(i)))
            objective += weight * error_variables[i]
        m.setObjective(objective, GRB.MAXIMIZE)

        detector_shot = np.asarray(detector_shot, dtype=int)
        for i, detector_vertex in enumerate(detector_vertices):
            detector_variables.append(
                m.addVar(
                    vtype=GRB.INTEGER,
                    name="h" + str(i),
                    ub=len(detector_vertex),
                    lb=0,
                )
            )
            constraint: gp.LinExpr = gp.LinExpr(0)
            for j in detector_vertex:
                constraint += error_variables[j]
            constraint -= 2 * detector_variables[i]
            m.addConstr(constraint == int(detector_shot[i]), name="c" + str(i))

        for obs_idx, observable_index in enumerate(observable_indices):
            logical_var = m.addVar(vtype=GRB.BINARY, name="l" + str(obs_idx))
            logical_variables.append(logical_var)
            if len(observable_index) == 0:
                m.addConstr(logical_var == 0, name="lfix" + str(obs_idx))
                continue
            slack_var = m.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                ub=len(observable_index),
                name="u" + str(obs_idx),
            )
            constraint = gp.LinExpr(0)
            for j in observable_index:
                constraint += error_variables[j]
            constraint -= 2 * slack_var
            m.addConstr(constraint == logical_var, name="lpar" + str(obs_idx))

        if forbidden_logical is not None:
            diff_variables: list[gp.Var] = []
            for obs_idx, forbidden_bit in enumerate(forbidden_logical.astype(int)):
                diff_var = m.addVar(vtype=GRB.BINARY, name="d" + str(obs_idx))
                diff_variables.append(diff_var)
                if forbidden_bit:
                    m.addConstr(
                        diff_var + logical_variables[obs_idx] == 1,
                        name="ddiff" + str(obs_idx),
                    )
                else:
                    m.addConstr(
                        diff_var == logical_variables[obs_idx],
                        name="ddiff" + str(obs_idx),
                    )
            m.addConstr(gp.quicksum(diff_variables) >= 1, name="logical_difference")

        m.optimize()
        if m.status == GRB.INFEASIBLE and forbidden_logical is not None:
            m.close()
            return None
        if m.status != GRB.OPTIMAL:
            if verbose:
                print("Did not find optimal solution", m.status)
            m.close()
            raise RuntimeError(
                f"Gurobi did not find an optimal solution. Status: {m.status}"
            )

        error = np.round(
            np.array([var.X for var in error_variables]), decimals=0
        ).astype(bool)
        logical = np.round(
            np.array([var.X for var in logical_variables]), decimals=0
        ).astype(bool)
        objective_value = float(m.ObjVal)
        m.close()
        return self._ConfidenceSolveResult(
            error=error,
            logical=logical,
            objective=objective_value,
        )

    def _decode_with_logical_gap(
        self,
        detector_bits: npt.NDArray[np.bool_],
        verbose: bool = False,
    ) -> tuple[npt.NDArray[np.bool_], np.ndarray]:
        """Decode detector bits and return the logical-gap confidence score."""

        parent_decode_with_logical_gap = getattr(
            super(),
            "_decode_with_logical_gap",
            None,
        )
        if callable(parent_decode_with_logical_gap):
            return cast(
                tuple[npt.NDArray[np.bool_], np.ndarray],
                parent_decode_with_logical_gap(detector_bits, verbose=verbose),
            )

        single_shot = detector_bits.ndim == 1
        det_shots = detector_bits.reshape(1, -1) if single_shot else detector_bits

        decoded_obs = np.zeros(
            (det_shots.shape[0], self.num_observables),
            dtype=np.bool_,
        )
        logical_gaps = np.zeros(det_shots.shape[0], dtype=float)

        for shot_idx, detector_shot in enumerate(det_shots.astype(int)):
            best = self._solve_single_shot_for_confidence(
                detector_shot,
                verbose=verbose,
            )
            assert best is not None
            second = self._solve_single_shot_for_confidence(
                detector_shot,
                verbose=verbose,
                forbidden_logical=best.logical,
            )
            decoded_obs[shot_idx] = best.logical
            logical_gaps[shot_idx] = (
                np.inf if second is None else best.objective - second.objective
            )

        if single_shot:
            return decoded_obs[0], logical_gaps
        return decoded_obs, logical_gaps

    def decode_with_confidence(
        self,
        detector_bits: npt.NDArray[np.bool_],
    ) -> tuple[npt.NDArray[np.bool_], np.float64]:
        """Decode a single shot and return the logical-gap confidence."""

        if detector_bits.ndim != 1:
            raise ValueError(
                "decode_with_confidence expects a single detector shot (1D array)."
            )
        decoded_obs, logical_gap = self._decode_with_logical_gap(detector_bits)
        logical_gap_arr = np.asarray(logical_gap, dtype=np.float64).reshape(-1)
        return decoded_obs.astype(np.bool_), np.float64(logical_gap_arr[0])


__all__ = [
    "ConfidenceDecoder",
    "GurobiDecoderWithConfidence",
]
