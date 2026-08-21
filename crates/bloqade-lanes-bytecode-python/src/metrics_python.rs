//! PyO3 binding for the FLAIR constant-jerk motion/timing model.
//!
//! Exposes [`bloqade_lanes_bytecode_core::arch::metrics::MotionModel`] to
//! Python as `bloqade.lanes.bytecode.MotionModel`, so the Python layer computes
//! lane durations through the same implementation as the Rust search rather
//! than a hand-maintained transcription.

use pyo3::prelude::*;

use bloqade_lanes_bytecode_core::arch::metrics as rs;

/// Constant-jerk motion/timing model for AOD move durations.
///
/// Defaults to the FLAIR constants; pass explicit values to model a different
/// motion profile. The pick/drop ramp and per-segment jerk math are computed
/// in Rust and shared with the move-search solver.
#[pyclass(
    name = "MotionModel",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone)]
pub struct PyMotionModel {
    pub(crate) inner: rs::MotionModel,
}

#[pymethods]
impl PyMotionModel {
    #[new]
    #[pyo3(signature = (
        max_ramp_us = rs::FLAIR_MAX_RAMP_US,
        max_jerk_um_per_us3 = rs::FLAIR_MAX_JERK_UM_PER_US3,
        max_accel_um_per_us2 = rs::FLAIR_MAX_ACCEL_UM_PER_US2,
    ))]
    fn new(
        max_ramp_us: f64,
        max_jerk_um_per_us3: f64,
        max_accel_um_per_us2: f64,
    ) -> PyResult<Self> {
        match rs::MotionModel::new(max_ramp_us, max_jerk_um_per_us3, max_accel_um_per_us2) {
            Some(inner) => Ok(Self { inner }),
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "MotionModel requires finite constants with max_ramp_us > 0, \
                 max_jerk_um_per_us3 > 0, and max_accel_um_per_us2 > 0",
            )),
        }
    }

    /// The FLAIR constant-jerk motion model (the default constants).
    #[staticmethod]
    fn flair() -> Self {
        Self {
            inner: rs::MotionModel::flair(),
        }
    }

    /// Maximum amplitude ramp rate (amplitude units per µs).
    #[getter]
    fn max_ramp_us(&self) -> f64 {
        self.inner.max_ramp_us
    }

    /// Maximum jerk in µm/µs³.
    #[getter]
    fn max_jerk_um_per_us3(&self) -> f64 {
        self.inner.max_jerk_um_per_us3
    }

    /// Maximum acceleration in µm/µs².
    #[getter]
    fn max_accel_um_per_us2(&self) -> f64 {
        self.inner.max_accel_um_per_us2
    }

    /// Minimum duration (µs) of a constant-jerk move over `max_dist_um`.
    ///
    /// The sign of `max_dist_um` is ignored; a distance below ~1e-8 µm is
    /// treated as no move and returns 0.0.
    fn const_jerk_min_duration_us(&self, max_dist_um: f64) -> f64 {
        self.inner.const_jerk_min_duration_us(max_dist_um)
    }

    /// Lane duration (µs) over a waypoint path: ramp + Σ segments + ramp.
    ///
    /// `waypoints` are `(x, y)` coordinates in µm. The pick/drop ramps are
    /// always charged, so a path with no motion segments (fewer than two
    /// waypoints) still costs `2 * ramp` rather than zero. `amplitude_delta`
    /// scales the ramp time; its sign is ignored.
    #[pyo3(signature = (waypoints, amplitude_delta = 1.0))]
    fn lane_duration_us(&self, waypoints: Vec<(f64, f64)>, amplitude_delta: f64) -> f64 {
        let waypoints: Vec<[f64; 2]> = waypoints.into_iter().map(|(x, y)| [x, y]).collect();
        self.inner.lane_duration_us(&waypoints, amplitude_delta)
    }

    fn __repr__(&self) -> String {
        format!(
            "MotionModel(max_ramp_us={}, max_jerk_um_per_us3={}, max_accel_um_per_us2={})",
            self.inner.max_ramp_us, self.inner.max_jerk_um_per_us3, self.inner.max_accel_um_per_us2,
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
