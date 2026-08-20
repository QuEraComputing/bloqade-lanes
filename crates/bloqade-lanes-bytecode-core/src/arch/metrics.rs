//! Move-duration metrics: the FLAIR constant-jerk motion/timing model.
//!
//! [`MotionModel`] owns the physical timing constants (ramp rate, max jerk,
//! max acceleration) and the derived move-duration formula. It is the single
//! source of truth for lane-duration timing across the whole workspace: the
//! Rust search ([`crate`] consumers via `bloqade-lanes-search`) and the Python
//! layer both compute durations through this type, the latter via the
//! `bloqade.lanes.bytecode.MotionModel` PyO3 binding.
//!
//! The model is *configurable*: [`MotionModel::default`] reproduces the FLAIR
//! constants exactly (see [`FLAIR_MAX_RAMP_US`] and friends), but callers may
//! construct a [`MotionModel`] with different constants to model a different
//! motion profile without touching this code.

/// FLAIR maximum amplitude ramp rate (amplitude units per µs).
///
/// The ramp time for a pick or drop is `amplitude / max_ramp_us`.
/// Extracted from bloqade-flair's constant-jerk motion model.
pub const FLAIR_MAX_RAMP_US: f64 = 0.2;
/// FLAIR maximum jerk in µm/µs³. Extracted from bloqade-flair.
pub const FLAIR_MAX_JERK_UM_PER_US3: f64 = 0.0004;
/// FLAIR maximum acceleration in µm/µs². Extracted from bloqade-flair.
pub const FLAIR_MAX_ACCEL_UM_PER_US2: f64 = 0.0015;

/// Distance below which a move is treated as zero-duration (µm).
const MIN_MOVE_DISTANCE_UM: f64 = 1e-8;

/// Constant-jerk motion/timing model for AOD moves.
///
/// Holds the three timing constants and computes move durations from a
/// waypoint path. Defaults to the FLAIR constants; construct with
/// [`MotionModel::new`] to override them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionModel {
    /// Maximum amplitude ramp rate (amplitude units per µs). Must be > 0.
    pub max_ramp_us: f64,
    /// Maximum jerk in µm/µs³. Must be > 0.
    pub max_jerk_um_per_us3: f64,
    /// Maximum acceleration in µm/µs². Must be > 0.
    pub max_accel_um_per_us2: f64,
}

impl Default for MotionModel {
    /// The FLAIR constant-jerk motion model.
    fn default() -> Self {
        Self {
            max_ramp_us: FLAIR_MAX_RAMP_US,
            max_jerk_um_per_us3: FLAIR_MAX_JERK_UM_PER_US3,
            max_accel_um_per_us2: FLAIR_MAX_ACCEL_UM_PER_US2,
        }
    }
}

impl MotionModel {
    /// Construct a motion model from explicit constants.
    ///
    /// All three constants must be positive and finite. `max_ramp_us` and
    /// `max_jerk_um_per_us3` appear as divisors; a non-positive
    /// `max_accel_um_per_us2` drives `t1` to zero (or negative), which makes
    /// the trajectory solver return `NaN`/negative durations. Returns `None`
    /// for any non-physical constant.
    pub fn new(
        max_ramp_us: f64,
        max_jerk_um_per_us3: f64,
        max_accel_um_per_us2: f64,
    ) -> Option<Self> {
        if !max_ramp_us.is_finite()
            || !max_jerk_um_per_us3.is_finite()
            || !max_accel_um_per_us2.is_finite()
            || max_ramp_us <= 0.0
            || max_jerk_um_per_us3 <= 0.0
            || max_accel_um_per_us2 <= 0.0
        {
            return None;
        }
        Some(Self {
            max_ramp_us,
            max_jerk_um_per_us3,
            max_accel_um_per_us2,
        })
    }

    /// The FLAIR constant-jerk motion model (alias for [`Default`]).
    pub fn flair() -> Self {
        Self::default()
    }

    /// Minimum duration (µs) for a constant-jerk move over `max_dist_um`.
    ///
    /// Solves the constant-jerk trajectory: below the acceleration cap the
    /// move is jerk-limited (four jerk phases); above it, two extra
    /// constant-acceleration phases are inserted.
    pub fn const_jerk_min_duration_us(&self, max_dist_um: f64) -> f64 {
        let max_dist_um = max_dist_um.abs();
        if max_dist_um < MIN_MOVE_DISTANCE_UM {
            return 0.0;
        }

        let t1 = self.max_accel_um_per_us2 / self.max_jerk_um_per_us3;
        let a = self.max_jerk_um_per_us3 * t1;
        let b = 3.0 * self.max_jerk_um_per_us3 * t1 * t1;
        let c = 2.0 * self.max_jerk_um_per_us3 * t1 * t1 * t1 - max_dist_um;

        if c >= 0.0 {
            let t1_jerk = (max_dist_um / (2.0 * self.max_jerk_um_per_us3)).cbrt();
            return 4.0 * t1_jerk;
        }

        let discriminant = b * b - 4.0 * a * c;
        let t2 = (-b + discriminant.sqrt()) / (2.0 * a);
        4.0 * t1 + 2.0 * t2
    }

    /// Compute lane duration (µs) from a waypoint path:
    /// `ramp + Σ per-segment const-jerk duration + ramp`.
    ///
    /// The pick and drop ramps are always charged, so a path with no motion
    /// segments (fewer than two waypoints) still costs `2 * ramp` rather than
    /// collapsing to a free move. `amplitude_delta` scales the ramp time; its
    /// sign is ignored.
    pub fn lane_duration_us(&self, waypoints: &[[f64; 2]], amplitude_delta: f64) -> f64 {
        let ramp = amplitude_delta.abs() / self.max_ramp_us;
        let segment_sum: f64 = waypoints
            .windows(2)
            .map(|w| {
                let dx = w[1][0] - w[0][0];
                let dy = w[1][1] - w[0][1];
                self.const_jerk_min_duration_us((dx * dx + dy * dy).sqrt())
            })
            .sum();
        ramp + segment_sum + ramp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_flair_constants() {
        let m = MotionModel::default();
        assert_eq!(m.max_ramp_us, FLAIR_MAX_RAMP_US);
        assert_eq!(m.max_jerk_um_per_us3, FLAIR_MAX_JERK_UM_PER_US3);
        assert_eq!(m.max_accel_um_per_us2, FLAIR_MAX_ACCEL_UM_PER_US2);
        assert_eq!(m, MotionModel::flair());
    }

    #[test]
    fn zero_and_tiny_distance_is_zero_duration() {
        let m = MotionModel::default();
        assert_eq!(m.const_jerk_min_duration_us(0.0), 0.0);
        assert_eq!(m.const_jerk_min_duration_us(1e-12), 0.0);
    }

    #[test]
    fn duration_is_even_in_sign_and_increasing() {
        let m = MotionModel::default();
        let d10 = m.const_jerk_min_duration_us(10.0);
        assert_eq!(d10, m.const_jerk_min_duration_us(-10.0));
        assert!(d10 > 0.0);
        assert!(m.const_jerk_min_duration_us(50.0) > d10);
    }

    #[test]
    fn both_trajectory_branches_are_exercised() {
        let m = MotionModel::default();
        // t1 = accel/jerk = 3.75 µs; jerk-only reach = 2*jerk*t1^3 ≈ 0.0422 µm.
        // A sub-threshold distance takes the pure-jerk (cbrt) branch; a large
        // one takes the constant-acceleration (quadratic) branch. Both must be
        // finite and ordered.
        let small = m.const_jerk_min_duration_us(0.01);
        let large = m.const_jerk_min_duration_us(100.0);
        assert!(small.is_finite() && small > 0.0);
        assert!(large.is_finite() && large > small);
    }

    #[test]
    fn lane_duration_is_ramp_plus_segments_plus_ramp() {
        let m = MotionModel::default();
        let waypoints = [[0.0, 0.0], [3.0, 4.0]]; // one 5 µm segment
        let ramp = 1.0 / m.max_ramp_us;
        let expected = ramp + m.const_jerk_min_duration_us(5.0) + ramp;
        assert_eq!(m.lane_duration_us(&waypoints, 1.0), expected);
    }

    #[test]
    fn lane_duration_of_pathless_move_is_two_ramps() {
        let m = MotionModel::default();
        // No motion segments — still pays the pick and drop ramps, never free.
        let two_ramps = 2.0 * (1.0 / m.max_ramp_us);
        assert_eq!(m.lane_duration_us(&[], 1.0), two_ramps);
        assert_eq!(m.lane_duration_us(&[[1.0, 2.0]], 1.0), two_ramps);
    }

    #[test]
    fn amplitude_scales_ramp_only() {
        let m = MotionModel::default();
        let waypoints = [[0.0, 0.0], [3.0, 4.0]];
        let seg = m.const_jerk_min_duration_us(5.0);
        // ramp = amp / max_ramp; total ramp contribution is 2 * ramp.
        assert_eq!(
            m.lane_duration_us(&waypoints, 2.0),
            2.0 * (2.0 / m.max_ramp_us) + seg
        );
        // Sign of amplitude is ignored.
        assert_eq!(
            m.lane_duration_us(&waypoints, -1.0),
            m.lane_duration_us(&waypoints, 1.0)
        );
    }

    #[test]
    fn new_rejects_non_physical_constants() {
        assert!(MotionModel::new(0.0, 1.0, 1.0).is_none());
        assert!(MotionModel::new(1.0, 0.0, 1.0).is_none());
        assert!(MotionModel::new(1.0, 1.0, 0.0).is_none());
        assert!(MotionModel::new(1.0, 1.0, -1.0).is_none());
        assert!(MotionModel::new(-1.0, 1.0, 1.0).is_none());
        assert!(MotionModel::new(f64::NAN, 1.0, 1.0).is_none());
        assert!(MotionModel::new(0.2, 0.0004, 0.0015).is_some());
    }
}
