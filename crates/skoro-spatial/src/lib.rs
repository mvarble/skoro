//! The rotations and rigid motions of space: the unit quaternions `S³`, the rigid motions
//! `S³ ⋉ ℝ³` and the extended poses `S³ ⋉ (ℝ³ × ℝ³)`, declared as embedded types with their
//! volumes, and their maps written as composites of the core's operations.

mod algebra;
mod euler;
mod extended;
pub mod host;
pub mod pose;
pub mod rotation;
mod series;
pub mod so3;
mod types;

pub use euler::{Angles, Axis, Euler, Frame, Sequence, angle_residual};
pub use extended::ExtendedPoseMaps;
pub use pose::PoseMaps;
pub use rotation::RotationMaps;
pub use types::{EXTENDED_POSE, ExtendedPose, POSE, Pose, ROTATION, Rotation, register};

/// The free functions of the extended poses.
pub mod extended_pose {
    pub use crate::extended::{exp, from_parts, identity, jl, jl_inv, jr, jr_inv};
}
