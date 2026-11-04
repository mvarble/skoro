//! The rotations and rigid motions of space: the unit quaternions `S³`, the rigid motions
//! `S³ ⋉ ℝ³` and the extended poses `S³ ⋉ (ℝ³ × ℝ³)`, declared as embedded types with their
//! volumes, and their maps written as composites of the core's operations.

mod algebra;
mod euler;
pub mod host;
pub mod rotation;
mod series;
pub mod so3;
mod types;

pub use euler::Axis;
pub use rotation::RotationMaps;
pub use types::{EXTENDED_POSE, ExtendedPose, POSE, Pose, ROTATION, Rotation, register};
