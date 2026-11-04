//! The rotations and rigid motions of space: the unit quaternions `S³`, the rigid motions
//! `S³ ⋉ ℝ³` and the extended poses `S³ ⋉ (ℝ³ × ℝ³)`, declared as embedded types with their
//! volumes, and their maps written as composites of the core's operations.

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the arithmetic of the declarations and maps that later tasks add"
    )
)]
mod algebra;
