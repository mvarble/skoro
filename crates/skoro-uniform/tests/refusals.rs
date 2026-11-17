#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! What the uniform refuses: a space without a finite mass, and a quantity of another space.

use skoro_core::scope::{quantity, sample};
use skoro_core::{Declares, Error, Failure, Integer, Positive, Real, Space, Vector, Writer};
use skoro_spatial::{Pose, Rotation};
use skoro_uniform::Uniform;

/// The failure of scoring a quantity of `marker`'s space by its uniform, and the line of the call.
fn refusal<T: Declares + Copy>(marker: T) -> (Failure, u32) {
    let line = line!() + 4;
    let failure = Writer::new()
        .build(|| {
            let x = quantity("x", marker);
            Uniform::on(marker).score(x);
        })
        .unwrap_err();
    (failure, line)
}

#[test]
fn a_space_of_infinite_mass_is_refused() {
    let refusals = [
        refusal(Real),
        refusal(Vector::<3>),
        refusal(Positive::<2>),
        refusal(Integer),
        refusal(Pose),
    ];
    let spaces = [
        Real.space(),
        Vector::<3>.space(),
        Positive::<2>.space(),
        Integer.space(),
        Pose.space(),
    ];
    for ((failure, line), space) in refusals.into_iter().zip(spaces) {
        assert_eq!(failure.error, Error::NoMass { space });
        assert_eq!(failure.location.line(), line);
        assert!(failure.location.file().ends_with("refusals.rs"));
    }
    // The same of a quantity declared and scored in one call.
    let line = line!() + 3;
    let failure = Writer::new()
        .build(|| {
            sample("x", Real, Uniform::on(Real));
        })
        .unwrap_err();
    assert_eq!(failure.error, Error::NoMass { space: Space::Real });
    assert_eq!(failure.location.line(), line);
}

#[test]
fn a_quantity_of_another_space_is_refused() {
    let line = line!() + 4;
    let failure = Writer::new()
        .build(|| {
            let x = quantity("x", Vector::<4>);
            Uniform::on(Rotation).score(x);
        })
        .unwrap_err();
    assert_eq!(
        failure.error,
        Error::SpaceMismatch {
            operation: "skoro.uniform.log_density",
            inputs: vec![Space::Vector(4)],
        }
    );
    assert_eq!(failure.location.line(), line);
}
