#[allow(dead_code)]
mod common;

type B = skoro_backend::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32>;

/// Rodrigues' formula and its derivatives, against forward differences at this backend's
/// precision. The small-angle branch is where the closed form loses its denominator, so the
/// tolerances are supplied per backend rather than shared.
mod exp {
    super::test_exp!(
        (skoro_so3::ops::exp::cube::exp, 1.0e-4),
        (skoro_so3::ops::exp::cube::exp_jvp, 1.0e-4),
        (skoro_so3::ops::exp::cube::exp_vjp, 1.0e-4),
        (1.0e-3, 1e-3),
    );
}

/// The inverse map, and its derivatives. `log` is where the angle is recovered from a matrix that
/// may have drifted off the group, so it is checked over the same span of angles `exp` is.
mod log {
    super::test_log!(
        (skoro_so3::ops::log::cube::log, 1.0e-4),
        (skoro_so3::ops::log::cube::log_jvp, 1.0e-4),
        (skoro_so3::ops::log::cube::log_vjp, 1.0e-4),
        (1.0e-3, 1e-2),
    );
}

/// `retract` on `SO3`: `exp` of the tangent, composed onto the point. This is the step an optimizer
/// takes, so what matters is that it lands on the group rather than near it.
mod retraction {
    super::test_retract!(1.0e-3, 1.0e-2);
}

/// The group operations on stored rotations, differentiated in the body-frame chart.
mod group {
    super::test_group!(1.0e-3, 1.0e-2);
}

/// `b ⊖ a` on `SO3`, the genuine `Log(aᵀb)` rather than a subtraction. This is the residual a
/// Gaussian factor over rotations whitens, and the sign convention is fixed by it.
mod inverse_retraction {
    super::test_inverse_retract!(1.0e-3, 1.0e-2);
}

/// The `hvp` coefficient series, against a frozen `f64` reference table rather than against
/// forward differences. The second derivative of the series is where catastrophic cancellation
/// lives, and a difference quotient at this precision cannot tell a correct series from a wrong
/// one -- so the reference is computed once, in `f64`, and compared to.
mod coefficients {
    super::test_hvp_reference!(
        exp_hvp_tracks_the_f64_reference,
        skoro_so3::ops::exp::cube::exp_hvp,
        EXP_HVP,
        1e-6,
    );
    super::test_hvp_reference!(
        log_hvp_tracks_the_f64_reference,
        skoro_so3::ops::log::cube::log_hvp,
        LOG_HVP,
        1e-6,
    );
}
