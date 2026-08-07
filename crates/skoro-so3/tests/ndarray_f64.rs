#[allow(dead_code)]
mod common;

type B = skoro_backend::NdArrayBackend<f64, i32>;

mod exp {
    super::test_exp!(
        (skoro_so3::ops::exp::ndarray::exp, 1.0e-5),
        (skoro_so3::ops::exp::ndarray::exp_jvp, 1.0e-5),
        (skoro_so3::ops::exp::ndarray::exp_vjp, 1.0e-5),
        (1.0e-6, 1e-5),
    );
}

mod log {
    super::test_log!(
        (skoro_so3::ops::log::ndarray::log, 1.0e-4),
        (skoro_so3::ops::log::ndarray::log_jvp, 1.0e-4),
        (skoro_so3::ops::log::ndarray::log_vjp, 1.0e-4),
        (1.0e-6, 1e-5),
    );
}

mod retraction {
    super::test_retract!(1.0e-6, 1.0e-5);
}

mod inverse_retraction {
    super::test_inverse_retract!(1.0e-6, 1.0e-5);
}

mod coefficients {
    super::test_hvp_reference!(
        exp_hvp_tracks_the_f64_reference,
        skoro_so3::ops::exp::ndarray::exp_hvp,
        EXP_HVP,
        1e-12,
    );
    super::test_hvp_reference!(
        log_hvp_tracks_the_f64_reference,
        skoro_so3::ops::log::ndarray::log_hvp,
        LOG_HVP,
        1e-12,
    );
}
