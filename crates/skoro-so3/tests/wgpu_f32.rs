#[allow(dead_code)]
mod common;

type B = skoro_backend::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32>;

mod exp {
    super::test_exp!(
        (skoro_so3::ops::exp::cube::exp, 1.0e-4),
        (skoro_so3::ops::exp::cube::exp_jvp, 1.0e-4),
        (skoro_so3::ops::exp::cube::exp_vjp, 1.0e-4),
        (1.0e-3, 1e-3),
    );
}

mod log {
    super::test_log!(
        (skoro_so3::ops::log::cube::log, 1.0e-4),
        (skoro_so3::ops::log::cube::log_jvp, 1.0e-4),
        (skoro_so3::ops::log::cube::log_vjp, 1.0e-4),
        (1.0e-3, 1e-2),
    );
}

mod retraction {
    super::test_retract!(1.0e-3, 1.0e-2);
}

mod inverse_retraction {
    super::test_inverse_retract!(1.0e-3, 1.0e-2);
}

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
