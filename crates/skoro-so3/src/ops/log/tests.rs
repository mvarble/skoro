use skoro_backend::{Backend, CubeBackend, NdArrayBackend, Tensor, TensorData};

use burn_wgpu::WgpuRuntime;
use skoro_graph::{point_map::points, quantity::ContinuousQuantity};
use skoro_model::model::ModelClient;
use skoro_riemann::euclidean::Euclidean;

use super::{SO3LogBackend, SO3LogOps, cube, ndarray};
use crate::quantity::SO3;

use crate::ops::exp::tests::{EXP_OMEGAS, OMEGAS, VS};

#[rustfmt::skip]
pub(crate) const LOG_R_GRADS: [f32; 24] = [
    1.19165, 0.839983, 0.946661,
    1.0, 1.22908, 0.729079,
    0.729079, 1.0, 1.22908,
    1.22908, 0.729079, 1.0,
    1.0, 1.0, 1.0,
    0.557636, 1.58222, 0.641549,
    0.415244, 1.0, 1.41524,
    1.17453, 0.2634, 1.33623,
];

#[rustfmt::skip]
pub(crate) const DLOG_OMEGA_VS: [f32; 48] = [
    1.89191, -0.978536, -0.513473,
    1.53601, -0.729345, -1.02445,
    -1.43612, 1.49531, -0.370745,
    1.07595, -0.368886, 0.57533,
    -0.861851, 0.172781, -0.273418,
    0.1344, -0.124875, 1.05335,
    -0.374246, 0.407579, 0.301074,
    -0.367138, -0.284672, 0.635578,
    -1.45024, -0.207513, -0.348411,
    -1.25148, 1.4975, -1.35245,
    -0.564793, 0.329759, -0.761193,
    -1.59154, -1.2199, 0.791805,
    0.716051, 0.902352, -1.54064,
    -0.927759, 0.30094, -0.818062,
    1.61153, -2.73518, 1.18719,
    -0.414265, -0.541505, 0.298255,
];

#[rustfmt::skip]
pub(crate) const DLOGT_OMEGA_VS: [f32; 48] = [
    1.45891, -1.48539, -0.682425,
    1.14393, -1.13472, -1.15957,
    -1.43612, 1.13451, -1.04225,
    1.07595, -0.0479003, 0.681753,
    -0.625262, 0.172781, -0.65314,
    -0.387055, -0.124875, 0.988833,
    -0.133027, 0.537108, 0.301074,
    -0.458673, -0.0738079, 0.635578,
    -1.45024, -0.207512, -0.348411,
    -1.25147, 1.4975, -1.35245,
    -0.246113, -0.567959, -0.789935,
    -1.30853, -1.08761, 1.32399,
    1.68329, 0.902352, -0.229875,
    0.187105, 0.30094, -1.22268,
    0.671003, -2.69979, -1.93612,
    -0.570463, -0.439215, -0.18831,
];

#[rustfmt::skip]
pub(crate) const ALPHAS: [f32; 24] = [
    -1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, -1.0,
    0.0, -1.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 0.0, -1.0,
    0.0, 1.0, 0.0,
    0.0, -1.0, 0.0,
];

#[test]
fn cube_log() {
    let expomega = Tensor::<CubeBackend<WgpuRuntime, f32, i32>, 2>::from_data(
        TensorData::new(EXP_OMEGAS[9..81].to_vec(), [8, 9]),
        &Default::default(),
    );
    let omega = cube::log(expomega);
    let omega = omega.into_data().to_vec::<f32>().unwrap();
    omega
        .into_iter()
        .zip(&OMEGAS[3..27])
        .for_each(|(x, xtrue)| approx::assert_abs_diff_eq!(x, xtrue, epsilon = 1e-3));
}

#[test]
fn ndarray_log() {
    let expomega = Tensor::<NdArrayBackend<f32, i32>, 2>::from_data(
        TensorData::new(EXP_OMEGAS[9..81].to_vec(), [8, 9]),
        &Default::default(),
    );
    let omega = ndarray::log(expomega);
    let omega = omega.into_data().to_vec::<f32>().unwrap();
    omega
        .into_iter()
        .zip(&OMEGAS[3..27])
        .for_each(|(x, xtrue)| approx::assert_abs_diff_eq!(x, xtrue, epsilon = 1e-3));
}

#[test]
fn cube_log_jvp() {
    let expomega = Tensor::<CubeBackend<WgpuRuntime, f32, i32>, 2>::from_data(
        TensorData::new(EXP_OMEGAS[9..81].to_vec(), [8, 9]),
        &Default::default(),
    );
    let v = Tensor::<CubeBackend<WgpuRuntime, f32, i32>, 3>::from_data(
        TensorData::new(VS[6..54].to_vec(), [8, 2, 3]),
        &Default::default(),
    )
    .transpose();
    let omega = cube::log(expomega);
    let dlogomegav = cube::log_jvp(omega, v).transpose();
    let dlogomegav = dlogomegav.into_data().to_vec::<f32>().unwrap();
    dlogomegav
        .into_iter()
        .zip(DLOG_OMEGA_VS)
        .for_each(|(x, xtrue)| approx::assert_abs_diff_eq!(x, xtrue, epsilon = 1e-3));
}

#[test]
fn ndarray_log_jvp() {
    let expomega = Tensor::<NdArrayBackend<f32, i32>, 2>::from_data(
        TensorData::new(EXP_OMEGAS[9..81].to_vec(), [8, 9]),
        &Default::default(),
    );
    let v = Tensor::<NdArrayBackend<f32, i32>, 3>::from_data(
        TensorData::new(VS[6..54].to_vec(), [8, 2, 3]),
        &Default::default(),
    )
    .transpose();
    let omega = ndarray::log(expomega);
    let dlogomegav = ndarray::log_jvp(omega, v).transpose();
    let dlogomegav = dlogomegav.into_data().to_vec::<f32>().unwrap();
    dlogomegav
        .into_iter()
        .zip(DLOG_OMEGA_VS)
        .for_each(|(x, xtrue)| approx::assert_abs_diff_eq!(x, xtrue, epsilon = 1e-3));
}

#[test]
fn cube_log_vjp() {
    let expomega = Tensor::<CubeBackend<WgpuRuntime, f32, i32>, 2>::from_data(
        TensorData::new(EXP_OMEGAS[9..81].to_vec(), [8, 9]),
        &Default::default(),
    );
    let v = Tensor::<CubeBackend<WgpuRuntime, f32, i32>, 3>::from_data(
        TensorData::new(VS[6..54].to_vec(), [8, 2, 3]),
        &Default::default(),
    )
    .transpose();
    let omega = cube::log(expomega);
    let dlogomegav = cube::log_vjp(omega, v).transpose();
    let dlogomegav = dlogomegav.into_data().to_vec::<f32>().unwrap();
    dlogomegav
        .into_iter()
        .zip(DLOGT_OMEGA_VS)
        .for_each(|(x, xtrue)| approx::assert_abs_diff_eq!(x, xtrue, epsilon = 1e-3));
}

#[test]
fn ndarray_log_vjp() {
    let expomega = Tensor::<NdArrayBackend<f32, i32>, 2>::from_data(
        TensorData::new(EXP_OMEGAS[9..81].to_vec(), [8, 9]),
        &Default::default(),
    );
    let v = Tensor::<NdArrayBackend<f32, i32>, 3>::from_data(
        TensorData::new(VS[6..54].to_vec(), [8, 2, 3]),
        &Default::default(),
    )
    .transpose();
    let omega = ndarray::log(expomega);
    let dlogomegav = ndarray::log_vjp(omega, v).transpose();
    let dlogomegav = dlogomegav.into_data().to_vec::<f32>().unwrap();
    dlogomegav
        .into_iter()
        .zip(DLOGT_OMEGA_VS)
        .for_each(|(x, xtrue)| approx::assert_abs_diff_eq!(x, xtrue, epsilon = 1e-3));
}

#[test]
fn cube_log_hvp() {
    let omega0 = Tensor::<CubeBackend<WgpuRuntime, f32, i32>, 2>::from_data(
        TensorData::new(OMEGAS[..12].to_vec(), [4, 3]),
        &Default::default(),
    );
    let v = Tensor::<CubeBackend<WgpuRuntime, f32, i32>, 2>::from_data(
        TensorData::new(VS[..12].to_vec(), [4, 3]),
        &Default::default(),
    );
    let vdir = v.clone() / burn::tensor::linalg::l2_norm(v.clone(), 1);

    let delta: f32 = 1e-4;
    let omega1 = omega0.clone() + delta * vdir.clone();

    let alpha = Tensor::<CubeBackend<WgpuRuntime, f32, i32>, 3>::from_data(
        TensorData::new(ALPHAS.to_vec(), [4, 2, 3]),
        &Default::default(),
    )
    .transpose();

    let vjp0 = cube::log_vjp(omega0.clone(), alpha.clone());
    let vjp1 = cube::log_vjp(omega1.clone(), alpha.clone());
    let hvp_numeric = ((vjp1 - vjp0) / delta).into_data().to_vec::<f32>().unwrap();
    let hvp = cube::log_hvp(omega0, vdir, alpha)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    hvp_numeric
        .into_iter()
        .zip(hvp)
        .for_each(|(x, y)| approx::assert_abs_diff_eq!(x, y, epsilon = 1e-2));
}

#[test]
fn ndarray_log_hvp() {
    let omega0 = Tensor::<NdArrayBackend<f32, i32>, 2>::from_data(
        TensorData::new(OMEGAS[..12].to_vec(), [4, 3]),
        &Default::default(),
    );
    let v = Tensor::<NdArrayBackend<f32, i32>, 2>::from_data(
        TensorData::new(VS[..12].to_vec(), [4, 3]),
        &Default::default(),
    );
    let vdir = v.clone() / burn::tensor::linalg::l2_norm(v.clone(), 1);

    let delta: f32 = 1e-4;
    let omega1 = omega0.clone() + delta * vdir.clone();

    let alpha = Tensor::<NdArrayBackend<f32, i32>, 3>::from_data(
        TensorData::new(ALPHAS.to_vec(), [4, 2, 3]),
        &Default::default(),
    )
    .transpose();

    let vjp0 = ndarray::log_vjp(omega0.clone(), alpha.clone());
    let vjp1 = ndarray::log_vjp(omega1.clone(), alpha.clone());
    let hvp_numeric = ((vjp1 - vjp0) / delta).into_data().to_vec::<f32>().unwrap();
    let hvp = ndarray::log_hvp(omega0, vdir, alpha)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    hvp_numeric
        .into_iter()
        .zip(hvp)
        .for_each(|(x, y)| approx::assert_abs_diff_eq!(x, y, epsilon = 1e-2));
}

fn log_generic<B: Backend + SO3LogBackend<Euclidean<3>>>()
where
    SO3: ContinuousQuantity<B>,
{
    let model = ModelClient::<B>::new(Default::default());
    let expomega = model.add_parameter(SO3);
    let mut eval = expomega
        .log()
        .evaluate_ord1(&points().with(
            expomega.id,
            Tensor::<B, 2>::from_data(
                TensorData::new(EXP_OMEGAS[9..72].to_vec(), [7, 9]),
                &Default::default(),
            ),
        ))
        .unwrap();
    let omega = eval.value.into_data().to_vec::<f32>().unwrap();
    omega
        .into_iter()
        .zip(&OMEGAS[3..21])
        .for_each(|(x, xtrue)| approx::assert_abs_diff_eq!(x, xtrue, epsilon = 1e-3));
    let omega_grad = eval
        .cotangents
        .remove(&expomega.id)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    omega_grad
        .into_iter()
        .zip(LOG_R_GRADS)
        .for_each(|(x, xtrue)| approx::assert_abs_diff_eq!(x, xtrue, epsilon = 1e-3));
}

#[test]
fn log() {
    log_generic::<CubeBackend<WgpuRuntime, f32, i32>>();
    log_generic::<NdArrayBackend<f32, i32>>();
}
