use approx::assert_abs_diff_eq;
use burn_wgpu::WgpuRuntime;
use skoro_backend::{Backend, CubeBackend, NdArrayBackend, Tensor, TensorData};
use skoro_graph::{
    graph::{EvalOrder2, PointEvalOrder1, PointJvpEval},
    point_map::{points, tangent_frames, tangents},
    quantity::ContinuousQuantity,
};
use skoro_model::model::ModelClient;
use skoro_riemann::euclidean::Euclidean3;

use crate::quantity::SO3;

#[rustfmt::skip]
const R0S: [f32; 9] = [
    0.877637, 0.479094, -0.0148907,
    -0.478116, 0.872793, -0.0981673,
    -0.0340349, 0.0932747, 0.995059,
];

#[rustfmt::skip]
const R1S: [f32; 9] = [
    0.955339, -0.00372177, -0.295489,
    -0.00372177, 0.99969, -0.0246241,
    0.295489, 0.0246241, 0.955029,
];

#[rustfmt::skip]
const VS: [f32; 3] = [0.000212108, -0.297136, 0.511331];

#[rustfmt::skip]
const DR0S: [f32; 9] = [
    -0.970683, -0.25566, -0.148577,
    0.255671, -0.978084, 0.0126296,
    0.148559, 0.0128417, -0.992599,
];

#[rustfmt::skip]
const DR1S: [f32; 9] = [
    0.970683, -0.255671, -0.148559,
    0.25566, 0.978084, -0.0128417,
    0.148577, -0.0126296, 0.992599,
];

fn inverse_retract_impl<B>()
where
    B: Backend,
    SO3: ContinuousQuantity<B>,
{
    let model: ModelClient<B> = Default::default();
    let r0 = model.add_parameter(SO3);
    let r1 = model.add_parameter(SO3);
    let v = r0.inverse_retract::<Euclidean3>(&r1);
    let points = points()
        .with(
            r0.id,
            Tensor::<B, 2>::from_data(TensorData::new(R0S.to_vec(), [1, 9]), &Default::default()),
        )
        .with(
            r1.id,
            Tensor::<B, 2>::from_data(TensorData::new(R1S.to_vec(), [1, 9]), &Default::default()),
        );
    let value = v.evaluate(&points).unwrap();
    value
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&VS)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));

    let ts = tangent_frames().with(
        r0.id,
        Tensor::<B, 3>::from_floats(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
            &Default::default(),
        ),
    );
    let PointJvpEval {
        value,
        tangents: mut ts,
    } = v.evaluate_jvp(&points, &ts).unwrap();
    value
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&VS)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));
    ts.remove(&v.id)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&DR0S)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));
    let ts = tangent_frames().with(
        r1.id,
        Tensor::<B, 3>::from_floats(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
            &Default::default(),
        ),
    );
    let PointJvpEval {
        value,
        tangents: mut ts,
    } = v.evaluate_jvp(&points, &ts).unwrap();
    value
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&VS)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));
    ts.remove(&v.id)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&DR1S)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));
    let PointEvalOrder1 {
        value,
        mut cotangents,
    } = v.evaluate_ord1(&points).unwrap();
    value
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&VS)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));
    cotangents
        .remove(&r0.id)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&[-0.566453, -1.2209, -1.12855])
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));
    cotangents
        .remove(&r1.id)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&[1.37492, 0.709783, 0.831198])
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));

    let dr0 = Tensor::<B, 2>::from_floats([[0.1, 0.5, -0.3]], &Default::default());
    let dr0 = dr0.clone() / burn::tensor::linalg::l2_norm(dr0, 1);
    let dr1 = Tensor::<B, 2>::from_floats([[-0.03, 0.01, 0.1]], &Default::default());
    let dr1 = dr1.clone() / burn::tensor::linalg::l2_norm(dr1, 1);
    let ts = tangents().with(r0.id, dr0.clone()).with(r1.id, dr1.clone());
    let EvalOrder2 {
        mut primal,
        mut ord1_cotangents,
        mut ord2_cotangents,
        ..
    } = model.evaluate_ids_ord2(&points, &ts, &[v.id]).unwrap();
    primal
        .evaluations
        .remove(&v.id)
        .unwrap()
        .0
        .into_float()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&VS)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));
    ord1_cotangents
        .get(&r0.id)
        .unwrap()
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&[-0.566453, -1.2209, -1.12855])
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));
    ord1_cotangents
        .get(&r1.id)
        .unwrap()
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&[1.37492, 0.709783, 0.831198])
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));

    let delta = 1e-4;
    let ts = tangents()
        .with(r0.id, delta * dr0)
        .with(r1.id, delta * dr1)
        .build();
    let points = primal.variables.retract(&ts).unwrap();

    let PointEvalOrder1 {
        cotangents: cotangents1,
        ..
    } = v.evaluate_ord1(points.as_ref()).unwrap();

    for (key, vjp1) in cotangents1 {
        let vjp0 = ord1_cotangents.remove(&key).unwrap();
        let hvp_numeric = ((vjp1 - vjp0) / delta).into_data().to_vec::<f32>().unwrap();
        let hvp = ord2_cotangents
            .remove(&key)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        hvp.into_iter()
            .zip(hvp_numeric)
            .for_each(|(x, y)| assert_abs_diff_eq!(x, y, epsilon = 0.1));
    }
}

#[test]
fn inverse_retract() {
    inverse_retract_impl::<CubeBackend<WgpuRuntime, f32, i32>>();
    inverse_retract_impl::<NdArrayBackend<f32, i32>>();
}
