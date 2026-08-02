use approx::assert_abs_diff_eq;
use skoro_backend::{Backend, CubeBackend, NdArrayBackend, Tensor, TensorData};

use burn_wgpu::WgpuRuntime;
use skoro_graph::{
    graph::{EvalOrder2, PointEvalOrder1, PointJvpEval},
    point_map::{points, tangent_frames, tangents},
    quantity::ContinuousQuantity,
};
use skoro_model::model::ModelClient;
use skoro_riemann::euclidean::Euclidean;

use crate::quantity::SO3;

#[rustfmt::skip]
const R: [f32; 9] = [
    0.877637, 0.479094, -0.0148907,
    -0.478116, 0.872793, -0.0981673,
    -0.0340349, 0.0932747, 0.995059,
];

const OMEGA: [f32; 3] = [0.025, -0.3, 0.0];

#[rustfmt::skip]
const REXPOMEGA: [f32; 9] = [
    0.832258, 0.475313, -0.285351,
    -0.489018, 0.871885, 0.0260337,
    0.261167, 0.117875, 0.95807,
];

#[rustfmt::skip]
const JVP_DR: [f32; 9] = [
    0.955339, -0.00372177, 0.295489,
    -0.00372177, 0.99969, 0.0246241,
    -0.295489, -0.0246241, 0.955029,
];

#[rustfmt::skip]
const JVP_DOMEGA: [f32; 9] = [
    0.985068, -0.00124435, 0.148871,
    -0.00124435, 0.999896, 0.0124059,
    -0.148871, -0.0124059, 0.984964,
];

const VJP_R: [f32; 3] = [0.656128, 0.971344, 1.27514];
const VJP_OMEGA: [f32; 3] = [0.834953, 0.986246, 1.14624];

fn retract_generic<B: Backend>()
where
    SO3: ContinuousQuantity<B>,
{
    let model: ModelClient<B> = Default::default();
    let r = model.add_parameter(SO3);
    let omega = model.add_parameter(Euclidean::<3>);
    let r1 = r.retract(&omega);
    let points = points()
        .with(
            r.id,
            Tensor::<B, 2>::from_data(TensorData::new(R.to_vec(), [1, 9]), &Default::default()),
        )
        .with(
            omega.id,
            Tensor::<B, 2>::from_data(TensorData::new(OMEGA.to_vec(), [1, 3]), &Default::default()),
        );
    let value = r1.evaluate(&points).unwrap();
    value
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&REXPOMEGA)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));

    let ts = tangent_frames().with(
        r.id,
        Tensor::<B, 3>::from_floats(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
            &Default::default(),
        ),
    );
    let PointJvpEval {
        value,
        tangents: mut ts,
    } = r1.evaluate_jvp(&points, &ts).unwrap();
    value
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&REXPOMEGA)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));
    ts.remove(&r1.id)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&JVP_DR)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));
    let ts = tangent_frames().with(
        omega.id,
        Tensor::<B, 3>::from_floats(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
            &Default::default(),
        ),
    );
    let PointJvpEval {
        value,
        tangents: mut ts,
    } = r1.evaluate_jvp(&points, &ts).unwrap();
    value
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&REXPOMEGA)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));
    ts.remove(&r1.id)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&JVP_DOMEGA)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));
    let PointEvalOrder1 {
        value,
        mut cotangents,
    } = r1.evaluate_ord1(&points).unwrap();
    value
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&REXPOMEGA)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));
    cotangents
        .remove(&r.id)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&VJP_R)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));
    cotangents
        .remove(&omega.id)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&VJP_OMEGA)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));

    let dr = Tensor::<B, 2>::from_floats([[0.1, 0.5, -0.3]], &Default::default());
    let dr = dr.clone() / burn::tensor::linalg::l2_norm(dr, 1);
    let domega = Tensor::<B, 2>::from_floats([[-0.03, 0.01, 0.1]], &Default::default());
    let domega = domega.clone() / burn::tensor::linalg::l2_norm(domega, 1);
    let ts = tangents()
        .with(r.id, dr.clone())
        .with(omega.id, domega.clone());
    let EvalOrder2 {
        mut primal,
        mut ord1_cotangents,
        mut ord2_cotangents,
        ..
    } = model.evaluate_ids_ord2(&points, &ts, &[r1.id]).unwrap();
    primal
        .evaluations
        .remove(&r1.id)
        .unwrap()
        .0
        .into_float()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&REXPOMEGA)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));
    ord1_cotangents
        .get(&r.id)
        .unwrap()
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&VJP_R)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));
    ord1_cotangents
        .get(&omega.id)
        .unwrap()
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(&VJP_OMEGA)
        .for_each(|(x, &y)| assert_abs_diff_eq!(x, y, epsilon = 1e-4));

    let delta = 1e-4;
    let ts = tangents()
        .with(r.id, delta * dr)
        .with(omega.id, delta * domega)
        .build();
    let points = primal.variables.retract(&ts).unwrap();

    let PointEvalOrder1 {
        cotangents: cotangents1,
        ..
    } = r1.evaluate_ord1(points.as_ref()).unwrap();

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
            .for_each(|(x, y)| assert_abs_diff_eq!(x, y, epsilon = 0.05));
    }
}

#[test]
fn retract() {
    retract_generic::<CubeBackend<WgpuRuntime, f32, i32>>();
    retract_generic::<NdArrayBackend<f32, i32>>();
}
