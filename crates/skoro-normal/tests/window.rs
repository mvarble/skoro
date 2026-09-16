//! A sliding window over a chain model, grown and retired tick by tick through the
//! builder — the model-side half of the loop in `docs/skoro.md` §3.
//!
//! What is asserted is the bookkeeping, not the statistics — nothing marginalizes yet, and
//! §3.3 is what is missing: each tick demotes the blanket member, replaces its conditional with a
//! joint factor over it, and retires the oldest state-measurement pair with everything
//! below it. The records, the row count of `J`, and the graph's size must all hold
//! exactly steady afterwards — a leak or a dangling record is invisible in one tick and
//! unmissable over twelve.

use skoro_backend::{Flex, Point, TensorData};
use skoro_linalg::matrix::matmul;
use skoro_model::{
    ModelBuilderOps, ModelClientOps,
    hash_map_builders::{point_map, points},
    least_squares::residual_layout,
    model::ModelClient,
    q::Q,
};
use skoro_normal::{
    distribution::Normal,
    facade::constant_covariance_identity,
    joint::{JointNormal, add_joint_factor},
};
use skoro_operation::{Input, ValueMap};
use skoro_riemann::{
    euclidean::{EuclideanD, VectorD},
    matrix::{EuclideanMatrixD, MatrixD},
};

type B = Flex;
skoro_backend::f64_test_binary!(Flex);

const N: usize = 3; // state dimension
const M: usize = 2; // measurement dimension
const WINDOW: usize = 3; // state-measurement pairs kept live
const TICKS: usize = 12;

/// Steady state: one joint over the freshly demoted blanket member, the two newest
/// states' own conditionals, and each window member's measurement conditional.
const FACTORS: usize = 1 + 2 + WINDOW;
const ROWS: usize = 3 * N + WINDOW * M;

fn point(values: &[f64]) -> Input<B> {
    Input::float(Point::<B>::from_data(
        TensorData::new(values.to_vec(), [1, values.len()]),
        &Default::default(),
    ))
}

fn scalar(t: skoro_backend::Tensor<B, 2>) -> f64 {
    t.into_data().to_vec::<f64>().unwrap()[0]
}

/// One tick of the window: a state and the measurement conditioned on it.
type Tick = (Q<B, EuclideanD>, Q<B, EuclideanD>);

/// The points a log-density evaluation needs: the demoted states and the newest one —
/// everything still live that is neither constant nor observed.
fn live_points(window: &[Tick]) -> ValueMap<B> {
    let mut builder = points();
    for (x, _) in window {
        builder = builder.with_quantity(x, point(&[0.0; N]));
    }
    builder.build()
}

#[test]
fn a_sliding_window_grows_and_retires_without_leaking() {
    let model = ModelClient::<B>::default();
    let prior_mean = model.add_constant(EuclideanD(N), &VectorD::new(&[0.0; N]));
    let dyn_cov = constant_covariance_identity(&model, N);
    let z_cov = constant_covariance_identity(&model, M);
    // The joint factors' whitening. Statistics are irrelevant here — the test watches
    // bookkeeping — so the identity stands in for a marginal precision.
    let whitening = constant_covariance_identity(&model, N);

    // The dynamics and measurement maps are ordinary quantity-level operations, exactly
    // as in the doc: each `matmul` becomes a `Mapped` node with a real `DiffOperation`.
    let f_mat = model.add_constant(
        EuclideanMatrixD(N, N),
        &MatrixD::new(N, N, &[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]),
    );
    let h_mat = model.add_constant(
        EuclideanMatrixD(M, N),
        &MatrixD::new(M, N, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    );

    // t = 0: the prior and its first measurement.
    let x = model
        .add_quantity(Normal::new(prior_mean.clone(), dyn_cov.clone()))
        .unwrap();
    let z_mean = matmul(&h_mat, &x).unwrap();
    let z = model
        .add_quantity(Normal::new(z_mean.clone(), z_cov.clone()))
        .unwrap();
    model.observe(&z, &VectorD::new(&[0.0; M])).unwrap();
    let mut window = vec![(x, z)];

    let mut last_node_count = None;
    for _ in 1..TICKS {
        let x_mean = matmul(&f_mat, &window.last().unwrap().0).unwrap();
        let x = model
            .add_quantity(Normal::new(x_mean.clone(), dyn_cov.clone()))
            .unwrap();
        let z_mean = matmul(&h_mat, &x).unwrap();
        let z = model
            .add_quantity(Normal::new(z_mean.clone(), z_cov.clone()))
            .unwrap();
        model.observe(&z, &VectorD::new(&[0.0; M])).unwrap();
        window.push((x, z));

        if window.len() <= WINDOW {
            continue;
        }

        // ---- The window slides ----
        //
        // 1. Demote the *blanket* member — not the state being retired. Retiring
        //    `oldest` takes everything downstream of it, and the blanket is downstream
        //    of it (its mean was built from `oldest`); the demotion is what severs that
        //    edge while keeping the blanket's identity, its measurement, and the rest
        //    of the chain intact. Its conditional's term leaves the model in
        //    the same call.
        let blanket = window[1].0.clone();
        model.demote_to_parameter(&blanket).unwrap();

        // 2. Attach the replacement: a joint factor over the blanket, linearized at a
        //    frozen estimate.
        let at = point_map()
            .with_quantity(&blanket, point(&[0.0; N]))
            .build();
        add_joint_factor(
            &model,
            &JointNormal::new(vec![blanket.id()], at, whitening.clone(), 0.0),
        )
        .unwrap();

        // 3. Retire the oldest pair with everything below it — including last tick's
        //    joint factor, whose whole body is downstream of the state it scored.
        let (old_x, old_z) = window.remove(0);
        model.remove_closure(&[old_x.id(), old_z.id()]).unwrap();

        assert_eq!(
            model.factors().len(),
            FACTORS,
            "the model holds exactly the live factors"
        );
        assert_eq!(
            residual_layout(&model).unwrap().total(),
            ROWS,
            "the rows of `J` are exactly the live factors'"
        );

        let points = live_points(&window);
        assert!(scalar(model.log_density(&points).unwrap()).is_finite());

        let node_count = model.graph_node_count();
        if let Some(last) = last_node_count {
            assert_eq!(
                node_count, last,
                "the window holds steady: every node a tick builds is reclaimed by the next"
            );
        }
        last_node_count = Some(node_count);
    }
}
