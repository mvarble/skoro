//! `demote_to_parameter` over a real (Gaussian) model.
//!
//! Demotion is how a quantity is re-rooted when its marginal does not factorize: it keeps its
//! identity and every dependent quantity, but gives up its sampler, its parents, and its own
//! log-density — the joint being supplied separately as a likelihood factor.

use std::collections::HashMap;

use skoro_backend::{CotangentFrame, NdArrayBackend, Point, TensorData};
use skoro_core::error::SkoroError;
use skoro_graph::node::Node;
use skoro_linalg::{tril::Tril, vector::LinalgVectorOps};
use skoro_model::{hash_map_builders::points, model::ModelClient, q::Q};
use skoro_operation::Input;
use skoro_riemann::euclidean::{EuclideanD, VectorD};

use skoro_normal::distribution::Normal;

pub type B = NdArrayBackend<f32, i32>;

const DIM: usize = 3;

/// `x ~ Normal(mean, diag(std_dev))` over `EuclideanD(3)`, with constant parameters.
fn unit_model(std_dev: &[f32]) -> (ModelClient<B>, Q<B, EuclideanD>) {
    let model = ModelClient::<B>::new(Default::default());
    let mean = model.add_constant(EuclideanD(DIM), &VectorD::new(&[0.0, 0.0, 0.0]));
    // `Tril` packs the lower triangle row-major; a diagonal factor zeroes the rest.
    let [a, b, c] = [std_dev[0], std_dev[1], std_dev[2]];
    let cov = model.add_constant(Tril(DIM), &VectorD::new(&[a, 0.0, b, 0.0, 0.0, c]));
    let x = model.add_quantity(Normal::new(mean, cov)).unwrap();
    (model, x)
}

fn at(x: &Q<B, EuclideanD>, value: [f32; DIM]) -> HashMap<skoro_core::id::Id, Input<B>> {
    points()
        .with_tensor(
            x.id,
            Point::<B>::from_data(
                TensorData::new(value.to_vec(), [1, DIM]),
                &Default::default(),
            ),
        )
        .build()
}

fn log_density(model: &ModelClient<B>, pts: &HashMap<skoro_core::id::Id, Input<B>>) -> f32 {
    model
        .log_density(pts)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0]
}

#[test]
fn demotion_drops_the_quantity_own_density() {
    let (model, x) = unit_model(&[1.0, 2.0, 3.0]);
    let pts = at(&x, [0.5, -1.0, 2.0]);

    let before = log_density(&model, &pts);
    // A non-degenerate Gaussian density; the model has exactly this one term.
    assert!(before.is_finite() && before != 0.0);

    model.demote_to_parameter(&x).unwrap();

    // With its density unregistered and no factor supplied in its place, the model
    // has no likelihood terms left.
    let after = log_density(&model, &pts);
    assert_eq!(after, 0.0);
}

#[test]
fn demotion_preserves_identity_and_dependents() {
    let (model, x) = unit_model(&[1.0, 1.0, 1.0]);
    let y = x.clone() * 2.0f32;
    let id_before = x.id;

    let pts = at(&x, [1.0, 2.0, 3.0]);
    let before = model
        .evaluate_ids(&pts, &[y.id])
        .unwrap()
        .evaluations
        .get(&y.id)
        .unwrap()
        .0
        .clone()
        .into_float()
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    model.demote_to_parameter(&x).unwrap();

    // Same `Id`, and the dependent still evaluates to the same thing.
    assert_eq!(x.id, id_before);
    let after = model
        .evaluate_ids(&pts, &[y.id])
        .unwrap()
        .evaluations
        .get(&y.id)
        .unwrap()
        .0
        .clone()
        .into_float()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(before, after);
    assert_eq!(after, vec![2.0, 4.0, 6.0]);
}

#[test]
fn demoted_quantity_becomes_a_parameter_node_with_no_parents() {
    let (model, x) = unit_model(&[1.0, 1.0, 1.0]);

    model.server_call(|server| {
        assert!(matches!(
            server.graph.nodes.get(&x.id),
            Some(Node::Stochastic(_))
        ));
        // The distribution's mean and covariance are its parents.
        assert_eq!(server.graph.nodes[&x.id].parent_ids().len(), 2);
    });

    model.demote_to_parameter(&x).unwrap();

    model.server_call(|server| {
        assert!(matches!(
            server.graph.nodes.get(&x.id),
            Some(Node::Parameter(_))
        ));
        assert!(server.graph.nodes[&x.id].parent_ids().is_empty());
        // No likelihood terms remain, and the density back-reference is gone.
        assert!(server.graph.log_likelihoods.is_empty());
        assert!(!server.graph.stochastic_to_log_density.contains_key(&x.id));
    });
}

#[test]
fn demoted_quantity_is_still_a_vjp_column() {
    // The property the joint-factor design rests on: a demoted quantity carries no
    // distribution of its own, yet gradients must still flow back to it so that a
    // filter can differentiate a measurement with respect to a marginalized state.
    let (model, x) = unit_model(&[1.0, 1.0, 1.0]);
    let y = x.clone() * 3.0f32;
    model.demote_to_parameter(&x).unwrap();

    let pts = at(&x, [1.0, 2.0, 3.0]);
    let seed: HashMap<_, CotangentFrame<B>> = HashMap::from([(
        y.id,
        CotangentFrame::<B>::from_data(
            TensorData::new(vec![1.0f32, 0.0, 0.0], [1, DIM, 1]),
            &Default::default(),
        ),
    )]);

    let out = model.evaluate_ids_vjp(&pts, &seed, &[y.id]).unwrap();
    let grad = out
        .cotangents
        .get(&x.id)
        .expect("a demoted quantity must receive a cotangent")
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    // d(3x)/dx applied to e_0 is 3 e_0.
    assert_eq!(grad, vec![3.0, 0.0, 0.0]);
}

#[test]
fn demotion_is_idempotent_and_rejects_non_stochastic() {
    let (model, x) = unit_model(&[1.0, 1.0, 1.0]);
    model.demote_to_parameter(&x).unwrap();
    // Already a parameter: a no-op, not an error.
    model.demote_to_parameter(&x).unwrap();

    let c = model.add_constant(EuclideanD(DIM), &VectorD::new(&[1.0, 1.0, 1.0]));
    assert_eq!(
        model.demote_to_parameter(&c),
        Err(SkoroError::NotStochastic { id: c.id })
    );

    let mapped = c.clone() * 2.0f32;
    assert_eq!(
        model.demote_to_parameter(&mapped),
        Err(SkoroError::NotStochastic { id: mapped.id })
    );
}

#[test]
fn a_joint_factor_over_demoted_quantities_scores_the_model() {
    // The shape a marginalization leaves behind: two quantities with no distributions
    // of their own, and one free-standing likelihood factor covering both.
    let model = ModelClient::<B>::new(Default::default());
    let mean = model.add_constant(EuclideanD(DIM), &VectorD::new(&[0.0, 0.0, 0.0]));
    let cov = model.add_constant(Tril(DIM), &VectorD::new(&[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]));
    let a = model
        .add_quantity(Normal::new(mean.clone(), cov.clone()))
        .unwrap();
    let b = model.add_quantity(Normal::new(mean, cov)).unwrap();

    model.demote_to_parameter(&a).unwrap();
    model.demote_to_parameter(&b).unwrap();
    model.server_call(|server| assert!(server.graph.log_likelihoods.is_empty()));

    // One factor over both, standing in for a correlated joint marginal.
    let factor = a.clone().norm_squared() + b.clone().norm_squared();
    model.mark_likelihood(&factor).unwrap();

    let pts = points()
        .with_tensor(
            a.id,
            Point::<B>::from_data(
                TensorData::new(vec![1.0f32, 0.0, 0.0], [1, DIM]),
                &Default::default(),
            ),
        )
        .with_tensor(
            b.id,
            Point::<B>::from_data(
                TensorData::new(vec![0.0f32, 2.0, 0.0], [1, DIM]),
                &Default::default(),
            ),
        )
        .build();

    // 1^2 + 2^2
    assert_eq!(log_density(&model, &pts), 5.0);

    // And gradients reach both demoted quantities through the shared factor.
    let grad = model.evaluate_ord1(&pts, &factor).unwrap();
    assert_eq!(
        grad.cotangents[&a.id]
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        vec![2.0, 0.0, 0.0]
    );
    assert_eq!(
        grad.cotangents[&b.id]
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        vec![0.0, 4.0, 0.0]
    );
}
