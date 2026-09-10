//! Reading a term's Fisher root, whatever kind of node it is.
//!
//! `information_blocks` evaluates the root it finds bound under `FISHER_ROOT`, and the evaluation
//! reports a node's value in one of three places depending on its kind: a mapped node's among the
//! evaluations, a constant's among the constants, and a parameter's among the variables. Each
//! test binds a `1 × 1` root `C` of one kind to a term over a scalar `θ`, so the information over
//! `θ` is exactly `C²`.

use skoro_backend::{NdArrayBackend, Tensor, TensorData};
use skoro_model::{
    ModelClientOps,
    distribution::Factor,
    hash_map_builders::points,
    information::information,
    map::register_map1,
    model::ModelClient,
    q::Q,
    roles::FISHER_ROOT,
};
use skoro_operation::{ValueMap, ops::LiteralScalarMultiplication};
use skoro_riemann::{
    euclidean::EuclideanD,
    matrix::{EuclideanMatrixD, MatrixD},
    real::Real,
};

type B = NdArrayBackend<f64, i32>;

fn row(value: f64) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(vec![value], [1, 1]), &Default::default())
}

/// A term over `θ` binding `root`, and the information over `θ` at `values`.
fn information_over_theta(
    model: &ModelClient<B>,
    theta: &Q<B, EuclideanD>,
    root: &Q<B, EuclideanMatrixD>,
    values: &ValueMap<B>,
) -> f64 {
    let term = register_map1(Real, LiteralScalarMultiplication::new(-0.5), theta).unwrap();
    model
        .add_factor(Factor::new(term).bind(FISHER_ROOT, root), &[theta.id()])
        .unwrap();
    let free = model.tangent_layout(&[theta.id()]).unwrap();
    let info = information(model, values, &free).unwrap();
    assert_eq!(info.dims(), [1, 1, 1]);
    info.into_data().to_vec::<f64>().unwrap()[0]
}

#[test]
fn a_parameter_root_is_read_from_the_points() {
    let model = ModelClient::<B>::default();
    let theta = model.add_parameter(EuclideanD(1));
    let root = model.add_parameter(EuclideanMatrixD(1, 1));
    let values = points()
        .with_quantity(&theta, row(0.3))
        .with_quantity(&root, row(2.0))
        .build();
    assert_eq!(information_over_theta(&model, &theta, &root, &values), 4.0);
}

#[test]
fn a_constant_root_is_read_from_the_constants() {
    let model = ModelClient::<B>::default();
    let theta = model.add_parameter(EuclideanD(1));
    let root = model.add_constant(EuclideanMatrixD(1, 1), &MatrixD::new(1, 1, &[3.0]));
    let values = points().with_quantity(&theta, row(0.3)).build();
    assert_eq!(information_over_theta(&model, &theta, &root, &values), 9.0);
}

#[test]
fn a_mapped_root_is_evaluated() {
    let model = ModelClient::<B>::default();
    let theta = model.add_parameter(EuclideanD(1));
    let c = model.add_parameter(EuclideanMatrixD(1, 1));
    let root = register_map1(
        EuclideanMatrixD(1, 1),
        LiteralScalarMultiplication::new(2.0),
        &c,
    )
    .unwrap();
    let values = points()
        .with_quantity(&theta, row(0.3))
        .with_quantity(&c, row(1.5))
        .build();
    assert_eq!(information_over_theta(&model, &theta, &root, &values), 9.0);
}
