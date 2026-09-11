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
    information::{information, information_blocks, information_root},
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

/// Terms that share arguments, list them in different orders, or bind no root at all: each block
/// is still the term's own root times the term's own Jacobian.
///
/// The free layout is `[φ, θ]` and `ψ = 3φ` is a mapped argument, so every row of `J` is known:
/// `θ` is `[0, 0, 1]`, `ψ₀` is `[3, 0, 0]` and `ψ₁` is `[0, 3, 0]`. The first term lists `[θ, ψ]`,
/// the rootless term `[ψ]`, and the last `[ψ, θ]`; the roots are constants, so each block can be
/// multiplied out by hand.
#[test]
fn terms_sharing_arguments_in_different_orders_get_their_own_blocks() {
    let model = ModelClient::<B>::default();
    let theta = model.add_parameter(EuclideanD(1));
    let phi = model.add_parameter(EuclideanD(2));
    let psi = register_map1(EuclideanD(2), LiteralScalarMultiplication::new(3.0), &phi).unwrap();
    let density = || register_map1(Real, LiteralScalarMultiplication::new(-0.5), &theta).unwrap();

    let c1 = model.add_constant(
        EuclideanMatrixD(2, 3),
        &MatrixD::new(2, 3, &[1.0, 2.0, 0.0, 0.0, -1.0, 4.0]),
    );
    let first = model
        .add_factor(
            Factor::new(density()).bind(FISHER_ROOT, &c1),
            &[theta.id(), psi.id()],
        )
        .unwrap();
    model
        .add_factor(Factor::new(density()), &[psi.id()])
        .unwrap();
    let c2 = model.add_constant(
        EuclideanMatrixD(1, 3),
        &MatrixD::new(1, 3, &[0.5, -2.0, 5.0]),
    );
    let last = model
        .add_factor(
            Factor::new(density()).bind(FISHER_ROOT, &c2),
            &[psi.id(), theta.id()],
        )
        .unwrap();

    let values = points()
        .with_quantity(&theta, row(0.3))
        .with_quantity(
            &phi,
            Tensor::<B, 2>::from_data(
                TensorData::new(vec![0.1, -0.2], [1, 2]),
                &Default::default(),
            ),
        )
        .build();
    let free = model.tangent_layout(&[phi.id(), theta.id()]).unwrap();

    let blocks = information_blocks(&model, &values, &free).unwrap();
    let factors: Vec<_> = blocks.iter().map(|block| block.factor).collect();
    assert_eq!(
        factors,
        [first.id(), last.id()],
        "the rootless term is skipped"
    );

    // C₁ J₁ = [[1, 2, 0], [0, −1, 4]] · [θ; ψ₀; ψ₁] and C₂ J₂ = [0.5, −2, 5] · [ψ₀; ψ₁; θ].
    let expected: [&[f64]; 2] = [&[6.0, 0.0, 1.0, -3.0, 12.0, 0.0], &[1.5, -6.0, 5.0]];
    for ((block, rows), want) in blocks.iter().zip([2, 1]).zip(expected) {
        assert_eq!(block.root.dims(), [1, rows, 3]);
        assert_eq!(
            block.root.clone().into_data().to_vec::<f64>().unwrap(),
            want
        );
    }

    let stacked = information_root(&model, &values, &free).unwrap();
    assert_eq!(stacked.dims(), [1, 3, 3]);
    assert_eq!(
        stacked.into_data().to_vec::<f64>().unwrap(),
        expected.concat()
    );
}
