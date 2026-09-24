//! Linearizing a model into a block-sparse factor graph, against the dense readers it replaces.

use skoro_backend::{Flex, Point, Tensor, TensorData};
use skoro_core::{error::SkoroError, id::Id};
use skoro_dense::{
    bayes::eliminate,
    dense::Dense,
    elimination::JunctionTree,
    factor::{lstsq, spd_solve},
    ordering::Ordering,
};
use skoro_linalg::{matrix::matmul, tril::Tril};
use skoro_linearize::{Curvature, linearize};
use skoro_model::{
    ModelBuilderOps, ModelReaderOps,
    hash_map_builders::points,
    information::{information, information_blocks},
    jacobian::{jacobian_blocks, jacobian_vjp},
    least_squares::{residual_blocks, residual_layout, residual_vector},
    model::ModelClient,
    q::Q,
};
use skoro_normal::{distribution::Normal, facade::constant_covariance_diagonal};
use skoro_operation::{Input, ValueMap};
use skoro_riemann::{
    euclidean::{EuclideanD, VectorD},
    matrix::{EuclideanMatrixD, MatrixD},
};

type B = Flex;
skoro_backend::f64_test_binary!(Flex);

const N: usize = 3;
const STATES: usize = 6;

fn row(values: &[f64]) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(values.to_vec(), [1, values.len()]),
        &Default::default(),
    )
}

fn input(values: &[f64]) -> Input<B> {
    Input::float(Point::<B>::from_data(
        TensorData::new(values.to_vec(), [1, values.len()]),
        &Default::default(),
    ))
}

fn dense3(t: Tensor<B, 3>) -> Dense {
    Dense::from_tensor(t).unwrap()
}

/// A nonlinear-free chain whose states are read by two terms each (their own conditional and their
/// successor's), plus a measurement on every other state: the shape that makes rows share keys.
fn chain() -> (ModelClient<B>, Vec<Q<B, EuclideanD>>, ValueMap<B>) {
    let model = ModelClient::<B>::default();
    let f = model.add_constant(
        EuclideanMatrixD(N, N),
        &MatrixD::new(N, N, &[0.9, 0.2, 0.0, -0.1, 0.8, 0.3, 0.05, 0.0, 1.1]),
    );
    let h = model.add_constant(
        EuclideanMatrixD(2, N),
        &MatrixD::new(2, N, &[1.0, 0.0, 0.5, 0.0, 1.0, -0.25]),
    );
    let q = constant_covariance_diagonal(&model, &[0.3, 0.5, 0.2]);
    let r = constant_covariance_diagonal(&model, &[0.4, 0.25]);
    let prior = model.add_constant(EuclideanD(N), &VectorD::new(&[0.5, -1.0, 0.25]));
    let mut states = vec![model.add_quantity(Normal::new(prior, q.clone())).unwrap()];
    for k in 1..STATES {
        let mean = matmul(&f, states.last().unwrap()).unwrap();
        states.push(model.add_quantity(Normal::new(mean, q.clone())).unwrap());
        if k % 2 == 0 {
            let z_mean = matmul(&h, states.last().unwrap()).unwrap();
            let z = model.add_quantity(Normal::new(z_mean, r.clone())).unwrap();
            model
                .observe(&z, &VectorD::new(&[0.1 * k as f64, -0.3]))
                .unwrap();
        }
    }
    let mut builder = points();
    for (k, x) in states.iter().enumerate() {
        let t = k as f64;
        builder = builder.with_quantity(x, input(&[0.2 * t, -0.1 * t, 0.3]));
    }
    (model, states, builder.build())
}

fn ids(states: &[Q<B, EuclideanD>]) -> Vec<Id> {
    states.iter().map(|x| x.id()).collect()
}

#[test]
fn block_jacobians_are_the_dense_jacobian_restricted_to_their_keys() {
    let (model, states, values) = chain();
    let free = model.tangent_layout(&ids(&states)).unwrap();
    let rows = residual_layout(&model).unwrap();
    let row_ids: Vec<Id> = rows.ids().collect();

    let blocks = jacobian_blocks(&model, &values, &free, &row_ids).unwrap();
    let dense = dense3(jacobian_vjp(&model, &values, &free, &rows).unwrap());

    for block in &blocks.rows {
        let at = rows.block(block.row).unwrap();
        let j = dense3(block.jacobian.clone());
        assert!(
            block.keys.total() <= 2 * N,
            "a row reaches at most two states"
        );
        for column in free.blocks() {
            let expected = dense.block(at.offset, column.offset, at.dimension, column.dimension);
            match block.keys.block(column.id) {
                Some(key) => {
                    let got = j.block(0, key.offset, at.dimension, key.dimension);
                    assert!(got.max_abs_diff(&expected).unwrap() < 1e-12);
                }
                None => assert_eq!(expected.norm(), 0.0, "a column outside the keys is zero"),
            }
        }
    }
    // Chain rows need a handful of colours however long the chain is, not one frame per row.
    assert!(
        blocks.frames <= 3 * N,
        "{} frames for {} rows",
        blocks.frames,
        rows.total()
    );
    assert!(blocks.frames < rows.total());
}

#[test]
fn residual_blocks_pair_each_residual_with_its_rows() {
    let (model, states, values) = chain();
    let free = model.tangent_layout(&ids(&states)).unwrap();
    let rows = residual_layout(&model).unwrap();
    let stacked = residual_vector(&model, &values).unwrap();
    assert_eq!(stacked.dims(), [1, rows.total()]);
    let stacked = Dense::from_vector(stacked).unwrap();

    let blocks = residual_blocks(&model, &values, &free).unwrap();
    assert_eq!(
        blocks.iter().map(|b| b.residual).collect::<Vec<_>>(),
        rows.ids().collect::<Vec<_>>(),
        "residual_layout's row order"
    );
    for block in &blocks {
        let at = rows.block(block.residual).unwrap();
        let value = Dense::from_vector(block.value.clone()).unwrap();
        let expected = stacked.block(at.offset, 0, at.dimension, 1);
        assert!(value.max_abs_diff(&expected).unwrap() < 1e-12);
    }
}

#[test]
fn the_residual_step_is_the_dense_least_squares_step() {
    let (model, states, values) = chain();
    let order = ids(&states);
    let free = model.tangent_layout(&order).unwrap();
    let rows = residual_layout(&model).unwrap();
    let j = dense3(jacobian_vjp(&model, &values, &free, &rows).unwrap());
    let r = Dense::from_vector(residual_vector(&model, &values).unwrap()).unwrap();
    let expected = lstsq(&j, &r.neg()).unwrap();

    let linearized = linearize(&model, &values, &free, Curvature::Residual).unwrap();
    let graph = &linearized[0].graph;
    assert_eq!(graph.len(), rows.blocks().len(), "one factor per residual");
    let tree = JunctionTree::new(graph, &Ordering::Colamd).unwrap();
    let step = eliminate(&tree, graph)
        .unwrap()
        .tree
        .back_substitute()
        .unwrap()
        .flatten(&order)
        .unwrap();
    assert!(step.max_abs_diff(&expected).unwrap() < 1e-10);
}

#[test]
fn information_blocks_are_keyed_by_the_states_they_reach() {
    let (model, states, values) = chain();
    let free = model.tangent_layout(&ids(&states)).unwrap();
    for block in information_blocks(&model, &values, &free).unwrap() {
        assert!(block.keys.total() <= 2 * N);
        assert_eq!(block.root.dims()[2], block.keys.total());
    }
}

/// A Gaussian whose covariance factor is itself being estimated.
fn heteroscedastic() -> (ModelClient<B>, Vec<Id>, ValueMap<B>) {
    let model = ModelClient::<B>::default();
    let mu = model.add_parameter(EuclideanD(2));
    let l = model.add_parameter(Tril(2));
    for y in [[0.3, -0.6], [1.1, 0.2], [-0.4, 0.9], [0.5, 0.5]].iter() {
        let x = model
            .add_quantity(Normal::new(mu.clone(), l.clone()))
            .unwrap();
        model.observe(&x, &VectorD::new(y)).unwrap();
    }
    let values = points()
        .with_quantity(&mu, row(&[0.1, 0.2]))
        .with_quantity(&l, row(&[1.5, -0.4, 0.8]))
        .build();
    (model, vec![mu.id(), l.id()], values)
}

#[test]
fn a_moving_normalizer_is_refused_by_the_residual_curvature() {
    let (model, free_ids, values) = heteroscedastic();
    let free = model.tangent_layout(&free_ids).unwrap();
    assert!(matches!(
        linearize(&model, &values, &free, Curvature::Residual).map(|l| l.len()),
        Err(SkoroError::NormalizerNotConstant { parent, .. }) if parent == free_ids[1]
    ));
}

#[test]
fn the_fisher_step_is_the_dense_fisher_step() {
    let (model, free_ids, values) = heteroscedastic();
    let free = model.tangent_layout(&free_ids).unwrap();

    let info = dense3(information(&model, &values, &free).unwrap());
    let gradient = model.log_density_ord1(&values).unwrap().cotangents;
    let g = Dense::from_vector(
        free.flatten_vectors(&gradient, 1, &Default::default())
            .unwrap(),
    )
    .unwrap();
    let expected = spd_solve(&info, &g).unwrap();

    let linearized = linearize(&model, &values, &free, Curvature::Fisher)
        .unwrap()
        .remove(0);
    let graph = &linearized.graph;
    let tree = JunctionTree::new(graph, &Ordering::Colamd).unwrap();
    let step = eliminate(&tree, graph)
        .unwrap()
        .tree
        .solve_normal(linearized.gradient.as_ref().unwrap())
        .unwrap()
        .flatten(&free_ids)
        .unwrap();
    assert!(
        step.max_abs_diff(&expected).unwrap() < 1e-9,
        "{:?} vs {:?}",
        step.to_row_major(),
        expected.to_row_major()
    );
}
