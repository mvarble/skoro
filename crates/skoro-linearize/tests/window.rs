//! A sliding window that marginalizes for real: on a linear-Gaussian chain, the windowed estimate
//! of the live states must equal the full-batch estimate to rounding.
//!
//! This is the statistics `skoro-normal/tests/window.rs` deliberately left out. Each tick computes
//! what retiring the oldest state leaves on its blanket (`marginalize_out`), attaches it as a
//! `JointNormal` (`joint_normal_from`), demotes the blanket and retires the oldest pair. A
//! linear-Gaussian marginal is exact, so nothing is lost: any disagreement with the model that never
//! forgot anything is a bug — a mis-ordered `W`, a dropped right-hand side, a factor counted twice.
//!
//! Every state is linearized at zero, far from its estimate, so the marginals carry a non-zero
//! right-hand side and the offset `joint_normal_from` attaches is exercised on every tick.

use skoro_backend::{Flex, Point, TensorData};
use skoro_core::id::Id;
use skoro_dense::{bayes::eliminate, elimination::JunctionTree, ordering::Ordering};
use skoro_linalg::matrix::matmul;
use skoro_linearize::{Curvature, joint_normal_from, linearize, marginalize_out};
use skoro_model::{
    ModelBuilderOps, ModelReaderOps,
    hash_map_builders::{point_map, points},
    model::ModelClient,
    q::Q,
};
use skoro_normal::{
    distribution::Normal, facade::constant_covariance_diagonal, joint::add_joint_factor,
};
use skoro_operation::{Input, ValueMap};
use skoro_riemann::{
    euclidean::{EuclideanD, VectorD},
    matrix::{EuclideanMatrixD, MatrixD},
};

type B = Flex;
skoro_backend::f64_test_binary!(Flex);

const N: usize = 3;
const M: usize = 2;
const WINDOW: usize = 3;
const TICKS: usize = 10;

const F: [f64; 9] = [0.9, 0.2, 0.0, -0.1, 0.8, 0.3, 0.05, 0.0, 1.1];
const H: [f64; 6] = [1.0, 0.0, 0.5, 0.0, 1.0, -0.25];

/// A deterministic, distinct measurement per tick.
fn measurement(k: usize) -> [f64; M] {
    let t = k as f64;
    [(0.7 * t).sin() + 0.1 * t, (0.3 * t).cos() - 0.2]
}

fn zeros(n: usize) -> Input<B> {
    Input::float(Point::<B>::from_data(
        TensorData::new(vec![0.0; n], [1, n]),
        &Default::default(),
    ))
}

type Tick = (Q<B, EuclideanD>, Q<B, EuclideanD>);

struct Chain {
    model: ModelClient<B>,
    f: Q<B, EuclideanMatrixD>,
    h: Q<B, EuclideanMatrixD>,
    q: Q<B, skoro_linalg::tril::Tril>,
    r: Q<B, skoro_linalg::tril::Tril>,
}

impl Chain {
    fn new() -> (Self, Tick) {
        let model = ModelClient::<B>::default();
        let f = model.add_constant(EuclideanMatrixD(N, N), &MatrixD::new(N, N, &F));
        let h = model.add_constant(EuclideanMatrixD(M, N), &MatrixD::new(M, N, &H));
        let q = constant_covariance_diagonal(&model, &[0.3, 0.5, 0.2]);
        let r = constant_covariance_diagonal(&model, &[0.4, 0.25]);
        let p0 = constant_covariance_diagonal(&model, &[1.0, 2.0, 0.5]);
        let prior_mean = model.add_constant(EuclideanD(N), &VectorD::new(&[0.5, -1.0, 0.25]));
        let x = model.add_quantity(Normal::new(prior_mean, p0)).unwrap();
        let chain = Self { model, f, h, q, r };
        let z = chain.measure(&x, 0);
        (chain, (x, z))
    }

    fn measure(&self, x: &Q<B, EuclideanD>, k: usize) -> Q<B, EuclideanD> {
        let mean = matmul(&self.h, x).unwrap();
        let z = self
            .model
            .add_quantity(Normal::new(mean, self.r.clone()))
            .unwrap();
        self.model
            .observe(&z, &VectorD::new(&measurement(k)))
            .unwrap();
        z
    }

    fn step(&self, previous: &Q<B, EuclideanD>, k: usize) -> Tick {
        let mean = matmul(&self.f, previous).unwrap();
        let x = self
            .model
            .add_quantity(Normal::new(mean, self.q.clone()))
            .unwrap();
        let z = self.measure(&x, k);
        (x, z)
    }
}

fn at_zero(states: &[Q<B, EuclideanD>]) -> ValueMap<B> {
    let mut builder = points();
    for x in states {
        builder = builder.with_quantity(x, zeros(N));
    }
    builder.build()
}

/// The MAP estimate of every live state: one Gauss–Newton step from zero, exact for a linear model.
fn estimate(model: &ModelClient<B>, states: &[Q<B, EuclideanD>]) -> Vec<(Id, Vec<f64>)> {
    let ids: Vec<Id> = states.iter().map(|x| x.id()).collect();
    let free = model.tangent_layout(&ids).unwrap();
    let linearized = linearize(model, &at_zero(states), &free, Curvature::Residual).unwrap();
    assert_eq!(linearized.len(), 1);
    let graph = &linearized[0].graph;
    let tree = JunctionTree::new(graph, &Ordering::Colamd).unwrap();
    let delta = eliminate(&tree, graph)
        .unwrap()
        .tree
        .back_substitute()
        .unwrap();
    ids.iter()
        .map(|id| (*id, delta.get(*id).unwrap().to_row_major()))
        .collect()
}

#[test]
fn a_marginalizing_window_agrees_with_the_full_batch() {
    // The full batch: every state and measurement, nothing forgotten.
    let (full, first) = Chain::new();
    let mut full_states = vec![first.0];
    for k in 1..TICKS {
        let (x, _) = full.step(full_states.last().unwrap(), k);
        full_states.push(x);
    }
    let batch = estimate(&full.model, &full_states);

    // The window: the same chain, marginalizing the oldest state each tick.
    let (win, first) = Chain::new();
    let mut window = vec![first];
    let mut last_node_count = None;
    for k in 1..TICKS {
        let tick = win.step(&window.last().unwrap().0, k);
        window.push(tick);
        if window.len() <= WINDOW {
            continue;
        }

        let states: Vec<_> = window.iter().map(|(x, _)| x.clone()).collect();
        let ids: Vec<Id> = states.iter().map(|x| x.id()).collect();
        let free = win.model.tangent_layout(&ids).unwrap();
        let (oldest, blanket) = (window[0].0.clone(), window[1].0.clone());

        let marginal = marginalize_out(
            &win.model,
            &at_zero(&states),
            &free,
            &[oldest.id()],
            Curvature::Residual,
        )
        .unwrap()
        .remove(0);
        assert_eq!(
            marginal.keys(),
            &[blanket.id()],
            "the blanket is the successor"
        );
        assert!(
            marginal.rhs().norm() > 1e-3,
            "linearized away from the optimum, the marginal carries a right-hand side"
        );

        win.model.demote_to_parameter(&blanket).unwrap();
        let at = point_map().with_quantity(&blanket, zeros(N)).build();
        let joint = joint_normal_from(&win.model, &marginal, &at).unwrap();
        add_joint_factor(&win.model, &joint).unwrap();
        let (old_x, old_z) = window.remove(0);
        win.model.remove_closure(&[old_x.id(), old_z.id()]).unwrap();
        win.model.cleanup();

        let node_count = win.model.graph_node_count();
        if let Some(last) = last_node_count {
            assert_eq!(node_count, last, "the window holds steady");
        }
        last_node_count = Some(node_count);
    }

    let states: Vec<_> = window.iter().map(|(x, _)| x.clone()).collect();
    let windowed = estimate(&win.model, &states);
    let expected = &batch[TICKS - WINDOW..];
    for ((_, got), (_, want)) in windowed.iter().zip(expected) {
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g - w).abs() < 1e-9,
                "windowed {got:?} vs full batch {want:?}"
            );
        }
    }
}
