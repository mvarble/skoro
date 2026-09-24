//! Block-sparse elimination against the dense answers it must reproduce.
//!
//! Every fixture mixes key dimensions 1, 2, 3 and 6 — SO(3) and SE(3) tangents are 3 and 6, never
//! their storage sizes — so a transposed block, a dimension read off the wrong key, or a column
//! scattered to its neighbour's offset lands somewhere visible. Host-only: no backend, no tensor.

use faer::linalg::solvers::SolveLstsq;
use faer::sparse::{SparseColMat, Triplet};
use skoro_core::error::SkoroError;
use skoro_dense::bayes::eliminate;
use skoro_dense::dense::Dense;
use skoro_dense::elimination::JunctionTree;
use skoro_dense::factor::{cholesky, lstsq, spd_solve};
use skoro_dense::factor_graph::{GaussianFactor, GaussianFactorGraph, JacobianFactor};
use skoro_dense::marginal::{marginal, marginal_with, marginalize_out};
use skoro_dense::ordering::Ordering;
use skoro_dense::values::VectorValues;

const DIMS: [usize; 9] = [3, 6, 1, 3, 6, 2, 3, 1, 6];

/// A small linear congruential generator: deterministic, and enough to make blocks generic.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn matrix(&mut self, rows: usize, cols: usize) -> Dense {
        Dense::from_fn(rows, cols, |_, _| self.next())
    }
}

/// A chain over nine keys with three loop closures and a prior on key 0: the shape of a pose graph.
/// Each factor has as many rows as its keys have columns, plus one, so every front is determined.
fn pose_graph(seed: u64) -> GaussianFactorGraph<u32> {
    let mut rng = Lcg(seed);
    let mut graph = GaussianFactorGraph::new();
    for (k, &d) in DIMS.iter().enumerate() {
        graph.declare(k as u32, d).unwrap();
    }
    let mut edges: Vec<Vec<u32>> = vec![vec![0]];
    edges.extend((0..8).map(|k| vec![k, k + 1]));
    edges.extend([vec![0, 4], vec![2, 7, 8], vec![5, 1]]);
    for keys in edges {
        let rows: usize = keys.iter().map(|&k| DIMS[k as usize]).sum::<usize>() + 1;
        let blocks = keys
            .iter()
            .map(|&k| (k, rng.matrix(rows, DIMS[k as usize])))
            .collect();
        graph
            .add(JacobianFactor::new(blocks, rng.matrix(rows, 1)).unwrap())
            .unwrap();
    }
    graph
}

fn natural(graph: &GaussianFactorGraph<u32>) -> Vec<u32> {
    graph.keys().collect()
}

fn assert_close(actual: &Dense, expected: &Dense, tolerance: f64, what: &str) {
    let diff = actual.max_abs_diff(expected).unwrap();
    assert!(diff <= tolerance, "{what}: off by {diff}");
}

/// The dense least-squares solution, over the keys in natural order.
fn dense_solution(graph: &GaussianFactorGraph<u32>) -> Dense {
    let (a, b) = graph.to_dense(&natural(graph)).unwrap();
    lstsq(&a, &b).unwrap()
}

#[test]
fn every_ordering_and_both_eliminations_reproduce_the_dense_least_squares_step() {
    let graph = pose_graph(1);
    let expected = dense_solution(&graph);
    let reversed: Vec<u32> = natural(&graph).into_iter().rev().collect();
    for ordering in [
        Ordering::Colamd,
        Ordering::Natural,
        Ordering::Custom(reversed),
        Ordering::ConstrainedLast(vec![4, 0]),
    ] {
        for tree in [
            JunctionTree::new(&graph, &ordering).unwrap(),
            JunctionTree::sequential(&graph, &ordering).unwrap(),
        ] {
            let delta = eliminate(&tree, &graph)
                .unwrap()
                .tree
                .back_substitute()
                .unwrap();
            let flat = delta.flatten(&natural(&graph)).unwrap();
            assert_close(&flat, &expected, 1e-10, &format!("{ordering:?}"));
        }
    }
}

#[test]
fn the_block_solve_agrees_with_faers_scalar_sparse_qr() {
    let graph = pose_graph(2);
    let (a, b) = graph.to_dense(&natural(&graph)).unwrap();
    let mut triplets = Vec::new();
    for r in 0..a.rows() {
        for c in 0..a.cols() {
            if a.get(r, c) != 0.0 {
                triplets.push(Triplet {
                    row: r,
                    col: c,
                    val: a.get(r, c),
                });
            }
        }
    }
    let sparse =
        SparseColMat::<usize, f64>::try_new_from_triplets(a.rows(), a.cols(), &triplets).unwrap();
    let mut rhs = b.clone().into_faer();
    sparse.sp_qr().unwrap().solve_lstsq_in_place(rhs.as_mut());
    let scalar = Dense::from_faer_ref(rhs.as_ref().get(..a.cols(), ..));

    let tree = JunctionTree::new(&graph, &Ordering::Colamd).unwrap();
    let block = eliminate(&tree, &graph)
        .unwrap()
        .tree
        .back_substitute()
        .unwrap()
        .flatten(&natural(&graph))
        .unwrap();
    assert_close(&block, &scalar, 1e-10, "block vs scalar sparse QR");
}

#[test]
fn multifrontal_elimination_merges_chains_into_fewer_cliques() {
    let graph = pose_graph(3);
    let multifrontal = JunctionTree::new(&graph, &Ordering::Colamd).unwrap();
    let sequential = JunctionTree::sequential(&graph, &Ordering::Colamd).unwrap();
    assert_eq!(sequential.clique_count(), DIMS.len());
    assert!(
        multifrontal.clique_count() < sequential.clique_count(),
        "{} cliques for {} keys",
        multifrontal.clique_count(),
        DIMS.len()
    );
}

#[test]
fn colamd_keeps_a_hub_key_away_from_the_front() {
    // A star: every factor touches the hub, key 5. Eliminating it first fills every front with
    // every key; COLAMD must defer it. The hub is not key 0, so a permutation read backwards
    // (new-to-old taken as old-to-new) would not land it late by accident.
    let hub = 5u32;
    let mut rng = Lcg(4);
    let mut graph = GaussianFactorGraph::new();
    for k in 0..8u32 {
        graph.declare(k, 2).unwrap();
    }
    for k in (0..8u32).filter(|&k| k != hub) {
        graph
            .add(
                JacobianFactor::new(
                    vec![(hub, rng.matrix(5, 2)), (k, rng.matrix(5, 2))],
                    rng.matrix(5, 1),
                )
                .unwrap(),
            )
            .unwrap();
    }
    let order = Ordering::Colamd.resolve(&graph.structure()).unwrap();
    let at = order.iter().position(|&k| k == hub).unwrap();
    assert!(at >= order.len() - 2, "hub at {at} in {order:?}");

    let mut hub_first = vec![hub];
    hub_first.extend((0..8u32).filter(|&k| k != hub));
    let bad = JunctionTree::new(&graph, &Ordering::Custom(hub_first)).unwrap();
    let good = JunctionTree::new(&graph, &Ordering::Colamd).unwrap();
    assert_eq!(bad.max_front_width(), 16);
    assert!(good.max_front_width() <= 4, "{}", good.max_front_width());
}

#[test]
fn the_elimination_error_is_the_least_squares_minimum() {
    let graph = pose_graph(5);
    let tree = JunctionTree::new(&graph, &Ordering::Colamd).unwrap();
    let elimination = eliminate(&tree, &graph).unwrap();
    let delta = elimination.tree.back_substitute().unwrap();
    let minimum = graph.error(&delta).unwrap();
    assert!(
        (elimination.error - minimum).abs() <= 1e-10 * minimum.max(1.0),
        "{} vs {minimum}",
        elimination.error
    );
    // And the gradient vanishes there.
    for (_, g) in graph.gradient(&delta).unwrap().iter() {
        assert!(g.norm() <= 1e-9, "gradient {}", g.norm());
    }
}

#[test]
fn solve_normal_is_the_normal_equations_solve_for_an_arbitrary_gradient() {
    let graph = pose_graph(6);
    let order = natural(&graph);
    let (a, _) = graph.to_dense(&order).unwrap();
    let info = a.transpose().matmul(&a).unwrap();
    let mut rng = Lcg(60);
    let g_flat = rng.matrix(info.rows(), 1);
    let g = VectorValues::unflatten(&order, &DIMS, &g_flat).unwrap();

    let tree = JunctionTree::new(&graph, &Ordering::Colamd).unwrap();
    let bayes = eliminate(&tree, &graph).unwrap().tree;
    let step = bayes.solve_normal(&g).unwrap().flatten(&order).unwrap();
    assert_close(
        &step,
        &spd_solve(&info, &g_flat).unwrap(),
        1e-9,
        "(AᵀA)⁻¹ g",
    );
}

#[test]
fn solve_upper_spreads_unit_draws_into_the_inverse_information() {
    // Σ_i (R⁻¹ eᵢ)(R⁻¹ eᵢ)ᵀ = R⁻¹R⁻ᵀ = (AᵀA)⁻¹ exactly — the covariance of R⁻¹z, z ~ N(0, I),
    // without drawing anything.
    let graph = pose_graph(7);
    let order = natural(&graph);
    let (a, _) = graph.to_dense(&order).unwrap();
    let info = a.transpose().matmul(&a).unwrap();
    let n = info.rows();
    let tree = JunctionTree::new(&graph, &Ordering::Colamd).unwrap();
    let bayes = eliminate(&tree, &graph).unwrap().tree;

    let mut covariance = Dense::zeros(n, n);
    for i in 0..n {
        let e = Dense::from_fn(n, 1, |r, _| if r == i { 1.0 } else { 0.0 });
        let z = VectorValues::unflatten(&order, &DIMS, &e).unwrap();
        let x = bayes.solve_upper(&z).unwrap().flatten(&order).unwrap();
        covariance = covariance.add(&x.matmul(&x.transpose()).unwrap()).unwrap();
    }
    let expected = spd_solve(&info, &Dense::identity(n)).unwrap();
    assert_close(&covariance, &expected, 1e-8, "Σ R⁻¹eᵢeᵢᵀR⁻ᵀ");

    let chol = cholesky(&info).unwrap();
    let half_log_det: f64 = (0..n).map(|i| chol.get(i, i).ln()).sum();
    assert!((bayes.log_det_r() - half_log_det).abs() <= 1e-9);
}

#[test]
fn damping_is_the_augmented_normal_equations() {
    let graph = pose_graph(8);
    let order = natural(&graph);
    let lambda = 0.37;
    let (a, b) = graph.to_dense(&order).unwrap();
    let n = a.cols();
    let lhs = a
        .transpose()
        .matmul(&a)
        .unwrap()
        .add(&Dense::scaled_identity(n, lambda))
        .unwrap();
    let expected = spd_solve(&lhs, &a.transpose().matmul(&b).unwrap()).unwrap();

    let damped = graph.damped(lambda).unwrap();
    let tree = JunctionTree::new(&damped, &Ordering::Colamd).unwrap();
    let step = eliminate(&tree, &damped)
        .unwrap()
        .tree
        .back_substitute()
        .unwrap()
        .flatten(&order)
        .unwrap();
    assert_close(&step, &expected, 1e-9, "damped step");
}

/// `Λ_kk − Λ_ke Λ_ee⁻¹ Λ_ek` over `keep`, formed densely.
fn dense_schur(graph: &GaussianFactorGraph<u32>, keep: &[u32]) -> Dense {
    let eliminated: Vec<u32> = graph.keys().filter(|k| !keep.contains(k)).collect();
    let mut order = eliminated.clone();
    order.extend_from_slice(keep);
    let (a, _) = graph.to_dense(&order).unwrap();
    let info = a.transpose().matmul(&a).unwrap();
    let e: usize = eliminated.iter().map(|&k| DIMS[k as usize]).sum();
    let k = info.rows() - e;
    let aa = info.block(0, 0, e, e);
    let ab = info.block(0, e, e, k);
    let bb = info.block(e, e, k, k);
    bb.sub(
        &ab.transpose()
            .matmul(&spd_solve(&aa, &ab).unwrap())
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn a_marginal_is_the_schur_complement_as_a_lower_whitening() {
    let graph = pose_graph(9);
    // Deliberately not in key order: the marginal follows the caller's order.
    let keep = [7u32, 1, 4];
    let m = marginal(&graph, &keep).unwrap();
    assert_eq!(m.keys(), &keep);
    assert_eq!(m.dims(), &[1, 6, 6]);

    let w = m.whitening();
    for i in 0..w.rows() {
        assert!(w.get(i, i) > 0.0, "diagonal {i} is {}", w.get(i, i));
        for j in i + 1..w.cols() {
            assert_eq!(w.get(i, j), 0.0, "W is not lower at ({i}, {j})");
        }
    }
    let schur = dense_schur(&graph, &keep);
    assert_close(&m.information(), &schur, 1e-9, "WᵀW vs Schur complement");

    let chol = cholesky(&schur).unwrap();
    let half_log_det: f64 = (0..schur.rows()).map(|i| chol.get(i, i).ln()).sum();
    assert!((m.log_det() - half_log_det).abs() <= 1e-9);

    // The marginal's mean is the joint optimum restricted to the kept keys.
    let joint = VectorValues::unflatten(&natural(&graph), &DIMS, &dense_solution(&graph)).unwrap();
    assert_close(
        &m.offset().unwrap(),
        &joint.flatten(&keep).unwrap(),
        1e-9,
        "W⁻¹d vs the joint optimum",
    );

    let packed = m.packed_whitening();
    assert_eq!(packed.len(), 13 * 14 / 2);
    assert_eq!(packed[2], w.get(1, 1));
}

#[test]
fn partial_elimination_leaves_untouched_factors_alone() {
    let graph = pose_graph(10);
    // Key 3 touches only the chain factors (2, 3) and (3, 4).
    let tree = JunctionTree::partial(&graph, &[3]).unwrap();
    let elimination = eliminate(&tree, &graph).unwrap();
    let remaining = &elimination.remaining;
    assert_eq!(remaining.keys().count(), DIMS.len() - 1);
    // Every original factor but the two on key 3, plus one separator factor on {2, 4}.
    assert_eq!(remaining.len(), graph.len() - 2 + 1);
    for original in graph.factors() {
        if !original.keys().contains(&3) {
            assert!(remaining.factors().contains(original));
        }
    }
    let GaussianFactor::Jacobian(separator) = remaining.factors().last().unwrap() else {
        unreachable!()
    };
    let mut keys = separator.keys().to_vec();
    keys.sort_unstable();
    assert_eq!(keys, vec![2, 4]);
}

#[test]
fn one_symbolic_elimination_serves_every_graph_of_its_structure() {
    let (first, second) = (pose_graph(11), pose_graph(12));
    let keep = [0u32, 8];
    let eliminated: Vec<u32> = first.keys().filter(|k| !keep.contains(k)).collect();
    let tree = JunctionTree::partial(&first, &eliminated).unwrap();
    for graph in [&first, &second] {
        let reused = marginal_with(&tree, graph, &keep).unwrap();
        let fresh = marginal(graph, &keep).unwrap();
        assert_close(reused.whitening(), fresh.whitening(), 1e-10, "reused tree");
    }

    // A graph of another structure is refused, not silently mis-eliminated.
    let mut other = pose_graph(11);
    other
        .add(
            JacobianFactor::new(
                vec![(3, Dense::identity(3)), (6, Dense::identity(3))],
                Dense::zeros(3, 1),
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(
        eliminate(&tree, &other).map(|e| e.error),
        Err(SkoroError::StructureMismatch)
    );
    assert_eq!(
        marginal_with(&tree, &first, &[0]).map(|m| m.log_det()),
        Err(SkoroError::StructureMismatch)
    );
}

#[test]
fn gauge_freedom_and_an_untouched_key_are_refused() {
    // Two keys related only through their difference: nothing fixes where they are.
    let mut graph = GaussianFactorGraph::new();
    graph.declare(0u32, 2).unwrap();
    graph.declare(1u32, 2).unwrap();
    graph
        .add(
            JacobianFactor::new(
                vec![
                    (0, Dense::identity(2)),
                    (1, Dense::scaled_identity(2, -1.0)),
                ],
                Dense::zeros(2, 1),
            )
            .unwrap(),
        )
        .unwrap();
    let tree = JunctionTree::new(&graph, &Ordering::Natural).unwrap();
    assert!(matches!(
        eliminate(&tree, &graph).map(|e| e.error),
        Err(SkoroError::RankDeficient { .. })
    ));
    // Its marginal on one key is singular too, though eliminating the other is fine.
    assert!(matches!(
        marginal(&graph, &[1]).map(|m| m.log_det()),
        Err(SkoroError::RankDeficient { .. })
    ));

    graph.declare(2u32, 1).unwrap();
    assert_eq!(
        JunctionTree::new(&graph, &Ordering::Natural).map(|t| t.clique_count()),
        Err(SkoroError::KeyUnconstrained { index: 2 })
    );
}

#[test]
fn a_factor_must_match_its_keys() {
    let mut graph = GaussianFactorGraph::new();
    graph.declare(0u32, 2).unwrap();
    let wrong_width = JacobianFactor::new(vec![(0, Dense::identity(3))], Dense::zeros(3, 1));
    assert_eq!(
        graph.add(wrong_width.unwrap()),
        Err(SkoroError::InvalidFactorKeys { factor: 0 })
    );
    let repeated = JacobianFactor::new(
        vec![(0, Dense::identity(2)), (0, Dense::identity(2))],
        Dense::zeros(2, 1),
    );
    assert_eq!(
        graph.add(repeated.unwrap()),
        Err(SkoroError::InvalidFactorKeys { factor: 0 })
    );
    assert_eq!(
        graph.declare(0, 3),
        Err(SkoroError::ShapeMismatch {
            expected: 2,
            found: 3
        })
    );
}

#[test]
fn marginalizing_out_reads_only_the_eliminated_keys_factors() {
    let graph = pose_graph(13);
    // Key 3 touches the chain factors (2, 3) and (3, 4) only.
    let out = marginalize_out(&graph, &[3]).unwrap();
    assert_eq!(out.keys(), &[2, 4]);

    // The same marginal, from a graph holding only those two factors.
    let mut local = GaussianFactorGraph::new();
    for k in [2u32, 3, 4] {
        local.declare(k, DIMS[k as usize]).unwrap();
    }
    for factor in graph.factors() {
        if factor.keys().contains(&3) {
            let GaussianFactor::Jacobian(f) = factor else {
                unreachable!()
            };
            local.add(f.clone()).unwrap();
        }
    }
    let expected = marginal(&local, &[2, 4]).unwrap();
    assert_close(out.whitening(), expected.whitening(), 1e-12, "W");
    assert_close(out.rhs(), expected.rhs(), 1e-12, "d");

    // It is not the marginal of the whole graph on the blanket, which folds in everything else.
    let global = marginal(&graph, &[2, 4]).unwrap();
    assert!(out.whitening().max_abs_diff(global.whitening()).unwrap() > 1e-3);
}
