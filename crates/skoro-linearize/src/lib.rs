//! # skoro-linearize
//!
//! The bridge from a model to the host elimination in `skoro-dense`: linearize a model at a point
//! into a block-sparse [`GaussianFactorGraph`] keyed by graph [`Id`], and marginalize it onto a set
//! of quantities.
//!
//! `skoro-dense` knows nothing of models and `skoro-model` nothing of `faer`; this crate is where
//! the two meet, so that neither has to depend on the other. The estimators built on it — a
//! Gauss–Newton or Levenberg–Marquardt loop, Fisher scoring, a fixed-lag smoother — depend on this
//! crate rather than living in it.
//!
//! ## Two curvatures
//!
//! [`Curvature::Residual`] reads the rows off the terms' `RESIDUAL` bindings: `J_t δ ≈ −r_t`, the
//! Gauss–Newton rows of `½Σ‖r_t‖²`. They carry their own right-hand side, so
//! [`BayesTree::back_substitute`](skoro_dense::bayes::BayesTree::back_substitute) gives the step at
//! the conditioning of `J`, not of `JᵀJ`. They are the negative log-density only when no term's
//! normalizer moves with the free quantities, so a term whose `NORMALIZER_PARENT` reaches a free
//! leaf — a heteroscedastic Gaussian, whose covariance is being estimated — is refused rather than
//! linearized wrongly.
//!
//! [`Curvature::Fisher`] reads the rows off the terms' `FISHER_ROOT` bindings: `C_t J_t` with no
//! right-hand side, plus the gradient of `log p`. The step is
//! [`BayesTree::solve_normal`](skoro_dense::bayes::BayesTree::solve_normal) against that gradient.
//! It is correct for any distribution that binds a root, heteroscedastic Gaussians included, since
//! the Gaussian root carries the covariance's own information block.
//!
//! Either way the curvature is `AᵀA` for rows `A`, and a [`marginalize`] of it is the Schur
//! complement of that curvature — Gauss–Newton or expected information respectively.
//!
//! ## Batches
//!
//! A batch of `N` points is `N` independent problems, and becomes `N` graphs of one structure, so one
//! [`JunctionTree`](skoro_dense::elimination::JunctionTree) serves them all. Each term's block
//! crosses to the host in one transfer for the whole batch — one transfer per term, not per problem.
//! On a cube backend each is a queue fence, so a model of many small terms pays for that; batching
//! the transfer is left for when it shows.
//!
//! ## What is local, and what is not
//!
//! The host elimination is local: [`marginalize_out`] reads only the factors adjacent to what it
//! eliminates. The device side is not: [`linearize`] differentiates every scored term, so a
//! marginalization pays for the whole model's Jacobian blocks once.

use std::collections::HashSet;
use std::f64::consts::PI;

use skoro_backend::{Backend, Tensor, TensorData};
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
};
use skoro_dense::{
    dense::Dense,
    factor_graph::{GaussianFactorGraph, JacobianFactor},
    marginal::{self, Marginal, marginal},
    values::VectorValues,
};
use skoro_graph::{point_map::PointMap, tangent_layout::TangentLayout};
use skoro_linalg::tril::Tril;
use skoro_model::{
    ModelReaderOps,
    coverage::require_role,
    information::information_blocks,
    least_squares::residual_blocks,
    roles::{FISHER_ROOT, NORMALIZER_PARENT, RESIDUAL},
};
use skoro_normal::joint::JointNormal;
use skoro_operation::ValueMap;
use skoro_riemann::euclidean::{EuclideanD, VectorD};

/// Which rows a linearization is built from.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Curvature {
    /// Gauss–Newton: each term's whitened residual Jacobian, with `b = −r`.
    Residual,
    /// Fisher scoring: each term's Fisher root composed with its Jacobian, with `b = 0`, and the
    /// gradient of `log p` alongside.
    Fisher,
}

/// One batch row's linearization.
#[derive(Debug, Clone, PartialEq)]
pub struct Linearized {
    /// One factor per scored term that reaches a free quantity, keyed by the free quantities' ids.
    pub graph: GaussianFactorGraph<Id>,
    /// `∇ log p` per free quantity, for [`Curvature::Fisher`]; `None` for
    /// [`Curvature::Residual`], whose rows carry their own right-hand side.
    pub gradient: Option<VectorValues<Id>>,
}

/// The model's linearization at `points` over the free quantities `free`, one per batch row.
///
/// Every free quantity is declared in every graph, with its manifold dimension. A term that
/// reaches no free quantity contributes nothing but a constant, and is left out.
///
/// Refuses a model any of whose scored terms does not bind the role the curvature reads —
/// [`SkoroError::MissingBinding`], by [`require_role`] — rather than linearizing a sub-model
/// silently. [`Curvature::Residual`] also refuses, with [`SkoroError::NormalizerNotConstant`], a
/// term whose normalizer depends on a free quantity.
pub fn linearize<B: Backend, C: ModelReaderOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    free: &TangentLayout,
    curvature: Curvature,
) -> SkoroResult<Vec<Linearized>> {
    match curvature {
        Curvature::Residual => {
            require_role(client, RESIDUAL)?;
            refuse_moving_normalizers(client, free)?;
            let blocks = residual_blocks(client, points, free)?;
            let batch = batch_of(points);
            let mut graphs = empty_graphs(free, batch)?;
            for block in blocks {
                if block.keys.total() == 0 {
                    continue;
                }
                let jacobians = Dense::split_tensor(block.jacobian)?;
                let values = Dense::split_vector(block.value)?;
                for ((graph, j), r) in graphs.iter_mut().zip(&jacobians).zip(&values) {
                    graph.add(JacobianFactor::new(split_columns(j, &block.keys), r.neg())?)?;
                }
            }
            Ok(graphs
                .into_iter()
                .map(|graph| Linearized {
                    graph,
                    gradient: None,
                })
                .collect())
        }
        Curvature::Fisher => {
            require_role(client, FISHER_ROOT)?;
            let blocks = information_blocks(client, points, free)?;
            let batch = batch_of(points);
            let mut graphs = empty_graphs(free, batch)?;
            for block in blocks {
                if block.keys.total() == 0 {
                    continue;
                }
                for (graph, root) in graphs.iter_mut().zip(Dense::split_tensor(block.root)?) {
                    graph.add(JacobianFactor::homogeneous(split_columns(
                        &root,
                        &block.keys,
                    ))?)?;
                }
            }
            let gradient = client.log_density_ord1(points)?.cotangents;
            let mut gradients = vec![VectorValues::new(); batch];
            for key in free.blocks() {
                let rows = match gradient.get(&key.id) {
                    Some(g) => Dense::split_vector(g.clone())?,
                    None => vec![Dense::zeros(key.dimension, 1); batch],
                };
                for (values, g) in gradients.iter_mut().zip(rows) {
                    values.insert(key.id, g)?;
                }
            }
            Ok(graphs
                .into_iter()
                .zip(gradients)
                .map(|(graph, gradient)| Linearized {
                    graph,
                    gradient: Some(gradient),
                })
                .collect())
        }
    }
}

/// The marginal on `keep` of the model's linearization at `points`, one per batch row.
///
/// `keep` must be among `free`; every other free quantity is eliminated. The marginal's
/// [`keys`](Marginal::keys) are `keep`, in `keep`'s order. With [`Curvature::Residual`] this is the
/// Laplace (Gauss–Newton) marginal, exact for a linear-Gaussian model; with [`Curvature::Fisher`]
/// it is the expected-information marginal.
pub fn marginalize<B: Backend, C: ModelReaderOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    free: &TangentLayout,
    keep: &[Id],
    curvature: Curvature,
) -> SkoroResult<Vec<Marginal<Id>>> {
    for id in keep {
        if !free.contains(*id) {
            return Err(SkoroError::IdNotInLayout { id: *id });
        }
    }
    linearize(client, points, free, curvature)?
        .iter()
        .map(|linearized| marginal(&linearized.graph, keep))
        .collect()
}

/// What retiring `eliminate` leaves on its Markov blanket, one per batch row: the marginal, over
/// the other free quantities the eliminated ones' terms touch, of those terms alone.
///
/// The sliding-window step. Every term adjacent to `eliminate` is summarized, and nothing else is
/// read, so the result is exactly what replaces them: attach it with [`joint_normal_from`], demote
/// the blanket, and retire the eliminated quantities with everything below them. See
/// [`marginalize_out`](skoro_dense::marginal::marginalize_out) for why this is not [`marginalize`]
/// onto the blanket.
pub fn marginalize_out<B: Backend, C: ModelReaderOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    free: &TangentLayout,
    eliminate: &[Id],
    curvature: Curvature,
) -> SkoroResult<Vec<Marginal<Id>>> {
    for id in eliminate {
        if !free.contains(*id) {
            return Err(SkoroError::IdNotInLayout { id: *id });
        }
    }
    linearize(client, points, free, curvature)?
        .iter()
        .map(|linearized| marginal::marginalize_out(&linearized.graph, eliminate))
        .collect()
}

/// A [`JointNormal`] over a marginal's quantities: the factor a sliding window attaches to the
/// blanket of what it retires.
///
/// The quantities, and the order of `W`'s blocks, are taken from the marginal — never supplied
/// separately — so the correspondence `JointNormal` cannot check is kept by construction. `W`
/// becomes a `Tril` constant, and the normalizer is `−½ D ln(2π) + ln|det W|`.
///
/// `at` is the linearization point the marginal was computed at, and the factor is anchored there
/// with the marginal's right-hand side as its offset: `r = W (x ⊖ at) − d`. At `at` its value and
/// Jacobian are exactly the eliminated rows, on any manifold, whether or not `at` was the optimum
/// of what was eliminated (GTSAM keeps a marginal factor's linearization point for the same
/// reason). Moving the anchor to the marginal's mean instead would agree only to first order off a
/// flat manifold. When `d = 0` no offset is attached.
///
/// Batch 1 only: a marginal is one problem. A batch of `N` needs `N` factors on `N` models, and
/// is [`SkoroError::BatchNotSingleton`] here.
pub fn joint_normal_from<B: Backend, C: ModelReaderOps<B>>(
    client: &C,
    marginal: &Marginal<Id>,
    at: &PointMap<B>,
) -> SkoroResult<JointNormal<B>> {
    let n = client.tangent_layout(marginal.keys())?.total();
    let mut anchor = PointMap::default();
    for id in marginal.keys() {
        let (point, quantity_type) = at.get(id).ok_or(SkoroError::MissingPoint { id: *id })?;
        let batch = point.unwrap_float_ref().dims()[0];
        if batch != 1 {
            return Err(SkoroError::BatchNotSingleton { found: batch });
        }
        anchor.insert(*id, point.clone(), quantity_type.clone());
    }
    let whitening = client.add_constant(Tril(n), &VectorD::from_vec(marginal.packed_whitening()));
    let normalizer = -0.5 * n as f64 * (2.0 * PI).ln() + marginal.log_det();
    let joint = JointNormal::new(marginal.keys().to_vec(), anchor, whitening, normalizer);
    let d = marginal.rhs();
    if d.norm() == 0.0 {
        return Ok(joint);
    }
    let offset = client.add_constant(EuclideanD(n), &VectorD::from_vec(d.to_row_major()));
    Ok(joint.with_offset(offset))
}

/// [`SkoroError::NormalizerNotConstant`] for the first scored term whose normalizer depends on a
/// free quantity: its `½‖r‖²` is then not its negative log-density, and a residual step would
/// optimize the wrong objective.
fn refuse_moving_normalizers<B: Backend, C: ModelReaderOps<B>>(
    client: &C,
    free: &TangentLayout,
) -> SkoroResult<()> {
    let free: HashSet<Id> = free.ids().collect();
    for factor in client.factors() {
        if !factor.in_density {
            continue;
        }
        for parent in client.bound(factor.density, NORMALIZER_PARENT) {
            if let Some(&leaf) = client
                .leaf_support(&[parent])
                .iter()
                .find(|id| free.contains(id))
            {
                return Err(SkoroError::NormalizerNotConstant {
                    density: factor.density,
                    parent: leaf,
                });
            }
        }
    }
    Ok(())
}

fn batch_of<B: Backend>(points: &ValueMap<B>) -> usize {
    points.values().next().map_or(1, |point| point.dims()[0])
}

fn empty_graphs(free: &TangentLayout, batch: usize) -> SkoroResult<Vec<GaussianFactorGraph<Id>>> {
    let mut graph = GaussianFactorGraph::new();
    for key in free.blocks() {
        graph.declare(key.id, key.dimension)?;
    }
    Ok(vec![graph; batch])
}

/// `j`'s columns split into one block per key of `keys`.
fn split_columns(j: &Dense, keys: &TangentLayout) -> Vec<(Id, Dense)> {
    keys.blocks()
        .iter()
        .map(|key| (key.id, j.block(0, key.offset, j.rows(), key.dimension)))
        .collect()
}

/// Per-key values as a flat `[1, layout.total()]` tensor, for retracting a [`PointMap`] by a host
/// step.
pub fn to_flat_tensor<B: Backend>(
    values: &VectorValues<Id>,
    layout: &TangentLayout,
    device: &B::Device,
) -> SkoroResult<Tensor<B, 2>> {
    let ids: Vec<Id> = layout.ids().collect();
    let flat = values.flatten(&ids)?;
    Ok(Tensor::from_data(
        TensorData::new(flat.to_row_major(), [1, flat.rows()]),
        device,
    ))
}
