//! Marginals: the Schur complement, in square-root form, by partial elimination.
//!
//! Eliminating every key but the kept ones leaves a set of factors on the kept keys whose
//! curvature is exactly `Λ_kk − Λ_ke Λ_ee⁻¹ Λ_ek` — the Schur complement of the eliminated block —
//! and whose right-hand side is what the eliminated keys' optimum implies for the kept ones. No `Λ`
//! is formed and nothing is inverted: the factors are already a square root, and one more QR makes
//! it square.
//!
//! That last QR is done on the columns **reversed**, then the result flipped on both axes. QR gives
//! an upper `R'` with `R'ᵀR' = PΛP` (`P` the reversal); `W = P R' P` is then *lower*-triangular
//! with `WᵀW = Λ`, which is what a `Tril` whitening — and so a `JointNormal` — needs. The route
//! `Λ → Σ → chol(Σ) → inverse` that `whitening_from_precision` used to take arrives at the same
//! `W` through an inversion that cancels analytically but not numerically.

use faer::linalg::solvers::Qr;
use skoro_core::error::{SkoroError, SkoroResult};

use crate::bayes::{Elimination, eliminate};
use crate::dense::Dense;
use crate::elimination::JunctionTree;
use crate::factor::solve_lower;
use crate::factor_graph::{GaussianFactor, GaussianFactorGraph};
use crate::values::Key;

/// See [`crate::bayes`]; the same threshold, applied to the marginal's own front.
const RANK_TOLERANCE: f64 = 1e-12;

/// A Gaussian over some keys, as `½‖W δ − d‖²` with `W` lower-triangular.
///
/// The keys come *with* `W`: its block rows and columns follow [`keys`](Self::keys), and a
/// consumer that builds a factor from it should take the order from here rather than supplying one,
/// since nothing can check that a `W` of the right width is in the right coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Marginal<K: Key> {
    keys: Vec<K>,
    dims: Vec<usize>,
    w: Dense,
    d: Dense,
}

impl<K: Key> Marginal<K> {
    pub fn keys(&self) -> &[K] {
        &self.keys
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// `W`, lower-triangular with a positive diagonal, `WᵀW = Λ`.
    pub fn whitening(&self) -> &Dense {
        &self.w
    }

    /// `d`. Zero when the rows carried no right-hand side (a Fisher root) or when the kept keys'
    /// linearization point is already their conditional optimum.
    pub fn rhs(&self) -> &Dense {
        &self.d
    }

    /// `W`'s lower triangle, row-major, `(i, j)` at `i(i+1)/2 + j` — the packing of a `Tril`.
    pub fn packed_whitening(&self) -> Vec<f64> {
        let n = self.w.rows();
        let mut packed = Vec::with_capacity(n * (n + 1) / 2);
        for i in 0..n {
            for j in 0..=i {
                packed.push(self.w.get(i, j));
            }
        }
        packed
    }

    /// `ln |det W|`, which is `½ ln det Λ`.
    pub fn log_det(&self) -> f64 {
        (0..self.w.rows()).map(|i| self.w.get(i, i).ln()).sum()
    }

    /// `Λ = WᵀW`, the marginal information.
    pub fn information(&self) -> Dense {
        self.w
            .transpose()
            .matmul(&self.w)
            .expect("internal: W is square")
    }

    /// `Σ = W⁻¹W⁻ᵀ`, the marginal covariance, by two triangular solves.
    pub fn covariance(&self) -> SkoroResult<Dense> {
        let n = self.w.rows();
        let w_inv = solve_lower(&self.w, &Dense::identity(n))?;
        w_inv.matmul(&w_inv.transpose())
    }

    /// `W⁻¹ d`: the marginal's mean, as an offset in the tangent coordinates the rows were
    /// linearized in. Zero exactly when [`rhs`](Self::rhs) is.
    pub fn offset(&self) -> SkoroResult<Dense> {
        solve_lower(&self.w, &self.d)
    }
}

/// The marginal of `graph` on `keep`, in `keep`'s order.
///
/// Eliminates every other key in a COLAMD order, then squares up what is left. Errors
/// [`SkoroError::StructureMismatch`] if `keep` names an undeclared key or one twice, and
/// [`SkoroError::RankDeficient`] if either the eliminated keys or the marginal itself are not
/// determined by the rows — a marginal of fewer rows than `Σ dims(keep)` is singular.
pub fn marginal<K: Key>(graph: &GaussianFactorGraph<K>, keep: &[K]) -> SkoroResult<Marginal<K>> {
    let eliminate_keys: Vec<K> = graph.keys().filter(|k| !keep.contains(k)).collect();
    let tree = JunctionTree::partial(graph, &eliminate_keys)?;
    marginal_with(&tree, graph, keep)
}

/// [`marginal`] along a symbolic elimination computed earlier — by
/// [`JunctionTree::partial`] over every key but `keep` — so that a caller marginalizing the same
/// structure repeatedly pays for the ordering once.
///
/// Errors [`SkoroError::StructureMismatch`] if `tree` keeps a different set of keys than `keep`.
pub fn marginal_with<K: Key>(
    tree: &JunctionTree<K>,
    graph: &GaussianFactorGraph<K>,
    keep: &[K],
) -> SkoroResult<Marginal<K>> {
    let order = tree.order();
    let mut kept: Vec<K> = order[tree.eliminated()..].to_vec();
    let mut wanted = keep.to_vec();
    kept.sort_unstable();
    wanted.sort_unstable();
    wanted.dedup();
    if kept != wanted || wanted.len() != keep.len() {
        return Err(SkoroError::StructureMismatch);
    }
    let Elimination { remaining, .. } = eliminate(tree, graph)?;
    square_up(&remaining, keep, tree.eliminated())
}

/// What eliminating `eliminate` leaves on its Markov blanket: the marginal, over every other key
/// the eliminated keys' factors touch, of **those factors alone**.
///
/// This is the sliding-window primitive. Retiring a state replaces exactly the factors adjacent to
/// it — its prior, its measurements, its successor's conditional — with one factor on its blanket,
/// and that factor must summarize those and nothing else: [`marginal`] over the blanket would also
/// fold in every factor elsewhere in the graph, which the model keeps, and count them twice.
/// Factors that touch no eliminated key are never read.
///
/// The blanket is in key order. An empty blanket — eliminated keys touching nothing else — gives a
/// marginal over no keys.
pub fn marginalize_out<K: Key>(
    graph: &GaussianFactorGraph<K>,
    eliminate: &[K],
) -> SkoroResult<Marginal<K>> {
    for key in eliminate {
        graph.dim(*key).ok_or(SkoroError::StructureMismatch)?;
    }
    let mut local = GaussianFactorGraph::new();
    let mut blanket = std::collections::BTreeSet::new();
    for factor in graph.factors() {
        let keys = factor.keys();
        if !keys.iter().any(|k| eliminate.contains(k)) {
            continue;
        }
        let GaussianFactor::Jacobian(jacobian) = factor;
        for (key, dim) in keys.iter().zip(jacobian.dims()) {
            local.declare(*key, *dim)?;
            if !eliminate.contains(key) {
                blanket.insert(*key);
            }
        }
        local.add(jacobian.clone())?;
    }
    for key in eliminate {
        if local.dim(*key).is_none() {
            let index = graph.keys().position(|k| k == *key).expect("checked above");
            return Err(SkoroError::KeyUnconstrained { index });
        }
    }
    let blanket: Vec<K> = blanket.into_iter().collect();
    if blanket.is_empty() {
        return Ok(Marginal {
            keys: Vec::new(),
            dims: Vec::new(),
            w: Dense::zeros(0, 0),
            d: Dense::zeros(0, 1),
        });
    }
    marginal(&local, &blanket)
}

/// One lower-triangular square root of every factor in `remaining`, over `keep`.
fn square_up<K: Key>(
    remaining: &GaussianFactorGraph<K>,
    keep: &[K],
    first_position: usize,
) -> SkoroResult<Marginal<K>> {
    let dims: Vec<usize> = keep
        .iter()
        .map(|k| remaining.dim(*k).ok_or(SkoroError::StructureMismatch))
        .collect::<SkoroResult<_>>()?;
    let n: usize = dims.iter().sum();
    // Which kept key owns scalar column `i`, as an elimination position: kept keys follow the
    // eliminated prefix.
    let owner = |i: usize| {
        let mut acc = 0;
        for (j, dim) in dims.iter().enumerate() {
            acc += dim;
            if i < acc {
                return first_position + j;
            }
        }
        unreachable!("column {i} is kept")
    };

    let (a, b) = remaining.to_dense(keep)?;
    let rows = a.rows();
    // `[A P | b]`, `P` reversing the scalar columns.
    let reversed = Dense::from_fn(rows, n + 1, |r, c| {
        if c < n {
            a.get(r, n - 1 - c)
        } else {
            b.get(r, 0)
        }
    });
    let mut r = if rows == 0 {
        Dense::zeros(0, n + 1)
    } else {
        Dense::from_faer_ref(Qr::new(reversed.as_faer()).thin_R())
    };
    for i in 0..n {
        let norm = (0..rows)
            .map(|row| reversed.get(row, i).powi(2))
            .sum::<f64>()
            .sqrt();
        if i >= r.rows() || norm == 0.0 || r.get(i, i).abs() <= RANK_TOLERANCE * norm {
            // Reversed column `i` is original column `n − 1 − i`.
            return Err(SkoroError::RankDeficient {
                position: owner(n - 1 - i),
            });
        }
        if r.get(i, i) < 0.0 {
            for c in i..=n {
                r.set(i, c, -r.get(i, c));
            }
        }
    }

    let w = Dense::from_fn(n, n, |i, j| r.get(n - 1 - i, n - 1 - j));
    let d = Dense::from_fn(n, 1, |i, _| r.get(n - 1 - i, n));
    Ok(Marginal {
        keys: keep.to_vec(),
        dims,
        w,
        d,
    })
}
