//! A linear-Gaussian factor graph: rows `‖Σ_k A_k δ_k − b‖²` over named keys.
//!
//! A factor is a block of whitened rows over the few keys it touches. The graph is the list of
//! them, with each key's dimension declared once. Nothing here asks where the rows came from: a
//! whitened residual Jacobian with `b = −r`, a Fisher root `C·J` with `b = 0`, and a damping prior
//! `√λ I` are all the same object, a factor contributing `AᵀA` to a curvature and `Aᵀb` to a step.
//! That is what lets one elimination serve Gauss–Newton, Fisher scoring, and a Laplace marginal.

use std::collections::{BTreeMap, HashSet};

use skoro_core::error::{SkoroError, SkoroResult};

use crate::dense::Dense;
use crate::values::{Key, VectorValues};

/// Whitened rows over a few keys: `‖Σ_k A_k δ_k − b‖²`.
///
/// Stored as one `m × Σ d_k` matrix `A`, its column blocks in [`keys`](Self::keys) order, and an
/// `m × 1` right-hand side `b`.
#[derive(Debug, Clone, PartialEq)]
pub struct JacobianFactor<K: Key> {
    keys: Vec<K>,
    dims: Vec<usize>,
    a: Dense,
    b: Dense,
}

impl<K: Key> JacobianFactor<K> {
    /// From one `m × d_k` block per key and an `m × 1` right-hand side.
    ///
    /// Errors [`SkoroError::ShapeMismatch`] if a block's height, or `b`'s, disagrees, or `b` is not
    /// a column. Duplicate keys are caught when the factor joins a graph, which can say which
    /// factor it was.
    pub fn new(blocks: Vec<(K, Dense)>, b: Dense) -> SkoroResult<Self> {
        if b.cols() != 1 {
            return Err(SkoroError::ShapeMismatch {
                expected: 1,
                found: b.cols(),
            });
        }
        let m = b.rows();
        let mut keys = Vec::with_capacity(blocks.len());
        let mut dims = Vec::with_capacity(blocks.len());
        let mut total = 0;
        for (key, block) in &blocks {
            if block.rows() != m {
                return Err(SkoroError::ShapeMismatch {
                    expected: m,
                    found: block.rows(),
                });
            }
            keys.push(*key);
            dims.push(block.cols());
            total += block.cols();
        }
        let mut a = Dense::zeros(m, total);
        let mut at = 0;
        for (_, block) in &blocks {
            a.set_block(0, at, block);
            at += block.cols();
        }
        Ok(Self { keys, dims, a, b })
    }

    /// A factor with `b = 0`: rows that contribute curvature and no step of their own, such as a
    /// Fisher root `C·J`.
    pub fn homogeneous(blocks: Vec<(K, Dense)>) -> SkoroResult<Self> {
        let m = blocks.first().map_or(0, |(_, block)| block.rows());
        Self::new(blocks, Dense::zeros(m, 1))
    }

    pub fn keys(&self) -> &[K] {
        &self.keys
    }

    /// The column count of each key's block, in [`keys`](Self::keys) order.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rows(&self) -> usize {
        self.b.rows()
    }

    /// All the blocks side by side: `m × Σ d_k`.
    pub fn a(&self) -> &Dense {
        &self.a
    }

    pub fn b(&self) -> &Dense {
        &self.b
    }

    /// The `i`th key's block.
    pub fn block(&self, i: usize) -> Dense {
        let at: usize = self.dims[..i].iter().sum();
        self.a.block(0, at, self.rows(), self.dims[i])
    }

    /// `Σ_k A_k δ_k − b` at `values`. Errors [`SkoroError::StructureMismatch`] if a key has no
    /// value.
    pub fn residual(&self, values: &VectorValues<K>) -> SkoroResult<Dense> {
        let delta = values.flatten(&self.keys)?;
        self.a.matmul(&delta)?.sub(&self.b)
    }
}

/// A linear-Gaussian factor, in whichever form it arrives.
///
/// Only the Jacobian (square-root) form exists today. A Hessian form `[G g; gᵀ f]` — what a term
/// with no closed-form root would contribute once `hessian_*` assembly exists — joins here, and
/// front elimination would dispatch on it (QR for an all-Jacobian front, Cholesky otherwise).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum GaussianFactor<K: Key> {
    Jacobian(JacobianFactor<K>),
}

impl<K: Key> GaussianFactor<K> {
    pub fn keys(&self) -> &[K] {
        match self {
            Self::Jacobian(f) => f.keys(),
        }
    }
}

/// The part of a factor graph a symbolic elimination depends on: each key's dimension, and which
/// keys each factor touches.
///
/// Two graphs with equal structures can share one [`JunctionTree`](crate::elimination::JunctionTree)
/// — successive iterations of a solver, the rows of a batch, or ticks of a filter whose topology
/// did not change. The tree checks this on use rather than trusting it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Structure<K: Key> {
    pub(crate) keys: Vec<K>,
    pub(crate) dims: Vec<usize>,
    /// For each factor, its keys as indices into `keys`, in the factor's own order.
    pub(crate) factors: Vec<Vec<usize>>,
}

impl<K: Key> Structure<K> {
    pub fn keys(&self) -> &[K] {
        &self.keys
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn factor_count(&self) -> usize {
        self.factors.len()
    }
}

/// Factors over declared keys.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianFactorGraph<K: Key> {
    dims: BTreeMap<K, usize>,
    factors: Vec<GaussianFactor<K>>,
}

impl<K: Key> Default for GaussianFactorGraph<K> {
    fn default() -> Self {
        Self {
            dims: BTreeMap::new(),
            factors: Vec::new(),
        }
    }
}

impl<K: Key> GaussianFactorGraph<K> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare `key` with tangent dimension `dim`. Redeclaring with the same dimension is a no-op;
    /// with a different one it is [`SkoroError::ShapeMismatch`].
    pub fn declare(&mut self, key: K, dim: usize) -> SkoroResult<()> {
        match self.dims.get(&key) {
            Some(&existing) if existing != dim => Err(SkoroError::ShapeMismatch {
                expected: existing,
                found: dim,
            }),
            _ => {
                self.dims.insert(key, dim);
                Ok(())
            }
        }
    }

    /// Add a factor, returning its index. Errors [`SkoroError::InvalidFactorKeys`] if it names an
    /// undeclared key, names one twice, or has a block whose width is not that key's dimension.
    pub fn add(&mut self, factor: JacobianFactor<K>) -> SkoroResult<usize> {
        let index = self.factors.len();
        let mut seen = HashSet::new();
        for (key, dim) in factor.keys.iter().zip(&factor.dims) {
            if self.dims.get(key) != Some(dim) || !seen.insert(*key) {
                return Err(SkoroError::InvalidFactorKeys { factor: index });
            }
        }
        self.factors.push(GaussianFactor::Jacobian(factor));
        Ok(index)
    }

    /// Every declared key, in `Ord` order — the order [`SkoroError::KeyUnconstrained`] indexes.
    pub fn keys(&self) -> impl Iterator<Item = K> + '_ {
        self.dims.keys().copied()
    }

    pub fn dim(&self, key: K) -> Option<usize> {
        self.dims.get(&key).copied()
    }

    /// The summed dimension of every declared key.
    pub fn total_dim(&self) -> usize {
        self.dims.values().sum()
    }

    pub fn factors(&self) -> &[GaussianFactor<K>] {
        &self.factors
    }

    pub fn len(&self) -> usize {
        self.factors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }

    /// The structure a symbolic elimination is computed from.
    pub fn structure(&self) -> Structure<K> {
        let keys: Vec<K> = self.dims.keys().copied().collect();
        let dims: Vec<usize> = self.dims.values().copied().collect();
        let factors = self
            .factors
            .iter()
            .map(|f| {
                f.keys()
                    .iter()
                    .map(|k| {
                        keys.binary_search(k)
                            .expect("internal: a factor's keys were checked on add")
                    })
                    .collect()
            })
            .collect();
        Structure {
            keys,
            dims,
            factors,
        }
    }

    /// This graph plus the Levenberg–Marquardt prior `√λ I` on every key.
    ///
    /// `(AᵀA + λI) δ = Aᵀb` is exactly the least-squares problem of the augmented rows, so a
    /// damped step is the undamped elimination of this graph, at `κ` rather than `κ²`.
    pub fn damped(&self, lambda: f64) -> SkoroResult<Self> {
        let mut out = self.clone();
        let s = lambda.sqrt();
        for (&key, &dim) in &self.dims {
            out.add(JacobianFactor::new(
                vec![(key, Dense::scaled_identity(dim, s))],
                Dense::zeros(dim, 1),
            )?)?;
        }
        Ok(out)
    }

    /// This graph plus the prior `√λ D^{1/2}` on every key, `D` a positive diagonal given per key
    /// in `diagonal` (Marquardt's scaling, typically the diagonal of `AᵀA`).
    ///
    /// Errors [`SkoroError::StructureMismatch`] if a key has no diagonal, and
    /// [`SkoroError::ShapeMismatch`] if one has the wrong height.
    pub fn damped_diagonal(&self, lambda: f64, diagonal: &VectorValues<K>) -> SkoroResult<Self> {
        let mut out = self.clone();
        for (&key, &dim) in &self.dims {
            let d = diagonal.get(key).ok_or(SkoroError::StructureMismatch)?;
            if d.rows() != dim {
                return Err(SkoroError::ShapeMismatch {
                    expected: dim,
                    found: d.rows(),
                });
            }
            let block = Dense::from_fn(dim, dim, |r, c| {
                if r == c {
                    (lambda * d.get(r, 0)).sqrt()
                } else {
                    0.0
                }
            });
            out.add(JacobianFactor::new(
                vec![(key, block)],
                Dense::zeros(dim, 1),
            )?)?;
        }
        Ok(out)
    }

    /// The whole system as one dense `[A | b]` pair, columns in `order` — for tests and for models
    /// small enough not to care.
    ///
    /// Errors [`SkoroError::StructureMismatch`] if `order` omits a key some factor touches or names
    /// an undeclared one.
    pub fn to_dense(&self, order: &[K]) -> SkoroResult<(Dense, Dense)> {
        let mut offset = BTreeMap::new();
        let mut total = 0;
        for key in order {
            let dim = self.dim(*key).ok_or(SkoroError::StructureMismatch)?;
            if offset.insert(*key, total).is_some() {
                return Err(SkoroError::StructureMismatch);
            }
            total += dim;
        }
        let rows: usize = self
            .factors
            .iter()
            .map(|GaussianFactor::Jacobian(f)| f.rows())
            .sum();
        let mut a = Dense::zeros(rows, total);
        let mut b = Dense::zeros(rows, 1);
        let mut at = 0;
        for GaussianFactor::Jacobian(f) in &self.factors {
            for (i, key) in f.keys.iter().enumerate() {
                let col = *offset.get(key).ok_or(SkoroError::StructureMismatch)?;
                a.set_block(at, col, &f.block(i));
            }
            b.set_block(at, 0, &f.b);
            at += f.rows();
        }
        Ok((a, b))
    }

    /// `½ Σ ‖A δ − b‖²` at `values`, the least-squares objective a step is judged by.
    pub fn error(&self, values: &VectorValues<K>) -> SkoroResult<f64> {
        let mut total = 0.0;
        for GaussianFactor::Jacobian(f) in &self.factors {
            total += f.residual(values)?.half_sum_of_squares();
        }
        Ok(total)
    }

    /// `Aᵀ(Aδ − b)` summed over the factors, per key: the gradient of [`error`](Self::error).
    pub fn gradient(&self, values: &VectorValues<K>) -> SkoroResult<VectorValues<K>> {
        let mut out = VectorValues::new();
        for (&key, &dim) in &self.dims {
            out.insert(key, Dense::zeros(dim, 1))?;
        }
        for GaussianFactor::Jacobian(f) in &self.factors {
            let r = f.residual(values)?;
            for (i, key) in f.keys.iter().enumerate() {
                let g = f.block(i).transpose().matmul(&r)?;
                let acc = out.get(*key).expect("every key was seeded").add(&g)?;
                out.insert(*key, acc)?;
            }
        }
        Ok(out)
    }
}
