//! Numeric elimination: dense QR on each front, giving a Bayes tree of Gaussian conditionals.
//!
//! Each clique of a [`JunctionTree`] gathers its factors and its children's separator factors into
//! one dense `[A_f | A_s | b]`, frontal columns first, and factors it by **unpivoted** Householder
//! QR. The top rows are the conditional `R x_f + S x_s = d`; the rows after them are a new factor
//! on the separator, handed to the parent clique (or, past the eliminated prefix, to the remaining
//! graph); anything below is residual error. Column pivoting would be wrong here, not merely
//! different: it would mix frontal and separator columns, and the partition is the whole point.
//!
//! What the rows mean is not this module's business. Rows with `b = −r` give the Gauss–Newton step
//! by [`BayesTree::back_substitute`]; rows with `b = 0` and a gradient `g` give the Fisher-scoring
//! step by [`BayesTree::solve_normal`]; and in either case `R` is a square root of the curvature,
//! so [`BayesTree::solve_upper`] turns standard normal draws into draws from the Laplace
//! approximation.

use faer::linalg::solvers::Qr;
use skoro_core::error::{SkoroError, SkoroResult};

use crate::dense::Dense;
use crate::elimination::JunctionTree;
use crate::factor::{solve_lower, solve_upper};
use crate::factor_graph::{GaussianFactor, GaussianFactorGraph, JacobianFactor};
use crate::values::{Key, VectorValues};

/// A frontal diagonal smaller than this, relative to the norm of its column before elimination,
/// is a direction the front does not determine.
///
/// Exact dependence — gauge freedom, a key reached by no informative row — leaves a diagonal at
/// the level of rounding, around `1e-16` relative, far below this; a merely ill-conditioned but
/// determined problem stays far above it. That assumes rows computed in `f64`. Rows that passed
/// through `f32` carry `1e-7` relative noise, so an exact dependence there survives this test and
/// comes back as a huge step instead of [`SkoroError::RankDeficient`].
const RANK_TOLERANCE: f64 = 1e-12;

/// `R x_f + S x_s = d`: what eliminating a clique's frontals leaves, conditioned on its separator.
///
/// `R` is upper-triangular with a positive diagonal. Its row blocks follow
/// [`frontals`](Self::frontals); `S`'s column blocks follow [`separator`](Self::separator).
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianConditional<K: Key> {
    frontals: Vec<K>,
    frontal_dims: Vec<usize>,
    separator: Vec<K>,
    separator_dims: Vec<usize>,
    r: Dense,
    s: Dense,
    d: Dense,
    // Key indices of the graph, for the solves.
    frontal_idx: Vec<usize>,
    separator_idx: Vec<usize>,
}

impl<K: Key> GaussianConditional<K> {
    pub fn frontals(&self) -> &[K] {
        &self.frontals
    }

    pub fn frontal_dims(&self) -> &[usize] {
        &self.frontal_dims
    }

    pub fn separator(&self) -> &[K] {
        &self.separator
    }

    pub fn separator_dims(&self) -> &[usize] {
        &self.separator_dims
    }

    pub fn r(&self) -> &Dense {
        &self.r
    }

    pub fn s(&self) -> &Dense {
        &self.s
    }

    pub fn d(&self) -> &Dense {
        &self.d
    }

    /// `x_f = R⁻¹ (rhs − S x_s)`.
    fn solve_frontals(&self, rhs: &Dense, x_s: &Dense) -> SkoroResult<Dense> {
        let rhs = if self.s.cols() == 0 {
            rhs.clone()
        } else {
            rhs.sub(&self.s.matmul(x_s)?)?
        };
        solve_upper(&self.r, &rhs)
    }
}

/// The conditionals of an elimination, arranged as its junction tree was.
///
/// Stored children before parents. When every key was eliminated this is a square root of the
/// whole curvature: `RᵀR = AᵀA` with `R` block upper-triangular in elimination order, whose block
/// rows are the conditionals.
#[derive(Debug, Clone, PartialEq)]
pub struct BayesTree<K: Key> {
    keys: Vec<K>,
    dims: Vec<usize>,
    conditionals: Vec<GaussianConditional<K>>,
    parents: Vec<Option<usize>>,
    complete: bool,
}

/// What [`eliminate`] returns.
#[derive(Debug, Clone, PartialEq)]
pub struct Elimination<K: Key> {
    /// One conditional per clique of the junction tree.
    pub tree: BayesTree<K>,
    /// Every kept key, with the factors that touched no eliminated key (untouched) and the
    /// separator factors the elimination left on the kept keys. Empty after a full elimination.
    pub remaining: GaussianFactorGraph<K>,
    /// `½‖e‖²` of the rows the eliminated fronts could not absorb: the least-squares objective at
    /// the optimum of the eliminated keys, given the kept ones at zero. After a full elimination
    /// of rows `[A | b]` it is `min_δ ½‖Aδ − b‖²`.
    pub error: f64,
}

/// Rows over key indices, in flight between a clique and its parent.
struct Rows {
    keys: Vec<usize>,
    a: Dense,
    b: Dense,
}

/// Eliminate `graph` along `tree`.
///
/// Errors [`SkoroError::StructureMismatch`] if `graph`'s structure is not the one `tree` was built
/// for, and [`SkoroError::RankDeficient`] naming the elimination position of the first key a front
/// cannot determine.
pub fn eliminate<K: Key>(
    tree: &JunctionTree<K>,
    graph: &GaussianFactorGraph<K>,
) -> SkoroResult<Elimination<K>> {
    if graph.structure() != tree.structure {
        return Err(SkoroError::StructureMismatch);
    }
    let structure = &tree.structure;
    let factors: Vec<&JacobianFactor<K>> = graph
        .factors()
        .iter()
        .map(|GaussianFactor::Jacobian(f)| f)
        .collect();
    let n = structure.keys.len();
    let mut position = vec![0; n];
    for (p, &k) in tree.order.iter().enumerate() {
        position[k] = p;
    }

    let mut pending: Vec<Vec<Rows>> = (0..tree.cliques.len()).map(|_| Vec::new()).collect();
    let mut roots: Vec<Rows> = Vec::new();
    let mut conditionals = Vec::with_capacity(tree.cliques.len());
    let mut error = 0.0;
    let mut offset = vec![usize::MAX; n];

    for (id, clique) in tree.cliques.iter().enumerate() {
        // Column layout: frontals, then separator, then b.
        let mut width = 0;
        for &k in clique.frontals.iter().chain(&clique.separator) {
            offset[k] = width;
            width += structure.dims[k];
        }
        let frontal_width: usize = clique.frontals.iter().map(|&k| structure.dims[k]).sum();
        let incoming = std::mem::take(&mut pending[id]);
        let rows: usize = clique
            .factors
            .iter()
            .map(|&f| factors[f].rows())
            .chain(incoming.iter().map(|r| r.b.rows()))
            .sum();

        let mut m = Dense::zeros(rows, width + 1);
        let mut at = 0;
        for &f in &clique.factors {
            let factor = factors[f];
            let mut col = 0;
            for (&k, &dim) in structure.factors[f].iter().zip(factor.dims()) {
                m.set_block(at, offset[k], &factor.a().block(0, col, factor.rows(), dim));
                col += dim;
            }
            m.set_block(at, width, factor.b());
            at += factor.rows();
        }
        for part in &incoming {
            let mut col = 0;
            for &k in &part.keys {
                let dim = structure.dims[k];
                m.set_block(at, offset[k], &part.a.block(0, col, part.a.rows(), dim));
                col += dim;
            }
            m.set_block(at, width, &part.b);
            at += part.b.rows();
        }

        // Which frontal key owns scalar column `i`.
        let owner = |i: usize| {
            let mut acc = 0;
            for &k in &clique.frontals {
                acc += structure.dims[k];
                if i < acc {
                    return position[k];
                }
            }
            unreachable!("column {i} is frontal")
        };
        let column_norm = |i: usize| (0..rows).map(|r| m.get(r, i).powi(2)).sum::<f64>().sqrt();

        let mut r = if rows == 0 {
            Dense::zeros(0, width + 1)
        } else {
            Dense::from_faer_ref(Qr::new(m.as_faer()).thin_R())
        };
        for i in 0..frontal_width {
            let norm = column_norm(i);
            if i >= r.rows() || norm == 0.0 || r.get(i, i).abs() <= RANK_TOLERANCE * norm {
                return Err(SkoroError::RankDeficient { position: owner(i) });
            }
            if r.get(i, i) < 0.0 {
                for c in i..=width {
                    r.set(i, c, -r.get(i, c));
                }
            }
        }

        let separator_width = width - frontal_width;
        let separator_rows = r.rows().min(width).saturating_sub(frontal_width);
        for row in frontal_width + separator_rows..r.rows() {
            error += 0.5 * r.get(row, width).powi(2);
        }
        if separator_rows > 0 {
            let rows = Rows {
                keys: clique.separator.clone(),
                a: r.block(
                    frontal_width,
                    frontal_width,
                    separator_rows,
                    separator_width,
                ),
                b: r.block(frontal_width, width, separator_rows, 1),
            };
            match clique.parent {
                Some(parent) => pending[parent].push(rows),
                None => roots.push(rows),
            }
        }

        conditionals.push(GaussianConditional {
            frontals: clique.frontals.iter().map(|&k| structure.keys[k]).collect(),
            frontal_dims: clique.frontals.iter().map(|&k| structure.dims[k]).collect(),
            separator: clique
                .separator
                .iter()
                .map(|&k| structure.keys[k])
                .collect(),
            separator_dims: clique
                .separator
                .iter()
                .map(|&k| structure.dims[k])
                .collect(),
            r: r.block(0, 0, frontal_width, frontal_width),
            s: r.block(0, frontal_width, frontal_width, separator_width),
            d: r.block(0, width, frontal_width, 1),
            frontal_idx: clique.frontals.clone(),
            separator_idx: clique.separator.clone(),
        });
    }

    let mut remaining = GaussianFactorGraph::new();
    for &k in &tree.order[tree.eliminated..] {
        remaining.declare(structure.keys[k], structure.dims[k])?;
    }
    for &f in &tree.passthrough {
        remaining.add(factors[f].clone())?;
    }
    for rows in roots {
        let mut blocks = Vec::with_capacity(rows.keys.len());
        let mut col = 0;
        for &k in &rows.keys {
            let dim = structure.dims[k];
            blocks.push((structure.keys[k], rows.a.block(0, col, rows.a.rows(), dim)));
            col += dim;
        }
        remaining.add(JacobianFactor::new(blocks, rows.b)?)?;
    }

    Ok(Elimination {
        tree: BayesTree {
            keys: structure.keys.clone(),
            dims: structure.dims.clone(),
            conditionals,
            parents: tree.cliques.iter().map(|c| c.parent).collect(),
            complete: tree.eliminated == n,
        },
        remaining,
        error,
    })
}

impl<K: Key> BayesTree<K> {
    /// The conditionals, children before parents.
    pub fn conditionals(&self) -> &[GaussianConditional<K>] {
        &self.conditionals
    }

    /// The parent of each conditional, by index into [`conditionals`](Self::conditionals).
    pub fn parents(&self) -> &[Option<usize>] {
        &self.parents
    }

    /// Whether every key of the graph was eliminated, so that `R` is a square root of the whole
    /// curvature. The solves other than [`back_substitute_given`](Self::back_substitute_given)
    /// need it.
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// `Σ ln R_ii`, which is `½ ln det(AᵀA)` for a complete tree: the log-determinant a Laplace
    /// evidence needs.
    pub fn log_det_r(&self) -> f64 {
        self.conditionals
            .iter()
            .map(|c| (0..c.r.rows()).map(|i| c.r.get(i, i).ln()).sum::<f64>())
            .sum()
    }

    /// The least-squares solution `δ = R⁻¹ d`, top-down: the Gauss–Newton step when the rows were
    /// `[J | −r]`.
    ///
    /// Errors [`SkoroError::StructureMismatch`] on an incomplete tree; use
    /// [`back_substitute_given`](Self::back_substitute_given) there.
    pub fn back_substitute(&self) -> SkoroResult<VectorValues<K>> {
        self.require_complete()?;
        self.back_substitute_given(&VectorValues::new())
    }

    /// The optimum of the eliminated keys given values for the kept ones.
    ///
    /// Errors [`SkoroError::StructureMismatch`] if a kept key the tree conditions on has no value.
    pub fn back_substitute_given(&self, kept: &VectorValues<K>) -> SkoroResult<VectorValues<K>> {
        self.top_down(kept, |c| Ok(c.d.clone()))
    }

    /// `R⁻¹ z`, top-down. With `z ~ N(0, I)` per key this is a draw from `N(0, (RᵀR)⁻¹)` — the
    /// Laplace approximation's spread, to be retracted onto its mode.
    pub fn solve_upper(&self, z: &VectorValues<K>) -> SkoroResult<VectorValues<K>> {
        self.require_complete()?;
        self.top_down(&VectorValues::new(), |c| {
            self.stack(z, &c.frontal_idx)
                .ok_or(SkoroError::StructureMismatch)
        })
    }

    /// `R⁻ᵀ g`, bottom-up.
    pub fn solve_upper_transpose(&self, g: &VectorValues<K>) -> SkoroResult<VectorValues<K>> {
        self.require_complete()?;
        let mut acc: Vec<Option<Dense>> = vec![None; self.keys.len()];
        for (k, value) in g.iter() {
            let i = self
                .keys
                .binary_search(&k)
                .map_err(|_| SkoroError::StructureMismatch)?;
            if value.rows() != self.dims[i] {
                return Err(SkoroError::ShapeMismatch {
                    expected: self.dims[i],
                    found: value.rows(),
                });
            }
            acc[i] = Some(value.clone());
        }
        let mut out = VectorValues::new();
        for c in &self.conditionals {
            let rhs = stack_indices(&acc, &c.frontal_idx).ok_or(SkoroError::StructureMismatch)?;
            let y = solve_lower(&c.r.transpose(), &rhs)?;
            if c.s.cols() > 0 {
                let push = c.s.transpose().matmul(&y)?;
                let mut at = 0;
                for &k in &c.separator_idx {
                    let dim = self.dims[k];
                    let part = push.block(at, 0, dim, 1);
                    let current = acc[k].take().ok_or(SkoroError::StructureMismatch)?;
                    acc[k] = Some(current.sub(&part)?);
                    at += dim;
                }
            }
            self.split_into(&mut out, &c.frontal_idx, &y)?;
        }
        Ok(out)
    }

    /// `(RᵀR)⁻¹ g`: the step `δ` solving `AᵀA δ = g` for a gradient `g` that does not come from
    /// the rows' own right-hand sides — the Fisher-scoring step, `g = ∇ log p`.
    ///
    /// Two triangular passes against a factor computed at `κ(A)`; the solve itself is sensitive at
    /// `κ(A)²`, as any normal-equations solve is. When the rows carry their own residual,
    /// [`back_substitute`](Self::back_substitute) avoids that.
    pub fn solve_normal(&self, g: &VectorValues<K>) -> SkoroResult<VectorValues<K>> {
        let y = self.solve_upper_transpose(g)?;
        self.solve_upper(&y)
    }

    fn require_complete(&self) -> SkoroResult<()> {
        if self.complete {
            Ok(())
        } else {
            Err(SkoroError::StructureMismatch)
        }
    }

    /// Top-down: each conditional's frontals from `rhs(c)` and its already-solved separator.
    fn top_down(
        &self,
        kept: &VectorValues<K>,
        rhs: impl Fn(&GaussianConditional<K>) -> SkoroResult<Dense>,
    ) -> SkoroResult<VectorValues<K>> {
        let mut solved: Vec<Option<Dense>> = vec![None; self.keys.len()];
        for (k, value) in kept.iter() {
            if let Ok(i) = self.keys.binary_search(&k) {
                solved[i] = Some(value.clone());
            }
        }
        let mut out = VectorValues::new();
        for c in self.conditionals.iter().rev() {
            let x_s = if c.separator_idx.is_empty() {
                Dense::zeros(0, 1)
            } else {
                stack_indices(&solved, &c.separator_idx).ok_or(SkoroError::StructureMismatch)?
            };
            let x_f = c.solve_frontals(&rhs(c)?, &x_s)?;
            let mut at = 0;
            for &k in &c.frontal_idx {
                let dim = self.dims[k];
                solved[k] = Some(x_f.block(at, 0, dim, 1));
                at += dim;
            }
            self.split_into(&mut out, &c.frontal_idx, &x_f)?;
        }
        Ok(out)
    }

    fn stack(&self, values: &VectorValues<K>, idx: &[usize]) -> Option<Dense> {
        let parts: Option<Vec<&Dense>> = idx.iter().map(|&k| values.get(self.keys[k])).collect();
        Dense::vstack_all(&parts?).ok()
    }

    fn split_into(&self, out: &mut VectorValues<K>, idx: &[usize], x: &Dense) -> SkoroResult<()> {
        let mut at = 0;
        for &k in idx {
            let dim = self.dims[k];
            out.insert(self.keys[k], x.block(at, 0, dim, 1))?;
            at += dim;
        }
        Ok(())
    }
}

fn stack_indices(values: &[Option<Dense>], idx: &[usize]) -> Option<Dense> {
    let parts: Option<Vec<&Dense>> = idx.iter().map(|&k| values[k].as_ref()).collect();
    Dense::vstack_all(&parts?).ok()
}
