//! The numeric elimination: each clique's rows stacked into one front, frontal columns first,
//! and reduced by unpivoted Householder QR, as the lemma of eliminating one block says.

use faer::Mat;
use faer::linalg::solvers::Qr;

use crate::{Damping, Error, Matrix, Problem, Rows, Schedule};

/// `R τ_F + T τ_S = rhs`: what eliminating a clique's frontal blocks leaves, given its separator.
/// `R` is upper triangular with a positive diagonal, its columns the frontal blocks' coordinates
/// in order; `T`'s columns are the separator blocks' in order.
#[derive(Clone, PartialEq, Debug)]
pub struct Conditional {
    pub frontals: Vec<usize>,
    pub separator: Vec<usize>,
    pub r: Matrix,
    pub t: Matrix,
    /// `d - y`: the reduced right-hand side less the pushes of the frontal blocks' linear terms.
    pub rhs: Vec<f64>,
}

/// What an elimination leaves: a conditional per clique, the factors left on the blocks it did not
/// eliminate, and the constants.
#[derive(Clone, Debug)]
pub struct Elimination {
    pub(crate) schedule: Schedule,
    /// One per clique, in the schedule's order.
    pub(crate) conditionals: Vec<Conditional>,
    /// The new factors on the kept blocks, and the original factors joined to none eliminated.
    pub(crate) left: Vec<(Vec<usize>, Rows)>,
    /// The linear term, over all blocks, as the eliminations pushed it: what is left of it on the
    /// kept blocks; `None` when the problem had none.
    pub(crate) left_linear: Option<Vec<f64>>,
    /// `c + Σγ + ½|e|²`: the constant of what is left.
    pub(crate) constant: f64,
    /// `Σ log R_ii` over the eliminated blocks.
    pub(crate) log_determinant: f64,
    /// The dimension of the eliminated blocks.
    pub(crate) eliminated: usize,
}

impl Elimination {
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    /// The conditionals, one per clique of the schedule, children before parents.
    pub fn conditionals(&self) -> &[Conditional] {
        &self.conditionals
    }

    /// The factors left on the kept blocks: the new factors of the cliques whose separators are
    /// kept, and the original factors joined to no eliminated block, each with its blocks.
    pub fn left(&self) -> &[(Vec<usize>, Rows)] {
        &self.left
    }

    /// The linear term left on the kept blocks, stacked over all blocks; `None` when the problem
    /// had none.
    pub fn left_linear(&self) -> Option<&[f64]> {
        self.left_linear.as_deref()
    }

    /// The constant of what is left: the problem's, with every `γ` and every residual `½|e|²`.
    pub fn constant(&self) -> f64 {
        self.constant
    }

    /// `Σ log R_ii` over the eliminated blocks: half the log-determinant of `AᵀA` over them.
    pub fn log_determinant(&self) -> f64 {
        self.log_determinant
    }

    /// The dimension of the blocks eliminated.
    pub fn eliminated(&self) -> usize {
        self.eliminated
    }

    pub(crate) fn complete(&self) -> Result<(), Error> {
        if self.schedule.kept.is_empty() {
            Ok(())
        } else {
            Err(Error::Incomplete)
        }
    }

    /// `τ` with each clique's frontal blocks solved from its conditional, given its separator,
    /// from the roots to the leaves, the right-hand side of each given by `rhs`.
    pub(crate) fn back_substitute(&self, rhs: impl Fn(&Conditional) -> Vec<f64>) -> Vec<f64> {
        let structure = &self.schedule.structure;
        let mut tangent = vec![0.0; structure.dimension()];
        for conditional in self.conditionals.iter().rev() {
            let given: Vec<f64> = conditional
                .separator
                .iter()
                .flat_map(|&k| tangent[structure.range(k)].to_vec())
                .collect();
            let t = conditional.t.apply(&given);
            let v: Vec<f64> = rhs(conditional)
                .iter()
                .zip(&t)
                .map(|(r, t)| r - t)
                .collect();
            let x = upper_solve(&conditional.r, &v);
            let mut at = 0;
            for &k in &conditional.frontals {
                let range = structure.range(k);
                let n = range.len();
                tangent[range].copy_from_slice(&x[at..at + n]);
                at += n;
            }
        }
        tangent
    }

    /// The minimizer of a complete elimination, back-substituted from the roots: `τ*`
    /// (`Incomplete` when blocks were kept).
    pub fn minimizer(&self) -> Result<Vec<f64>, Error> {
        self.complete()?;
        Ok(self.back_substitute(|c| c.rhs.clone()))
    }

    /// `min φ`: the constant left by a complete elimination (`Incomplete` when blocks were kept).
    pub fn minimum(&self) -> Result<f64, Error> {
        self.complete()?;
        Ok(self.constant)
    }
}

impl Problem {
    /// Eliminates the schedule's blocks, with the damping rows appended if given.
    pub fn eliminate(
        &self,
        schedule: &Schedule,
        damping: Option<&Damping>,
    ) -> Result<Elimination, Error> {
        let structure = self.structure();
        if **schedule.structure() != **structure {
            return Err(Error::Shape { factor: None });
        }
        if damping.is_some_and(|d| d.diagonal.len() != structure.dimension()) {
            return Err(Error::Shape { factor: None });
        }
        let mut linear: Option<Vec<f64>> = self.linear().map(<[f64]>::to_vec);
        let mut constant = self.constant();
        let mut log_determinant = 0.0;
        let mut eliminated = 0;
        let mut conditionals = Vec::with_capacity(schedule.cliques.len());
        // The new factor each clique leaves for its parent.
        let mut handed: Vec<Option<(Vec<usize>, Rows)>> = vec![None; schedule.cliques.len()];
        let mut left: Vec<(Vec<usize>, Rows)> = schedule
            .left
            .iter()
            .map(|&f| (structure.factors()[f].clone(), self.factors()[f].clone()))
            .collect();
        for (i, clique) in schedule.cliques.iter().enumerate() {
            // The front's columns: the frontal blocks in order, then the separator's.
            let blocks: Vec<usize> = clique
                .frontals
                .iter()
                .chain(&clique.separator)
                .copied()
                .collect();
            let mut offset = vec![usize::MAX; structure.blocks().len()];
            let mut width = 0;
            for &k in &blocks {
                offset[k] = width;
                width += structure.blocks()[k];
            }
            let frontal: usize = clique.frontals.iter().map(|&k| structure.blocks()[k]).sum();
            let separator = width - frontal;
            let mut pieces: Vec<(&[usize], &Rows)> = clique
                .factors
                .iter()
                .map(|&f| (structure.factors()[f].as_slice(), &self.factors()[f]))
                .collect();
            let from_children: Vec<(Vec<usize>, Rows)> = clique
                .children
                .iter()
                .filter_map(|&c| handed[c].take())
                .collect();
            pieces.extend(from_children.iter().map(|(b, r)| (b.as_slice(), r)));
            let damping_rows: Vec<(Vec<usize>, Rows)> = match damping {
                Some(damping) => clique
                    .frontals
                    .iter()
                    .map(|&k| (vec![k], damping_rows(structure.range(k), damping)))
                    .collect(),
                None => Vec::new(),
            };
            pieces.extend(damping_rows.iter().map(|(b, r)| (b.as_slice(), r)));
            let m: usize = pieces.iter().map(|(_, r)| r.b.len()).sum();
            let mut front = Mat::<f64>::zeros(m, width + 1);
            let mut row = 0;
            for (factor_blocks, rows) in &pieces {
                let mut column = 0;
                for &k in factor_blocks.iter() {
                    let d = structure.blocks()[k];
                    for r in 0..rows.b.len() {
                        for j in 0..d {
                            front[(row + r, offset[k] + j)] = rows.a.get(r, column + j);
                        }
                    }
                    column += d;
                }
                for r in 0..rows.b.len() {
                    front[(row + r, width)] = rows.b[r];
                }
                row += rows.b.len();
            }
            let norms: Vec<f64> = (0..frontal)
                .map(|j| (0..m).map(|r| front[(r, j)].powi(2)).sum::<f64>().sqrt())
                .collect();
            let reduced = if m == 0 {
                Matrix::zeros(0, width + 1)
            } else {
                Matrix::from_faer(Qr::new(front.as_ref()).thin_R())
            };
            let at = |r: usize, c: usize| {
                if r < reduced.rows() {
                    reduced.get(r, c)
                } else {
                    0.0
                }
            };
            for (j, norm) in norms.iter().enumerate() {
                let diagonal = at(j, j).abs();
                if diagonal <= self.threshold() * norm || !diagonal.is_finite() {
                    let mut start = 0;
                    let block = clique.frontals.iter().copied().find(|&k| {
                        start += structure.blocks()[k];
                        j < start
                    });
                    return Err(Error::Undetermined {
                        block: block.unwrap_or(0),
                    });
                }
            }
            // R, T and d, each frontal row's sign changed so that R's diagonal is positive.
            let sign: Vec<f64> = (0..frontal).map(|j| at(j, j).signum()).collect();
            let r = Matrix::new(
                frontal,
                frontal,
                (0..frontal)
                    .flat_map(|i| (0..frontal).map(move |j| (i, j)))
                    .map(|(i, j)| if j >= i { sign[i] * at(i, j) } else { 0.0 })
                    .collect(),
            )
            .unwrap_or_else(|| Matrix::zeros(frontal, frontal));
            let t = Matrix::new(
                frontal,
                separator,
                (0..frontal)
                    .flat_map(|i| (0..separator).map(move |j| (i, j)))
                    .map(|(i, j)| sign[i] * at(i, frontal + j))
                    .collect(),
            )
            .unwrap_or_else(|| Matrix::zeros(frontal, separator));
            let d: Vec<f64> = (0..frontal).map(|i| sign[i] * at(i, width)).collect();
            // The pushes of the frontal blocks' linear terms: `Rᵀ y = η_F`.
            let mut rhs = d.clone();
            if let Some(eta) = &mut linear {
                let eta_f: Vec<f64> = clique
                    .frontals
                    .iter()
                    .flat_map(|&k| eta[structure.range(k)].to_vec())
                    .collect();
                let y = upper_transpose_solve(&r, &eta_f);
                constant += y.iter().zip(&d).map(|(y, d)| y * d).sum::<f64>()
                    - 0.5 * y.iter().map(|y| y * y).sum::<f64>();
                let pushed = transpose_apply(&t, &y);
                let mut at = 0;
                for &k in &clique.separator {
                    for c in structure.range(k) {
                        eta[c] -= pushed[at];
                        at += 1;
                    }
                }
                for (r, y) in rhs.iter_mut().zip(&y) {
                    *r -= y;
                }
            }
            // The new factor on the separator, and the residual below it.
            let last = reduced.rows().min(frontal + separator);
            let new_rows = last.saturating_sub(frontal);
            let a_new = Matrix::new(
                new_rows,
                separator,
                (frontal..last)
                    .flat_map(|i| (0..separator).map(move |j| (i, j)))
                    .map(|(i, j)| at(i, frontal + j))
                    .collect(),
            )
            .unwrap_or_else(|| Matrix::zeros(new_rows, separator));
            let b_new: Vec<f64> = (frontal..last).map(|i| at(i, width)).collect();
            if reduced.rows() > frontal + separator {
                constant += 0.5 * at(frontal + separator, width).powi(2);
            }
            log_determinant += (0..frontal).map(|j| at(j, j).abs().ln()).sum::<f64>();
            eliminated += frontal;
            if !clique.separator.is_empty() {
                let factor = (clique.separator.clone(), Rows { a: a_new, b: b_new });
                match clique.parent {
                    Some(_) => handed[i] = Some(factor),
                    None => left.push(factor),
                }
            }
            conditionals.push(Conditional {
                frontals: clique.frontals.clone(),
                separator: clique.separator.clone(),
                r,
                t,
                rhs,
            });
        }
        // Damping rows of the kept blocks stay with what is left.
        if let Some(damping) = damping {
            for &k in &schedule.kept {
                left.push((vec![k], damping_rows(structure.range(k), damping)));
            }
        }
        Ok(Elimination {
            schedule: schedule.clone(),
            conditionals,
            left,
            left_linear: linear,
            constant,
            log_determinant,
            eliminated,
        })
    }

    /// A frontal diagonal entry at most this, relative to its column's norm before the reduction,
    /// is a direction the rows do not determine.
    fn threshold(&self) -> f64 {
        0.0 * self.roundoff()
    }
}

/// `√(λ D_k)` on block `k`'s coordinates, with a right-hand side of zeros.
fn damping_rows(range: std::ops::Range<usize>, damping: &Damping) -> Rows {
    let n = range.len();
    let mut a = Matrix::zeros(n, n);
    for (i, c) in range.enumerate() {
        a.set(i, i, (damping.lambda * damping.diagonal[c]).sqrt());
    }
    Rows { a, b: vec![0.0; n] }
}

/// `x` with `R x = v`, `R` upper triangular, by back substitution.
pub(crate) fn upper_solve(r: &Matrix, v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let rest: f64 = (i + 1..n).map(|j| r.get(i, j) * x[j]).sum();
        x[i] = (v[i] - rest) / r.get(i, i);
    }
    x
}

/// `y` with `Rᵀ y = v`, `R` upper triangular, by forward substitution.
pub(crate) fn upper_transpose_solve(r: &Matrix, v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let rest: f64 = (0..i).map(|j| r.get(j, i) * y[j]).sum();
        y[i] = (v[i] - rest) / r.get(i, i);
    }
    y
}

/// `Aᵀ y`.
pub(crate) fn transpose_apply(a: &Matrix, y: &[f64]) -> Vec<f64> {
    (0..a.columns())
        .map(|j| (0..a.rows()).map(|i| a.get(i, j) * y[i]).sum())
        .collect()
}
