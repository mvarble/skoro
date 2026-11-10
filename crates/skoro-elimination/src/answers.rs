//! What an elimination answers: traversals of its conditionals, and one more factorization of
//! what it left.

use std::f64::consts::PI;

use faer::Mat;
use faer::linalg::solvers::Qr;

use crate::{Elimination, Error, Matrix};

/// A Gaussian marginal over the kept blocks, in their order: `W` lower triangular with a positive
/// diagonal, its precision `WᵀW`, and its mean.
#[derive(Clone, PartialEq, Debug)]
pub struct Marginal {
    pub blocks: Vec<usize>,
    pub whitening: Matrix,
    pub mean: Vec<f64>,
}

impl Marginal {
    /// `W⁻¹ W⁻ᵀ`, by triangular solves.
    pub fn covariance(&self) -> Matrix {
        let n = self.whitening.rows();
        // The columns of W⁻¹, by forward substitution of the identity's.
        let inverse: Vec<Vec<f64>> = (0..n)
            .map(|j| {
                let e: Vec<f64> = (0..n).map(|i| if i == j { 1.0 } else { 0.0 }).collect();
                lower_solve(&self.whitening, &e)
            })
            .collect();
        let mut covariance = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let s: f64 = (0..n).map(|k| inverse[k][i] * inverse[k][j]).sum();
                covariance.set(i, j, s);
            }
        }
        covariance
    }

    /// `Σ log W_ii`: half the log-determinant of the precision.
    pub fn log_determinant(&self) -> f64 {
        (0..self.whitening.rows())
            .map(|i| self.whitening.get(i, i).ln())
            .sum()
    }
}

/// `-½|A τ - b|² - ηᵀτ + c` over the kept blocks: what integrating the eliminated blocks out
/// leaves, in numbers.
#[derive(Clone, PartialEq, Debug)]
pub struct Reduced {
    pub blocks: Vec<usize>,
    /// `min(m', D)` rows by the kept blocks' `D` columns, in their order.
    pub a: Matrix,
    pub b: Vec<f64>,
    /// `η` over the kept coordinates, in their order, where the problem had a linear term.
    pub linear: Option<Vec<f64>>,
    pub constant: f64,
}

impl Reduced {
    /// `-½|A τ - b|² - ηᵀτ + c` at `tangent`, over the kept coordinates.
    pub fn value(&self, tangent: &[f64]) -> f64 {
        let squares: f64 = self
            .a
            .apply(tangent)
            .iter()
            .zip(&self.b)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let linear: f64 = self
            .linear
            .as_ref()
            .map_or(0.0, |eta| eta.iter().zip(tangent).map(|(e, t)| e * t).sum());
        -0.5 * squares - linear + self.constant
    }
}

impl Elimination {
    /// What is left of a `Schedule::eliminating` elimination, reduced by one more QR with no
    /// pivoting and no decision of rank, with the constant of integrating the eliminated blocks
    /// out: `c = (d_E/2) log 2π - Σ log R_ii - c₀ - Γ - ½|e|²`.
    pub fn reduced(&self) -> Reduced {
        let (stacked, coordinates) = self.stacked(false);
        let n = coordinates.len();
        let m = stacked.nrows();
        let reduced = if m == 0 {
            Matrix::zeros(0, n + 1)
        } else {
            Matrix::from_faer(Qr::new(stacked.as_ref()).thin_R())
        };
        let rows = m.min(n);
        let a = Matrix::new(
            rows,
            n,
            (0..rows)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .map(|(i, j)| reduced.get(i, j))
                .collect(),
        )
        .unwrap_or_else(|| Matrix::zeros(rows, n));
        let b = (0..rows).map(|i| reduced.get(i, n)).collect();
        let residual = if m > n { reduced.get(n, n) } else { 0.0 };
        let linear = self
            .left_linear
            .as_ref()
            .map(|eta| coordinates.iter().map(|&c| eta[c]).collect());
        let d_e = self.eliminated as f64;
        let constant = 0.5 * d_e * (2.0 * PI).ln()
            - self.log_determinant
            - self.constant
            - 0.5 * residual * residual;
        Reduced {
            blocks: self.schedule.kept.clone(),
            a,
            b,
            linear,
            constant,
        }
    }

    /// `log ∫ e^{-φ}` of a complete elimination: `(D/2) log 2π - Σ log R_ii - min φ`
    /// (`Incomplete` when blocks were kept).
    pub fn log_integral(&self) -> Result<f64, Error> {
        self.complete()?;
        let d = self.eliminated as f64;
        Ok(0.5 * d * (2.0 * PI).ln() - self.log_determinant - self.constant)
    }

    /// `R⁻¹ z`, from the roots to the leaves: a draw from `N(0, (AᵀA)⁻¹)` for standard normal `z`,
    /// stacked over all blocks (`Incomplete` when blocks were kept, `Shape` for a `z` not of the
    /// structure's dimension).
    pub fn solve(&self, z: &[f64]) -> Result<Vec<f64>, Error> {
        self.complete()?;
        let structure = &self.schedule.structure;
        if z.len() != structure.dimension() {
            return Err(Error::Shape { factor: None });
        }
        Ok(self.back_substitute(|c| {
            c.frontals
                .iter()
                .flat_map(|&k| z[structure.range(k)].to_vec())
                .collect()
        }))
    }

    /// The factors left on the kept blocks stacked into one matrix, its columns the kept blocks'
    /// coordinates in the schedule's order of them, `reversed` if asked, with the right-hand side
    /// last.
    pub(crate) fn stacked(&self, reversed: bool) -> (Mat<f64>, Vec<usize>) {
        let structure = &self.schedule.structure;
        let kept = &self.schedule.kept;
        let mut column = vec![usize::MAX; structure.dimension()];
        let mut width = 0;
        for &k in kept {
            for c in structure.range(k) {
                column[c] = width;
                width += 1;
            }
        }
        let place = |c: usize| {
            if reversed {
                width - 1 - column[c]
            } else {
                column[c]
            }
        };
        let m: usize = self.left.iter().map(|(_, rows)| rows.b.len()).sum();
        let mut stacked = Mat::<f64>::zeros(m, width + 1);
        let mut row = 0;
        for (blocks, rows) in &self.left {
            let mut at = 0;
            for &k in blocks {
                for c in structure.range(k) {
                    for r in 0..rows.b.len() {
                        stacked[(row + r, place(c))] = rows.a.get(r, at);
                    }
                    at += 1;
                }
            }
            for r in 0..rows.b.len() {
                stacked[(row + r, width)] = rows.b[r];
            }
            row += rows.b.len();
        }
        let coordinates = kept.iter().flat_map(|&k| structure.range(k)).collect();
        (stacked, coordinates)
    }

    /// The marginal on the kept blocks of a `Schedule::keeping` elimination, by one more QR of
    /// what is left with the kept coordinates reversed, as the note says: refuses kept blocks the
    /// rows do not determine (`Undetermined`).
    pub fn marginal(&self) -> Result<Marginal, Error> {
        let structure = &self.schedule.structure;
        let (stacked, coordinates) = self.stacked(true);
        let n = coordinates.len();
        let norms: Vec<f64> = (0..n)
            .map(|j| {
                (0..stacked.nrows())
                    .map(|r| stacked[(r, j)].powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        let reduced = if stacked.nrows() == 0 {
            Matrix::zeros(0, n + 1)
        } else {
            Matrix::from_faer(Qr::new(stacked.as_ref()).thin_R())
        };
        let at = |r: usize, c: usize| {
            if r < reduced.rows() {
                reduced.get(r, c)
            } else {
                0.0
            }
        };
        // The block of the `i`-th kept coordinate, for a refusal.
        let block_of = |i: usize| {
            let c = coordinates[i];
            (0..structure.blocks().len())
                .find(|&k| structure.range(k).contains(&c))
                .unwrap_or(0)
        };
        let threshold = self.roundoff.powf(0.75);
        for (j, norm) in norms.iter().enumerate() {
            let diagonal = at(j, j).abs();
            if diagonal <= threshold * norm || !diagonal.is_finite() {
                return Err(Error::Undetermined {
                    block: block_of(n - 1 - j),
                });
            }
        }
        // W = P R' P, P the reversal, each row's sign changed so that the diagonal is positive.
        let mut whitening = Matrix::zeros(n, n);
        let mut d = Vec::with_capacity(n);
        for i in 0..n {
            let ri = n - 1 - i;
            let sign = at(ri, ri).signum();
            for j in 0..=i {
                whitening.set(i, j, sign * at(ri, n - 1 - j));
            }
            d.push(sign * at(ri, n));
        }
        // With a linear term, complete the square: y = W⁻ᵀ η, mean = W⁻¹ (d - y).
        if let Some(eta) = &self.left_linear {
            let eta_k: Vec<f64> = coordinates.iter().map(|&c| eta[c]).collect();
            let y = lower_transpose_solve(&whitening, &eta_k);
            for (d, y) in d.iter_mut().zip(&y) {
                *d -= y;
            }
        }
        let mean = lower_solve(&whitening, &d);
        Ok(Marginal {
            blocks: self.schedule.kept.clone(),
            whitening,
            mean,
        })
    }
}

/// `x` with `W x = v`, `W` lower triangular, by forward substitution.
pub(crate) fn lower_solve(w: &Matrix, v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut x = vec![0.0; n];
    for i in 0..n {
        let rest: f64 = (0..i).map(|j| w.get(i, j) * x[j]).sum();
        x[i] = (v[i] - rest) / w.get(i, i);
    }
    x
}

/// `y` with `Wᵀ y = v`, `W` lower triangular, by back substitution.
pub(crate) fn lower_transpose_solve(w: &Matrix, v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut y = vec![0.0; n];
    for i in (0..n).rev() {
        let rest: f64 = (i + 1..n).map(|j| w.get(j, i) * y[j]).sum();
        y[i] = (v[i] - rest) / w.get(i, i);
    }
    y
}
