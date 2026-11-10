use std::sync::Arc;

use crate::{Error, Matrix, Structure};

/// One factor's rows: `A_f`, its columns those of its blocks in the structure's order, and `b_f`.
#[derive(Clone, PartialEq, Debug)]
pub struct Rows {
    pub a: Matrix,
    pub b: Vec<f64>,
}

/// `½ λ τᵀ D τ`: `λ` and the positive diagonal `D`, one entry per coordinate.
#[derive(Clone, PartialEq, Debug)]
pub struct Damping {
    pub lambda: f64,
    pub diagonal: Vec<f64>,
}

/// A problem with a constant, `c + Σ_f ½|A_f τ_{K_f} - b_f|² + ηᵀτ`, over a structure.
#[derive(Clone, Debug)]
pub struct Problem {
    structure: Arc<Structure>,
    factors: Vec<Rows>,
    /// `η`, stacked over all blocks, or `None`: absent by structure, and then exactly zero.
    linear: Option<Vec<f64>>,
    constant: f64,
    /// The unit roundoff of the precision the rows were computed at, for the rank test.
    roundoff: f64,
}

impl Problem {
    /// Refuses rows whose shapes do not fit the structure (`Shape`, naming the factor), and a
    /// linear term not of its dimension (`Shape` of no factor).
    pub fn new(
        structure: Arc<Structure>,
        factors: Vec<Rows>,
        linear: Option<Vec<f64>>,
        constant: f64,
        roundoff: f64,
    ) -> Result<Problem, Error> {
        if factors.len() != structure.factors().len() {
            return Err(Error::Shape { factor: None });
        }
        for (f, rows) in factors.iter().enumerate() {
            let width = structure.width(f);
            if rows.a.columns() != width || rows.a.rows() != rows.b.len() {
                return Err(Error::Shape { factor: Some(f) });
            }
        }
        if linear
            .as_ref()
            .is_some_and(|eta| eta.len() != structure.dimension())
        {
            return Err(Error::Shape { factor: None });
        }
        Ok(Problem {
            structure,
            factors,
            linear,
            constant,
            roundoff,
        })
    }

    pub fn structure(&self) -> &Arc<Structure> {
        &self.structure
    }

    pub fn factors(&self) -> &[Rows] {
        &self.factors
    }

    pub fn linear(&self) -> Option<&[f64]> {
        self.linear.as_deref()
    }

    pub fn constant(&self) -> f64 {
        self.constant
    }

    pub fn roundoff(&self) -> f64 {
        self.roundoff
    }

    /// `A_f τ_{K_f}`, the factor's rows applied to its blocks of `tangent`.
    fn applied(&self, f: usize, tangent: &[f64]) -> Vec<f64> {
        let x = self.structure.gather(f, tangent);
        self.factors[f].a.apply(&x)
    }

    /// `c + Σ ½|A_f τ - b_f|² + ηᵀτ` at `tangent`, of the structure's dimension.
    pub fn value(&self, tangent: &[f64]) -> f64 {
        let squares: f64 = (0..self.factors.len())
            .map(|f| {
                let ax = self.applied(f, tangent);
                let b = &self.factors[f].b;
                0.5 * ax.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f64>()
            })
            .sum();
        let linear: f64 = self
            .linear
            .as_ref()
            .map_or(0.0, |eta| eta.iter().zip(tangent).map(|(e, t)| e * t).sum());
        self.constant + squares + linear
    }

    /// `½|J τ|² + λ τᵀ D τ`: the decrease of the undamped problem that the damped minimizer
    /// `tangent` promises.
    pub fn promised(&self, tangent: &[f64], damping: &Damping) -> f64 {
        let squares: f64 = (0..self.factors.len())
            .map(|f| 0.5 * self.applied(f, tangent).iter().map(|v| v * v).sum::<f64>())
            .sum();
        let penalty: f64 = damping
            .diagonal
            .iter()
            .zip(tangent)
            .map(|(d, t)| d * t * t)
            .sum();
        squares + damping.lambda * penalty
    }
}
