//! A batch's eliminations, mapped back to the linearizer's quantities.

use skoro_core::{Key, Layout, NodeId, Values};
use skoro_elimination::{Damping, Elimination, Marginal, Schedule};
use skoro_tensor::{Points, Target};

use crate::{Assembly, Error, Linearized};

/// A batch's eliminations, one per row, mapped back to the linearizer's quantities.
#[derive(Clone, Debug)]
pub struct Solved {
    pub(crate) eliminations: Vec<Elimination>,
    pub(crate) assembly: Assembly,
    pub(crate) layout: Layout,
    pub(crate) curved: Vec<NodeId>,
}

/// The elimination's refusal of a block, as a refusal of its quantity.
pub(crate) fn undetermined(error: skoro_elimination::Error, layout: &Layout) -> Error {
    match error {
        skoro_elimination::Error::Undetermined { block } => Error::Undetermined {
            quantity: layout.quantities()[block],
        },
        other => Error::Elimination(other),
    }
}

/// A marginal of the linearization, over named quantities.
#[derive(Clone, Debug)]
pub struct QuantityMarginal {
    /// The kept quantities, in the order of the marginal's blocks.
    pub quantities: Vec<NodeId>,
    /// Those of them of a curved type, whose covariance is reported with an error of order `σ²`.
    pub curved: Vec<NodeId>,
    pub marginal: Marginal,
}

impl Linearized {
    /// Eliminates every row along `schedule`, which must be of the linearizer's structure, with
    /// the damping rows appended if given; refuses a free quantity the rows do not determine
    /// (`Undetermined`).
    pub fn eliminate(
        &self,
        schedule: &Schedule,
        damping: Option<&Damping>,
    ) -> Result<Solved, Error> {
        let eliminations = self
            .problems
            .iter()
            .map(|problem| {
                problem
                    .eliminate(schedule, damping)
                    .map_err(|e| undetermined(e, &self.layout))
            })
            .collect::<Result<_, Error>>()?;
        Ok(Solved {
            eliminations,
            assembly: self.assembly,
            layout: self.layout.clone(),
            curved: self.curved.clone(),
        })
    }
}

impl Solved {
    pub fn eliminations(&self) -> &[Elimination] {
        &self.eliminations
    }

    pub fn assembly(&self) -> Assembly {
        self.assembly
    }

    /// The free quantities of a curved type.
    pub fn curved(&self) -> &[NodeId] {
        &self.curved
    }

    /// Each row's minimizer as a `Points` of `[rows, D]` on `target`, for `Layout::retract`.
    pub fn minimizer(&self, target: &Target) -> Result<Points, Error> {
        let mut data = Vec::new();
        for elimination in &self.eliminations {
            data.extend(elimination.minimizer()?);
        }
        let d = self.layout.dimension();
        Ok(Points::from_host(&data, self.eliminations.len(), d, target))
    }
}

impl Solved {
    fn integral(&self) -> Result<(), Error> {
        match self.assembly {
            Assembly::Fisher => Err(Error::NotAnIntegral),
            Assembly::Residual => Ok(()),
        }
    }

    /// Each row's marginal, of a `schedule_keeping` elimination, with its quantities. The
    /// covariance of a curved quantity errs by a relative amount of order `σ²`, the noise's
    /// variance in its chart, against the second-order approximation and the posterior alike.
    pub fn marginal(&self) -> Result<Vec<QuantityMarginal>, Error> {
        self.eliminations
            .iter()
            .map(|elimination| {
                let marginal = elimination
                    .marginal()
                    .map_err(|e| undetermined(e, &self.layout))?;
                let quantities: Vec<NodeId> = marginal
                    .blocks
                    .iter()
                    .map(|&k| self.layout.quantities()[k])
                    .collect();
                let curved = quantities
                    .iter()
                    .copied()
                    .filter(|q| self.curved.contains(q))
                    .collect();
                Ok(QuantityMarginal {
                    quantities,
                    curved,
                    marginal,
                })
            })
            .collect()
    }

    /// `log ∫ e^{-m}` per row: with a chart linearization at a mode, the Laplace evidence
    /// (`NotAnIntegral` of a Fisher linearization).
    pub fn log_integral(&self) -> Result<Vec<f64>, Error> {
        self.integral()?;
        self.eliminations
            .iter()
            .map(|e| Ok(e.log_integral()?))
            .collect()
    }

    /// A draw per row, `τ* + R⁻¹ z` with `z` standard normal from `key`, one key per row by
    /// `Key::split`, retracted onto the quantities at `at` (`NotAnIntegral` of a Fisher
    /// linearization). On a curved quantity its spread errs as the covariance does.
    pub fn draw(&self, at: &Values, key: &Key, target: &Target) -> Result<Values, Error> {
        self.integral()?;
        let d = self.layout.dimension();
        let keys = key.split(self.eliminations.len());
        let mut data = Vec::with_capacity(self.eliminations.len() * d);
        for (elimination, key) in self.eliminations.iter().zip(&keys) {
            let z: Vec<f64> = key
                .uniforms(d)
                .into_iter()
                .map(skoro_normal::quantile)
                .collect();
            let spread = elimination.solve(&z)?;
            let centre = elimination.minimizer()?;
            data.extend(centre.iter().zip(&spread).map(|(c, s)| c + s));
        }
        let tangent = Points::from_host(&data, self.eliminations.len(), d, target);
        Ok(self.layout.retract(at, &tangent, target)?)
    }
}
