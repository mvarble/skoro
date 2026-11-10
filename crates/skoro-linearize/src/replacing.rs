//! A retirement's replacement: the linearized integral of its retired terms over the quantities
//! it integrates, as an improper Gaussian over the blanket quantities it keeps.

use std::collections::HashSet;

use skoro_core::scope::constant;
use skoro_core::{
    NodeId, Observations, QDyn, Query, Real, Replacement, Retirement, Rows, Space, Values,
};
use skoro_elimination::{Reduced, Schedule};
use skoro_normal::Stacked;
use skoro_tensor::{HostData, HostValue, Target};

use crate::assemble::host;
use crate::{Assembly, Error, Linearizer};

/// A retirement's replacement, ready to be linearized at any batch of points: the query of the
/// retired terms alone, free in the quantities to integrate and to keep.
#[derive(Debug)]
pub struct Replacing {
    pub(crate) linearizer: Linearizer,
    /// `E_F`.
    pub(crate) integrated: Vec<NodeId>,
    /// The elimination of `E_F`'s blocks from the retired terms' factors.
    pub(crate) schedule: Schedule,
    /// The kept quantities' handles and spaces, in the order of the kept blocks.
    pub(crate) kept: Vec<(QDyn, Space)>,
}

impl Replacing {
    /// `keep` the continuous blanket quantities the replacement stays a function of; `fixed` the
    /// values of the other blanket quantities the retired terms involve. The values of `E_O` are
    /// the retirement's own, which a value in `fixed` for the same quantity does not replace.
    /// Refuses a kept quantity that is not a continuous blanket quantity, and a query of the
    /// retired terms whose free quantities are not exactly `E_F` and `keep` (`Blanket`, naming
    /// the first quantity that is not), and anything the linearizer refuses.
    pub fn new(
        retirement: &Retirement<'_>,
        keep: &[QDyn],
        fixed: &Observations,
        target: &Target,
    ) -> Result<Replacing, Error> {
        let plan = retirement.plan();
        let snapshot = retirement.snapshot();
        for q in keep {
            let continuous = plan
                .blanket
                .iter()
                .any(|(b, space, _)| b.id() == q.id() && space.is_continuous());
            if !continuous {
                return Err(Error::Blanket { quantity: q.id() });
            }
        }
        let observations = plan.observed.clone().union(fixed);
        let query =
            Query::log_density_of(plan.terms.iter().map(|t| t.log_density)).given(observations);
        let linearizer = Linearizer::new(snapshot, &query, target, Assembly::Residual)?;
        let integrated: Vec<NodeId> = plan.integrated.iter().map(QDyn::id).collect();
        let expected: HashSet<NodeId> = integrated
            .iter()
            .copied()
            .chain(keep.iter().map(QDyn::id))
            .collect();
        let free: HashSet<NodeId> = linearizer.layout().quantities().iter().copied().collect();
        let mismatch = linearizer
            .layout()
            .quantities()
            .iter()
            .copied()
            .find(|q| !expected.contains(q))
            .or_else(|| {
                integrated
                    .iter()
                    .chain(&keep.iter().map(QDyn::id).collect::<Vec<_>>())
                    .copied()
                    .find(|q| !free.contains(q))
            });
        if let Some(quantity) = mismatch {
            return Err(Error::Blanket { quantity });
        }
        let blocks = linearizer.blocks_of(&integrated)?;
        let schedule = Schedule::eliminating(linearizer.structure(), &blocks)?;
        let kept = schedule
            .kept()
            .iter()
            .map(|&k| {
                let id = linearizer.layout().quantities()[k];
                plan.blanket
                    .iter()
                    .find(|(q, ..)| q.id() == id)
                    .map(|(q, space, _)| (*q, space.clone()))
                    .ok_or(Error::Blanket { quantity: id })
            })
            .collect::<Result<_, Error>>()?;
        Ok(Replacing {
            linearizer,
            integrated,
            schedule,
            kept,
        })
    }

    /// The linearizer of the retired terms' query.
    pub fn linearizer(&self) -> &Linearizer {
        &self.linearizer
    }

    /// `E_F`: the retired quantities the replacement integrates out.
    pub fn integrated(&self) -> &[NodeId] {
        &self.integrated
    }

    /// The elimination of `E_F`'s blocks from the retired terms' factors.
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }
}

impl Replacing {
    /// The replacement linearized at `at`, which holds `p̂_E` and the anchors `p̂_B`: in the chart
    /// of `E_F` alone, `E_F` eliminated by integration, and what is left reduced to
    /// `min(m', D)` rows with no decision of rank. A retired quantity its terms do not determine
    /// is `RetiredUndetermined`. On a curved quantity of the blanket its precision errs by a
    /// relative amount of order `σ²`, the noise's variance in its chart, and more over a sequence
    /// of retirements that measure nothing.
    pub fn at(&self, at: &Values) -> Result<LinearizedReplacement, Error> {
        let linearized = self.linearizer.at_in_chart(at, &self.integrated)?;
        let layout = self.linearizer.layout();
        let mut reduced = Vec::with_capacity(linearized.problems().len());
        for problem in linearized.problems() {
            let elimination = problem
                .eliminate(&self.schedule, None)
                .map_err(|e| match e {
                    skoro_elimination::Error::Undetermined { block } => {
                        Error::RetiredUndetermined {
                            quantity: layout.quantities()[block],
                        }
                    }
                    other => Error::Elimination(other),
                })?;
            reduced.push(elimination.reduced());
        }
        let rows = linearized.problems().len();
        let mut blocks = Vec::new();
        for (q, space) in &self.kept {
            let data = host(at, *q)?;
            let anchor = HostValue::new(
                (rows > 1).then_some(rows),
                space.storage(),
                HostData::Continuous(data),
            )
            .ok_or(Error::Core(skoro_core::Error::ValueShape {
                space: space.clone(),
            }))?;
            blocks.push((*q, space.clone(), anchor));
        }
        Ok(LinearizedReplacement { blocks, reduced })
    }
}

/// The numbers of the improper Gaussian `-½|A δ - b|² - ηᵀδ + c` per row, with the anchors of
/// the kept quantities at the linearization point.
#[derive(Clone, Debug)]
pub struct LinearizedReplacement {
    /// Each kept quantity, its space and its anchors, in the order of the stacked tangent.
    blocks: Vec<(QDyn, Space, HostValue)>,
    reduced: Vec<Reduced>,
}

impl LinearizedReplacement {
    /// The number of rows of `A`, `min(m', D)`: the same for every row of the batch.
    pub fn rows(&self) -> usize {
        self.reduced.first().map_or(0, |r| r.a.rows())
    }

    /// Each row's reduced factor, over the kept quantities in the order of the stacked tangent.
    pub fn reduced(&self) -> &[Reduced] {
        &self.reduced
    }

    /// The kept quantities, in the order of the stacked tangent.
    pub fn quantities(&self) -> Vec<NodeId> {
        self.blocks.iter().map(|(q, ..)| q.id()).collect()
    }

    /// The improper factor as the core's `Replacement`, its roles bound: to be called inside
    /// `Retirement::replace`'s body. The blocks are stacked in the order of `A`'s columns, which
    /// both come from the layout, and `A` is stored column by column, as a `MatrixDyn` is.
    #[track_caller]
    pub fn build(&self) -> Replacement {
        let constants: Vec<f64> = self.reduced.iter().map(|r| r.constant).collect();
        if self.blocks.is_empty() {
            let c = match constants.as_slice() {
                [c] => constant(Real, *c),
                _ => constant(Real, Rows(constants.clone())),
            };
            return Replacement {
                log_density: c + 0.0,
                proposals: Vec::new(),
                bindings: Vec::new(),
            };
        }
        let stacked = self
            .blocks
            .iter()
            .fold(Stacked::new(), |stacked, (q, space, anchor)| {
                stacked.block_in(*q, space, anchor.clone())
            });
        let rows = self.rows();
        let a: Vec<Vec<f64>> = self.reduced.iter().map(|r| r.a.by_columns()).collect();
        let b: Vec<Vec<f64>> = self.reduced.iter().map(|r| r.b.clone()).collect();
        let improper = if self.reduced.len() == 1 {
            stacked.improper(rows, a[0].as_slice(), b[0].as_slice(), constants[0])
        } else {
            let a: Vec<&[f64]> = a.iter().map(Vec::as_slice).collect();
            let b: Vec<&[f64]> = b.iter().map(Vec::as_slice).collect();
            stacked.improper(rows, Rows(a), Rows(b), Rows(constants.clone()))
        };
        let linear: Option<Vec<Vec<f64>>> = self.reduced.iter().map(|r| r.linear.clone()).collect();
        let improper = match linear {
            Some(eta) if self.reduced.len() == 1 => improper.linear(eta[0].as_slice()),
            Some(eta) => {
                improper.linear(Rows(eta.iter().map(Vec::as_slice).collect::<Vec<&[f64]>>()))
            }
            None => improper,
        };
        improper.replacement()
    }
}
