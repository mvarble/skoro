//! A retirement's replacement: the linearized integral of its retired terms over the quantities
//! it integrates, as an improper Gaussian over the blanket quantities it keeps.

use std::collections::HashSet;

use skoro_core::{NodeId, Observations, QDyn, Query, Retirement};
use skoro_elimination::Schedule;
use skoro_tensor::Target;

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
        Ok(Replacing {
            linearizer,
            integrated,
            schedule,
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
