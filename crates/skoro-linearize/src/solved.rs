//! A batch's eliminations, mapped back to the linearizer's quantities.

use skoro_core::{Layout, NodeId};
use skoro_elimination::{Damping, Elimination, Schedule};
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
