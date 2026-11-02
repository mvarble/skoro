//! How the normal refuses a space: an operation that is never added, whose `output` gives the
//! error it was made with.

use std::sync::Arc;

use skoro_core::scope::apply;
use skoro_core::{Error, Lowered, Lowering, Operation, Param, QDyn, Space};

/// An operation whose `output` gives its error, so that `scope::apply` records it at the caller's
/// line and gives a failed handle.
#[derive(Debug)]
pub(crate) struct Refused(pub(crate) Error);

impl Operation for Refused {
    fn name(&self) -> &'static str {
        "skoro.normal"
    }

    fn params(&self) -> Vec<Param> {
        Vec::new()
    }

    fn output(&self, _: &[Space]) -> Result<Space, Error> {
        Err(self.0.clone())
    }

    fn lower(&self, _: &[Space], _: &mut Lowering<'_>) -> Result<Lowered, Error> {
        Err(self.0.clone())
    }
}

/// A failed handle recording `error` at the caller's line, in the open scope.
#[track_caller]
pub(crate) fn refuse(error: Error, inputs: &[QDyn]) -> QDyn {
    apply(Arc::new(Refused(error)), inputs)
}
