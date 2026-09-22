//! Demoting is a *builder* capability.
//!
//! `demote_to_parameter` cannot break a reference, which once put it on the read tier, but it
//! deletes a term of the log-density. Done without the compensating factor it is half a
//! marginalization, so an estimator bounded on `ModelReaderOps` must not reach it.

use skoro_backend::Flex;
use skoro_core::error::SkoroResult;
use skoro_model::{ModelReaderOps, q::Q};
use skoro_riemann::real::Real;

type B = Flex;

fn an_estimator<M: ModelReaderOps<B>>(model: &M, x: &Q<B, Real>) -> SkoroResult<()> {
    model.demote_to_parameter(x)
}

fn main() {
    let _ = an_estimator::<skoro_model::model::ModelClient<B>>;
}
