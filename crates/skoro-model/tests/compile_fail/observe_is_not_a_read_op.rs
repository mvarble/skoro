//! Observing is a *builder* capability.
//!
//! Observing a quantity moves it out of the free set and changes the posterior, so an estimator
//! bounded on `ModelReaderOps` must not be able to fix one of the quantities it is solving for.

use skoro_backend::Flex;
use skoro_core::error::SkoroResult;
use skoro_model::{ModelReaderOps, q::Q};
use skoro_riemann::real::Real;

type B = Flex;

fn an_estimator<M: ModelReaderOps<B>>(model: &M, x: &Q<B, Real>) -> SkoroResult<()> {
    model.observe(x, &1.0f64)
}

fn main() {
    let _ = an_estimator::<skoro_model::model::ModelClient<B>>;
}
