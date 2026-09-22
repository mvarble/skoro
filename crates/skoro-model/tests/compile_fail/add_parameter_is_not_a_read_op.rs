//! Adding a parameter is a *builder* capability.
//!
//! A parameter grows the free set, which is what defines an estimator's unknowns. So an
//! estimator bounded on `ModelReaderOps` must not be able to add one. Constants, which change
//! nothing but the node count, stay on the read tier.

use skoro_backend::Flex;
use skoro_model::{ModelReaderOps, q::Q};
use skoro_riemann::real::Real;

type B = Flex;

fn an_estimator<M: ModelReaderOps<B>>(model: &M) -> Q<B, Real> {
    let _constant = model.add_constant(Real, &1.0f64);
    model.add_parameter(Real)
}

fn main() {
    let _ = an_estimator::<skoro_model::model::ModelClient<B>>;
}
