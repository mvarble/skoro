//! A `Q` is not a way around the read tier.
//!
//! A `Q` carries a full client, so any method on it is reachable by whoever holds the quantity,
//! whatever bound the model came under. An estimator is handed the quantities it solves for;
//! if `Q` could observe, the estimator could fix one of them without `ModelBuilderOps`.

use skoro_backend::Flex;
use skoro_core::error::SkoroResult;
use skoro_model::{ModelReaderOps, q::Q};
use skoro_riemann::real::Real;

type B = Flex;

fn an_estimator<M: ModelReaderOps<B>>(_model: &M, x: &Q<B, Real>) -> SkoroResult<()> {
    x.observe(&1.0f64)
}

fn main() {
    let _ = an_estimator::<skoro_model::model::ModelClient<B>>;
}
