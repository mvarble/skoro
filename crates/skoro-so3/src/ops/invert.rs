use skoro_backend::Backend;
use skoro_core::error::SkoroResult;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_linalg::matrix::TransposeOps;
use skoro_model::q::Q;

use crate::quantity::SO3;

pub trait SO3InvertOps<B: Backend> {
    fn invert(&self) -> SkoroResult<Self>
    where
        Self: Sized;
}

impl<B: Backend> SO3InvertOps<B> for Q<B, SO3>
where
    SO3: ContinuousQuantity<B>,
{
    fn invert(&self) -> SkoroResult<Self> {
        self.transpose()
    }
}
