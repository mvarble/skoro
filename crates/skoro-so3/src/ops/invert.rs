use skoro_backend::Backend;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_linalg::matrix::TransposeOps;
use skoro_model::q::Q;

use crate::quantity::SO3;

pub trait SO3InvertOps<B: Backend> {
    fn invert(&self) -> Self;
}

impl<B: Backend> SO3InvertOps<B> for Q<B, SO3>
where
    SO3: ContinuousQuantity<B>,
{
    fn invert(&self) -> Self {
        self.transpose()
    }
}
