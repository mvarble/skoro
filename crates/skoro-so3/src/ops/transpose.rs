use skoro_backend::Backend;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_linalg::matrix::{Transpose, TransposeType};

use crate::quantity::SO3;

impl<B: Backend> TransposeType<B> for SO3
where
    SO3: ContinuousQuantity<B>,
{
    type OutputType = SO3;

    fn build_transpose(&self) -> (Self::OutputType, Transpose) {
        (SO3, Transpose(3, 3))
    }
}
