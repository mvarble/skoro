use skoro_backend::Backend;
use skoro_core::error::SkoroResult;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_linalg::matrix::Matmul;
use skoro_model::{map::register_map2, q::Q};

use crate::quantity::SO3;

#[inline(always)]
pub fn compose<B>(r0: &Q<B, SO3>, r1: &Q<B, SO3>) -> SkoroResult<Q<B, SO3>>
where
    B: Backend,
    SO3: ContinuousQuantity<B>,
{
    r0.compose(r1)
}

pub trait SO3ComposeOps<B: Backend>
where
    SO3: ContinuousQuantity<B>,
{
    fn compose(&self, other: &Q<B, SO3>) -> SkoroResult<Q<B, SO3>>;
}

impl<B: Backend> SO3ComposeOps<B> for Q<B, SO3>
where
    SO3: ContinuousQuantity<B>,
{
    fn compose(&self, other: &Q<B, SO3>) -> SkoroResult<Q<B, SO3>> {
        register_map2(SO3, Matmul(3, 3, 3), (self, other))
    }
}
