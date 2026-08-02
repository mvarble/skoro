use skoro_backend::{
    Backend, CubeBackend, CubeFloat, CubeInt, CubeRuntime, NdArrayBackend, NdArrayFloat,
    NdArrayInt, NdArrayTensor, SharedArray,
};
use skoro_model::{map::register_map2, q::Q};
use skoro_riemann::euclidean::{Euclidean, Euclidean3};

use super::CrossProduct;

pub trait CrossProductBackend: Backend {
    fn cross_product(lhs: &Q<Self, Euclidean3>, rhs: &Q<Self, Euclidean3>) -> Q<Self, Euclidean3>;
}

#[inline(always)]
pub fn cross_product<B: CrossProductBackend>(
    lhs: &Q<B, Euclidean3>,
    rhs: &Q<B, Euclidean3>,
) -> Q<B, Euclidean3> {
    B::cross_product(lhs, rhs)
}

pub trait CrossProductOps<B: Backend> {
    fn cross_product(&self, other: &Q<B, Euclidean3>) -> Q<B, Euclidean3>;
}

impl<B: CrossProductBackend> CrossProductOps<B> for Q<B, Euclidean3> {
    #[inline(always)]
    fn cross_product(&self, other: &Q<B, Euclidean3>) -> Q<B, Euclidean3> {
        B::cross_product(self, other)
    }
}

impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> CrossProductBackend for CubeBackend<R, F, I> {
    fn cross_product(lhs: &Q<Self, Euclidean3>, rhs: &Q<Self, Euclidean3>) -> Q<Self, Euclidean3> {
        register_map2(Euclidean::<3>, CrossProduct, (lhs, rhs))
    }
}

impl<F, I> CrossProductBackend for NdArrayBackend<F, I>
where
    F: NdArrayFloat + num_traits::Float,
    I: NdArrayInt,
    NdArrayTensor: From<SharedArray<F>> + From<SharedArray<I>>,
{
    fn cross_product(lhs: &Q<Self, Euclidean3>, rhs: &Q<Self, Euclidean3>) -> Q<Self, Euclidean3> {
        register_map2(Euclidean::<3>, CrossProduct, (lhs, rhs))
    }
}
