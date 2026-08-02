use skoro_backend::{
    Backend, CubeBackend, CubeFloat, CubeInt, CubeRuntime, NdArrayBackend, NdArrayFloat,
    NdArrayInt, NdArrayTensor, SharedArray,
};
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{map::register_map1, q::Q};
use skoro_riemann::euclidean::{Euclidean3, EuclideanD};

use super::SO3Exp;
use crate::quantity::SO3;

#[inline(always)]
pub fn exp<B, T>(omega: &Q<B, T>) -> Q<B, SO3>
where
    B: SO3ExpBackend<T>,
    T: ContinuousQuantity<B>,
    SO3: ContinuousQuantity<B>,
{
    <B as SO3ExpBackend<T>>::exp(omega)
}

pub trait SO3ExpBackend<T>: Backend
where
    SO3: ContinuousQuantity<Self>,
    T: ContinuousQuantity<Self>,
{
    fn exp(omega: &Q<Self, T>) -> Q<Self, SO3>;
}

pub trait SO3ExpOps<B: Backend>
where
    SO3: ContinuousQuantity<B>,
{
    fn exp(&self) -> Q<B, SO3>;
}

impl<B: SO3ExpBackend<Euclidean3>> SO3ExpOps<B> for Q<B, Euclidean3>
where
    SO3: ContinuousQuantity<B>,
{
    #[inline(always)]
    fn exp(&self) -> Q<B, SO3> {
        <B as SO3ExpBackend<Euclidean3>>::exp(self)
    }
}

impl<B: SO3ExpBackend<EuclideanD>> SO3ExpOps<B> for Q<B, EuclideanD>
where
    SO3: ContinuousQuantity<B>,
{
    #[inline(always)]
    fn exp(&self) -> Q<B, SO3> {
        <B as SO3ExpBackend<EuclideanD>>::exp(self)
    }
}

impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> SO3ExpBackend<Euclidean3> for CubeBackend<R, F, I> {
    fn exp(omega: &Q<Self, Euclidean3>) -> Q<Self, SO3> {
        register_map1(SO3, SO3Exp, omega)
    }
}

impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> SO3ExpBackend<EuclideanD> for CubeBackend<R, F, I> {
    fn exp(omega: &Q<Self, EuclideanD>) -> Q<Self, SO3> {
        register_map1(SO3, SO3Exp, omega)
    }
}

impl<F, I> SO3ExpBackend<Euclidean3> for NdArrayBackend<F, I>
where
    F: NdArrayFloat + num_traits::Float,
    I: NdArrayInt,
    NdArrayTensor: From<SharedArray<F>> + From<SharedArray<I>>,
{
    fn exp(omega: &Q<Self, Euclidean3>) -> Q<Self, SO3> {
        register_map1(SO3, SO3Exp, omega)
    }
}

impl<F, I> SO3ExpBackend<EuclideanD> for NdArrayBackend<F, I>
where
    F: NdArrayFloat + num_traits::Float,
    I: NdArrayInt,
    NdArrayTensor: From<SharedArray<F>> + From<SharedArray<I>>,
{
    fn exp(omega: &Q<Self, EuclideanD>) -> Q<Self, SO3> {
        register_map1(SO3, SO3Exp, omega)
    }
}
