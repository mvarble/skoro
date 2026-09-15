use skoro_backend::{Backend, CubeBackend, CubeFloat, CubeInt, CubeRuntime, Flex};
use skoro_core::error::SkoroResult;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{map::register_map1, q::Q};
use skoro_riemann::euclidean::{Euclidean, Euclidean3, EuclideanD};

use super::SO3Log;
use crate::quantity::SO3;

#[inline(always)]
pub fn log<B, T>(r: &Q<B, SO3>) -> SkoroResult<Q<B, T>>
where
    B: SO3LogBackend<T>,
    T: ContinuousQuantity<B>,
    SO3: ContinuousQuantity<B>,
{
    <B as SO3LogBackend<T>>::log(r)
}

pub trait SO3LogBackend<T>: Backend
where
    SO3: ContinuousQuantity<Self>,
    T: ContinuousQuantity<Self>,
{
    fn log(r: &Q<Self, SO3>) -> SkoroResult<Q<Self, T>>;
}

pub trait SO3LogOps<B: Backend, T: ContinuousQuantity<B>>
where
    SO3: ContinuousQuantity<B>,
{
    fn log(&self) -> SkoroResult<Q<B, T>>;
}

impl<B: SO3LogBackend<Euclidean3>> SO3LogOps<B, Euclidean3> for Q<B, SO3>
where
    SO3: ContinuousQuantity<B>,
{
    #[inline(always)]
    fn log(&self) -> SkoroResult<Q<B, Euclidean3>> {
        <B as SO3LogBackend<Euclidean3>>::log(self)
    }
}

impl<B: SO3LogBackend<EuclideanD>> SO3LogOps<B, EuclideanD> for Q<B, SO3>
where
    SO3: ContinuousQuantity<B>,
{
    #[inline(always)]
    fn log(&self) -> SkoroResult<Q<B, EuclideanD>> {
        <B as SO3LogBackend<EuclideanD>>::log(self)
    }
}

impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> SO3LogBackend<Euclidean3> for CubeBackend<R, F, I> {
    fn log(r: &Q<Self, SO3>) -> SkoroResult<Q<Self, Euclidean3>> {
        register_map1(Euclidean::<3>, SO3Log, r)
    }
}

impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> SO3LogBackend<EuclideanD> for CubeBackend<R, F, I> {
    fn log(r: &Q<Self, SO3>) -> SkoroResult<Q<Self, EuclideanD>> {
        register_map1(EuclideanD(3), SO3Log, r)
    }
}

impl SO3LogBackend<Euclidean3> for Flex {
    fn log(r: &Q<Self, SO3>) -> SkoroResult<Q<Self, Euclidean3>> {
        register_map1(Euclidean::<3>, SO3Log, r)
    }
}

impl SO3LogBackend<EuclideanD> for Flex {
    fn log(r: &Q<Self, SO3>) -> SkoroResult<Q<Self, EuclideanD>> {
        register_map1(EuclideanD(3), SO3Log, r)
    }
}
