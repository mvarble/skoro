use skoro_backend::Backend;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{map::register_map1, q::Q};
use skoro_riemann::{
    euclidean::{Euclidean, EuclideanD},
    real::Real,
};

pub mod cross_product;
pub mod norm;
pub mod norm_squared;
pub mod sum_components;

pub use cross_product::{CrossProduct, CrossProductBackend, CrossProductOps};
pub use norm::Norm;
pub use norm_squared::NormSquared;
pub use sum_components::SumComponents;

/// Marker for Euclidean vector quantities that support linear-algebra
/// reductions ([`LinalgVectorOps`]). Implemented for the runtime- and
/// static-dimension Euclidean manifolds.
pub trait LinalgVector<B: Backend>: ContinuousQuantity<B> {}

impl<B: Backend> LinalgVector<B> for EuclideanD {}
impl<B: Backend, const D: usize> LinalgVector<B> for Euclidean<D> {}

pub trait LinalgVectorOps<B: Backend> {
    fn norm_squared(&self) -> Q<B, Real>;
    fn norm(&self) -> Q<B, Real>;
    fn sum(&self) -> Q<B, Real>;
}

impl<B: Backend, M: LinalgVector<B>> LinalgVectorOps<B> for Q<B, M> {
    #[inline(always)]
    fn norm_squared(&self) -> Q<B, Real> {
        register_map1(Real, NormSquared, self)
    }
    #[inline(always)]
    fn norm(&self) -> Q<B, Real> {
        register_map1(Real, Norm, self)
    }
    #[inline(always)]
    fn sum(&self) -> Q<B, Real> {
        register_map1(Real, SumComponents, self)
    }
}

#[cfg(test)]
mod tests;
