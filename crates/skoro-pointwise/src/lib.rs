//! # skoro-pointwise
//!
//! Differentiable elementwise operations on quantities ([`PointwiseOps`]):
//! `log` and `abs`. Each is a `DiffKernel` generic over the backend.

mod abs;
mod log;

pub use abs::*;
pub use log::*;

use skoro_backend::Backend;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{map::register_map1, q::Q};
use skoro_riemann::{
    euclidean::{Euclidean, EuclideanD},
    real::Real,
};

/// Marker for quantities that support elementwise operations
/// ([`PointwiseOps`]). The operations are type-preserving, so each is
/// implemented for the real line and the Euclidean manifolds.
pub trait PointwiseQuantity<B: Backend>: ContinuousQuantity<B> {}

impl<B: Backend> PointwiseQuantity<B> for Real {}
impl<B: Backend> PointwiseQuantity<B> for EuclideanD {}
impl<B: Backend, const D: usize> PointwiseQuantity<B> for Euclidean<D> {}

pub trait PointwiseOps<B: Backend, M: ContinuousQuantity<B>> {
    fn log(&self) -> Q<B, M>;
    fn abs(&self) -> Q<B, M>;
}

impl<B: Backend, M: PointwiseQuantity<B>> PointwiseOps<B, M> for Q<B, M> {
    #[inline(always)]
    fn log(&self) -> Q<B, M> {
        register_map1(self.get_quantity_type(), PointwiseLog, self)
    }

    #[inline(always)]
    fn abs(&self) -> Q<B, M> {
        register_map1(self.get_quantity_type(), PointwiseAbs, self)
    }
}
