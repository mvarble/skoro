mod diag;
mod invert;
mod manifold;
mod utils;
mod vector_product;

pub use diag::*;
pub use invert::*;
pub use manifold::*;
pub use utils::*;
pub use vector_product::*;

use skoro_backend::Backend;
use skoro_core::error::SkoroResult;
use skoro_model::q::Q;
use skoro_riemann::euclidean::EuclideanD;

pub trait TrilOps<B: Backend> {
    fn vector_product<T: TrilVector<B>>(&self, vector: &Q<B, T>) -> SkoroResult<Q<B, T>>;
    fn invert(&self) -> SkoroResult<Q<B, Tril>>;
    fn diag(&self) -> SkoroResult<Q<B, EuclideanD>>;
}

impl<B: Backend> TrilOps<B> for Q<B, Tril>
where
    B::FloatElem: std::ops::Div<B::FloatElem, Output = B::FloatElem>
        + std::ops::Mul<B::FloatElem, Output = B::FloatElem>
        + std::ops::Add<B::FloatElem, Output = B::FloatElem>
        + std::ops::Sub<B::FloatElem, Output = B::FloatElem>
        + std::ops::Neg<Output = B::FloatElem>
        + std::ops::AddAssign<B::FloatElem>,
{
    #[inline(always)]
    fn vector_product<T: TrilVector<B>>(&self, vector: &Q<B, T>) -> SkoroResult<Q<B, T>> {
        vector_product(self, vector)
    }
    #[inline(always)]
    fn invert(&self) -> SkoroResult<Q<B, Tril>> {
        invert(self)
    }
    #[inline(always)]
    fn diag(&self) -> SkoroResult<Q<B, EuclideanD>> {
        diag(self)
    }
}
