use skoro_backend::Backend;
use skoro_core::error::{SkoroError, SkoroResult};
use skoro_graph::quantity::ContinuousQuantity;
use skoro_linalg::matrix::{Matmul, TransposeOps};
use skoro_model::{map::register_map2, q::Q};
use skoro_riemann::{
    euclidean::{Euclidean, EuclideanD},
    matrix::{EuclideanMatrix, EuclideanMatrixD},
};

use crate::quantity::SO3;

/// Rotate a vector (or frame of vectors) by an `SO(3)` rotation.
///
/// Returns [`SkoroError::InvalidRotationFrame`] if the operand's leading
/// dimension is not 3.
#[inline(always)]
pub fn rotate<B, T>(r: &Q<B, SO3>, v: &Q<B, T>) -> SkoroResult<Q<B, T>>
where
    B: Backend,
    SO3: ContinuousQuantity<B>,
    T: RotationVector<B>,
{
    r.rotate(v)
}

/// Apply the inverse rotation to a vector (or frame of vectors).
///
/// Returns [`SkoroError::InvalidRotationFrame`] if the operand's leading
/// dimension is not 3.
#[inline(always)]
pub fn unrotate<B, T>(r: &Q<B, SO3>, v: &Q<B, T>) -> SkoroResult<Q<B, T>>
where
    B: Backend,
    SO3: ContinuousQuantity<B>,
    T: RotationVector<B>,
{
    r.unrotate(v)
}

pub trait SO3RotateOps<B: Backend> {
    fn rotate<T>(&self, other: &Q<B, T>) -> SkoroResult<Q<B, T>>
    where
        T: RotationVector<B>;

    fn unrotate<T>(&self, other: &Q<B, T>) -> SkoroResult<Q<B, T>>
    where
        T: RotationVector<B>;
}

impl<B> SO3RotateOps<B> for Q<B, SO3>
where
    B: Backend,
    SO3: ContinuousQuantity<B>,
{
    fn rotate<T>(&self, other: &Q<B, T>) -> SkoroResult<Q<B, T>>
    where
        T: RotationVector<B>,
    {
        let otype = other.get_quantity_type();
        let matmul = otype.build_matmul()?;
        Ok(register_map2(otype, matmul, (self, other)))
    }

    fn unrotate<T>(&self, other: &Q<B, T>) -> SkoroResult<Q<B, T>>
    where
        T: RotationVector<B>,
    {
        self.transpose().rotate(other)
    }
}

pub trait RotationVector<B: Backend>: ContinuousQuantity<B> {
    fn build_matmul(&self) -> SkoroResult<Matmul>;
}

impl<B: Backend> RotationVector<B> for EuclideanD {
    #[inline(always)]
    fn build_matmul(&self) -> SkoroResult<Matmul> {
        Ok(Matmul(3, 3, 1))
    }
}

impl<B: Backend> RotationVector<B> for Euclidean<3> {
    #[inline(always)]
    fn build_matmul(&self) -> SkoroResult<Matmul> {
        Ok(Matmul(3, 3, 1))
    }
}

impl<B: Backend> RotationVector<B> for EuclideanMatrixD {
    fn build_matmul(&self) -> SkoroResult<Matmul> {
        if self.0 != 3 {
            return Err(SkoroError::InvalidRotationFrame { rows: self.0 });
        }
        Ok(Matmul(3, 3, self.1))
    }
}

impl<B: Backend, const N: usize> RotationVector<B> for EuclideanMatrix<3, N> {
    #[inline(always)]
    fn build_matmul(&self) -> SkoroResult<Matmul> {
        Ok(Matmul(3, 3, N))
    }
}
