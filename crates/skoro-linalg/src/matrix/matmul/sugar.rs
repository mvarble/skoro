use skoro_backend::Backend;
use skoro_core::error::{SkoroError, SkoroResult};
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{map::register_map2, q::Q};
use skoro_riemann::{
    euclidean::{Euclidean, EuclideanD},
    matrix::{EuclideanMatrix, EuclideanMatrixD},
};

use super::Matmul;

/// Build a matrix-multiplication quantity `lhs @ rhs`.
///
/// Returns [`SkoroError::IncompatibleMatmul`] if the operands' inner
/// dimensions do not agree.
#[inline(always)]
pub fn matmul<B, T, S>(
    lhs: &Q<B, T>,
    rhs: &Q<B, S>,
) -> SkoroResult<Q<B, <T as MatmulQuantityBuilder<B, S>>::OutputType>>
where
    B: Backend,
    T: ContinuousQuantity<B>,
    S: ContinuousQuantity<B>,
    T: MatmulQuantityBuilder<B, S>,
{
    lhs.matmul(rhs)
}

pub trait MatmulOps<B: Backend> {
    type InnerType: MatmulQuantity<B>;

    #[allow(clippy::type_complexity)]
    fn matmul<S: ContinuousQuantity<B>>(
        &self,
        rhs: &Q<B, S>,
    ) -> SkoroResult<Q<B, <Self::InnerType as MatmulQuantityBuilder<B, S>>::OutputType>>
    where
        Self::InnerType: MatmulQuantityBuilder<B, S>;
}

impl<B: Backend, T: ContinuousQuantity<B>> MatmulOps<B> for Q<B, T>
where
    T: MatmulQuantity<B>,
{
    type InnerType = T;

    fn matmul<S: ContinuousQuantity<B>>(
        &self,
        rhs: &Q<B, S>,
    ) -> SkoroResult<Q<B, <T as MatmulQuantityBuilder<B, S>>::OutputType>>
    where
        T: MatmulQuantityBuilder<B, S>,
    {
        let (out_type, matmul) = self
            .get_quantity_type()
            .build_matmul(&rhs.get_quantity_type())?;
        register_map2(out_type, matmul, (self, rhs))
    }
}

pub trait MatmulQuantity<B: Backend>: ContinuousQuantity<B> {}
pub trait MatmulQuantityBuilder<B: Backend, Rhs: ContinuousQuantity<B>>: MatmulQuantity<B> {
    type OutputType: ContinuousQuantity<B>;

    fn build_matmul(&self, rhs: &Rhs) -> SkoroResult<(Self::OutputType, Matmul)>;
}

impl<B: Backend> MatmulQuantity<B> for EuclideanD {}
impl<B: Backend> MatmulQuantityBuilder<B, EuclideanD> for EuclideanMatrixD {
    type OutputType = EuclideanD;

    fn build_matmul(&self, rhs: &EuclideanD) -> SkoroResult<(Self::OutputType, Matmul)> {
        if self.1 != rhs.0 {
            return Err(SkoroError::IncompatibleMatmul {
                lhs: format!("{self:?}"),
                rhs: format!("{rhs:?}"),
            });
        }
        Ok((EuclideanD(self.0), Matmul(self.0, self.1, 1)))
    }
}

impl<B: Backend> MatmulQuantity<B> for EuclideanMatrixD {}
impl<B: Backend, const N: usize> MatmulQuantityBuilder<B, Euclidean<N>> for EuclideanMatrixD {
    type OutputType = EuclideanD;

    fn build_matmul(&self, rhs: &Euclidean<N>) -> SkoroResult<(Self::OutputType, Matmul)> {
        if self.1 != N {
            return Err(SkoroError::IncompatibleMatmul {
                lhs: format!("{self:?}"),
                rhs: format!("{rhs:?}"),
            });
        }
        Ok((EuclideanD(self.0), Matmul(self.0, N, 1)))
    }
}

impl<B: Backend> MatmulQuantityBuilder<B, EuclideanMatrixD> for EuclideanMatrixD {
    type OutputType = EuclideanMatrixD;

    fn build_matmul(&self, rhs: &EuclideanMatrixD) -> SkoroResult<(Self::OutputType, Matmul)> {
        if self.1 != rhs.0 {
            return Err(SkoroError::IncompatibleMatmul {
                lhs: format!("{self:?}"),
                rhs: format!("{rhs:?}"),
            });
        }
        Ok((
            EuclideanMatrixD(self.0, rhs.1),
            Matmul(self.0, self.1, rhs.1),
        ))
    }
}

impl<B: Backend, const M: usize, const N: usize> MatmulQuantity<B> for EuclideanMatrix<M, N> {}
impl<B: Backend, const M: usize, const N: usize> MatmulQuantityBuilder<B, EuclideanMatrixD>
    for EuclideanMatrix<M, N>
{
    type OutputType = EuclideanMatrixD;

    fn build_matmul(&self, rhs: &EuclideanMatrixD) -> SkoroResult<(Self::OutputType, Matmul)> {
        if N != rhs.0 {
            return Err(SkoroError::IncompatibleMatmul {
                lhs: format!("{self:?}"),
                rhs: format!("{rhs:?}"),
            });
        }
        Ok((EuclideanMatrixD(M, rhs.1), Matmul(M, N, rhs.1)))
    }
}

impl<B: Backend, const M: usize, const N: usize> MatmulQuantityBuilder<B, EuclideanMatrix<M, N>>
    for EuclideanMatrixD
{
    type OutputType = EuclideanMatrixD;

    fn build_matmul(&self, rhs: &EuclideanMatrix<M, N>) -> SkoroResult<(Self::OutputType, Matmul)> {
        if self.1 != M {
            return Err(SkoroError::IncompatibleMatmul {
                lhs: format!("{self:?}"),
                rhs: format!("{rhs:?}"),
            });
        }
        Ok((EuclideanMatrixD(self.1, N), Matmul(self.0, M, N)))
    }
}

macro_rules! impl_matmul_quantity {
    ($M:literal, $N:literal) => {
        impl<B: Backend> MatmulQuantityBuilder<B, Euclidean<$N>> for EuclideanMatrix<$M, $N> {
            type OutputType = Euclidean<$M>;

            fn build_matmul(&self, _: &Euclidean<$N>) -> SkoroResult<(Self::OutputType, Matmul)> {
                Ok((Euclidean::<$M>, Matmul($M, $N, 1)))
            }
        }
    };
    ($M:literal, $N:literal, $K:literal) => {
        impl<B: Backend> MatmulQuantityBuilder<B, EuclideanMatrix<$N, $K>>
            for EuclideanMatrix<$M, $N>
        {
            type OutputType = EuclideanMatrix<$M, $K>;

            fn build_matmul(
                &self,
                _: &EuclideanMatrix<$N, $K>,
            ) -> SkoroResult<(Self::OutputType, Matmul)> {
                Ok((EuclideanMatrix::<$M, $K>, Matmul($M, $N, $K)))
            }
        }
    };
}

impl_matmul_quantity!(1, 1);
impl_matmul_quantity!(1, 2);
impl_matmul_quantity!(1, 3);
impl_matmul_quantity!(1, 4);
impl_matmul_quantity!(1, 5);
impl_matmul_quantity!(1, 6);

impl_matmul_quantity!(2, 1);
impl_matmul_quantity!(2, 2);
impl_matmul_quantity!(2, 3);
impl_matmul_quantity!(2, 4);
impl_matmul_quantity!(2, 5);
impl_matmul_quantity!(2, 6);

impl_matmul_quantity!(3, 1);
impl_matmul_quantity!(3, 2);
impl_matmul_quantity!(3, 3);
impl_matmul_quantity!(3, 4);
impl_matmul_quantity!(3, 5);
impl_matmul_quantity!(3, 6);

impl_matmul_quantity!(4, 1);
impl_matmul_quantity!(4, 2);
impl_matmul_quantity!(4, 3);
impl_matmul_quantity!(4, 4);
impl_matmul_quantity!(4, 5);
impl_matmul_quantity!(4, 6);

impl_matmul_quantity!(5, 1);
impl_matmul_quantity!(5, 2);
impl_matmul_quantity!(5, 3);
impl_matmul_quantity!(5, 4);
impl_matmul_quantity!(5, 5);
impl_matmul_quantity!(5, 6);

impl_matmul_quantity!(6, 1);
impl_matmul_quantity!(6, 2);
impl_matmul_quantity!(6, 3);
impl_matmul_quantity!(6, 4);
impl_matmul_quantity!(6, 5);
impl_matmul_quantity!(6, 6);

impl_matmul_quantity!(1, 1, 1);
impl_matmul_quantity!(1, 1, 2);
impl_matmul_quantity!(1, 1, 3);
impl_matmul_quantity!(1, 1, 4);
impl_matmul_quantity!(1, 1, 5);
impl_matmul_quantity!(1, 1, 6);

impl_matmul_quantity!(1, 2, 1);
impl_matmul_quantity!(1, 2, 2);
impl_matmul_quantity!(1, 2, 3);
impl_matmul_quantity!(1, 2, 4);
impl_matmul_quantity!(1, 2, 5);
impl_matmul_quantity!(1, 2, 6);

impl_matmul_quantity!(1, 3, 1);
impl_matmul_quantity!(1, 3, 2);
impl_matmul_quantity!(1, 3, 3);
impl_matmul_quantity!(1, 3, 4);
impl_matmul_quantity!(1, 3, 5);
impl_matmul_quantity!(1, 3, 6);

impl_matmul_quantity!(1, 4, 1);
impl_matmul_quantity!(1, 4, 2);
impl_matmul_quantity!(1, 4, 3);
impl_matmul_quantity!(1, 4, 4);
impl_matmul_quantity!(1, 4, 5);
impl_matmul_quantity!(1, 4, 6);

impl_matmul_quantity!(1, 5, 1);
impl_matmul_quantity!(1, 5, 2);
impl_matmul_quantity!(1, 5, 3);
impl_matmul_quantity!(1, 5, 4);
impl_matmul_quantity!(1, 5, 5);
impl_matmul_quantity!(1, 5, 6);

impl_matmul_quantity!(1, 6, 1);
impl_matmul_quantity!(1, 6, 2);
impl_matmul_quantity!(1, 6, 3);
impl_matmul_quantity!(1, 6, 4);
impl_matmul_quantity!(1, 6, 5);
impl_matmul_quantity!(1, 6, 6);

impl_matmul_quantity!(2, 1, 1);
impl_matmul_quantity!(2, 1, 2);
impl_matmul_quantity!(2, 1, 3);
impl_matmul_quantity!(2, 1, 4);
impl_matmul_quantity!(2, 1, 5);
impl_matmul_quantity!(2, 1, 6);

impl_matmul_quantity!(2, 2, 1);
impl_matmul_quantity!(2, 2, 2);
impl_matmul_quantity!(2, 2, 3);
impl_matmul_quantity!(2, 2, 4);
impl_matmul_quantity!(2, 2, 5);
impl_matmul_quantity!(2, 2, 6);

impl_matmul_quantity!(2, 3, 1);
impl_matmul_quantity!(2, 3, 2);
impl_matmul_quantity!(2, 3, 3);
impl_matmul_quantity!(2, 3, 4);
impl_matmul_quantity!(2, 3, 5);
impl_matmul_quantity!(2, 3, 6);

impl_matmul_quantity!(2, 4, 1);
impl_matmul_quantity!(2, 4, 2);
impl_matmul_quantity!(2, 4, 3);
impl_matmul_quantity!(2, 4, 4);
impl_matmul_quantity!(2, 4, 5);
impl_matmul_quantity!(2, 4, 6);

impl_matmul_quantity!(2, 5, 1);
impl_matmul_quantity!(2, 5, 2);
impl_matmul_quantity!(2, 5, 3);
impl_matmul_quantity!(2, 5, 4);
impl_matmul_quantity!(2, 5, 5);
impl_matmul_quantity!(2, 5, 6);

impl_matmul_quantity!(2, 6, 1);
impl_matmul_quantity!(2, 6, 2);
impl_matmul_quantity!(2, 6, 3);
impl_matmul_quantity!(2, 6, 4);
impl_matmul_quantity!(2, 6, 5);
impl_matmul_quantity!(2, 6, 6);

impl_matmul_quantity!(3, 1, 1);
impl_matmul_quantity!(3, 1, 2);
impl_matmul_quantity!(3, 1, 3);
impl_matmul_quantity!(3, 1, 4);
impl_matmul_quantity!(3, 1, 5);
impl_matmul_quantity!(3, 1, 6);

impl_matmul_quantity!(3, 2, 1);
impl_matmul_quantity!(3, 2, 2);
impl_matmul_quantity!(3, 2, 3);
impl_matmul_quantity!(3, 2, 4);
impl_matmul_quantity!(3, 2, 5);
impl_matmul_quantity!(3, 2, 6);

impl_matmul_quantity!(3, 3, 1);
impl_matmul_quantity!(3, 3, 2);
impl_matmul_quantity!(3, 3, 3);
impl_matmul_quantity!(3, 3, 4);
impl_matmul_quantity!(3, 3, 5);
impl_matmul_quantity!(3, 3, 6);

impl_matmul_quantity!(3, 4, 1);
impl_matmul_quantity!(3, 4, 2);
impl_matmul_quantity!(3, 4, 3);
impl_matmul_quantity!(3, 4, 4);
impl_matmul_quantity!(3, 4, 5);
impl_matmul_quantity!(3, 4, 6);

impl_matmul_quantity!(3, 5, 1);
impl_matmul_quantity!(3, 5, 2);
impl_matmul_quantity!(3, 5, 3);
impl_matmul_quantity!(3, 5, 4);
impl_matmul_quantity!(3, 5, 5);
impl_matmul_quantity!(3, 5, 6);

impl_matmul_quantity!(3, 6, 1);
impl_matmul_quantity!(3, 6, 2);
impl_matmul_quantity!(3, 6, 3);
impl_matmul_quantity!(3, 6, 4);
impl_matmul_quantity!(3, 6, 5);
impl_matmul_quantity!(3, 6, 6);

impl_matmul_quantity!(4, 1, 1);
impl_matmul_quantity!(4, 1, 2);
impl_matmul_quantity!(4, 1, 3);
impl_matmul_quantity!(4, 1, 4);
impl_matmul_quantity!(4, 1, 5);
impl_matmul_quantity!(4, 1, 6);

impl_matmul_quantity!(4, 2, 1);
impl_matmul_quantity!(4, 2, 2);
impl_matmul_quantity!(4, 2, 3);
impl_matmul_quantity!(4, 2, 4);
impl_matmul_quantity!(4, 2, 5);
impl_matmul_quantity!(4, 2, 6);

impl_matmul_quantity!(4, 3, 1);
impl_matmul_quantity!(4, 3, 2);
impl_matmul_quantity!(4, 3, 3);
impl_matmul_quantity!(4, 3, 4);
impl_matmul_quantity!(4, 3, 5);
impl_matmul_quantity!(4, 3, 6);

impl_matmul_quantity!(4, 4, 1);
impl_matmul_quantity!(4, 4, 2);
impl_matmul_quantity!(4, 4, 3);
impl_matmul_quantity!(4, 4, 4);
impl_matmul_quantity!(4, 4, 5);
impl_matmul_quantity!(4, 4, 6);

impl_matmul_quantity!(4, 5, 1);
impl_matmul_quantity!(4, 5, 2);
impl_matmul_quantity!(4, 5, 3);
impl_matmul_quantity!(4, 5, 4);
impl_matmul_quantity!(4, 5, 5);
impl_matmul_quantity!(4, 5, 6);

impl_matmul_quantity!(4, 6, 1);
impl_matmul_quantity!(4, 6, 2);
impl_matmul_quantity!(4, 6, 3);
impl_matmul_quantity!(4, 6, 4);
impl_matmul_quantity!(4, 6, 5);
impl_matmul_quantity!(4, 6, 6);

impl_matmul_quantity!(5, 1, 1);
impl_matmul_quantity!(5, 1, 2);
impl_matmul_quantity!(5, 1, 3);
impl_matmul_quantity!(5, 1, 4);
impl_matmul_quantity!(5, 1, 5);
impl_matmul_quantity!(5, 1, 6);

impl_matmul_quantity!(5, 2, 1);
impl_matmul_quantity!(5, 2, 2);
impl_matmul_quantity!(5, 2, 3);
impl_matmul_quantity!(5, 2, 4);
impl_matmul_quantity!(5, 2, 5);
impl_matmul_quantity!(5, 2, 6);

impl_matmul_quantity!(5, 3, 1);
impl_matmul_quantity!(5, 3, 2);
impl_matmul_quantity!(5, 3, 3);
impl_matmul_quantity!(5, 3, 4);
impl_matmul_quantity!(5, 3, 5);
impl_matmul_quantity!(5, 3, 6);

impl_matmul_quantity!(5, 4, 1);
impl_matmul_quantity!(5, 4, 2);
impl_matmul_quantity!(5, 4, 3);
impl_matmul_quantity!(5, 4, 4);
impl_matmul_quantity!(5, 4, 5);
impl_matmul_quantity!(5, 4, 6);

impl_matmul_quantity!(5, 5, 1);
impl_matmul_quantity!(5, 5, 2);
impl_matmul_quantity!(5, 5, 3);
impl_matmul_quantity!(5, 5, 4);
impl_matmul_quantity!(5, 5, 5);
impl_matmul_quantity!(5, 5, 6);

impl_matmul_quantity!(5, 6, 1);
impl_matmul_quantity!(5, 6, 2);
impl_matmul_quantity!(5, 6, 3);
impl_matmul_quantity!(5, 6, 4);
impl_matmul_quantity!(5, 6, 5);
impl_matmul_quantity!(5, 6, 6);

impl_matmul_quantity!(6, 1, 1);
impl_matmul_quantity!(6, 1, 2);
impl_matmul_quantity!(6, 1, 3);
impl_matmul_quantity!(6, 1, 4);
impl_matmul_quantity!(6, 1, 5);
impl_matmul_quantity!(6, 1, 6);

impl_matmul_quantity!(6, 2, 1);
impl_matmul_quantity!(6, 2, 2);
impl_matmul_quantity!(6, 2, 3);
impl_matmul_quantity!(6, 2, 4);
impl_matmul_quantity!(6, 2, 5);
impl_matmul_quantity!(6, 2, 6);

impl_matmul_quantity!(6, 3, 1);
impl_matmul_quantity!(6, 3, 2);
impl_matmul_quantity!(6, 3, 3);
impl_matmul_quantity!(6, 3, 4);
impl_matmul_quantity!(6, 3, 5);
impl_matmul_quantity!(6, 3, 6);

impl_matmul_quantity!(6, 4, 1);
impl_matmul_quantity!(6, 4, 2);
impl_matmul_quantity!(6, 4, 3);
impl_matmul_quantity!(6, 4, 4);
impl_matmul_quantity!(6, 4, 5);
impl_matmul_quantity!(6, 4, 6);

impl_matmul_quantity!(6, 5, 1);
impl_matmul_quantity!(6, 5, 2);
impl_matmul_quantity!(6, 5, 3);
impl_matmul_quantity!(6, 5, 4);
impl_matmul_quantity!(6, 5, 5);
impl_matmul_quantity!(6, 5, 6);

impl_matmul_quantity!(6, 6, 1);
impl_matmul_quantity!(6, 6, 2);
impl_matmul_quantity!(6, 6, 3);
impl_matmul_quantity!(6, 6, 4);
impl_matmul_quantity!(6, 6, 5);
impl_matmul_quantity!(6, 6, 6);
