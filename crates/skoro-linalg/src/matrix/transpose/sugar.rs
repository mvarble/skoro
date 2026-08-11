use skoro_backend::Backend;
use skoro_core::error::SkoroResult;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{map::register_map1, q::Q};
use skoro_riemann::matrix::{EuclideanMatrix, EuclideanMatrixD};

use super::Transpose;

#[inline(always)]
pub fn transpose<B, M>(matrix: &Q<B, M>) -> SkoroResult<Q<B, <M as TransposeType<B>>::OutputType>>
where
    B: Backend,
    M: TransposeType<B>,
{
    matrix.transpose()
}

pub trait TransposeOps<B: Backend> {
    type InnerType: ContinuousQuantity<B>;
    fn transpose(&self) -> SkoroResult<Q<B, Self::InnerType>>;
}

impl<B: Backend, M: TransposeType<B>> TransposeOps<B> for Q<B, M> {
    type InnerType = <M as TransposeType<B>>::OutputType;
    fn transpose(&self) -> SkoroResult<Q<B, Self::InnerType>> {
        let (out_type, transpose) = self.get_quantity_type().build_transpose();
        register_map1(out_type, transpose, self)
    }
}

pub trait TransposeType<B: Backend>: ContinuousQuantity<B> {
    type OutputType: ContinuousQuantity<B>;

    fn build_transpose(&self) -> (Self::OutputType, Transpose);
}

impl<B: Backend> TransposeType<B> for EuclideanMatrixD {
    type OutputType = EuclideanMatrixD;

    fn build_transpose(&self) -> (Self::OutputType, Transpose) {
        (EuclideanMatrixD(self.1, self.0), Transpose(self.0, self.1))
    }
}

macro_rules! impl_transpose_type {
    ($M:literal, $N:literal) => {
        impl<B: Backend> TransposeType<B> for EuclideanMatrix<$M, $N> {
            type OutputType = EuclideanMatrix<$N, $M>;

            fn build_transpose(&self) -> (Self::OutputType, Transpose) {
                (EuclideanMatrix::<$N, $M>, Transpose($M, $N))
            }
        }
    };
}

impl_transpose_type!(1, 1);
impl_transpose_type!(1, 2);
impl_transpose_type!(1, 3);
impl_transpose_type!(1, 4);
impl_transpose_type!(1, 5);
impl_transpose_type!(1, 6);

impl_transpose_type!(2, 1);
impl_transpose_type!(2, 2);
impl_transpose_type!(2, 3);
impl_transpose_type!(2, 4);
impl_transpose_type!(2, 5);
impl_transpose_type!(2, 6);

impl_transpose_type!(3, 1);
impl_transpose_type!(3, 2);
impl_transpose_type!(3, 3);
impl_transpose_type!(3, 4);
impl_transpose_type!(3, 5);
impl_transpose_type!(3, 6);

impl_transpose_type!(4, 1);
impl_transpose_type!(4, 2);
impl_transpose_type!(4, 3);
impl_transpose_type!(4, 4);
impl_transpose_type!(4, 5);
impl_transpose_type!(4, 6);

impl_transpose_type!(5, 1);
impl_transpose_type!(5, 2);
impl_transpose_type!(5, 3);
impl_transpose_type!(5, 4);
impl_transpose_type!(5, 5);
impl_transpose_type!(5, 6);

impl_transpose_type!(6, 1);
impl_transpose_type!(6, 2);
impl_transpose_type!(6, 3);
impl_transpose_type!(6, 4);
impl_transpose_type!(6, 5);
impl_transpose_type!(6, 6);
