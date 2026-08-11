use skoro_backend::Backend;
use skoro_core::error::SkoroResult;
use skoro_operation::ops::{
    Addition, LiteralPointwiseAdd, LiteralPointwiseDiv, LiteralPointwiseMul, LiteralPointwiseSub,
    PointwiseDiv, PointwiseMul, Subtraction,
};
use skoro_riemann::{
    matrix::{EuclideanMatrix, EuclideanMatrixD},
    real::Real,
};

use crate::ops::{
    AssociatedTangentVector, AssociatedTangentVectorBuilder, QLiteralAdd, QLiteralDiv, QLiteralMul,
    QLiteralSub,
};
use crate::q::Q;

use super::{QAdd, QDiv, QMul, QSub};

impl<B: Backend> AssociatedTangentVector<EuclideanMatrixD, B> for EuclideanMatrixD {}
impl<B: Backend> AssociatedTangentVectorBuilder<EuclideanMatrixD, B> for EuclideanMatrixD {
    fn build_tangent(&self) -> EuclideanMatrixD {
        *self
    }
}

impl<B: Backend, const M: usize, const N: usize> AssociatedTangentVector<EuclideanMatrix<M, N>, B>
    for EuclideanMatrix<M, N>
{
}
impl<B: Backend, const M: usize, const N: usize>
    AssociatedTangentVectorBuilder<EuclideanMatrix<M, N>, B> for EuclideanMatrix<M, N>
{
    fn build_tangent(&self) -> EuclideanMatrix<M, N> {
        *self
    }
}

impl<B: Backend> QAdd<B, EuclideanMatrixD> for EuclideanMatrixD {
    type OutputType = EuclideanMatrixD;
    type OpType = Addition;
    fn quantity_type(&self, _: &EuclideanMatrixD) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &EuclideanMatrixD) -> Self::OpType {
        Addition
    }
    fn register(
        lhs: &Q<B, Self>,
        rhs: &Q<B, EuclideanMatrixD>,
    ) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QAdd<B, EuclideanMatrixD>>::quantity_type(&t0, &t1);
        let op = <Self as QAdd<B, EuclideanMatrixD>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QSub<B, EuclideanMatrixD> for EuclideanMatrixD {
    type OutputType = EuclideanMatrixD;
    type OpType = Subtraction;
    fn quantity_type(&self, _: &EuclideanMatrixD) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &EuclideanMatrixD) -> Self::OpType {
        Subtraction
    }
    fn register(
        lhs: &Q<B, Self>,
        rhs: &Q<B, EuclideanMatrixD>,
    ) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QSub<B, EuclideanMatrixD>>::quantity_type(&t0, &t1);
        let op = <Self as QSub<B, EuclideanMatrixD>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QMul<B, Real> for EuclideanMatrixD {
    type OutputType = EuclideanMatrixD;
    type OpType = PointwiseMul;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &Real) -> Self::OpType {
        PointwiseMul
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Real>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QMul<B, Real>>::quantity_type(&t0, &t1);
        let op = <Self as QMul<B, Real>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QMul<B, EuclideanMatrixD> for Real {
    type OutputType = EuclideanMatrixD;
    type OpType = PointwiseMul;
    fn quantity_type(&self, other: &EuclideanMatrixD) -> Self::OutputType {
        *other
    }
    fn op_type(&self, _: &EuclideanMatrixD) -> Self::OpType {
        PointwiseMul
    }
    fn register(
        lhs: &Q<B, Self>,
        rhs: &Q<B, EuclideanMatrixD>,
    ) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QMul<B, EuclideanMatrixD>>::quantity_type(&t0, &t1);
        let op = <Self as QMul<B, EuclideanMatrixD>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QDiv<B, Real> for EuclideanMatrixD {
    type OutputType = EuclideanMatrixD;
    type OpType = PointwiseDiv;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &Real) -> Self::OpType {
        PointwiseDiv
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Real>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QDiv<B, Real>>::quantity_type(&t0, &t1);
        let op = <Self as QDiv<B, Real>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QDiv<B, EuclideanMatrixD> for Real {
    type OutputType = EuclideanMatrixD;
    type OpType = PointwiseDiv;
    fn quantity_type(&self, other: &EuclideanMatrixD) -> Self::OutputType {
        *other
    }
    fn op_type(&self, _: &EuclideanMatrixD) -> Self::OpType {
        PointwiseDiv
    }
    fn register(
        lhs: &Q<B, Self>,
        rhs: &Q<B, EuclideanMatrixD>,
    ) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QDiv<B, EuclideanMatrixD>>::quantity_type(&t0, &t1);
        let op = <Self as QDiv<B, EuclideanMatrixD>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QLiteralAdd<B> for EuclideanMatrixD {
    type OutputType = EuclideanMatrixD;
    type OpType = LiteralPointwiseAdd<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseAdd::new(float)
    }
}

impl<B: Backend> QLiteralSub<B> for EuclideanMatrixD {
    type OutputType = EuclideanMatrixD;
    type OpType = LiteralPointwiseSub<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseSub::new(float)
    }
}

impl<B: Backend> QLiteralMul<B> for EuclideanMatrixD {
    type OutputType = EuclideanMatrixD;
    type OpType = LiteralPointwiseMul<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseMul::new(float)
    }
}

impl<B: Backend> QLiteralDiv<B> for EuclideanMatrixD {
    type OutputType = EuclideanMatrixD;
    type OpType = LiteralPointwiseDiv<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseDiv::new(float)
    }
}

impl<B: Backend, const M: usize, const N: usize> QAdd<B, EuclideanMatrix<M, N>>
    for EuclideanMatrix<M, N>
{
    type OutputType = EuclideanMatrix<M, N>;
    type OpType = Addition;
    fn quantity_type(&self, _: &EuclideanMatrix<M, N>) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &EuclideanMatrix<M, N>) -> Self::OpType {
        Addition
    }
    fn register(
        lhs: &Q<B, Self>,
        rhs: &Q<B, EuclideanMatrix<M, N>>,
    ) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QAdd<B, EuclideanMatrix<M, N>>>::quantity_type(&t0, &t1);
        let op = <Self as QAdd<B, EuclideanMatrix<M, N>>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend, const M: usize, const N: usize> QSub<B, EuclideanMatrix<M, N>>
    for EuclideanMatrix<M, N>
{
    type OutputType = EuclideanMatrix<M, N>;
    type OpType = Subtraction;
    fn quantity_type(&self, _: &EuclideanMatrix<M, N>) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &EuclideanMatrix<M, N>) -> Self::OpType {
        Subtraction
    }
    fn register(
        lhs: &Q<B, Self>,
        rhs: &Q<B, EuclideanMatrix<M, N>>,
    ) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QSub<B, EuclideanMatrix<M, N>>>::quantity_type(&t0, &t1);
        let op = <Self as QSub<B, EuclideanMatrix<M, N>>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend, const M: usize, const N: usize> QMul<B, Real> for EuclideanMatrix<M, N> {
    type OutputType = EuclideanMatrix<M, N>;
    type OpType = PointwiseMul;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &Real) -> Self::OpType {
        PointwiseMul
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Real>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QMul<B, Real>>::quantity_type(&t0, &t1);
        let op = <Self as QMul<B, Real>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend, const M: usize, const N: usize> QMul<B, EuclideanMatrix<M, N>> for Real {
    type OutputType = EuclideanMatrix<M, N>;
    type OpType = PointwiseMul;
    fn quantity_type(&self, other: &EuclideanMatrix<M, N>) -> Self::OutputType {
        *other
    }
    fn op_type(&self, _: &EuclideanMatrix<M, N>) -> Self::OpType {
        PointwiseMul
    }
    fn register(
        lhs: &Q<B, Self>,
        rhs: &Q<B, EuclideanMatrix<M, N>>,
    ) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QMul<B, EuclideanMatrix<M, N>>>::quantity_type(&t0, &t1);
        let op = <Self as QMul<B, EuclideanMatrix<M, N>>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend, const M: usize, const N: usize> QDiv<B, Real> for EuclideanMatrix<M, N> {
    type OutputType = EuclideanMatrix<M, N>;
    type OpType = PointwiseDiv;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &Real) -> Self::OpType {
        PointwiseDiv
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Real>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QDiv<B, Real>>::quantity_type(&t0, &t1);
        let op = <Self as QDiv<B, Real>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend, const M: usize, const N: usize> QDiv<B, EuclideanMatrix<M, N>> for Real {
    type OutputType = EuclideanMatrix<M, N>;
    type OpType = PointwiseDiv;
    fn quantity_type(&self, other: &EuclideanMatrix<M, N>) -> Self::OutputType {
        *other
    }
    fn op_type(&self, _: &EuclideanMatrix<M, N>) -> Self::OpType {
        PointwiseDiv
    }
    fn register(
        lhs: &Q<B, Self>,
        rhs: &Q<B, EuclideanMatrix<M, N>>,
    ) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QDiv<B, EuclideanMatrix<M, N>>>::quantity_type(&t0, &t1);
        let op = <Self as QDiv<B, EuclideanMatrix<M, N>>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend, const M: usize, const N: usize> QLiteralMul<B> for EuclideanMatrix<M, N> {
    type OutputType = EuclideanMatrix<M, N>;
    type OpType = LiteralPointwiseMul<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseMul::new(float)
    }
}

impl<B: Backend, const M: usize, const N: usize> QLiteralDiv<B> for EuclideanMatrix<M, N> {
    type OutputType = EuclideanMatrix<M, N>;
    type OpType = LiteralPointwiseDiv<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseDiv::new(float)
    }
}
