use skoro_backend::Backend;
use skoro_core::error::SkoroResult;
use skoro_operation::ops::{
    Addition, LiteralScalarDivisionRight, LiteralScalarMultiplication, Negation, ScalarDivision,
    ScalarMultiplication, Subtraction,
};
use skoro_riemann::{
    matrix::{EuclideanMatrix, EuclideanMatrixD},
    real::Real,
};

use crate::map::register_map1;
use crate::ops::{
    AssociatedTangentVector, AssociatedTangentVectorBuilder, QLiteralDivRight, QLiteralMul,
};
use crate::q::Q;

use super::{QAdd, QDiv, QMul, QNeg, QSub};

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

impl<B: Backend> QNeg<B> for EuclideanMatrixD {
    type OutputType = EuclideanMatrixD;
    type OpType = Negation;
    fn quantity_type(&self) -> Self::OutputType {
        *self
    }
    fn op_type(&self) -> Self::OpType {
        Negation
    }
    fn register(node: &Q<B, Self>) -> Q<B, Self::OutputType> {
        let typin = node.get_quantity_type();
        let typout = <Self as QNeg<B>>::quantity_type(&typin);
        let op = <Self as QNeg<B>>::op_type(&typin);
        register_map1(typout, op, node).expect("Negation of a matrix should never error.")
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
    type OpType = ScalarMultiplication;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &Real) -> Self::OpType {
        ScalarMultiplication
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Real>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QMul<B, Real>>::quantity_type(&t0, &t1);
        let op = <Self as QMul<B, Real>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(rhs, lhs, t, op)
    }
}

impl<B: Backend> QMul<B, EuclideanMatrixD> for Real {
    type OutputType = EuclideanMatrixD;
    type OpType = ScalarMultiplication;
    fn quantity_type(&self, other: &EuclideanMatrixD) -> Self::OutputType {
        *other
    }
    fn op_type(&self, _: &EuclideanMatrixD) -> Self::OpType {
        ScalarMultiplication
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
    type OpType = ScalarDivision;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &Real) -> Self::OpType {
        ScalarDivision
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Real>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QDiv<B, Real>>::quantity_type(&t0, &t1);
        let op = <Self as QDiv<B, Real>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(rhs, lhs, t, op)
    }
}

impl<B: Backend> QLiteralMul<B> for EuclideanMatrixD {
    type OutputType = EuclideanMatrixD;
    type OpType = LiteralScalarMultiplication<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralScalarMultiplication::new(float)
    }
}

impl<B: Backend> QLiteralDivRight<B> for EuclideanMatrixD {
    type OutputType = EuclideanMatrixD;
    type OpType = LiteralScalarDivisionRight<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralScalarDivisionRight::new(float)
    }
}

impl<B: Backend, const M: usize, const N: usize> QNeg<B> for EuclideanMatrix<M, N> {
    type OutputType = EuclideanMatrix<M, N>;
    type OpType = Negation;
    fn quantity_type(&self) -> Self::OutputType {
        *self
    }
    fn op_type(&self) -> Self::OpType {
        Negation
    }
    fn register(node: &Q<B, Self>) -> Q<B, Self::OutputType> {
        let typin = node.get_quantity_type();
        let typout = <Self as QNeg<B>>::quantity_type(&typin);
        let op = <Self as QNeg<B>>::op_type(&typin);
        register_map1(typout, op, node).expect("Negation of a matrix should never error.")
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
    type OpType = ScalarMultiplication;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &Real) -> Self::OpType {
        ScalarMultiplication
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Real>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QMul<B, Real>>::quantity_type(&t0, &t1);
        let op = <Self as QMul<B, Real>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(rhs, lhs, t, op)
    }
}

impl<B: Backend, const M: usize, const N: usize> QMul<B, EuclideanMatrix<M, N>> for Real {
    type OutputType = EuclideanMatrix<M, N>;
    type OpType = ScalarMultiplication;
    fn quantity_type(&self, other: &EuclideanMatrix<M, N>) -> Self::OutputType {
        *other
    }
    fn op_type(&self, _: &EuclideanMatrix<M, N>) -> Self::OpType {
        ScalarMultiplication
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
    type OpType = ScalarDivision;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &Real) -> Self::OpType {
        ScalarDivision
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Real>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QDiv<B, Real>>::quantity_type(&t0, &t1);
        let op = <Self as QDiv<B, Real>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(rhs, lhs, t, op)
    }
}

impl<B: Backend, const M: usize, const N: usize> QLiteralMul<B> for EuclideanMatrix<M, N> {
    type OutputType = EuclideanMatrix<M, N>;
    type OpType = LiteralScalarMultiplication<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralScalarMultiplication::new(float)
    }
}

impl<B: Backend, const M: usize, const N: usize> QLiteralDivRight<B> for EuclideanMatrix<M, N> {
    type OutputType = EuclideanMatrix<M, N>;
    type OpType = LiteralScalarDivisionRight<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralScalarDivisionRight::new(float)
    }
}
