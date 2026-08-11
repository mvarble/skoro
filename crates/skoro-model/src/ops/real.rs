use skoro_backend::Backend;
use skoro_core::error::SkoroResult;
use skoro_operation::ops::{
    Addition, LiteralPointwiseAdd, LiteralPointwiseDiv, LiteralPointwiseMul, LiteralPointwiseSub,
    PointwiseDiv, PointwiseMul, Subtraction,
};
use skoro_riemann::real::Real;

use crate::ops::{QLiteralAdd, QLiteralDiv, QLiteralMul, QLiteralSub};
use crate::q::Q;

use super::{QAdd, QDiv, QMul, QSub};

impl<B: Backend> QAdd<B, Real> for Real {
    type OutputType = Real;
    type OpType = Addition;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        Real
    }
    fn op_type(&self, _: &Real) -> Self::OpType {
        Addition
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Real>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QAdd<B, Real>>::quantity_type(&t0, &t1);
        let op = <Self as QAdd<B, Real>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QSub<B, Real> for Real {
    type OutputType = Real;
    type OpType = Subtraction;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        Real
    }
    fn op_type(&self, _: &Real) -> Self::OpType {
        Subtraction
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Real>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QSub<B, Real>>::quantity_type(&t0, &t1);
        let op = <Self as QSub<B, Real>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QMul<B, Real> for Real {
    type OutputType = Real;
    type OpType = PointwiseMul;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        Real
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

impl<B: Backend> QDiv<B, Real> for Real {
    type OutputType = Real;
    type OpType = PointwiseDiv;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        Real
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

impl<B: Backend> QLiteralAdd<B> for Real {
    type OutputType = Real;
    type OpType = LiteralPointwiseAdd<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        Real
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseAdd::new(float)
    }
}

impl<B: Backend> QLiteralSub<B> for Real {
    type OutputType = Real;
    type OpType = LiteralPointwiseSub<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        Real
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseSub::new(float)
    }
}

impl<B: Backend> QLiteralMul<B> for Real {
    type OutputType = Real;
    type OpType = LiteralPointwiseMul<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        Real
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseMul::new(float)
    }
}

impl<B: Backend> QLiteralDiv<B> for Real {
    type OutputType = Real;
    type OpType = LiteralPointwiseDiv<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        Real
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseDiv::new(float)
    }
}
