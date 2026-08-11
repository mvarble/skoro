use skoro_backend::Backend;
use skoro_core::error::SkoroResult;
use skoro_operation::ops::{
    Addition, LiteralPointwiseAdd, LiteralPointwiseDiv, LiteralPointwiseMul, LiteralPointwiseSub,
    PointwiseDiv, PointwiseMul, Subtraction,
};
use skoro_riemann::{
    euclidean::{Euclidean, EuclideanD},
    real::Real,
};

use crate::ops::{QLiteralAdd, QLiteralDiv, QLiteralMul, QLiteralSub};
use crate::q::Q;

use super::{QAdd, QDiv, QMul, QSub};

impl<B: Backend> QAdd<B, EuclideanD> for EuclideanD {
    type OutputType = EuclideanD;
    type OpType = Addition;
    fn quantity_type(&self, _: &EuclideanD) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &EuclideanD) -> Self::OpType {
        Addition
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, EuclideanD>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QAdd<B, EuclideanD>>::quantity_type(&t0, &t1);
        let op = <Self as QAdd<B, EuclideanD>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QSub<B, EuclideanD> for EuclideanD {
    type OutputType = EuclideanD;
    type OpType = Subtraction;
    fn quantity_type(&self, _: &EuclideanD) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &EuclideanD) -> Self::OpType {
        Subtraction
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, EuclideanD>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QSub<B, EuclideanD>>::quantity_type(&t0, &t1);
        let op = <Self as QSub<B, EuclideanD>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QMul<B, Real> for EuclideanD {
    type OutputType = EuclideanD;
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

impl<B: Backend> QMul<B, EuclideanD> for Real {
    type OutputType = EuclideanD;
    type OpType = PointwiseMul;
    fn quantity_type(&self, other: &EuclideanD) -> Self::OutputType {
        *other
    }
    fn op_type(&self, _: &EuclideanD) -> Self::OpType {
        PointwiseMul
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, EuclideanD>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QMul<B, EuclideanD>>::quantity_type(&t0, &t1);
        let op = <Self as QMul<B, EuclideanD>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QDiv<B, Real> for EuclideanD {
    type OutputType = EuclideanD;
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

impl<B: Backend> QDiv<B, EuclideanD> for Real {
    type OutputType = EuclideanD;
    type OpType = PointwiseDiv;
    fn quantity_type(&self, other: &EuclideanD) -> Self::OutputType {
        *other
    }
    fn op_type(&self, _: &EuclideanD) -> Self::OpType {
        PointwiseDiv
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, EuclideanD>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QDiv<B, EuclideanD>>::quantity_type(&t0, &t1);
        let op = <Self as QDiv<B, EuclideanD>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QLiteralAdd<B> for EuclideanD {
    type OutputType = EuclideanD;
    type OpType = LiteralPointwiseAdd<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseAdd::new(float)
    }
}

impl<B: Backend> QLiteralSub<B> for EuclideanD {
    type OutputType = EuclideanD;
    type OpType = LiteralPointwiseSub<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseSub::new(float)
    }
}

impl<B: Backend> QLiteralMul<B> for EuclideanD {
    type OutputType = EuclideanD;
    type OpType = LiteralPointwiseMul<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseMul::new(float)
    }
}

impl<B: Backend> QLiteralDiv<B> for EuclideanD {
    type OutputType = EuclideanD;
    type OpType = LiteralPointwiseDiv<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseDiv::new(float)
    }
}

impl<B: Backend, const D: usize> QAdd<B, Euclidean<D>> for Euclidean<D> {
    type OutputType = Euclidean<D>;
    type OpType = Addition;
    fn quantity_type(&self, _: &Euclidean<D>) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &Euclidean<D>) -> Self::OpType {
        Addition
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Euclidean<D>>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QAdd<B, Euclidean<D>>>::quantity_type(&t0, &t1);
        let op = <Self as QAdd<B, Euclidean<D>>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend, const D: usize> QSub<B, Euclidean<D>> for Euclidean<D> {
    type OutputType = Euclidean<D>;
    type OpType = Subtraction;
    fn quantity_type(&self, _: &Euclidean<D>) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &Euclidean<D>) -> Self::OpType {
        Subtraction
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Euclidean<D>>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QSub<B, Euclidean<D>>>::quantity_type(&t0, &t1);
        let op = <Self as QSub<B, Euclidean<D>>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend, const D: usize> QMul<B, Real> for Euclidean<D> {
    type OutputType = Euclidean<D>;
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

impl<B: Backend, const D: usize> QMul<B, Euclidean<D>> for Real {
    type OutputType = Euclidean<D>;
    type OpType = PointwiseMul;
    fn quantity_type(&self, other: &Euclidean<D>) -> Self::OutputType {
        *other
    }
    fn op_type(&self, _: &Euclidean<D>) -> Self::OpType {
        PointwiseMul
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Euclidean<D>>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QMul<B, Euclidean<D>>>::quantity_type(&t0, &t1);
        let op = <Self as QMul<B, Euclidean<D>>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend, const D: usize> QDiv<B, Real> for Euclidean<D> {
    type OutputType = Euclidean<D>;
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

impl<B: Backend, const D: usize> QDiv<B, Euclidean<D>> for Real {
    type OutputType = Euclidean<D>;
    type OpType = PointwiseDiv;
    fn quantity_type(&self, other: &Euclidean<D>) -> Self::OutputType {
        *other
    }
    fn op_type(&self, _: &Euclidean<D>) -> Self::OpType {
        PointwiseDiv
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Euclidean<D>>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QDiv<B, Euclidean<D>>>::quantity_type(&t0, &t1);
        let op = <Self as QDiv<B, Euclidean<D>>>::op_type(&t0, &t1);
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend, const D: usize> QLiteralMul<B> for Euclidean<D> {
    type OutputType = Euclidean<D>;
    type OpType = LiteralPointwiseMul<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseMul::new(float)
    }
}

impl<B: Backend, const D: usize> QLiteralDiv<B> for Euclidean<D> {
    type OutputType = Euclidean<D>;
    type OpType = LiteralPointwiseDiv<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseDiv::new(float)
    }
}
