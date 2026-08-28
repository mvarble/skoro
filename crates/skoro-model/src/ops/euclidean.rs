use skoro_backend::Backend;
use skoro_core::error::SkoroResult;
use skoro_operation::ops::{
    Addition, LiteralScalarDivisionRight, LiteralScalarMultiplication, Negation, ScalarDivision,
    ScalarMultiplication, Subtraction,
};
use skoro_riemann::{
    euclidean::{Euclidean, EuclideanD},
    real::Real,
};

use crate::q::Q;
use crate::{
    map::register_map1,
    ops::{QLiteralDivRight, QLiteralMul},
};

use super::{QAdd, QDiv, QMul, QNeg, QSub};

impl<B: Backend> QNeg<B> for EuclideanD {
    type OutputType = EuclideanD;
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
        register_map1(typout, op, node).expect("Negation of a vector should never error.")
    }
}

impl<B: Backend> QAdd<B, EuclideanD> for EuclideanD {
    type OutputType = EuclideanD;
    fn quantity_type(&self, _: &EuclideanD) -> Self::OutputType {
        *self
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, EuclideanD>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QAdd<B, EuclideanD>>::quantity_type(&t0, &t1);
        let op = Addition;
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QSub<B, EuclideanD> for EuclideanD {
    type OutputType = EuclideanD;
    fn quantity_type(&self, _: &EuclideanD) -> Self::OutputType {
        *self
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, EuclideanD>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QSub<B, EuclideanD>>::quantity_type(&t0, &t1);
        let op = Subtraction;
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QMul<B, Real> for EuclideanD {
    type OutputType = EuclideanD;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        *self
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Real>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QMul<B, Real>>::quantity_type(&t0, &t1);
        let op = ScalarMultiplication;
        crate::ops::base::continuous_binop(rhs, lhs, t, op)
    }
}

impl<B: Backend> QMul<B, EuclideanD> for Real {
    type OutputType = EuclideanD;
    fn quantity_type(&self, other: &EuclideanD) -> Self::OutputType {
        *other
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, EuclideanD>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QMul<B, EuclideanD>>::quantity_type(&t0, &t1);
        let op = ScalarMultiplication;
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QDiv<B, Real> for EuclideanD {
    type OutputType = EuclideanD;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        *self
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Real>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QDiv<B, Real>>::quantity_type(&t0, &t1);
        let op = ScalarDivision;
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QLiteralMul<B> for EuclideanD {
    type OutputType = EuclideanD;
    type OpType = LiteralScalarMultiplication<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralScalarMultiplication::new(float)
    }
}

impl<B: Backend> QLiteralDivRight<B> for EuclideanD {
    type OutputType = EuclideanD;
    type OpType = LiteralScalarDivisionRight<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralScalarDivisionRight::new(float)
    }
}

impl<B: Backend, const D: usize> QNeg<B> for Euclidean<D> {
    type OutputType = Euclidean<D>;
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
        register_map1(typout, op, node).expect("Negation of a vector should never error.")
    }
}

impl<B: Backend, const D: usize> QAdd<B, Euclidean<D>> for Euclidean<D> {
    type OutputType = Euclidean<D>;
    fn quantity_type(&self, _: &Euclidean<D>) -> Self::OutputType {
        *self
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Euclidean<D>>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QAdd<B, Euclidean<D>>>::quantity_type(&t0, &t1);
        let op = Addition;
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend, const D: usize> QSub<B, Euclidean<D>> for Euclidean<D> {
    type OutputType = Euclidean<D>;
    fn quantity_type(&self, _: &Euclidean<D>) -> Self::OutputType {
        *self
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Euclidean<D>>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QSub<B, Euclidean<D>>>::quantity_type(&t0, &t1);
        let op = Subtraction;
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend, const D: usize> QMul<B, Real> for Euclidean<D> {
    type OutputType = Euclidean<D>;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        *self
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Real>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QMul<B, Real>>::quantity_type(&t0, &t1);
        let op = ScalarMultiplication;
        crate::ops::base::continuous_binop(rhs, lhs, t, op)
    }
}

impl<B: Backend, const D: usize> QMul<B, Euclidean<D>> for Real {
    type OutputType = Euclidean<D>;
    fn quantity_type(&self, other: &Euclidean<D>) -> Self::OutputType {
        *other
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Euclidean<D>>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QMul<B, Euclidean<D>>>::quantity_type(&t0, &t1);
        let op = ScalarMultiplication;
        crate::ops::base::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend, const D: usize> QDiv<B, Real> for Euclidean<D> {
    type OutputType = Euclidean<D>;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        *self
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Real>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QDiv<B, Real>>::quantity_type(&t0, &t1);
        let op = ScalarDivision;
        crate::ops::base::continuous_binop(rhs, lhs, t, op)
    }
}

impl<B: Backend, const D: usize> QLiteralMul<B> for Euclidean<D> {
    type OutputType = Euclidean<D>;
    type OpType = LiteralScalarMultiplication<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralScalarMultiplication::new(float)
    }
}

impl<B: Backend, const D: usize> QLiteralDivRight<B> for Euclidean<D> {
    type OutputType = Euclidean<D>;
    type OpType = LiteralScalarDivisionRight<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralScalarDivisionRight::new(float)
    }
}
