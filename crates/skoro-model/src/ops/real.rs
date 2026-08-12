use skoro_backend::Backend;
use skoro_core::error::SkoroResult;
use skoro_operation::ops::{
    Addition, LiteralAddition, LiteralScalarDivisionLeft, LiteralScalarDivisionRight,
    LiteralScalarMultiplication, LiteralSubtractionLeft, LiteralSubtractionRight, Negation,
    ScalarDivision, ScalarMultiplication, Subtraction,
};
use skoro_riemann::real::Real;

use crate::map::register_map1;
use crate::ops::{
    QLiteralAdd, QLiteralDivLeft, QLiteralDivRight, QLiteralMul, QLiteralSubLeft, QLiteralSubRight,
};
use crate::q::Q;

use super::{QAdd, QDiv, QMul, QNeg, QSub};

impl<B: Backend> QNeg<B> for Real {
    type OutputType = Real;
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
        register_map1(typout, op, node).expect("Negation of a scalar should never error.")
    }
}

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
    type OpType = ScalarMultiplication;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        Real
    }
    fn op_type(&self, _: &Real) -> Self::OpType {
        ScalarMultiplication
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
    type OpType = ScalarDivision;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        Real
    }
    fn op_type(&self, _: &Real) -> Self::OpType {
        ScalarDivision
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
    type OpType = LiteralAddition<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        Real
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralAddition::new(float)
    }
}

impl<B: Backend> QLiteralSubLeft<B> for Real {
    type OutputType = Real;
    type OpType = LiteralSubtractionLeft<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        Real
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralSubtractionLeft::new(float)
    }
}

impl<B: Backend> QLiteralSubRight<B> for Real {
    type OutputType = Real;
    type OpType = LiteralSubtractionRight<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        Real
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralSubtractionRight::new(float)
    }
}

impl<B: Backend> QLiteralMul<B> for Real {
    type OutputType = Real;
    type OpType = LiteralScalarMultiplication<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        Real
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralScalarMultiplication::new(float)
    }
}

impl<B: Backend> QLiteralDivLeft<B> for Real {
    type OutputType = Real;
    type OpType = LiteralScalarDivisionLeft<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        Real
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralScalarDivisionLeft::new(float)
    }
}

impl<B: Backend> QLiteralDivRight<B> for Real {
    type OutputType = Real;
    type OpType = LiteralScalarDivisionRight<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        Real
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralScalarDivisionRight::new(float)
    }
}
