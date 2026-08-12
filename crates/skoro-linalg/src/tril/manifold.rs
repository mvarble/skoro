use skoro_backend::{Backend, Element};
use skoro_core::error::SkoroResult;
use skoro_core::pack::{NumComponents, Packable};
use skoro_model::{
    map::register_map1,
    ops::{
        AssociatedTangentVector, AssociatedTangentVectorBuilder, QAdd, QDiv, QLiteralDivRight,
        QLiteralMul, QMul, QNeg, QSub,
    },
    q::Q,
};
use skoro_operation::{
    diff::DiffOperation,
    ops::{
        Addition, LiteralScalarDivisionRight, LiteralScalarMultiplication, Negation,
        ScalarDivision, ScalarMultiplication, Subtraction,
    },
};
use skoro_riemann::{Manifold, euclidean::VectorD, real::Real};

/// A lower triangular matrix of a runtime dimension, specified by the number of diagonal
/// components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tril(pub usize);

impl<B: Backend> Manifold<B> for Tril {
    fn dimension(&self) -> usize {
        self.0 * (self.0 + 1) / 2
    }

    fn retraction(&self) -> impl DiffOperation<B> {
        Addition
    }

    fn inverse_retraction(&self) -> impl DiffOperation<B> {
        Subtraction
    }
}

impl NumComponents for Tril {
    fn num_components(&self) -> usize {
        self.0 * (self.0 + 1) / 2
    }
}

impl<F: Element> Packable<F, VectorD<F>> for Tril {
    fn pack(&self, data: &VectorD<F>, offset: usize, dst: &mut [F]) {
        let k = data.len();
        assert_eq!(k, self.num_components());
        dst[offset..(offset + k)].copy_from_slice(data.slice());
    }

    fn unpack(&self, tensor: &[F], offset: usize) -> VectorD<F> {
        VectorD::<F>::new(&tensor[offset..(offset + self.num_components())])
    }
}

impl<B: Backend> QNeg<B> for Tril {
    type OutputType = Tril;
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
        register_map1(typout, op, node).expect("Negation of a tril should never error.")
    }
}

impl<B: Backend> QAdd<B, Tril> for Tril {
    type OutputType = Tril;
    type OpType = Addition;
    fn quantity_type(&self, other: &Tril) -> Self::OutputType {
        assert_eq!(self, other);
        *self
    }
    fn op_type(&self, _: &Tril) -> Self::OpType {
        Addition
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Tril>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QAdd<B, Tril>>::quantity_type(&t0, &t1);
        let op = <Self as QAdd<B, Tril>>::op_type(&t0, &t1);
        skoro_model::ops::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QSub<B, Tril> for Tril {
    type OutputType = Tril;
    type OpType = Subtraction;
    fn quantity_type(&self, other: &Tril) -> Self::OutputType {
        assert_eq!(self, other);
        *self
    }
    fn op_type(&self, _: &Tril) -> Self::OpType {
        Subtraction
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Tril>) -> SkoroResult<Q<B, Self::OutputType>> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QSub<B, Tril>>::quantity_type(&t0, &t1);
        let op = <Self as QSub<B, Tril>>::op_type(&t0, &t1);
        skoro_model::ops::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QMul<B, Real> for Tril {
    type OutputType = Tril;
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
        skoro_model::ops::continuous_binop(rhs, lhs, t, op)
    }
}

impl<B: Backend> QDiv<B, Real> for Tril {
    type OutputType = Tril;
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
        skoro_model::ops::continuous_binop(rhs, lhs, t, op)
    }
}

impl<B: Backend> QLiteralMul<B> for Tril {
    type OutputType = Tril;
    type OpType = LiteralScalarMultiplication<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralScalarMultiplication::new(float)
    }
}

impl<B: Backend> QLiteralDivRight<B> for Tril {
    type OutputType = Tril;
    type OpType = LiteralScalarDivisionRight<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralScalarDivisionRight::new(float)
    }
}

impl<B: Backend> AssociatedTangentVector<Tril, B> for Tril {}
impl<B: Backend> AssociatedTangentVectorBuilder<Tril, B> for Tril {
    fn build_tangent(&self) -> Tril {
        *self
    }
}
