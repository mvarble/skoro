use skoro_backend::{Backend, Element};
use skoro_core::pack::{NumComponents, Packable};
use skoro_model::ops::{
    AssociatedTangentVector, AssociatedTangentVectorBuilder, QAdd, QDiv, QLiteralDiv, QLiteralMul,
    QMul, QSub,
};
use skoro_model::q::Q;
use skoro_operation::{
    diff::DiffOperation,
    ops::{
        Addition, LiteralPointwiseDiv, LiteralPointwiseMul, PointwiseDiv, PointwiseMul, Subtraction,
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
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Tril>) -> Q<B, Self::OutputType> {
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
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Tril>) -> Q<B, Self::OutputType> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QSub<B, Tril>>::quantity_type(&t0, &t1);
        let op = <Self as QSub<B, Tril>>::op_type(&t0, &t1);
        skoro_model::ops::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QMul<B, Real> for Tril {
    type OutputType = Tril;
    type OpType = PointwiseMul;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &Real) -> Self::OpType {
        PointwiseMul
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Real>) -> Q<B, Self::OutputType> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QMul<B, Real>>::quantity_type(&t0, &t1);
        let op = <Self as QMul<B, Real>>::op_type(&t0, &t1);
        skoro_model::ops::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QDiv<B, Real> for Tril {
    type OutputType = Tril;
    type OpType = PointwiseDiv;
    fn quantity_type(&self, _: &Real) -> Self::OutputType {
        *self
    }
    fn op_type(&self, _: &Real) -> Self::OpType {
        PointwiseDiv
    }
    fn register(lhs: &Q<B, Self>, rhs: &Q<B, Real>) -> Q<B, Self::OutputType> {
        let t0 = lhs.get_quantity_type();
        let t1 = rhs.get_quantity_type();
        let t = <Self as QDiv<B, Real>>::quantity_type(&t0, &t1);
        let op = <Self as QDiv<B, Real>>::op_type(&t0, &t1);
        skoro_model::ops::continuous_binop(lhs, rhs, t, op)
    }
}

impl<B: Backend> QLiteralMul<B> for Tril {
    type OutputType = Tril;
    type OpType = LiteralPointwiseMul<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseMul::new(float)
    }
}

impl<B: Backend> QLiteralDiv<B> for Tril {
    type OutputType = Tril;
    type OpType = LiteralPointwiseDiv<B>;
    fn quantity_type(&self, _: B::FloatElem) -> Self::OutputType {
        *self
    }
    fn op_type(&self, float: B::FloatElem) -> Self::OpType {
        LiteralPointwiseDiv::new(float)
    }
}

impl<B: Backend> AssociatedTangentVector<Tril, B> for Tril {}
impl<B: Backend> AssociatedTangentVectorBuilder<Tril, B> for Tril {
    fn build_tangent(&self) -> Tril {
        *self
    }
}
