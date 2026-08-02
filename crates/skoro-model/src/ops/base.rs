use std::ops::{Add, Div, Mul, Sub};

use skoro_backend::{Backend, ElementConversion, ToElement};
use skoro_graph::{operation::GraphOperation, quantity::ContinuousQuantity};
use skoro_riemann::euclidean::{Euclidean, EuclideanD};

use crate::{
    map::{register_map1, register_map2},
    q::Q,
};

pub trait AssociatedTangentVector<T, B: Backend>: ContinuousQuantity<B> {}
pub trait AssociatedTangentVectorBuilder<T, B: Backend>: AssociatedTangentVector<T, B> {
    fn build_tangent(&self) -> T;
}

impl<B: Backend, T: ContinuousQuantity<B>> AssociatedTangentVector<EuclideanD, B> for T {}
impl<B: Backend, T: ContinuousQuantity<B>> AssociatedTangentVectorBuilder<EuclideanD, B> for T {
    fn build_tangent(&self) -> EuclideanD {
        EuclideanD(self.dimension())
    }
}

impl<B: Backend, const D: usize> AssociatedTangentVector<Euclidean<D>, B> for Euclidean<D> {}
impl<B: Backend, const D: usize> AssociatedTangentVectorBuilder<Euclidean<D>, B> for Euclidean<D> {
    fn build_tangent(&self) -> Euclidean<D> {
        *self
    }
}

/// Generate a binary `Q op Q` operator trait + a single `std::ops` impl on `Q`.
///
/// The trait is generalized over unbounded operand types (so it covers both
/// continuous and discrete operands — the "β" unification). Each implementor
/// supplies `quantity_type`/`op_type` to describe the output type and graph
/// operation, plus a `register` method that builds the resulting node. A
/// default `register` is provided for *continuous* outputs (via
/// [`continuous_binop`]); discrete-output implementors call [`register_map2`]
/// with a [`ForwardOnly`](skoro_graph::operation::ForwardOnly)-wrapped kernel.
macro_rules! impl_q_binop {
    ($trait:ident, $std_trait:ident, $method:ident) => {
        pub trait $trait<B: Backend, T> {
            type OutputType;
            type OpType: GraphOperation<B>;
            fn quantity_type(&self, other: &T) -> Self::OutputType;
            fn op_type(&self, other: &T) -> Self::OpType;

            /// Build the resulting node from the two operand proxies. Continuous
            /// implementors use [`continuous_binop`]; discrete implementors use
            /// [`register_map2`] with a forward-only kernel.
            fn register(lhs: &Q<B, Self>, rhs: &Q<B, T>) -> Q<B, Self::OutputType>
            where
                Self: Sized;
        }

        impl<B, T0, T1> $std_trait<Q<B, T1>> for Q<B, T0>
        where
            B: Backend,
            T0: $trait<B, T1>,
        {
            type Output = Q<B, <T0 as $trait<B, T1>>::OutputType>;
            fn $method(self, rhs: Q<B, T1>) -> Self::Output {
                <T0 as $trait<B, T1>>::register(&self, &rhs)
            }
        }
    };
}

impl_q_binop!(QAdd, Add, add);
impl_q_binop!(QSub, Sub, sub);
impl_q_binop!(QMul, Mul, mul);
impl_q_binop!(QDiv, Div, div);

/// Helper for continuous binary operators: build a continuous mapped node from
/// the operand proxies, given the output quantity type and graph operation.
#[inline(always)]
pub fn continuous_binop<B, T0, T1, O, K>(
    lhs: &Q<B, T0>,
    rhs: &Q<B, T1>,
    out_type: O,
    op: K,
) -> Q<B, O>
where
    B: Backend,
    O: ContinuousQuantity<B>,
    K: GraphOperation<B>,
{
    register_map2(out_type, op, (lhs, rhs))
}

pub trait QLiteralAdd<B: Backend>: ContinuousQuantity<B> {
    type OutputType: ContinuousQuantity<B>;
    type OpType: GraphOperation<B>;
    fn quantity_type(&self, float: B::FloatElem) -> Self::OutputType;
    fn op_type(&self, float: B::FloatElem) -> Self::OpType;
}

impl<B, T, E> Add<E> for Q<B, T>
where
    B: Backend,
    T: ContinuousQuantity<B> + QLiteralAdd<B>,
    E: ToElement,
{
    type Output = Q<B, <T as QLiteralAdd<B>>::OutputType>;
    fn add(self, float: E) -> Self::Output {
        let elem = B::FloatElem::from_elem(float);
        let t0 = self.get_quantity_type();
        let t = t0.quantity_type(elem);
        let op = t0.op_type(elem);
        register_map1(t, op, &self)
    }
}

impl<B, T> Add<Q<B, T>> for f32
where
    B: Backend,
    T: ContinuousQuantity<B> + QLiteralAdd<B>,
{
    type Output = Q<B, <T as QLiteralAdd<B>>::OutputType>;
    fn add(self, rhs: Q<B, T>) -> Self::Output {
        let elem = B::FloatElem::from_elem(self);
        let t0 = rhs.get_quantity_type();
        let t = t0.quantity_type(elem);
        let op = t0.op_type(elem);
        register_map1(t, op, &rhs)
    }
}

impl<B, T> Add<Q<B, T>> for f64
where
    B: Backend,
    T: ContinuousQuantity<B> + QLiteralAdd<B>,
{
    type Output = Q<B, <T as QLiteralAdd<B>>::OutputType>;
    fn add(self, rhs: Q<B, T>) -> Self::Output {
        let elem = B::FloatElem::from_elem(self);
        let t0 = rhs.get_quantity_type();
        let t = t0.quantity_type(elem);
        let op = t0.op_type(elem);
        register_map1(t, op, &rhs)
    }
}

pub trait QLiteralSub<B: Backend>: ContinuousQuantity<B> {
    type OutputType: ContinuousQuantity<B>;
    type OpType: GraphOperation<B>;
    fn quantity_type(&self, float: B::FloatElem) -> Self::OutputType;
    fn op_type(&self, float: B::FloatElem) -> Self::OpType;
}

impl<B, T, E> Sub<E> for Q<B, T>
where
    B: Backend,
    T: ContinuousQuantity<B> + QLiteralSub<B>,
    E: ToElement,
{
    type Output = Q<B, <T as QLiteralSub<B>>::OutputType>;
    fn sub(self, float: E) -> Self::Output {
        let elem = B::FloatElem::from_elem(float);
        let t0 = self.get_quantity_type();
        let t = t0.quantity_type(elem);
        let op = t0.op_type(elem);
        register_map1(t, op, &self)
    }
}

impl<B, T> Sub<Q<B, T>> for f32
where
    B: Backend,
    T: ContinuousQuantity<B> + QLiteralSub<B>,
{
    type Output = Q<B, <T as QLiteralSub<B>>::OutputType>;
    fn sub(self, rhs: Q<B, T>) -> Self::Output {
        let elem = B::FloatElem::from_elem(self);
        let t0 = rhs.get_quantity_type();
        let t = t0.quantity_type(elem);
        let op = t0.op_type(elem);
        register_map1(t, op, &rhs)
    }
}

impl<B, T> Sub<Q<B, T>> for f64
where
    B: Backend,
    T: ContinuousQuantity<B> + QLiteralSub<B>,
{
    type Output = Q<B, <T as QLiteralSub<B>>::OutputType>;
    fn sub(self, rhs: Q<B, T>) -> Self::Output {
        let elem = B::FloatElem::from_elem(self);
        let t0 = rhs.get_quantity_type();
        let t = t0.quantity_type(elem);
        let op = t0.op_type(elem);
        register_map1(t, op, &rhs)
    }
}

pub trait QLiteralMul<B: Backend>: ContinuousQuantity<B> {
    type OutputType: ContinuousQuantity<B>;
    type OpType: GraphOperation<B>;
    fn quantity_type(&self, float: B::FloatElem) -> Self::OutputType;
    fn op_type(&self, float: B::FloatElem) -> Self::OpType;
}

impl<B, T, E> Mul<E> for Q<B, T>
where
    B: Backend,
    T: ContinuousQuantity<B> + QLiteralMul<B>,
    E: ToElement,
{
    type Output = Q<B, <T as QLiteralMul<B>>::OutputType>;
    fn mul(self, float: E) -> Self::Output {
        let elem = B::FloatElem::from_elem(float);
        let t0 = self.get_quantity_type();
        let t = t0.quantity_type(elem);
        let op = t0.op_type(elem);
        register_map1(t, op, &self)
    }
}

impl<B, T> Mul<Q<B, T>> for f32
where
    B: Backend,
    T: ContinuousQuantity<B> + QLiteralMul<B>,
{
    type Output = Q<B, <T as QLiteralMul<B>>::OutputType>;
    fn mul(self, rhs: Q<B, T>) -> Self::Output {
        let elem = B::FloatElem::from_elem(self);
        let t0 = rhs.get_quantity_type();
        let t = t0.quantity_type(elem);
        let op = t0.op_type(elem);
        register_map1(t, op, &rhs)
    }
}

impl<B, T> Mul<Q<B, T>> for f64
where
    B: Backend,
    T: ContinuousQuantity<B> + QLiteralMul<B>,
{
    type Output = Q<B, <T as QLiteralMul<B>>::OutputType>;
    fn mul(self, rhs: Q<B, T>) -> Self::Output {
        let elem = B::FloatElem::from_elem(self);
        let t0 = rhs.get_quantity_type();
        let t = t0.quantity_type(elem);
        let op = t0.op_type(elem);
        register_map1(t, op, &rhs)
    }
}

pub trait QLiteralDiv<B: Backend>: ContinuousQuantity<B> {
    type OutputType: ContinuousQuantity<B>;
    type OpType: GraphOperation<B>;
    fn quantity_type(&self, float: B::FloatElem) -> Self::OutputType;
    fn op_type(&self, float: B::FloatElem) -> Self::OpType;
}

impl<B, T, E> Div<E> for Q<B, T>
where
    B: Backend,
    T: ContinuousQuantity<B> + QLiteralDiv<B>,
    E: ToElement,
{
    type Output = Q<B, <T as QLiteralDiv<B>>::OutputType>;
    fn div(self, float: E) -> Self::Output {
        let elem = B::FloatElem::from_elem(float);
        let t0 = self.get_quantity_type();
        let t = t0.quantity_type(elem);
        let op = t0.op_type(elem);
        register_map1(t, op, &self)
    }
}

impl<B, T> Div<Q<B, T>> for f32
where
    B: Backend,
    T: ContinuousQuantity<B> + QLiteralDiv<B>,
{
    type Output = Q<B, <T as QLiteralDiv<B>>::OutputType>;
    fn div(self, rhs: Q<B, T>) -> Self::Output {
        let elem = B::FloatElem::from_elem(self);
        let t0 = rhs.get_quantity_type();
        let t = t0.quantity_type(elem);
        let op = t0.op_type(elem);
        register_map1(t, op, &rhs)
    }
}

impl<B, T> Div<Q<B, T>> for f64
where
    B: Backend,
    T: ContinuousQuantity<B> + QLiteralDiv<B>,
{
    type Output = Q<B, <T as QLiteralDiv<B>>::OutputType>;
    fn div(self, rhs: Q<B, T>) -> Self::Output {
        let elem = B::FloatElem::from_elem(self);
        let t0 = rhs.get_quantity_type();
        let t = t0.quantity_type(elem);
        let op = t0.op_type(elem);
        register_map1(t, op, &rhs)
    }
}
