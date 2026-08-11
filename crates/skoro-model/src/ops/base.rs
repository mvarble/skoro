use std::ops::{Add, Div, Mul, Sub};

use skoro_backend::{Backend, ElementConversion, ToElement};
use skoro_core::error::SkoroResult;
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
            fn register(lhs: &Q<B, Self>, rhs: &Q<B, T>) -> SkoroResult<Q<B, Self::OutputType>>
            where
                Self: Sized;
        }

        impl<B, T0, T1> $std_trait<Q<B, T1>> for Q<B, T0>
        where
            B: Backend,
            T0: $trait<B, T1>,
        {
            type Output = SkoroResult<Q<B, <T0 as $trait<B, T1>>::OutputType>>;
            fn $method(self, rhs: Q<B, T1>) -> Self::Output {
                <T0 as $trait<B, T1>>::register(&self, &rhs)
            }
        }

        // `Result op Q` is what keeps left-associative chains readable: `a + b + c`
        // parses as `(a + b) + c`, so without it every chain step would need its own
        // `?`.
        //
        // The reverse (`Q op Result`, as in `a + b * c`) is deliberately absent. It
        // would overlap the generic literal impl `Add<E: ToElement> for Q`, since
        // nothing tells the compiler a `SkoroResult` is not an element — and giving up
        // the generic literal impl would mean every bare float in model code needed an
        // `f32`/`f64` suffix. Right-nested products take an explicit `?` instead:
        // `a + (b * c)?`.
        impl<B, T0, T1> $std_trait<Q<B, T1>> for SkoroResult<Q<B, T0>>
        where
            B: Backend,
            T0: $trait<B, T1>,
        {
            type Output = SkoroResult<Q<B, <T0 as $trait<B, T1>>::OutputType>>;
            fn $method(self, rhs: Q<B, T1>) -> Self::Output {
                <T0 as $trait<B, T1>>::register(&self?, &rhs)
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
) -> SkoroResult<Q<B, O>>
where
    B: Backend,
    O: ContinuousQuantity<B>,
    K: GraphOperation<B>,
{
    register_map2(out_type, op, (lhs, rhs))
}

/// Generate a literal operator trait and its `std::ops` impls.
///
/// Four impls per operator: `Q op lit` and `lit op Q`, for `f32` and `f64`.
///
/// The `SkoroResult` forms are *not* expressible: `impl Add<f32> for SkoroResult<Q>`
/// has no local type before its uncovered parameters, so the orphan rule rejects it.
/// A literal combined with a chained result therefore needs a `?`:
/// `(a * b)? + 1.0`.
///
/// The literal side is written out per float type rather than generically over
/// `ToElement`: a blanket `impl<E: ToElement> Add<E> for Q` would overlap
/// `Add<SkoroResult<Q>> for Q`, since nothing tells the compiler a `SkoroResult`
/// is not an element.
macro_rules! impl_q_literal_binop {
    ($trait:ident, $std_trait:ident, $method:ident, $helper:ident) => {
        pub trait $trait<B: Backend>: ContinuousQuantity<B> {
            type OutputType: ContinuousQuantity<B>;
            type OpType: GraphOperation<B>;
            fn quantity_type(&self, float: B::FloatElem) -> Self::OutputType;
            fn op_type(&self, float: B::FloatElem) -> Self::OpType;
        }

        #[inline(always)]
        fn $helper<B, T, E>(
            q: &Q<B, T>,
            float: E,
        ) -> SkoroResult<Q<B, <T as $trait<B>>::OutputType>>
        where
            B: Backend,
            T: ContinuousQuantity<B> + $trait<B>,
            E: ToElement,
        {
            let elem = B::FloatElem::from_elem(float);
            let t0 = q.get_quantity_type();
            let t = t0.quantity_type(elem);
            let op = t0.op_type(elem);
            register_map1(t, op, q)
        }

        impl<B, T, E> $std_trait<E> for Q<B, T>
        where
            B: Backend,
            T: ContinuousQuantity<B> + $trait<B>,
            E: ToElement,
        {
            type Output = SkoroResult<Q<B, <T as $trait<B>>::OutputType>>;
            fn $method(self, float: E) -> Self::Output {
                $helper(&self, float)
            }
        }

        impl_q_literal_binop!(@lit $trait, $std_trait, $method, $helper, f32);
        impl_q_literal_binop!(@lit $trait, $std_trait, $method, $helper, f64);
    };

    (@lit $trait:ident, $std_trait:ident, $method:ident, $helper:ident, $float:ty) => {
        impl<B, T> $std_trait<Q<B, T>> for $float
        where
            B: Backend,
            T: ContinuousQuantity<B> + $trait<B>,
        {
            type Output = SkoroResult<Q<B, <T as $trait<B>>::OutputType>>;
            fn $method(self, rhs: Q<B, T>) -> Self::Output {
                $helper(&rhs, self)
            }
        }
    };
}

impl_q_literal_binop!(QLiteralAdd, Add, add, lit_add);
impl_q_literal_binop!(QLiteralSub, Sub, sub, lit_sub);
impl_q_literal_binop!(QLiteralMul, Mul, mul, lit_mul);
impl_q_literal_binop!(QLiteralDiv, Div, div, lit_div);
