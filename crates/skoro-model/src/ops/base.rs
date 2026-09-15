use std::ops::{Add, Div, Mul, Neg, Sub};

use skoro_backend::{Backend, ToElement};
use skoro_core::error::SkoroResult;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_operation::diff::DiffOperation;
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

/// Negation operator trait for `Q`.
pub trait QNeg<B: Backend> {
    type OutputType;
    type OpType: DiffOperation<B>;
    fn quantity_type(&self) -> Self::OutputType;
    fn op_type(&self) -> Self::OpType;
    fn register(node: &Q<B, Self>) -> Q<B, Self::OutputType>
    where
        Self: Sized;
}

impl<B, T> Neg for Q<B, T>
where
    B: Backend,
    T: QNeg<B>,
{
    type Output = Q<B, <T as QNeg<B>>::OutputType>;
    fn neg(self) -> Self::Output {
        <T as QNeg<B>>::register(&self)
    }
}

/// Generate a binary `Q op Q` operator trait + a single `std::ops` impl on `Q`.
///
/// The trait is generalized over unbounded operand types (so it covers both continuous and discrete
/// operands). Each implementor supplies `quantity_type` to describe the output type, plus a
/// `register` method that builds the resulting node: continuous implementors go through
/// [`register_map2`](crate::map::register_map2), discrete-output implementors
/// through [`register_discrete_map2`](crate::map::register_discrete_map2).
macro_rules! impl_q_binop {
    ($trait:ident, $std_trait:ident, $method:ident) => {
        pub trait $trait<B: Backend, T> {
            type OutputType;
            fn quantity_type(&self, other: &T) -> Self::OutputType;

            /// Build the resulting node from the two operand proxies.
            ///
            /// The kernel is named here rather than behind an associated type: nothing reads it
            /// generically -- `register` is the implementor's own -- and the two families of
            /// operation, [`DiffOperation`](skoro_operation::diff::DiffOperation) and
            /// [`DiscreteOperation`](skoro_operation::discrete::DiscreteOperation), have
            /// no common bound this trait could carry.
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
    K: DiffOperation<B>,
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
    // Declaration of the trait
    (declare_trait: @lit $trait:ident) => {
        pub trait $trait<B: Backend>: ContinuousQuantity<B> {
            type OutputType: ContinuousQuantity<B>;
            type OpType: DiffOperation<B>;
            fn quantity_type(&self, float: f64) -> Self::OutputType;
            fn op_type(&self, float: f64) -> Self::OpType;
        }
    };

    // Implementation of helper
    (impl_helper: $helper:ident, @lit $trait:ident) => {
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
            let elem = float.to_f64();
            let t0 = q.get_quantity_type();
            let t = t0.quantity_type(elem);
            let op = t0.op_type(elem);
            register_map1(t, op, q)
        }

    };

    // Implementation of the trait on the quantity type.
    (impl_quantity: $helper:ident, @lit $trait:ident, $std_trait:ident, $method:ident) => {
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
    };

    // Implementation of the operation on the float type.
    (impl_float: @lit $trait:ident, $std_trait:ident, $method:ident, $helper:ident, $float:ty) => {
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

    // Macro when the trait is a symmetric operation (addition, multiplication)
    (symmetric: $trait:ident, $std_trait:ident, $method:ident, $helper:ident) => {
        impl_q_literal_binop!(declare_trait: @lit $trait);
        impl_q_literal_binop!(impl_helper: $helper, @lit $trait);
        impl_q_literal_binop!(impl_quantity: $helper, @lit $trait, $std_trait, $method);
        impl_q_literal_binop!(impl_float: @lit $trait, $std_trait, $method, $helper, f32);
        impl_q_literal_binop!(impl_float: @lit $trait, $std_trait, $method, $helper, f64);
    };

    // Macro when the trait is an asymmetric operation (subtraction, division)
    (asymmetric: $left_trait:ident, $right_trait:ident, $std_trait:ident, $method:ident, $left_helper:ident, $right_helper:ident) => {
        impl_q_literal_binop!(declare_trait: @lit $left_trait);
        impl_q_literal_binop!(declare_trait: @lit $right_trait);
        impl_q_literal_binop!(impl_helper: $left_helper, @lit $left_trait);
        impl_q_literal_binop!(impl_helper: $right_helper, @lit $right_trait);
        impl_q_literal_binop!(impl_quantity: $right_helper, @lit $right_trait, $std_trait, $method);
        impl_q_literal_binop!(impl_float: @lit $left_trait, $std_trait, $method, $left_helper, f32);
        impl_q_literal_binop!(impl_float: @lit $left_trait, $std_trait, $method, $left_helper, f64);
    };
}

impl_q_literal_binop!(symmetric: QLiteralAdd, Add, add, lit_add);
impl_q_literal_binop!(symmetric: QLiteralMul, Mul, mul, lit_mul);
impl_q_literal_binop!(asymmetric: QLiteralSubLeft, QLiteralSubRight, Sub, sub, lit_sub_left, lit_sub_right);
impl_q_literal_binop!(asymmetric: QLiteralDivLeft, QLiteralDivRight, Div, div, lit_div_left, lit_div_right);
