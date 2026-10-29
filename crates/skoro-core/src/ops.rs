//! What an operator means, decided by the markers of its operands through traits any crate may
//! implement for a marker of its own; and the methods and functions for the operations that are
//! not operators.
//!
//! `std::ops::Mul` is `std`'s and `Q` is the core's, so a feature crate cannot implement `Mul`
//! for `Q<Complex>`; it can implement `QMul` for `Complex`, and the one generic `Mul` on `Q` gives
//! `*` that meaning. The core implements the traits for each of its concrete markers, never as a
//! blanket over a type parameter, so a feature crate's impl never collides with one of the core's.

use std::ops::Range;
use std::panic::Location;
use std::sync::Arc;

use crate::primitive::{
    Add, Atan2, Concatenate, Cos, Exp, Include, IntegerAdd, IntegerEq, IntegerLe, IntegerLt,
    IntegerMul, IntegerSub, IsNonnegative, Log, MatMul, Mul, Neg, PositiveAsReal, PositiveExp,
    PositiveLog, PositiveMul, PositiveRecip, Project, Scale, Select, Sin, Slice, Sub, Sum, ToReal,
    Transpose,
};
use crate::scope::{self, Draft, literal};
use crate::{
    Boolean, Categorical, CategoricalDyn, Declares, Error, Integer, Matrix, MatrixDyn, NodeId,
    Operation, Positive, PositiveDyn, ProductDyn, Q, QDyn, Real, Space, Typed, Vector, VectorDyn,
};

/// `lhs + rhs`.
pub trait QAdd<Rhs: Typed = Self>: Typed {
    type Output: Typed;
    fn add(lhs: Q<Self>, rhs: Q<Rhs>) -> Q<Self::Output>;
}

/// `lhs - rhs`.
pub trait QSub<Rhs: Typed = Self>: Typed {
    type Output: Typed;
    fn sub(lhs: Q<Self>, rhs: Q<Rhs>) -> Q<Self::Output>;
}

/// `lhs * rhs`.
pub trait QMul<Rhs: Typed = Self>: Typed {
    type Output: Typed;
    fn mul(lhs: Q<Self>, rhs: Q<Rhs>) -> Q<Self::Output>;
}

/// `lhs / rhs`.
pub trait QDiv<Rhs: Typed = Self>: Typed {
    type Output: Typed;
    fn div(lhs: Q<Self>, rhs: Q<Rhs>) -> Q<Self::Output>;
}

/// `-q`.
pub trait QNeg: Typed {
    type Output: Typed;
    fn neg(q: Q<Self>) -> Q<Self::Output>;
}

/// A literal on either side of `*`, or on the right of `/`.
pub trait QScale: Typed {
    fn scale(c: f64, q: Q<Self>) -> Q<Self>;
}

/// A literal on either side of `+` or `-`.
pub trait QShift: Typed {
    fn shift(c: f64, q: Q<Self>) -> Q<Self>;
}

impl<T: QAdd<U>, U: Typed> std::ops::Add<Q<U>> for Q<T> {
    type Output = Q<T::Output>;
    #[track_caller]
    fn add(self, rhs: Q<U>) -> Q<T::Output> {
        T::add(self, rhs)
    }
}

impl<T: QSub<U>, U: Typed> std::ops::Sub<Q<U>> for Q<T> {
    type Output = Q<T::Output>;
    #[track_caller]
    fn sub(self, rhs: Q<U>) -> Q<T::Output> {
        T::sub(self, rhs)
    }
}

impl<T: QMul<U>, U: Typed> std::ops::Mul<Q<U>> for Q<T> {
    type Output = Q<T::Output>;
    #[track_caller]
    fn mul(self, rhs: Q<U>) -> Q<T::Output> {
        T::mul(self, rhs)
    }
}

impl<T: QDiv<U>, U: Typed> std::ops::Div<Q<U>> for Q<T> {
    type Output = Q<T::Output>;
    #[track_caller]
    fn div(self, rhs: Q<U>) -> Q<T::Output> {
        T::div(self, rhs)
    }
}

impl<T: QNeg> std::ops::Neg for Q<T> {
    type Output = Q<T::Output>;
    #[track_caller]
    fn neg(self) -> Q<T::Output> {
        T::neg(self)
    }
}

impl<T: QScale> std::ops::Mul<f64> for Q<T> {
    type Output = Q<T>;
    #[track_caller]
    fn mul(self, c: f64) -> Q<T> {
        T::scale(c, self)
    }
}

impl<T: QScale> std::ops::Mul<Q<T>> for f64 {
    type Output = Q<T>;
    #[track_caller]
    fn mul(self, q: Q<T>) -> Q<T> {
        T::scale(self, q)
    }
}

impl<T: QScale> std::ops::Div<f64> for Q<T> {
    type Output = Q<T>;
    #[track_caller]
    fn div(self, c: f64) -> Q<T> {
        T::scale(1.0 / c, self)
    }
}

impl<T: QShift> std::ops::Add<f64> for Q<T> {
    type Output = Q<T>;
    #[track_caller]
    fn add(self, c: f64) -> Q<T> {
        T::shift(c, self)
    }
}

impl<T: QShift> std::ops::Add<Q<T>> for f64 {
    type Output = Q<T>;
    #[track_caller]
    fn add(self, q: Q<T>) -> Q<T> {
        T::shift(self, q)
    }
}

impl<T: QShift> std::ops::Sub<f64> for Q<T> {
    type Output = Q<T>;
    #[track_caller]
    fn sub(self, c: f64) -> Q<T> {
        T::shift(-c, self)
    }
}

impl<T: QShift + QNeg<Output = T>> std::ops::Sub<Q<T>> for f64 {
    type Output = Q<T>;
    #[track_caller]
    fn sub(self, q: Q<T>) -> Q<T> {
        T::shift(self, T::neg(q))
    }
}

/// One node of `operation` at `inputs`, built at the caller's line.
#[track_caller]
fn node(operation: impl Operation, inputs: &[QDyn]) -> NodeId {
    let operation: Arc<dyn Operation> = Arc::new(operation);
    scope::build(Location::caller(), |draft, _| {
        draft.mapped(operation, inputs)
    })
}

#[track_caller]
fn typed<T: Typed>(operation: impl Operation, inputs: &[QDyn]) -> Q<T> {
    Q::from_id(node(operation, inputs))
}

/// Builds at the caller's line with the draft and the inputs' spaces in hand.
#[track_caller]
fn with_spaces(
    inputs: &[QDyn],
    step: impl FnOnce(&mut Draft, &[Space]) -> Result<NodeId, Error>,
) -> NodeId {
    scope::build(Location::caller(), |draft, _| {
        let spaces = draft.spaces(inputs)?;
        step(draft, &spaces)
    })
}

fn mismatch(operation: &'static str, inputs: &[Space]) -> Error {
    Error::SpaceMismatch {
        operation,
        inputs: inputs.to_vec(),
    }
}

fn is_euclidean(space: &Space) -> bool {
    matches!(space, Space::Real | Space::Vector(_) | Space::Matrix(_, _))
}

/// `c q`, a literal times a Euclidean value: `scale` of a `Real` constant.
fn scale_by(draft: &mut Draft, c: f64, q: QDyn) -> Result<NodeId, Error> {
    let c = literal(draft, Space::Real, c)?;
    draft.mapped(Arc::new(Scale), &[QDyn::from_id(c), q])
}

/// `q + c`, a Euclidean value plus a literal: `add` of a constant of its space.
fn shift_by(draft: &mut Draft, space: &Space, c: f64, q: QDyn) -> Result<NodeId, Error> {
    let c = literal(draft, space.clone(), c)?;
    draft.mapped(Arc::new(Add), &[q, QDyn::from_id(c)])
}

/// `c p`, a literal times a positive value: `positive.mul` of a positive constant.
fn positive_scale_by(draft: &mut Draft, space: &Space, c: f64, q: QDyn) -> Result<NodeId, Error> {
    let c = literal(draft, space.clone(), c)?;
    draft.mapped(Arc::new(PositiveMul), &[QDyn::from_id(c), q])
}

/// `p / q` for positive values: `positive.mul` of `p` and `positive.recip(q)`.
fn positive_div(draft: &mut Draft, p: QDyn, q: QDyn) -> Result<NodeId, Error> {
    let recip = draft.mapped(Arc::new(PositiveRecip), &[q])?;
    draft.mapped(Arc::new(PositiveMul), &[p, QDyn::from_id(recip)])
}

/// The operators of a Euclidean marker, on itself, with literals, and with `Real` by `scale`.
macro_rules! euclidean {
    ($([$($generics:tt)*] $marker:ty),* $(,)?) => {$(
        impl<$($generics)*> QAdd for $marker {
            type Output = $marker;
            #[track_caller]
            fn add(lhs: Q<Self>, rhs: Q<Self>) -> Q<Self> {
                typed(Add, &[lhs.untyped(), rhs.untyped()])
            }
        }

        impl<$($generics)*> QSub for $marker {
            type Output = $marker;
            #[track_caller]
            fn sub(lhs: Q<Self>, rhs: Q<Self>) -> Q<Self> {
                typed(Sub, &[lhs.untyped(), rhs.untyped()])
            }
        }

        impl<$($generics)*> QMul for $marker {
            type Output = $marker;
            #[track_caller]
            fn mul(lhs: Q<Self>, rhs: Q<Self>) -> Q<Self> {
                typed(Mul, &[lhs.untyped(), rhs.untyped()])
            }
        }

        impl<$($generics)*> QNeg for $marker {
            type Output = $marker;
            #[track_caller]
            fn neg(q: Q<Self>) -> Q<Self> {
                typed(Neg, &[q.untyped()])
            }
        }

        impl<$($generics)*> QScale for $marker {
            #[track_caller]
            fn scale(c: f64, q: Q<Self>) -> Q<Self> {
                Q::from_id(with_spaces(&[q.untyped()], |draft, _| scale_by(draft, c, q.untyped())))
            }
        }

        impl<$($generics)*> QShift for $marker {
            #[track_caller]
            fn shift(c: f64, q: Q<Self>) -> Q<Self> {
                Q::from_id(with_spaces(&[q.untyped()], |draft, spaces| {
                    shift_by(draft, &spaces[0], c, q.untyped())
                }))
            }
        }

        impl<$($generics)*> Q<$marker> {
            #[track_caller]
            pub fn exp(self) -> Q<$marker> {
                typed(Exp, &[self.untyped()])
            }

            #[track_caller]
            pub fn log(self) -> Q<$marker> {
                typed(Log, &[self.untyped()])
            }

            #[track_caller]
            pub fn sin(self) -> Q<$marker> {
                typed(Sin, &[self.untyped()])
            }

            #[track_caller]
            pub fn cos(self) -> Q<$marker> {
                typed(Cos, &[self.untyped()])
            }

            /// `atan2(self, b)`, elementwise: the angle of the point `(b, self)`.
            #[track_caller]
            pub fn atan2(self, b: Q<$marker>) -> Q<$marker> {
                typed(Atan2, &[self.untyped(), b.untyped()])
            }

            #[track_caller]
            pub fn sum(self) -> Q<Real> {
                typed(Sum, &[self.untyped()])
            }

            /// The entries `range` of storage, as a vector.
            #[track_caller]
            pub fn slice(self, range: Range<usize>) -> Q<VectorDyn> {
                let space = Space::Vector(range.len());
                typed(Slice::new(space, range.clone(), range), &[self.untyped()])
            }

            /// The same entries, in storage order, as a value of `to`'s space.
            #[track_caller]
            pub fn reshape<U: Declares>(self, to: U) -> Q<U> {
                typed(Concatenate::new(to.space()), &[self.untyped()])
            }
        }
    )*};
}

euclidean!(
    [] Real,
    [const N: usize] Vector<N>,
    [] VectorDyn,
    [const M: usize, const N: usize] Matrix<M, N>,
    [] MatrixDyn,
);

/// `s * a` and `a * s` for a `Real` `s` and a Euclidean marker other than `Real`.
macro_rules! scaled_by_real {
    ($([$($generics:tt)*] $marker:ty),* $(,)?) => {$(
        impl<$($generics)*> QMul<$marker> for Real {
            type Output = $marker;
            #[track_caller]
            fn mul(lhs: Q<Real>, rhs: Q<$marker>) -> Q<$marker> {
                typed(Scale, &[lhs.untyped(), rhs.untyped()])
            }
        }

        impl<$($generics)*> QMul<Real> for $marker {
            type Output = $marker;
            #[track_caller]
            fn mul(lhs: Q<$marker>, rhs: Q<Real>) -> Q<$marker> {
                typed(Scale, &[rhs.untyped(), lhs.untyped()])
            }
        }
    )*};
}

scaled_by_real!(
    [const N: usize] Vector<N>,
    [] VectorDyn,
    [const M: usize, const N: usize] Matrix<M, N>,
    [] MatrixDyn,
);

/// The operators of a positive marker: products, quotients, and literal factors.
macro_rules! positive {
    ($([$($generics:tt)*] $marker:ty => $real:ty),* $(,)?) => {$(
        impl<$($generics)*> QMul for $marker {
            type Output = $marker;
            #[track_caller]
            fn mul(lhs: Q<Self>, rhs: Q<Self>) -> Q<Self> {
                typed(PositiveMul, &[lhs.untyped(), rhs.untyped()])
            }
        }

        impl<$($generics)*> QDiv for $marker {
            type Output = $marker;
            #[track_caller]
            fn div(lhs: Q<Self>, rhs: Q<Self>) -> Q<Self> {
                Q::from_id(with_spaces(&[lhs.untyped(), rhs.untyped()], |draft, _| {
                    positive_div(draft, lhs.untyped(), rhs.untyped())
                }))
            }
        }

        impl<$($generics)*> QScale for $marker {
            /// `c p`, refusing a `c` that is not positive (`ValueShape`).
            #[track_caller]
            fn scale(c: f64, q: Q<Self>) -> Q<Self> {
                Q::from_id(with_spaces(&[q.untyped()], |draft, spaces| {
                    positive_scale_by(draft, &spaces[0], c, q.untyped())
                }))
            }
        }

        impl<$($generics)*> Q<$marker> {
            /// The value read as a real.
            #[track_caller]
            pub fn as_real(self) -> Q<$real> {
                typed(PositiveAsReal, &[self.untyped()])
            }

            #[track_caller]
            pub fn log(self) -> Q<$real> {
                typed(PositiveLog, &[self.untyped()])
            }

            #[track_caller]
            pub fn recip(self) -> Q<$marker> {
                typed(PositiveRecip, &[self.untyped()])
            }
        }
    )*};
}

positive!(
    [const N: usize] Positive<N> => Vector<N>,
    [] PositiveDyn => VectorDyn,
);

impl<const N: usize> Q<Vector<N>> {
    /// `e^c`, a positive value.
    #[track_caller]
    pub fn exp_positive(self) -> Q<Positive<N>> {
        typed(PositiveExp, &[self.untyped()])
    }
}

impl Q<VectorDyn> {
    /// `e^c`, a positive value.
    #[track_caller]
    pub fn exp_positive(self) -> Q<PositiveDyn> {
        typed(PositiveExp, &[self.untyped()])
    }
}

impl Q<Real> {
    /// `e^c`, a positive value.
    #[track_caller]
    pub fn exp_positive(self) -> Q<Positive<1>> {
        typed(PositiveExp, &[self.untyped()])
    }

    /// `1[self ≥ 0]`: a discretization, passing no cotangent back.
    #[track_caller]
    pub fn is_nonnegative(self) -> Q<Boolean> {
        typed(IsNonnegative, &[self.untyped()])
    }
}

/// The matrix product's output marker, for each pair of markers it takes.
pub trait QMatMul<Rhs: Typed>: Typed {
    type Output: Typed;
}

impl<const M: usize, const K: usize, const N: usize> QMatMul<Matrix<K, N>> for Matrix<M, K> {
    type Output = Matrix<M, N>;
}

impl<const M: usize, const K: usize> QMatMul<Vector<K>> for Matrix<M, K> {
    type Output = Vector<M>;
}

impl QMatMul<MatrixDyn> for MatrixDyn {
    type Output = MatrixDyn;
}

impl QMatMul<VectorDyn> for MatrixDyn {
    type Output = VectorDyn;
}

impl<T: Typed> Q<T> {
    /// `self b`, a matrix times a matrix or a vector.
    #[track_caller]
    pub fn matmul<U: Typed>(self, b: Q<U>) -> Q<T::Output>
    where
        T: QMatMul<U>,
    {
        typed(MatMul, &[self.untyped(), b.untyped()])
    }
}

impl<const M: usize, const N: usize> Q<Matrix<M, N>> {
    #[track_caller]
    pub fn transpose(self) -> Q<Matrix<N, M>> {
        typed(Transpose, &[self.untyped()])
    }
}

impl Q<MatrixDyn> {
    #[track_caller]
    pub fn transpose(self) -> Q<MatrixDyn> {
        typed(Transpose, &[self.untyped()])
    }
}

impl QAdd for Integer {
    type Output = Integer;
    #[track_caller]
    fn add(lhs: Q<Integer>, rhs: Q<Integer>) -> Q<Integer> {
        typed(IntegerAdd, &[lhs.untyped(), rhs.untyped()])
    }
}

impl QSub for Integer {
    type Output = Integer;
    #[track_caller]
    fn sub(lhs: Q<Integer>, rhs: Q<Integer>) -> Q<Integer> {
        typed(IntegerSub, &[lhs.untyped(), rhs.untyped()])
    }
}

impl QMul for Integer {
    type Output = Integer;
    #[track_caller]
    fn mul(lhs: Q<Integer>, rhs: Q<Integer>) -> Q<Integer> {
        typed(IntegerMul, &[lhs.untyped(), rhs.untyped()])
    }
}

impl Q<Integer> {
    /// `self = j`.
    #[track_caller]
    pub fn eq(self, j: Q<Integer>) -> Q<Boolean> {
        typed(IntegerEq, &[self.untyped(), j.untyped()])
    }

    /// `self < j`.
    #[track_caller]
    pub fn lt(self, j: Q<Integer>) -> Q<Boolean> {
        typed(IntegerLt, &[self.untyped(), j.untyped()])
    }

    /// `self ≤ j`.
    #[track_caller]
    pub fn le(self, j: Q<Integer>) -> Q<Boolean> {
        typed(IntegerLe, &[self.untyped(), j.untyped()])
    }
}

/// `to_real` on every discrete marker.
macro_rules! to_real {
    ($([$($generics:tt)*] $marker:ty),* $(,)?) => {$(
        impl<$($generics)*> Q<$marker> {
            /// The value read as a real: an embedding, with no derivative.
            #[track_caller]
            pub fn to_real(self) -> Q<Real> {
                typed(ToReal, &[self.untyped()])
            }
        }
    )*};
}

to_real!(
    [] Integer,
    [const N: usize] Categorical<N>,
    [] CategoricalDyn,
    [] Boolean,
);

impl<T: Typed> Q<T> {
    /// `self ⊕ t`, by the retraction of the space of `self`, with `t` in its tangent
    /// coordinates, a `Vector` of its dimension.
    #[track_caller]
    pub fn retract(self, t: impl Into<QDyn>) -> Q<T> {
        Q::from_id(self.untyped().retract(t.into()).id())
    }

    /// `self ⊖ from`, by the inverse retraction of the space of `self`: the tangent coordinates
    /// at `from` that reach `self`.
    #[track_caller]
    pub fn local(self, from: Q<T>) -> Q<VectorDyn> {
        Q::from_id(self.untyped().local(from.untyped()).id())
    }

    /// An embedded value as the vector of its storage, `ι(self)` (`SpaceMismatch` on a value of
    /// any other space).
    #[track_caller]
    pub fn include(self) -> Q<VectorDyn> {
        Q::from_id(self.untyped().include().id())
    }
}

impl Q<ProductDyn> {
    /// The `i`-th factor of a product (`SpaceMismatch` when there is none).
    #[track_caller]
    pub fn factor(self, i: usize) -> QDyn {
        self.untyped().factor(i)
    }
}

impl QDyn {
    /// The `i`-th factor of a product (`SpaceMismatch` when there is none).
    #[track_caller]
    pub fn factor(self, i: usize) -> QDyn {
        QDyn::from_id(with_spaces(&[self], |draft, spaces| {
            let block = match &spaces[0] {
                Space::Product(_) => spaces[0].blocks().into_iter().nth(i),
                _ => None,
            };
            let block = block.ok_or_else(|| mismatch("factor", spaces))?;
            let slice = Slice::new(block.space, block.storage, block.tangent);
            draft.mapped(Arc::new(slice), &[self])
        }))
    }

    /// `self ⊕ t`, by the retraction of the space of `self`.
    #[track_caller]
    pub fn retract(self, t: QDyn) -> QDyn {
        QDyn::from_id(with_spaces(&[self, t], |draft, spaces| {
            let retraction = spaces[0].retraction()?;
            draft.mapped(retraction.operation(), &[self, t])
        }))
    }

    /// `self ⊖ from`, by the inverse retraction of the space of `self`.
    #[track_caller]
    pub fn local(self, from: QDyn) -> QDyn {
        QDyn::from_id(with_spaces(&[self, from], |draft, spaces| {
            let inverse = spaces[0].inverse_retraction()?;
            draft.mapped(inverse.operation(), &[self, from])
        }))
    }

    /// An embedded value as the vector of its storage, `ι(self)` (`SpaceMismatch` on a value of
    /// any other space).
    #[track_caller]
    pub fn include(self) -> QDyn {
        QDyn::from_id(with_spaces(&[self], |draft, spaces| {
            let Space::Embedded(declaration) = &spaces[0] else {
                return Err(mismatch("skoro.include", spaces));
            };
            draft.mapped(Arc::new(Include::new(declaration.clone())), &[self])
        }))
    }

    /// The same entries, in storage order, as a value of `space`.
    #[track_caller]
    pub fn reshape(self, space: Space) -> QDyn {
        QDyn::from_id(node(Concatenate::new(space), &[self]))
    }

    /// The entries `range` of a Euclidean value's storage, as a vector.
    #[track_caller]
    pub fn slice(self, range: Range<usize>) -> QDyn {
        let space = Space::Vector(range.len());
        QDyn::from_id(node(Slice::new(space, range.clone(), range), &[self]))
    }

    /// `self b`, a matrix times a matrix or a vector.
    #[track_caller]
    pub fn matmul(self, b: QDyn) -> QDyn {
        QDyn::from_id(node(MatMul, &[self, b]))
    }

    #[track_caller]
    pub fn transpose(self) -> QDyn {
        QDyn::from_id(node(Transpose, &[self]))
    }

    #[track_caller]
    pub fn sum(self) -> QDyn {
        QDyn::from_id(node(Sum, &[self]))
    }

    #[track_caller]
    pub fn exp(self) -> QDyn {
        QDyn::from_id(node(Exp, &[self]))
    }

    /// `log`: elementwise on a Euclidean value, `positive.log` on a positive one.
    #[track_caller]
    pub fn log(self) -> QDyn {
        QDyn::from_id(with_spaces(&[self], |draft, spaces| {
            let operation: Arc<dyn Operation> = match spaces[0] {
                Space::Positive(_) => Arc::new(PositiveLog),
                _ => Arc::new(Log),
            };
            draft.mapped(operation, &[self])
        }))
    }

    #[track_caller]
    pub fn sin(self) -> QDyn {
        QDyn::from_id(node(Sin, &[self]))
    }

    #[track_caller]
    pub fn cos(self) -> QDyn {
        QDyn::from_id(node(Cos, &[self]))
    }

    #[track_caller]
    pub fn atan2(self, b: QDyn) -> QDyn {
        QDyn::from_id(node(Atan2, &[self, b]))
    }

    #[track_caller]
    pub fn as_real(self) -> QDyn {
        QDyn::from_id(node(PositiveAsReal, &[self]))
    }

    #[track_caller]
    pub fn exp_positive(self) -> QDyn {
        QDyn::from_id(node(PositiveExp, &[self]))
    }

    #[track_caller]
    pub fn recip(self) -> QDyn {
        QDyn::from_id(node(PositiveRecip, &[self]))
    }

    #[track_caller]
    pub fn to_real(self) -> QDyn {
        QDyn::from_id(node(ToReal, &[self]))
    }

    #[track_caller]
    pub fn is_nonnegative(self) -> QDyn {
        QDyn::from_id(node(IsNonnegative, &[self]))
    }
}

/// The operation of a binary operator on the core's spaces, and its inputs in order; `None` for
/// spaces it has no meaning on, a feature crate's among them.
fn binary(operator: char, a: &Space, b: &Space) -> Option<(Arc<dyn Operation>, bool)> {
    let same = a == b;
    Some(match (operator, a, b) {
        ('+', Space::Integer, Space::Integer) => (Arc::new(IntegerAdd), false),
        ('-', Space::Integer, Space::Integer) => (Arc::new(IntegerSub), false),
        ('*', Space::Integer, Space::Integer) => (Arc::new(IntegerMul), false),
        ('+', a, _) if same && is_euclidean(a) => (Arc::new(Add), false),
        ('-', a, _) if same && is_euclidean(a) => (Arc::new(Sub), false),
        ('*', a, _) if same && is_euclidean(a) => (Arc::new(Mul), false),
        ('*', Space::Real, b) if is_euclidean(b) => (Arc::new(Scale), false),
        ('*', a, Space::Real) if is_euclidean(a) => (Arc::new(Scale), true),
        ('*', Space::Positive(_), Space::Positive(_)) if same => (Arc::new(PositiveMul), false),
        _ => return None,
    })
}

#[track_caller]
fn dyn_binary(operator: char, name: &'static str, a: QDyn, b: QDyn) -> QDyn {
    QDyn::from_id(with_spaces(&[a, b], |draft, spaces| {
        if operator == '/' {
            return match spaces {
                [Space::Positive(_), Space::Positive(_)] if spaces[0] == spaces[1] => {
                    positive_div(draft, a, b)
                }
                _ => Err(mismatch(name, spaces)),
            };
        }
        let (operation, swap) =
            binary(operator, &spaces[0], &spaces[1]).ok_or_else(|| mismatch(name, spaces))?;
        draft.mapped(operation, &if swap { [b, a] } else { [a, b] })
    }))
}

impl std::ops::Add for QDyn {
    type Output = QDyn;
    #[track_caller]
    fn add(self, rhs: QDyn) -> QDyn {
        dyn_binary('+', "+", self, rhs)
    }
}

impl std::ops::Sub for QDyn {
    type Output = QDyn;
    #[track_caller]
    fn sub(self, rhs: QDyn) -> QDyn {
        dyn_binary('-', "-", self, rhs)
    }
}

impl std::ops::Mul for QDyn {
    type Output = QDyn;
    #[track_caller]
    fn mul(self, rhs: QDyn) -> QDyn {
        dyn_binary('*', "*", self, rhs)
    }
}

impl std::ops::Div for QDyn {
    type Output = QDyn;
    #[track_caller]
    fn div(self, rhs: QDyn) -> QDyn {
        dyn_binary('/', "/", self, rhs)
    }
}

impl std::ops::Neg for QDyn {
    type Output = QDyn;
    #[track_caller]
    fn neg(self) -> QDyn {
        QDyn::from_id(with_spaces(&[self], |draft, spaces| {
            if !is_euclidean(&spaces[0]) {
                return Err(mismatch("-", spaces));
            }
            draft.mapped(Arc::new(Neg), &[self])
        }))
    }
}

/// `c q` for a literal `c`: `scale` on a Euclidean value, a positive constant on a positive one.
#[track_caller]
fn dyn_scale(c: f64, q: QDyn) -> QDyn {
    QDyn::from_id(with_spaces(&[q], |draft, spaces| match &spaces[0] {
        space if is_euclidean(space) => scale_by(draft, c, q),
        space @ Space::Positive(_) => positive_scale_by(draft, space, c, q),
        _ => Err(mismatch("*", spaces)),
    }))
}

/// `q + c` for a literal `c`, on a Euclidean value.
#[track_caller]
fn dyn_shift(c: f64, q: QDyn) -> QDyn {
    QDyn::from_id(with_spaces(&[q], |draft, spaces| match &spaces[0] {
        space if is_euclidean(space) => shift_by(draft, space, c, q),
        _ => Err(mismatch("+", spaces)),
    }))
}

impl std::ops::Mul<f64> for QDyn {
    type Output = QDyn;
    #[track_caller]
    fn mul(self, c: f64) -> QDyn {
        dyn_scale(c, self)
    }
}

impl std::ops::Mul<QDyn> for f64 {
    type Output = QDyn;
    #[track_caller]
    fn mul(self, q: QDyn) -> QDyn {
        dyn_scale(self, q)
    }
}

impl std::ops::Div<f64> for QDyn {
    type Output = QDyn;
    #[track_caller]
    fn div(self, c: f64) -> QDyn {
        dyn_scale(1.0 / c, self)
    }
}

impl std::ops::Add<f64> for QDyn {
    type Output = QDyn;
    #[track_caller]
    fn add(self, c: f64) -> QDyn {
        dyn_shift(c, self)
    }
}

impl std::ops::Add<QDyn> for f64 {
    type Output = QDyn;
    #[track_caller]
    fn add(self, q: QDyn) -> QDyn {
        dyn_shift(self, q)
    }
}

impl std::ops::Sub<f64> for QDyn {
    type Output = QDyn;
    #[track_caller]
    fn sub(self, c: f64) -> QDyn {
        dyn_shift(-c, self)
    }
}

/// Values laid end to end as a value of `space`, a Euclidean space or a product.
#[track_caller]
pub fn concatenate(parts: &[QDyn], space: Space) -> QDyn {
    QDyn::from_id(node(Concatenate::new(space), parts))
}

/// `π(w)`: a vector of storage projected onto the embedded type of `marker` (`NoProjection` when
/// the type declares none, `SpaceMismatch` when `marker`'s space is not embedded or `w` is not a
/// vector of its storage).
#[track_caller]
pub fn project<T: Declares>(marker: T, w: impl Into<QDyn>) -> Q<T> {
    let (space, w) = (marker.space(), w.into());
    Q::from_id(with_spaces(&[w], |draft, spaces| {
        let Space::Embedded(declaration) = &space else {
            return Err(mismatch("skoro.project", spaces));
        };
        draft.mapped(Arc::new(Project::new(declaration.clone())), &[w])
    }))
}

/// The value of `values` that `index` chooses, row by row.
#[track_caller]
pub fn select<T: Typed>(index: impl Into<QDyn>, values: &[Q<T>]) -> Q<T> {
    let inputs: Vec<QDyn> = std::iter::once(index.into())
        .chain(values.iter().map(|q| q.untyped()))
        .collect();
    typed(Select, &inputs)
}

/// The product of `parts`, each a factor (`NotContinuous` for a discrete one).
#[track_caller]
pub fn product(parts: &[QDyn]) -> Q<ProductDyn> {
    Q::from_id(with_spaces(parts, |draft, spaces| {
        let space = Space::product(spaces.iter().cloned())?;
        draft.mapped(Arc::new(Concatenate::new(space)), parts)
    }))
}
