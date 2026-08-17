//! Type-level tags distinguishing continuous and discrete quantities.
//!
//! Generic code is parameterized by a [`Kind`] type ([`Continuous`] or
//! [`Discrete`]), while type-erased contexts (graph nodes, dyn quantity types)
//! carry the runtime [`QuantityKind`] discriminator. Because the kind is a
//! *parameter* of the traits that use it (e.g. `QuantityType<B, K>`,
//! `Sampler<B, K>`), the blanket implementations for the two kinds live on
//! different trait instantiations and never conflict.

use skoro_backend::{
    Backend, DiscretePoint, Element, Point, UnbatchedDiscretePoint, UnbatchedPoint,
};

use crate::kind_erased_tensor::KindErasedTensor;

/// Runtime discriminator for the two quantity kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantityKind {
    /// A continuous (float-valued, differentiable) quantity.
    Continuous,
    /// A discrete (integer-valued, non-differentiable) quantity.
    Discrete,
}

mod private {
    pub trait Sealed {}
}

/// A type-level tag for a quantity kind.
///
/// Only [`Continuous`] and [`Discrete`] implement it.
pub trait Kind<B: Backend>: 'static + Send + Sync + std::fmt::Debug + private::Sealed {
    /// The runtime discriminator corresponding to this kind.
    const KIND: QuantityKind;
    /// The tensor element type that values of this kind pack into.
    type Elem: Element;
    /// The unbatched (rank-1) point type of this kind.
    type UnbatchedPoint: Clone;
    /// The batched (rank-2) point type of this kind.
    type Point: Clone;

    /// Wrap a kinded unbatched point into the erased [`KindErasedTensor`] operand.
    fn wrap_unbatched_point(point: Self::UnbatchedPoint) -> KindErasedTensor<B, 1>;

    /// Wrap a kinded point into the erased [`KindErasedTensor`] operand.
    fn wrap_point(point: Self::Point) -> KindErasedTensor<B, 2>;

    /// Borrow the kinded point inside an erased [`KindErasedTensor`], if the kind
    /// matches.
    fn as_point_ref(input: &KindErasedTensor<B, 2>) -> Option<&Self::Point>;

    /// Unwrap kinded point inside an erased [`KindErasedTensor`], if the kind
    /// matches.
    fn unwrap_point(input: KindErasedTensor<B, 2>) -> Option<Self::Point>;

    /// Extract the row-major elements of a kinded point.
    fn to_elements(point: &Self::Point) -> Vec<Self::Elem>;
}

/// The continuous kind: float-valued quantities with a manifold structure.
#[derive(Debug)]
pub struct Continuous;

/// The discrete kind: integer-valued, non-differentiable quantities.
#[derive(Debug)]
pub struct Discrete;

impl private::Sealed for Continuous {}
impl private::Sealed for Discrete {}

impl<B: Backend> Kind<B> for Continuous {
    const KIND: QuantityKind = QuantityKind::Continuous;
    type Elem = B::FloatElem;
    type UnbatchedPoint = UnbatchedPoint<B>;
    type Point = Point<B>;

    #[inline(always)]
    fn wrap_unbatched_point(point: Self::UnbatchedPoint) -> KindErasedTensor<B, 1> {
        KindErasedTensor::Float(point)
    }

    #[inline(always)]
    fn wrap_point(point: Self::Point) -> KindErasedTensor<B, 2> {
        KindErasedTensor::Float(point)
    }

    #[inline(always)]
    fn as_point_ref(input: &KindErasedTensor<B, 2>) -> Option<&Self::Point> {
        match input {
            KindErasedTensor::Float(t) => Some(t),
            _ => None,
        }
    }

    #[inline(always)]
    fn unwrap_point(input: KindErasedTensor<B, 2>) -> Option<Self::Point> {
        match input {
            KindErasedTensor::Float(t) => Some(t),
            _ => None,
        }
    }

    #[inline(always)]
    fn to_elements(point: &Self::Point) -> Vec<Self::Elem> {
        point
            .to_data()
            .to_vec::<B::FloatElem>()
            .expect("continuous points should have float elements")
    }
}

impl<B: Backend> Kind<B> for Discrete {
    const KIND: QuantityKind = QuantityKind::Discrete;
    type Elem = B::IntElem;
    type UnbatchedPoint = UnbatchedDiscretePoint<B>;
    type Point = DiscretePoint<B>;

    #[inline(always)]
    fn wrap_unbatched_point(point: Self::UnbatchedPoint) -> KindErasedTensor<B, 1> {
        KindErasedTensor::Int(point)
    }

    #[inline(always)]
    fn wrap_point(point: Self::Point) -> KindErasedTensor<B, 2> {
        KindErasedTensor::Int(point)
    }

    #[inline(always)]
    fn unwrap_point(input: KindErasedTensor<B, 2>) -> Option<Self::Point> {
        match input {
            KindErasedTensor::Int(t) => Some(t),
            _ => None,
        }
    }

    #[inline(always)]
    fn as_point_ref(input: &KindErasedTensor<B, 2>) -> Option<&Self::Point> {
        match input {
            KindErasedTensor::Int(t) => Some(t),
            _ => None,
        }
    }

    #[inline(always)]
    fn to_elements(point: &Self::Point) -> Vec<Self::Elem> {
        point
            .to_data()
            .to_vec::<B::IntElem>()
            .expect("discrete points should have integer elements")
    }
}
