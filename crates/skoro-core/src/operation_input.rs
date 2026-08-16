//! The shared operand type for the two operation tracks.
//!
//! A [`OperationInput`] is the value handed to a operation's `forward` method for each
//! input. It is either a continuous (float) point or a discrete (integer) point. Both the
//! differentiable (continuous-output) and discrete (discrete-output) operation tracks consume
//! [`OperationInput`] instances, so an operation can read inputs from either domain.
//!
//! The operations [`Add`], [`Sub`], [`Mul`], and [`Div`] only operate on the continuous type,
//! since most of the time we are operating on these values. Otherwise, one can unwrap.
//! For example, one can add two integer inputs via code like below
//!
//! ```rs
//! // where float-variants are as simple as `x + y`.
//! OperationInput::int(x.into_int() + y.into_int())
//! ```

use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Sub};

use skoro_backend::{Backend, DiscretePoint, Point};

use crate::error::{SkoroError, SkoroResult};
use crate::id::Id;

/// A map of node ids to batched values, continuous or discrete.
///
/// This is the single value map threaded through evaluation: both continuous (`Float`) and
/// discrete (`Int`) values travel under one key type.
pub type ValueMap<B> = HashMap<Id, OperationInput<B>>;

/// A single input to an operation: a continuous point or a discrete point.
#[derive(Debug, Clone)]
pub enum OperationInput<B: Backend> {
    /// A continuous value (`Tensor<B, 2>`, float-kind).
    Float(Point<B>),
    /// A discrete value (`Tensor<B, 2, Int>`).
    Int(DiscretePoint<B>),
}

impl<B: Backend> OperationInput<B> {
    /// Construct a continuous input.
    #[inline(always)]
    pub fn float(point: Point<B>) -> Self {
        OperationInput::Float(point)
    }

    /// Construct a discrete input.
    #[inline(always)]
    pub fn int(tensor: DiscretePoint<B>) -> Self {
        OperationInput::Int(tensor)
    }

    /// Whether this input is discrete.
    #[inline(always)]
    pub fn is_discrete(&self) -> bool {
        matches!(self, OperationInput::Int(_))
    }

    /// Borrow the continuous tensor if the input is continuous.
    #[inline(always)]
    pub fn as_float_opt(&self) -> Option<&Point<B>> {
        match self {
            OperationInput::Float(t) => Some(t),
            _ => None,
        }
    }

    /// Borrow the continuous tensor, panicking if the input is discrete.
    #[inline(always)]
    pub fn as_float(&self) -> &Point<B> {
        match self {
            OperationInput::Float(t) => t,
            OperationInput::Int(_) => {
                panic!("expected a continuous (float) operation input, found a discrete one")
            }
        }
    }

    /// Take the continuous tensor, panicking if the input is discrete.
    #[inline(always)]
    pub fn into_float(self) -> Point<B> {
        match self {
            OperationInput::Float(t) => t,
            OperationInput::Int(_) => {
                panic!("expected a continuous (float) operation input, found a discrete one")
            }
        }
    }

    /// Borrow the discrete tensor if the input is discrete.
    #[inline(always)]
    pub fn as_int_opt(&self) -> Option<&DiscretePoint<B>> {
        match self {
            OperationInput::Int(t) => Some(t),
            _ => None,
        }
    }

    /// Borrow the discrete tensor, panicking if the input is continuous.
    #[inline(always)]
    pub fn as_int(&self) -> &DiscretePoint<B> {
        match self {
            OperationInput::Int(t) => t,
            OperationInput::Float(_) => {
                panic!("expected a discrete (int) operation input, found a continuous one")
            }
        }
    }

    /// Take the discrete tensor, panicking if the input is continuous.
    #[inline(always)]
    pub fn into_int(self) -> DiscretePoint<B> {
        match self {
            OperationInput::Int(t) => t,
            OperationInput::Float(_) => {
                panic!("expected a discrete (int) operation input, found a continuous one")
            }
        }
    }

    /// The shape `[batch, components]` of the underlying tensor, regardless of
    /// kind.
    #[inline(always)]
    pub fn dims(&self) -> [usize; 2] {
        match self {
            OperationInput::Float(t) => t.dims(),
            OperationInput::Int(t) => t.dims(),
        }
    }

    /// This value, if its batch axis matches `batch_size`.
    ///
    /// The single batch check in the workspace. Everything that is *checked* rather than
    /// broadcast goes through here -- parameters, samplers, and batch-pinned constants -- so the
    /// three cannot come to disagree about what counts as a match or about which id gets blamed.
    #[inline(always)]
    pub fn ensure_batch(&self, batch_size: usize, id: Id) -> SkoroResult<Self> {
        let found = self.dims()[0];
        if found == batch_size {
            Ok(self.clone())
        } else {
            Err(SkoroError::BatchSizeMismatch {
                id,
                expected: batch_size,
                found,
            })
        }
    }

    /// The device the underlying tensor lives on, regardless of kind.
    #[inline(always)]
    pub fn device(&self) -> B::Device {
        match self {
            OperationInput::Float(t) => t.device(),
            OperationInput::Int(t) => t.device(),
        }
    }
}

impl<B: Backend> From<Point<B>> for OperationInput<B> {
    #[inline(always)]
    fn from(point: Point<B>) -> Self {
        OperationInput::Float(point)
    }
}

impl<B: Backend> From<DiscretePoint<B>> for OperationInput<B> {
    #[inline(always)]
    fn from(point: DiscretePoint<B>) -> Self {
        OperationInput::Int(point)
    }
}

/// Helper: extract the two float operands for a transparent arithmetic operator,
/// panicking with a clear message on any discrete operand.
#[inline(always)]
fn float_operands<B: Backend>(
    lhs: OperationInput<B>,
    rhs: OperationInput<B>,
) -> (Point<B>, Point<B>) {
    match (lhs, rhs) {
        (OperationInput::Float(a), OperationInput::Float(b)) => (a, b),
        _ => panic!(
            "arithmetic on a discrete operation input is not supported; \
             discrete operations belong to the discrete track (no implicit coercion)"
        ),
    }
}

/// As with all operations, this assumes both inputs were `Continuous` data.
impl<B: Backend> Add for OperationInput<B> {
    type Output = Point<B>;
    #[inline(always)]
    fn add(self, rhs: Self) -> Point<B> {
        let (a, b) = float_operands(self, rhs);
        a + b
    }
}

/// As with all operations, this assumes both inputs were `Continuous` data.
impl<B: Backend> Sub for OperationInput<B> {
    type Output = Point<B>;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Point<B> {
        let (a, b) = float_operands(self, rhs);
        a - b
    }
}

/// As with all operations, this assumes both inputs were `Continuous` data.
impl<B: Backend> Mul for OperationInput<B> {
    type Output = Point<B>;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Point<B> {
        let (a, b) = float_operands(self, rhs);
        a * b
    }
}

/// As with all operations, this assumes both inputs were `Continuous` data.
impl<B: Backend> Div for OperationInput<B> {
    type Output = Point<B>;
    #[inline(always)]
    fn div(self, rhs: Self) -> Point<B> {
        let (a, b) = float_operands(self, rhs);
        a / b
    }
}
