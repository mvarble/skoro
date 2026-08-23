//! The shared tensor type for the two kinds.

use std::collections::HashMap;

use skoro_backend::{Backend, Int, Tensor};

use crate::error::{SkoroError, SkoroResult};
use crate::id::Id;

/// A backend tensor enumeration that stores continuous and discrete variants, so that they are
/// opaque to the compiler and can be determined at runtime.
#[derive(Debug, Clone)]
pub enum AnyTensor<B: Backend, const D: usize> {
    /// A continuous kind `Tensor<B, D>`
    Float(Tensor<B, D>),
    /// A discrete kind `Tensor<B, 2, Int>`
    Int(Tensor<B, D, Int>),
}

impl<B: Backend, const D: usize> AnyTensor<B, D> {
    /// Construct a continuous input.
    #[inline(always)]
    pub fn float(point: Tensor<B, D>) -> Self {
        Self::Float(point)
    }

    /// Construct a discrete input.
    #[inline(always)]
    pub fn int(tensor: Tensor<B, D, Int>) -> Self {
        Self::Int(tensor)
    }

    /// Whether this input is discrete.
    #[inline(always)]
    pub fn is_discrete(&self) -> bool {
        matches!(self, Self::Int(_))
    }

    /// Borrow the continuous tensor if the input is continuous.
    #[inline(always)]
    pub fn as_float_opt(&self) -> Option<&Tensor<B, D>> {
        match self {
            Self::Float(t) => Some(t),
            _ => None,
        }
    }

    /// Borrow the continuous tensor, panicking if the input is discrete.
    #[inline(always)]
    pub fn unwrap_float_ref(&self) -> &Tensor<B, D> {
        match self {
            Self::Float(t) => t,
            Self::Int(_) => {
                panic!("expected a continuous (float) tensor, found a discrete one")
            }
        }
    }

    /// Take the continuous tensor, panicking if the input is discrete.
    #[inline(always)]
    pub fn unwrap_float(self) -> Tensor<B, D> {
        match self {
            Self::Float(t) => t,
            Self::Int(_) => {
                panic!("expected a continuous (float) tensor, found a discrete one")
            }
        }
    }

    /// Borrow the discrete tensor if the input is discrete.
    #[inline(always)]
    pub fn as_int_opt(&self) -> Option<&Tensor<B, D, Int>> {
        match self {
            Self::Int(t) => Some(t),
            _ => None,
        }
    }

    /// Borrow the discrete tensor, panicking if the input is continuous.
    #[inline(always)]
    pub fn unwrap_int_ref(&self) -> &Tensor<B, D, Int> {
        match self {
            Self::Int(t) => t,
            Self::Float(_) => {
                panic!("expected a discrete (int) tensor, found a continuous one")
            }
        }
    }

    /// Take the discrete tensor, panicking if the input is continuous.
    #[inline(always)]
    pub fn unwrap_int(self) -> Tensor<B, D, Int> {
        match self {
            Self::Int(t) => t,
            Self::Float(_) => {
                panic!("expected a discrete (int) tensor, found a continuous one")
            }
        }
    }

    /// The shape `[batch, components]` of the underlying tensor, regardless of
    /// kind.
    #[inline(always)]
    pub fn dims(&self) -> [usize; D] {
        match self {
            Self::Float(t) => t.dims(),
            Self::Int(t) => t.dims(),
        }
    }

    /// The device the underlying tensor lives on, regardless of kind.
    #[inline(always)]
    pub fn device(&self) -> B::Device {
        match self {
            Self::Float(t) => t.device(),
            Self::Int(t) => t.device(),
        }
    }
}

impl<B: Backend> AnyTensor<B, 2> {
    /// This value, if its batch axis matches `batch_size`.
    ///
    /// The single batch check in the workspace. Everything that is *checked* rather than
    /// broadcast goes through here -- parameters, samplers, and batch-pinned constants -- so the
    /// three cannot come to disagree about what counts as a match or about which id gets blamed.
    #[inline(always)]
    pub fn ensure_batch(&self, batch_size: usize) -> SkoroResult<Self> {
        let found = self.dims()[0];
        if found == batch_size {
            Ok(self.clone())
        } else {
            Err(SkoroError::BatchSizeMismatch {
                expected: batch_size,
                found,
            })
        }
    }

    /// Reduce a batched (rank-2) [`AnyTensor`] to a rank-1 one (i.e. squeeze).
    ///
    /// This assumes there is one batch.
    pub fn unbatch(&self) -> AnyTensor<B, 1> {
        match self {
            Self::Float(t) => AnyTensor::Float(t.clone().squeeze_dim::<1>(0)),
            Self::Int(t) => AnyTensor::Int(t.clone().squeeze_dim::<1>(0)),
        }
    }
}

impl<B: Backend> AnyTensor<B, 1> {
    /// Batch this stored (rank-1) [`AnyTensor`] into a rank-2 one.
    pub fn expand(&self, batch_size: usize) -> AnyTensor<B, 2> {
        match self {
            Self::Float(t) => {
                AnyTensor::Float(t.clone().unsqueeze_dim(0).repeat_dim(0, batch_size))
            }
            Self::Int(t) => AnyTensor::Int(t.clone().unsqueeze_dim(0).repeat_dim(0, batch_size)),
        }
    }
}

impl<B: Backend, const D: usize> From<Tensor<B, D>> for AnyTensor<B, D> {
    #[inline(always)]
    fn from(point: Tensor<B, D>) -> Self {
        Self::Float(point)
    }
}

impl<B: Backend, const D: usize> From<Tensor<B, D, Int>> for AnyTensor<B, D> {
    #[inline(always)]
    fn from(point: Tensor<B, D, Int>) -> Self {
        Self::Int(point)
    }
}

/// A map of [`AnyTensor`] instances of a fixed rank, by [`Id`].
pub type AnyTensorMap<B, const D: usize> = HashMap<Id, AnyTensor<B, D>>;

/// A stored quantity value, either batch-agnostic or batch-pinned.
///
/// [`Unbatched`](QuantityValue::Unbatched) is rank-1 storage that broadcasts to whatever batch a
/// pass runs at. That is the right default for a value shared by every row of a batch, and it is
/// what every constant was before this type existed.
///
/// [`Batched`](QuantityValue::Batched) is rank-2 and **pins** the pass to its batch size. It is how
/// a batch of `N` independent problems carries `N` different values -- one per row. A filter
/// running `N` multi-start optimizations needs `N` linearization points, not one broadcast across
/// all of them, and a model conditioned on per-row data needs `N` observations.
///
/// The distinction lives in the value rather than in a separate node kind, because a batch-pinned
/// constant plays exactly the same role in the graph as any other: concrete, non-differentiable,
/// no parents. Only the storage differs. Keeping it here is also what lets a mapped node whose
/// parents are batch-pinned still fold -- see `Graph::check_cache`.
#[derive(Clone, Debug)]
pub enum QuantityValue<B: Backend> {
    /// Rank-1 storage, broadcast to the running batch.
    Unbatched(AnyTensor<B, 1>),
    /// Rank-2 storage, which the running batch must match exactly.
    Batched(AnyTensor<B, 2>),
}

impl<B: Backend> QuantityValue<B> {
    /// The rank-2 operand this value contributes to a pass at `batch_size`.
    ///
    /// This is the single place the two representations diverge: `Unbatched` expands, `Batched`
    /// checks. Every read site goes through here, for the same reason both Jacobian paths go
    /// through `TangentLayout::identity_frames` -- a second, subtly different materialization is
    /// how two conventions drift apart without anything announcing it.
    ///
    /// A `Batched` value of batch 1 in a batch-`n` pass is an **error**, not a broadcast.
    /// Broadcasting is what `Unbatched` already is, and a rule where batch 1 broadcasts but batch
    /// `n` does not is the failure mode `PointMap::retract_flat_mut` guards against: a flat
    /// manifold's `Addition` broadcasts happily and returns a silently wrong answer.
    pub fn materialize(&self, batch_size: usize, id: Id) -> SkoroResult<AnyTensor<B, 2>> {
        match self {
            Self::Unbatched(value) => Ok(value.expand(batch_size)),
            Self::Batched(input) => input.ensure_batch(batch_size),
        }
    }

    /// `Some(n)` if this value can only be evaluated at batch `n`.
    pub fn pinned_batch(&self) -> Option<usize> {
        match self {
            Self::Unbatched(_) => None,
            Self::Batched(input) => Some(input.dims()[0]),
        }
    }

    /// Whether this value is discrete (integer-valued).
    pub fn is_discrete(&self) -> bool {
        match self {
            Self::Unbatched(value) => value.is_discrete(),
            Self::Batched(input) => input.is_discrete(),
        }
    }

    /// The rank-1 storage, if this value is batch-agnostic.
    ///
    /// The two variants hold tensors of different rank, so there is no accessor that serves both.
    /// A caller reaching for the value has to say which it expects.
    pub fn as_unbatched(&self) -> Option<&AnyTensor<B, 1>> {
        match self {
            Self::Unbatched(value) => Some(value),
            Self::Batched(_) => None,
        }
    }

    /// The rank-2 storage, if this value is batch-pinned.
    pub fn as_batched(&self) -> Option<&AnyTensor<B, 2>> {
        match self {
            Self::Unbatched(_) => None,
            Self::Batched(input) => Some(input),
        }
    }

    /// Number of components in the data (e.g. slice at last index)
    pub fn num_components(&self) -> usize {
        match self {
            Self::Unbatched(value) => value.dims()[0],
            Self::Batched(value) => value.dims()[1],
        }
    }
}

impl<B: Backend> From<Tensor<B, 1>> for QuantityValue<B> {
    fn from(value: Tensor<B, 1>) -> Self {
        Self::Unbatched(AnyTensor::Float(value))
    }
}

impl<B: Backend> From<Tensor<B, 1, Int>> for QuantityValue<B> {
    fn from(value: Tensor<B, 1, Int>) -> Self {
        Self::Unbatched(AnyTensor::Int(value))
    }
}

impl<B: Backend> From<Tensor<B, 2>> for QuantityValue<B> {
    fn from(value: Tensor<B, 2>) -> Self {
        Self::Batched(AnyTensor::Float(value))
    }
}

impl<B: Backend> From<Tensor<B, 2, Int>> for QuantityValue<B> {
    fn from(value: Tensor<B, 2, Int>) -> Self {
        Self::Batched(AnyTensor::Int(value))
    }
}
