use std::fmt::Debug;

use skoro_backend::{Backend, UnbatchedDiscretePoint, UnbatchedPoint};

use crate::error::SkoroResult;
use crate::id::Id;
use crate::operation_input::OperationInput;

/// A quantity value, which is either a continuous (float) tensor or a discrete (integer) tensor.
///
/// This is the storage form (rank 1, unbatched) of a quantity's value when the quantity is
/// constant. All functions map (via [`DiffOperation`](skoro_operation::diff::DiffOperation) or
/// [`DiscreteOperation`](skoro_operation::discrete::DiscreteOperation)) over batches of quantity
/// values (via [`Input`](OperationInput)).
#[derive(Clone, Debug)]
pub enum ConstantValue<B: Backend> {
    /// A continuous value packed into the backend float element type.
    Float(UnbatchedPoint<B>),
    /// A discrete value packed into the backend integer element type.
    Int(UnbatchedDiscretePoint<B>),
}

impl<B: Backend> ConstantValue<B> {
    /// Returns the contained float tensor, panicking if the value is discrete.
    pub fn unwrap_float(self) -> UnbatchedPoint<B> {
        match self {
            ConstantValue::Float(t) => t,
            ConstantValue::Int(_) => {
                panic!("expected a continuous (float) value, found a discrete one")
            }
        }
    }

    /// Returns the contained int tensor, panicking if the value is continuous.
    pub fn unwrap_int(self) -> UnbatchedDiscretePoint<B> {
        match self {
            ConstantValue::Int(t) => t,
            ConstantValue::Float(_) => {
                panic!("expected a discrete (int) value, found a continuous one")
            }
        }
    }

    /// Whether this value is discrete (integer-valued).
    pub fn is_discrete(&self) -> bool {
        matches!(self, ConstantValue::Int(_))
    }

    /// Batch this stored (rank-1) value into a rank-2 [`Input`](OperationInput).
    pub fn expand(&self, batch_size: usize) -> OperationInput<B> {
        match self {
            ConstantValue::Float(t) => {
                OperationInput::Float(t.clone().unsqueeze_dim(0).repeat_dim(0, batch_size))
            }
            ConstantValue::Int(t) => {
                OperationInput::Int(t.clone().unsqueeze_dim(0).repeat_dim(0, batch_size))
            }
        }
    }

    /// Reduce a batched (rank-2) [`Input`](OperationInput) to storage form.
    pub fn unbatch(input: OperationInput<B>) -> Self {
        match input {
            OperationInput::Float(t) => ConstantValue::Float(t.squeeze_dim::<1>(0)),
            OperationInput::Int(t) => ConstantValue::Int(t.squeeze_dim::<1>(0)),
        }
    }
}

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
    Unbatched(ConstantValue<B>),
    /// Rank-2 storage, which the running batch must match exactly.
    Batched(OperationInput<B>),
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
    pub fn materialize(&self, batch_size: usize, id: Id) -> SkoroResult<OperationInput<B>> {
        match self {
            QuantityValue::Unbatched(value) => Ok(value.expand(batch_size)),
            QuantityValue::Batched(input) => input.ensure_batch(batch_size, id),
        }
    }

    /// `Some(n)` if this value can only be evaluated at batch `n`.
    pub fn pinned_batch(&self) -> Option<usize> {
        match self {
            QuantityValue::Unbatched(_) => None,
            QuantityValue::Batched(input) => Some(input.dims()[0]),
        }
    }

    /// Whether this value is discrete (integer-valued).
    pub fn is_discrete(&self) -> bool {
        match self {
            QuantityValue::Unbatched(value) => value.is_discrete(),
            QuantityValue::Batched(input) => input.is_discrete(),
        }
    }

    /// The rank-1 storage, if this value is batch-agnostic.
    ///
    /// The two variants hold tensors of different rank, so there is no accessor that serves both.
    /// A caller reaching for the value has to say which it expects.
    pub fn as_unbatched(&self) -> Option<&ConstantValue<B>> {
        match self {
            QuantityValue::Unbatched(value) => Some(value),
            QuantityValue::Batched(_) => None,
        }
    }

    /// The rank-2 storage, if this value is batch-pinned.
    pub fn as_batched(&self) -> Option<&OperationInput<B>> {
        match self {
            QuantityValue::Unbatched(_) => None,
            QuantityValue::Batched(input) => Some(input),
        }
    }
}

impl<B: Backend> From<ConstantValue<B>> for QuantityValue<B> {
    fn from(value: ConstantValue<B>) -> Self {
        Self::Unbatched(value)
    }
}

impl<B: Backend> From<UnbatchedPoint<B>> for ConstantValue<B> {
    fn from(value: UnbatchedPoint<B>) -> Self {
        Self::Float(value)
    }
}

impl<B: Backend> From<UnbatchedDiscretePoint<B>> for ConstantValue<B> {
    fn from(value: UnbatchedDiscretePoint<B>) -> Self {
        Self::Int(value)
    }
}
