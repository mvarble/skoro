use std::fmt::Debug;

use skoro_backend::{Backend, UnbatchedDiscretePoint, UnbatchedPoint};

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
