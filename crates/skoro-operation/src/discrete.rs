//! Discrete operations.
//!
//! A [`DiscreteOperation`] is the discrete-output analogue of
//! [`DiffOperation`](crate::diff::DiffOperation): it is a forward-only, non-differentiable function
//! `Vec<Input> -> Tensor<Int>`. It carries no differentiation state (no `vjp`/`hvp`), though a
//! concrete operation struct may hold its own configuration.
//!
//! Discrete operations lower into the *same* sequence-operation interface as differentiable
//! operations ([`DynSequenceOperation`](crate::DynSequenceOperation)) via
//! [`SequencedDiscreteOperation`]: the value sequence is kind-erased ([`Input`](crate::Input)),
//! so a forward-only operation simply writes an [`Input::Int`](crate::Input) output and must never
//! be invoked in a differentiation pass (the compiler guarantees this via a `forward_only` flag on
//! the compiled step).

use std::fmt::Debug;
use std::marker::PhantomData;

use skoro_backend::{Backend, DiscretePoint, TangentFrame, TangentVector};
use skoro_core::sequence::SequenceMutator;

use crate::step::{BoxDynHvpStep, BoxDynVjpStep};
use crate::tape::Tape;
use crate::{DynSequenceOperation, Input};

/// A stateless, forward-only operation producing a discrete (integer) output.
pub trait DiscreteOperation<B: Backend>: Debug + Clone + Send + Sync + 'static {
    /// Evaluate the discrete operation over its inputs.
    ///
    /// Inputs may be continuous or discrete (e.g. a discretization bridge reads a continuous input
    /// and produces a discrete output).
    fn forward(&self, inputs: Vec<Input<B>>) -> DiscretePoint<B>;
}

/// A context-wrapper for a [`DiscreteOperation`] which positions its inputs and outputs in a
/// sequence.
pub struct SequencedDiscreteOperation<B: Backend, Op: DiscreteOperation<B>> {
    input_indices: Vec<usize>,
    input_diffs: Vec<bool>,
    output_index: usize,
    operation: Op,
    _phantom: PhantomData<B>,
}

impl<B: Backend, Op: DiscreteOperation<B>> std::fmt::Debug for SequencedDiscreteOperation<B, Op> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequencedDiscreteOperation")
            .field("input_indices", &self.input_indices)
            .field("output_index", &self.output_index)
            .field("operation", &self.operation)
            .finish_non_exhaustive()
    }
}

impl<B: Backend, Op: DiscreteOperation<B>> SequencedDiscreteOperation<B, Op> {
    pub fn new(input_indices: Vec<usize>, output_index: usize, operation: Op) -> Self {
        let input_diffs = vec![false; input_indices.len()];
        Self {
            input_indices,
            input_diffs,
            output_index,
            operation,
            _phantom: PhantomData,
        }
    }

    fn get_inputs(&self, points: &[Option<Input<B>>]) -> Vec<Input<B>> {
        self.input_indices
            .iter()
            .enumerate()
            .map(|(i, index)| {
                if let Some(point) = &points[*index] {
                    point.clone()
                } else {
                    let operation = &self.operation;
                    panic!("A discrete sequence operation {operation:?} requires parent at (relative) input index {i:?} to already exist.");
                }
            })
            .collect()
    }
}

impl<B: Backend, Op: DiscreteOperation<B>> SequenceMutator for SequencedDiscreteOperation<B, Op> {
    fn input_indices(&self) -> &[usize] {
        &self.input_indices
    }
    fn output_index(&self) -> usize {
        self.output_index
    }
}

impl<B: Backend, Op: DiscreteOperation<B>> DynSequenceOperation<B>
    for SequencedDiscreteOperation<B, Op>
{
    fn input_diffs(&self) -> &[bool] {
        &self.input_diffs
    }

    fn forward(&self, points: &mut [Option<Input<B>>]) {
        let output = self.operation.forward(self.get_inputs(points));
        points[self.output_index] = Some(Input::Int(output));
    }

    fn jvp_forward(&self, _: &mut [Option<Input<B>>], _: &mut [Option<TangentFrame<B>>]) {
        unreachable!("discrete operations never participate in differentiation passes")
    }

    fn vjp_forward(&self, _: &mut [Option<Input<B>>], _: &mut Tape<BoxDynVjpStep<B>>) {
        unreachable!("discrete operations never participate in differentiation passes")
    }

    fn hvp_forward(
        &self,
        _: &mut [Option<Input<B>>],
        _: &mut [Option<TangentVector<B>>],
        _: &mut Tape<BoxDynHvpStep<B>>,
    ) {
        unreachable!("discrete operations never participate in differentiation passes")
    }
}
