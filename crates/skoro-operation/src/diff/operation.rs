use std::marker::PhantomData;

use skoro_backend::{Backend, TangentFrame, TangentVector, Tensor};
use skoro_core::sequence::SequenceMutator;

use crate::step::{BoxDynHvpStep, BoxDynVjpStep};
use crate::tape::Tape;
use crate::{DynSequenceOperation, Input};

use super::{DiffModeState, DiffOperation, DiffOperationForward, HvpStep, VjpStep};

/// A context-wrapper for a [`DiffOperation`] which positions its inputs and outputs in a sequence.
pub struct SequencedDiffOperation<B: Backend, Op: DiffOperation<B>> {
    input_indices: Vec<usize>,
    input_diffs: Vec<bool>,
    output_index: usize,
    operation: Op,
    _phantom: PhantomData<B>,
}

impl<B: Backend, Op: DiffOperation<B>> std::fmt::Debug for SequencedDiffOperation<B, Op> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequenceOperation")
            .field("input_indices", &self.input_indices)
            .field("input_diffs", &self.input_diffs)
            .field("output_index", &self.output_index)
            .field("operation", &self.operation)
            .finish_non_exhaustive()
    }
}

impl<B: Backend, Op: DiffOperation<B>> SequencedDiffOperation<B, Op> {
    pub fn new(
        input_indices: Vec<usize>,
        input_diffs: Vec<bool>,
        output_index: usize,
        operation: Op,
    ) -> Self {
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
                    panic!("A sequence operation {operation:?} requires parent at (relative) input index {i:?} to already exist.");
                }
            })
            .collect()
    }

    fn get_tangents<const D: usize>(
        &self,
        tangents: &[Option<Tensor<B, D>>],
    ) -> Vec<Option<Tensor<B, D>>> {
        self.input_indices
            .iter()
            .zip(&self.input_diffs)
            .map(|(index, diff)| {
                if *diff {
                    tangents
                        .get(*index)
                        .expect(
                            "Input tangents should be large enough to index any operation input.",
                        )
                        .clone()
                } else {
                    None
                }
            })
            .collect()
    }
}

impl<B: Backend, K: DiffOperation<B>> SequenceMutator for SequencedDiffOperation<B, K> {
    fn input_indices(&self) -> &[usize] {
        &self.input_indices
    }
    fn output_index(&self) -> usize {
        self.output_index
    }
}

impl<B: Backend, K: DiffOperation<B>> DynSequenceOperation<B> for SequencedDiffOperation<B, K> {
    fn input_diffs(&self) -> &[bool] {
        &self.input_diffs
    }

    fn forward(&self, points: &mut [Option<Input<B>>]) {
        match self
            .operation
            .forward(self.get_inputs(points), DiffModeState::ForwardOnly)
        {
            DiffOperationForward::ForwardOnly { output } => {
                points[self.output_index] = Some(Input::Float(output));
            }
            _ => panic!(
                "{:?} must pass `DiffModeState::ForwardOnly` to `DiffOperationForward::ForwardOnly`.",
                self.operation
            ),
        };
    }

    fn jvp_forward(
        &self,
        points: &mut [Option<Input<B>>],
        tangents: &mut [Option<TangentFrame<B>>],
    ) {
        match self.operation.forward(
            self.get_inputs(points),
            DiffModeState::Jvp(self.get_tangents(tangents)),
        ) {
            DiffOperationForward::Jvp { output, tangent } => {
                points[self.output_index] = Some(Input::Float(output));
                tangents[self.output_index] = tangent;
            }
            _ => panic!(
                "{:?} must pass `DiffModeState::Jvp` to `DiffOperationForward::Jvp`.",
                self.operation
            ),
        };
    }

    fn vjp_forward(&self, points: &mut [Option<Input<B>>], tape: &mut Tape<BoxDynVjpStep<B>>) {
        match self.operation.forward(
            self.get_inputs(points),
            DiffModeState::Vjp(self.input_diffs.clone()),
        ) {
            DiffOperationForward::Vjp { output, vjp_state } => {
                points[self.output_index] = Some(Input::Float(output));
                tape.push(Box::new(VjpStep {
                    operation: self.operation.clone(),
                    state: vjp_state,
                    input_indices: self.input_indices.clone(),
                    output_index: self.output_index,
                }));
            }
            _ => panic!(
                "{:?} must pass `DiffModeState::Vjp` to `DiffOperationForward::Vjp`.",
                self.operation,
            ),
        };
    }

    fn hvp_forward(
        &self,
        points: &mut [Option<Input<B>>],
        tangents: &mut [Option<TangentVector<B>>],
        tape: &mut Tape<BoxDynHvpStep<B>>,
    ) {
        match self.operation.forward(
            self.get_inputs(points),
            DiffModeState::Hvp(self.get_tangents(tangents)),
        ) {
            DiffOperationForward::Hvp {
                output,
                tangent,
                vjp_state,
                hvp_state,
            } => {
                points[self.output_index] = Some(Input::Float(output));
                tangents[self.output_index] = tangent;
                tape.push(Box::new(HvpStep {
                    operation: self.operation.clone(),
                    vjp_state,
                    hvp_state,
                    input_indices: self.input_indices.clone(),
                    output_index: self.output_index,
                }));
            }
            _ => panic!(
                "{:?} must pass `DiffModeState::Hvp` to `DiffOperationForward::Hvp`.",
                self.operation
            ),
        };
    }
}
