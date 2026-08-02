use skoro_backend::{Backend, CotangentFrame};

use crate::step::{BoxDynHvpStep, BoxDynVjpStep};

/// A flat sequence of backward steps collected during the forward pass.
///
/// Steps are pushed in forward order and consumed in reverse during `backward`. The tape owns the
/// accumulation recurrence and handles fan-in (accumulating cotangents from multiple children at
/// the same Id) and fan-out (distributing cotangents to multiple parent Ids).
pub struct Tape<BoxStep> {
    steps: Vec<BoxStep>,
}

impl<BoxStep> Default for Tape<BoxStep> {
    fn default() -> Self {
        Self::new()
    }
}

impl<BoxStep> Tape<BoxStep> {
    /// Create empty tape.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Create tape with a fixed capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            steps: Vec::with_capacity(cap),
        }
    }

    /// Number of backward operations in the tape.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether tape is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Push a step onto the tape.
    pub fn push(&mut self, step: BoxStep) {
        self.steps.push(step);
    }
}

impl<B: Backend> Tape<BoxDynVjpStep<B>> {
    /// Fold backward over the tape without consuming it.
    ///
    /// - `cotangents`: vector of cotangents of appropriate size
    pub fn backward(&self, cotangents: &mut [Option<CotangentFrame<B>>]) {
        for step in self.steps.iter().rev() {
            let output_index = step.output_index();
            if cotangents[output_index].is_none() {
                panic!("Cotangent vector not present at index {output_index:?}");
            }
            step.step(cotangents);
        }
    }

    /// Fold backward over the tape, consuming all steps.
    ///
    /// - `cotangents`: vector of cotangents of appropriate size
    pub fn consume_backward(self, cotangents: &mut [Option<CotangentFrame<B>>]) {
        for step in self.steps.into_iter().rev() {
            let output_index = step.output_index();
            if cotangents[output_index].is_none() {
                panic!("Cotangent vector not present at index {output_index:?}");
            }
            step.step(cotangents);
        }
    }
}

impl<B: Backend> Tape<BoxDynHvpStep<B>> {
    /// Fold backward over the tape without consuming it.
    ///
    /// - `ord1_cotangents`: vector of cotangents `α` of order-1
    /// - `ord2_cotangents`: vector of cotangents `β` of order-2
    pub fn backward(
        &self,
        ord1_cotangents: &mut [Option<CotangentFrame<B>>],
        ord2_cotangents: &mut [Option<CotangentFrame<B>>],
    ) {
        for step in self.steps.iter().rev() {
            let output_index = step.output_index();
            if ord1_cotangents[output_index].is_none() || ord2_cotangents[output_index].is_none() {
                panic!("Cotangent vector not present at index {output_index:?}");
            }
            step.step(ord1_cotangents, ord2_cotangents);
        }
    }

    /// Fold backward over the tape without consuming it.
    ///
    /// - `ord1_cotangents`: vector of cotangents `α` of order-1
    /// - `ord2_cotangents`: vector of cotangents `β` of order-2
    pub fn consume_backward(
        self,
        ord1_cotangents: &mut [Option<CotangentFrame<B>>],
        ord2_cotangents: &mut [Option<CotangentFrame<B>>],
    ) {
        for step in self.steps.into_iter().rev() {
            let output_index = step.output_index();
            if ord1_cotangents[output_index].is_none() || ord2_cotangents[output_index].is_none() {
                panic!("Cotangent vector not present at index {output_index:?}");
            }
            step.step(ord1_cotangents, ord2_cotangents);
        }
    }
}
