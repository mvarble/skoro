//! `DiscreteOperation` and its sequence adapter.
//!
//! A discrete operation is forward-only by construction. It lowers into the *same*
//! `DynSequenceOperation` interface as a differentiable one, which is what lets integer values
//! flow through the same kind-erased value sequence -- and the three `unreachable!()` arms are
//! the guarantee that no differentiation pass ever reaches it. Asserting those panics is
//! asserting the compiler's `forward_only` flag actually holds.

mod common;

use common::*;
use skoro_backend::{Int, Tensor, TensorData};
use skoro_core::sequence::SequenceMutator;
use skoro_operation::{
    DynSequenceOperation, Input,
    discrete::{DiscreteOperation, SequencedDiscreteOperation},
    step::{BoxDynHvpStep, BoxDynVjpStep},
    tape::Tape,
};

fn ints(data: Vec<i32>, batch: usize, components: usize) -> Tensor<B, 2, Int> {
    Tensor::from_data(
        TensorData::new(data, [batch, components]),
        &Default::default(),
    )
}

/// Sums a continuous and a discrete input into a discrete one, by truncation.
///
/// Mixed-kind on purpose: the docs describe a discretization bridge that reads a continuous
/// input, and the adapter has to carry both without either kind leaking into the other.
#[derive(Debug, Clone)]
struct TruncatingSum;

impl DiscreteOperation<B> for TruncatingSum {
    fn forward(&self, inputs: Vec<Input<B>>) -> Tensor<B, 2, Int> {
        let [x, n] = <[Input<B>; 2]>::try_from(inputs).expect("two inputs");
        x.unwrap_float().int() + n.unwrap_int()
    }
}

fn sequenced() -> SequencedDiscreteOperation<B, TruncatingSum> {
    SequencedDiscreteOperation::new(vec![0, 1], 2, TruncatingSum)
}

/// A value sequence with the two inputs realized and the output slot empty.
fn seq() -> Vec<Option<Input<B>>> {
    vec![
        Some(Input::Float(scalar(2.7))),
        Some(Input::Int(ints(vec![5], 1, 1))),
        None,
    ]
}

/// The one pass a discrete operation does run: it writes an integer into its output slot, from a
/// mixed-kind input list. The output being discrete is asserted alongside its value, since an
/// integer stored in a float slot has the right number and the wrong meaning.
mod forward {
    use super::*;

    #[test]
    fn forward_writes_a_discrete_output_into_the_sequence() {
        let op = sequenced();
        let mut points = seq();
        op.forward(&mut points);

        let out = points[2].take().expect("the output slot is filled");
        assert!(out.is_discrete(), "a discrete operation produces integers");
        assert_eq!(
            out.unwrap_int().into_data().to_vec::<i32>().unwrap(),
            vec![7]
        );
    }
}

/// No input of a discrete operation is ever differentiable, whatever kind it is -- that is what
/// `forward_only` means at the adapter. An unrealized parent is a sequencing bug rather than a
/// user error, so it panics and names the index.
mod inputs {
    use super::*;

    /// Every input of a discrete operation is non-differentiable, whatever its kind.
    #[test]
    fn no_input_is_ever_differentiable() {
        let op = sequenced();
        assert_eq!(op.input_diffs(), &[false, false]);
        assert_eq!(op.input_indices(), &[0, 1]);
        assert_eq!(op.output_index(), 2);
    }

    /// A parent that has not been realized is a sequencing bug, and says which one.
    #[test]
    #[should_panic(expected = "requires parent at (relative) input index 1")]
    fn an_unrealized_parent_panics() {
        let op = sequenced();
        let mut points = vec![Some(Input::Float(scalar(1.0))), None, None];
        op.forward(&mut points);
    }
}

/// The wrapper rather than the operation: the name it prints, and its dyn-compatibility. Being
/// storable as a `Box<dyn DynSequenceOperation>` beside the differentiable adapters is the whole
/// reason it exists -- integer values flow through the same kind-erased sequence.
mod adapter {
    use super::*;

    #[test]
    fn the_adapter_names_the_operation_it_wraps() {
        let text = format!("{:?}", sequenced());
        assert!(text.contains("SequencedDiscreteOperation"));
        assert!(text.contains("TruncatingSum"));
        assert!(text.contains("input_indices"));
    }

    /// The adapter is dyn-compatible, which is the whole reason it exists.
    #[test]
    fn it_can_be_stored_as_a_trait_object() {
        let boxed: Box<dyn DynSequenceOperation<B>> = Box::new(sequenced());
        let mut points = seq();
        boxed.forward(&mut points);
        assert!(points[2].as_ref().unwrap().is_discrete());
    }
}

/// The three passes a discrete operation must never enter.
///
/// The compiler marks these steps `forward_only` and routes around them; reaching any of these
/// means that flag was lost, and a panic is the right answer because there is no meaningful
/// derivative to return.
mod never_differentiates {
    use super::*;

    #[test]
    #[should_panic(expected = "never participate in differentiation passes")]
    fn jvp() {
        let mut points = seq();
        let mut tangents = vec![None; 3];
        sequenced().jvp_forward(&mut points, &mut tangents);
    }

    #[test]
    #[should_panic(expected = "never participate in differentiation passes")]
    fn vjp() {
        let mut points = seq();
        let mut tape: Tape<BoxDynVjpStep<B>> = Tape::new();
        sequenced().vjp_forward(&mut points, &mut tape);
    }

    #[test]
    #[should_panic(expected = "never participate in differentiation passes")]
    fn hvp() {
        let mut points = seq();
        let mut tangents = vec![None; 3];
        let mut tape: Tape<BoxDynHvpStep<B>> = Tape::new();
        sequenced().hvp_forward(&mut points, &mut tangents, &mut tape);
    }
}
