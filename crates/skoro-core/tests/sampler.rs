//! `SequenceSampler`: a kinded sampler positioned in an index-based value sequence.
//!
//! The adapter's job is to erase the output kind for dyn-compatibility while keeping it correct:
//! a `Sampler<B, Continuous>` must write a float slot and a `Sampler<B, Discrete>` an integer
//! one, from the same call site. Its inputs stay erased throughout, so a discrete distribution
//! parameterized by a continuous logit works without either side knowing about the other.

use skoro_backend::{Int, NdArrayBackend, Tensor, TensorData};
use skoro_core::{
    any_tensor::AnyTensor,
    kind::{Continuous, Discrete},
    sampler::{DynSequenceSampler, Sampler, SequenceSampler},
    sequence::SequenceMutator,
};

type B = NdArrayBackend<f64, i32>;

fn floats(data: Vec<f64>, batch: usize, components: usize) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(data, [batch, components]),
        &Default::default(),
    )
}

/// Returns its parent's value plus an offset, one row per batch element.
///
/// Deterministic, because what is being tested is the routing, not the randomness.
#[derive(Debug, Clone)]
struct OffsetSampler(f64);

impl Sampler<B, Continuous> for OffsetSampler {
    fn sample(&self, batch_size: usize, parameters: Vec<AnyTensor<B, 2>>) -> Tensor<B, 2> {
        assert_eq!(
            parameters.len(),
            1,
            "OffsetSampler reads exactly one parent"
        );
        let parent = parameters[0].clone().unwrap_float();
        assert_eq!(parent.dims()[0], batch_size);
        parent.add_scalar(self.0)
    }
}

/// Reads a *continuous* parent and produces a discrete output: the mixed-kind case the erased
/// input list exists for.
#[derive(Debug, Clone)]
struct ThresholdSampler(f64);

impl Sampler<B, Discrete> for ThresholdSampler {
    fn sample(&self, _batch_size: usize, parameters: Vec<AnyTensor<B, 2>>) -> Tensor<B, 2, Int> {
        let logit = parameters[0].clone().unwrap_float();
        logit.greater_elem(self.0).int()
    }
}

/// A sampler with no parents at all.
#[derive(Debug, Clone)]
struct ConstantSampler(f64);

impl Sampler<B, Continuous> for ConstantSampler {
    fn sample(&self, batch_size: usize, parameters: Vec<AnyTensor<B, 2>>) -> Tensor<B, 2> {
        assert!(parameters.is_empty());
        floats(vec![self.0; batch_size], batch_size, 1)
    }
}

/// The output kind follows the sampler's type parameter, not its inputs: a `Sampler<B, Discrete>`
/// writes an integer slot even when every parent it read was continuous. That is the whole reason
/// the input list stays erased.
mod discrimination {
    use super::*;

    #[test]
    fn a_continuous_sampler_writes_a_float_slot() {
        let sampler = SequenceSampler::<B, Continuous, _>::new(vec![0], 1, OffsetSampler(0.5));
        let mut points = vec![Some(AnyTensor::Float(floats(vec![1.0, 2.0], 2, 1))), None];
        sampler.sample(2, &mut points);

        let out = points[1].take().expect("the output slot is filled");
        assert!(!out.is_discrete());
        assert_eq!(
            out.unwrap_float().into_data().to_vec::<f64>().unwrap(),
            vec![1.5, 2.5]
        );
    }

    /// The output kind follows the type parameter, not the input kinds.
    #[test]
    fn a_discrete_sampler_writes_an_int_slot_from_a_continuous_parent() {
        let sampler = SequenceSampler::<B, Discrete, _>::new(vec![0], 1, ThresholdSampler(0.0));
        let mut points = vec![Some(AnyTensor::Float(floats(vec![-1.0, 3.0], 2, 1))), None];
        sampler.sample(2, &mut points);

        let out = points[1].take().unwrap();
        assert!(out.is_discrete());
        assert_eq!(
            out.unwrap_int().into_data().to_vec::<i32>().unwrap(),
            vec![0, 1]
        );
    }
}

/// Gathering a sampler's parents out of the value sequence. The order is the declared one and not
/// slot order -- a wrapper that sorted or deduplicated the indices would silently permute a
/// distribution's arguments -- and gathering clones, so a parent feeding two children survives the
/// first. An unrealized parent is a sequencing bug, so it panics and says which index.
mod inputs {
    use super::*;

    #[test]
    fn a_parentless_sampler_receives_an_empty_input_list() {
        let sampler = SequenceSampler::<B, Continuous, _>::new(vec![], 0, ConstantSampler(4.0));
        let mut points = vec![None];
        sampler.sample(3, &mut points);
        assert_eq!(
            points[0]
                .take()
                .unwrap()
                .unwrap_float()
                .into_data()
                .to_vec::<f64>()
                .unwrap(),
            vec![4.0, 4.0, 4.0]
        );
    }

    /// Inputs are gathered in the order declared, not in slot order.
    ///
    /// The sampler's own parameter list is positional, so a wrapper that sorted or deduplicated the
    /// indices would silently permute a distribution's arguments.
    #[test]
    fn inputs_are_gathered_in_the_declared_order() {
        #[derive(Debug, Clone)]
        struct FirstMinusSecond;

        impl Sampler<B, Continuous> for FirstMinusSecond {
            fn sample(&self, _: usize, parameters: Vec<AnyTensor<B, 2>>) -> Tensor<B, 2> {
                parameters[0].clone().unwrap_float() - parameters[1].clone().unwrap_float()
            }
        }

        // Declared as (slot 2, slot 0), deliberately out of order.
        let sampler = SequenceSampler::<B, Continuous, _>::new(vec![2, 0], 1, FirstMinusSecond);
        let mut points = vec![
            Some(AnyTensor::Float(floats(vec![1.0], 1, 1))),
            None,
            Some(AnyTensor::Float(floats(vec![10.0], 1, 1))),
        ];
        sampler.sample(1, &mut points);
        assert_eq!(
            points[1]
                .take()
                .unwrap()
                .unwrap_float()
                .into_data()
                .to_vec::<f64>()
                .unwrap(),
            vec![9.0],
            "10 - 1, not 1 - 10"
        );
    }

    /// A parent that has not been realized is a sequencing bug, and the panic names which one.
    ///
    /// The compiler emits samplers in topological order so this cannot happen; reaching it means the
    /// ordering was wrong, which is an internal invariant rather than a recoverable user error.
    #[test]
    #[should_panic(expected = "requires parent at (relative) input index 1")]
    fn an_unrealized_parent_panics() {
        let sampler = SequenceSampler::<B, Continuous, _>::new(vec![0, 2], 1, ConstantSampler(0.0));
        let mut points = vec![Some(AnyTensor::Float(floats(vec![1.0], 1, 1))), None, None];
        let _ = sampler.get_inputs(&mut points);
    }

    /// `get_inputs` is public so a caller can gather without sampling, and it clones rather than
    /// taking -- a parent feeding two children must survive the first.
    #[test]
    fn gathering_inputs_leaves_them_in_place() {
        let sampler = SequenceSampler::<B, Continuous, _>::new(vec![0], 1, OffsetSampler(0.0));
        let mut points = vec![Some(AnyTensor::Float(floats(vec![1.0], 1, 1))), None];

        let gathered = sampler.get_inputs(&mut points);
        assert_eq!(gathered.len(), 1);
        assert!(
            points[0].is_some(),
            "the parent is still there for its other children"
        );
    }
}

/// The wrapper itself rather than the sampling: the positions it was built with, the name it
/// prints, and its dyn-compatibility. Being storable as a `Box<dyn DynSequenceSampler>` is the
/// whole reason the adapter exists.
mod adapter {
    use super::*;

    #[test]
    fn it_reports_the_positions_it_was_built_with() {
        let sampler = SequenceSampler::<B, Continuous, _>::new(vec![3, 7], 9, ConstantSampler(0.0));
        assert_eq!(sampler.input_indices(), &[3, 7]);
        assert_eq!(sampler.output_index(), 9);
    }

    #[test]
    fn it_names_the_sampler_it_wraps_when_printed() {
        let sampler = SequenceSampler::<B, Continuous, _>::new(vec![0], 1, OffsetSampler(0.5));
        let text = format!("{sampler:?}");
        assert!(text.contains("SequenceSampler"));
        assert!(text.contains("OffsetSampler"));
        assert!(text.contains("input_indices"));
    }

    /// The adapter is dyn-compatible, which is the whole reason it exists.
    #[test]
    fn it_can_be_stored_as_a_trait_object() {
        let boxed: Box<dyn DynSequenceSampler<B>> = Box::new(
            SequenceSampler::<B, Discrete, _>::new(vec![0], 1, ThresholdSampler(1.0)),
        );
        assert_eq!(boxed.input_indices(), &[0]);
        assert_eq!(boxed.output_index(), 1);

        let mut points = vec![Some(AnyTensor::Float(floats(vec![5.0], 1, 1))), None];
        boxed.sample(1, &mut points);
        assert!(points[1].as_ref().unwrap().is_discrete());
    }
}
