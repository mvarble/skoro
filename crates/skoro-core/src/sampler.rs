use std::marker::PhantomData;

use skoro_backend::Backend;

use crate::kind::Kind;
use crate::kind_erased_tensor::KindErasedTensor;
use crate::sequence::SequenceMutator;

/// A pseudorandom sampling procedure producing points of kind `K`.
///
/// The kind parameter fixes the *output* element type statically ([`Point`](skoro_backend::Point)
/// for continuous samplers, [`DiscretePoint`](skoro_backend::DiscretePoint) for discrete ones),
/// while the *inputs* are erased [`OperationInput`]s, so a sampler may read parents of either kind
/// (e.g. a discrete distribution parameterized by a continuous logit).
pub trait Sampler<B: Backend, K: Kind<B>>: std::fmt::Debug + Clone + Send + Sync + 'static {
    /// Sample a batch of data with the specified batch_size.
    fn sample(&self, batch_size: usize, parameters: Vec<KindErasedTensor<B, 2>>) -> K::Point;
}

/// A sampler which has obscured the output type for dyn-compatibility and has positioned itself and
/// its inputs in a compiled sequence.
pub trait DynSequenceSampler<B: Backend>: SequenceMutator {
    fn sample(&self, batch_size: usize, points: &mut [Option<KindErasedTensor<B, 2>>]);
}

/// A context-wrapper for a [`Sampler`] which positions its inputs and outputs in a sequence.
pub struct SequenceSampler<B: Backend, K: Kind<B>, S: Sampler<B, K>> {
    input_indices: Vec<usize>,
    output_index: usize,
    sampler: S,
    _phantom: PhantomData<(B, K)>,
}

impl<B: Backend, K: Kind<B>, S: Sampler<B, K>> std::fmt::Debug for SequenceSampler<B, K, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequenceSampler")
            .field("input_indices", &self.input_indices)
            .field("output_index", &self.output_index)
            .field("sampler", &self.sampler)
            .finish_non_exhaustive()
    }
}

impl<B: Backend, K: Kind<B>, S: Sampler<B, K>> SequenceSampler<B, K, S> {
    pub fn new(input_indices: Vec<usize>, output_index: usize, sampler: S) -> Self {
        Self {
            input_indices,
            output_index,
            sampler,
            _phantom: PhantomData,
        }
    }

    /// Get the inputs of the sampler within the larger compiled sequence of quantities.
    pub fn get_inputs(
        &self,
        points: &mut [Option<KindErasedTensor<B, 2>>],
    ) -> Vec<KindErasedTensor<B, 2>> {
        self.input_indices
            .iter()
            .enumerate()
            .map(|(i, index)| {
                if let Some(point) = &points[*index] {
                    point.clone()
                } else {
                    let sampler = &self.sampler;
                    panic!("A sequence sampler {sampler:?} requires parent at (relative) input index {i:?} to already exist.");
                }
            })
            .collect()
    }
}

impl<B: Backend, K: Kind<B>, S: Sampler<B, K>> SequenceMutator for SequenceSampler<B, K, S> {
    fn input_indices(&self) -> &[usize] {
        &self.input_indices
    }
    fn output_index(&self) -> usize {
        self.output_index
    }
}

impl<B: Backend, K: Kind<B>, S: Sampler<B, K>> DynSequenceSampler<B> for SequenceSampler<B, K, S> {
    fn sample(&self, batch_size: usize, points: &mut [Option<KindErasedTensor<B, 2>>]) {
        let out = self.sampler.sample(batch_size, self.get_inputs(points));
        points[self.output_index] = Some(K::wrap_point(out));
    }
}
