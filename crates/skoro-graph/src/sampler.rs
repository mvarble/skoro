use std::fmt::Debug;
use std::marker::PhantomData;

use skoro_backend::Backend;
use skoro_core::kind::Kind;
use skoro_core::sampler::{DynSequenceSampler, Sampler, SequenceSampler};

pub trait DynGraphSampler<B: Backend>: 'static + Send + Sync + Debug {
    /// Lowers this deferred sampler into a concrete [`DynSequenceSampler`].
    fn lower(
        &self,
        input_indices: Vec<usize>,
        output_index: usize,
    ) -> Box<dyn DynSequenceSampler<B>>;

    /// Returns a new boxed copy of this sampler.
    fn clone_box(&self) -> Box<dyn DynGraphSampler<B>>;
}

impl<B: Backend> Clone for Box<dyn DynGraphSampler<B>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Adapter binding a kinded [`Sampler<B, K>`] into the erased [`DynSampler`] trait.
///
/// The kind parameter `K` fixes the sampler's output element type at the registration site
/// [`Graph::add_stochastic`](crate::graph::Graph::add_stochastic), while the erased trait lets
/// nodes store samplers of either kind uniformly.
pub struct GraphSampler<B: Backend, K: Kind<B>, S: Sampler<B, K>> {
    sampler: S,
    _phantom: PhantomData<(B, K)>,
}

impl<B: Backend, K: Kind<B>, S: Sampler<B, K>> GraphSampler<B, K, S> {
    pub fn new(sampler: S) -> Self {
        Self {
            sampler,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, K: Kind<B>, S: Sampler<B, K>> Debug for GraphSampler<B, K, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KindedSampler")
            .field("sampler", &self.sampler)
            .finish_non_exhaustive()
    }
}

impl<B: Backend, K: Kind<B>, S: Sampler<B, K>> DynGraphSampler<B> for GraphSampler<B, K, S> {
    fn lower(
        &self,
        input_indices: Vec<usize>,
        output_index: usize,
    ) -> Box<dyn DynSequenceSampler<B>> {
        Box::new(SequenceSampler::<B, K, S>::new(
            input_indices,
            output_index,
            self.sampler.clone(),
        ))
    }

    fn clone_box(&self) -> Box<dyn DynGraphSampler<B>> {
        Box::new(GraphSampler::<B, K, S>::new(self.sampler.clone()))
    }
}
