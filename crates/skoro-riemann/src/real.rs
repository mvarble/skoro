use skoro_backend::Backend;
use skoro_core::pack::{NumComponents, Packable};
use skoro_operation::{diff::DiffOperation, ops::Addition};

use crate::Manifold;
use crate::euclidean::EuclideanInverseRetraction;

/// Real numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Real;

impl<B: Backend> Manifold<B> for Real {
    fn dimension(&self) -> usize {
        1
    }

    fn retraction(&self) -> impl DiffOperation<B> {
        Addition
    }

    fn inverse_retraction(&self) -> impl DiffOperation<B> {
        EuclideanInverseRetraction
    }
}

impl NumComponents for Real {
    fn num_components(&self) -> usize {
        1
    }
}

impl Packable<f32, f32> for Real {
    fn pack(&self, data: &f32, offset: usize, target: &mut [f32]) {
        target[offset] = *data;
    }
    fn unpack(&self, source: &[f32], offset: usize) -> f32 {
        source[offset]
    }
}

impl Packable<f64, f64> for Real {
    fn pack(&self, data: &f64, offset: usize, target: &mut [f64]) {
        target[offset] = *data;
    }
    fn unpack(&self, source: &[f64], offset: usize) -> f64 {
        source[offset]
    }
}
