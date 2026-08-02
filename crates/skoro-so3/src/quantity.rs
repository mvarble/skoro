use num_traits::Float;
use skoro_backend::{
    Backend, CubeBackend, CubeFloat, CubeInt, CubeRuntime, Element, NdArrayBackend, NdArrayFloat,
    NdArrayInt, NdArrayTensor, SharedArray,
};
use skoro_core::pack::{NumComponents, Packable};
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::ops::{AssociatedTangentVector, AssociatedTangentVectorBuilder};
use skoro_operation::diff::DiffOperation;
use skoro_riemann::{
    Manifold,
    euclidean::{Euclidean, Euclidean3},
};

use crate::ops::{SO3InverseRetraction, SO3Retraction};

#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub struct SO3Matrix<F: Element>(pub [F; 9]);

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SO3;

impl NumComponents for SO3 {
    fn num_components(&self) -> usize {
        9
    }
}

impl<F: Element> Packable<F, SO3Matrix<F>> for SO3 {
    fn pack(&self, data: &SO3Matrix<F>, offset: usize, dst: &mut [F]) {
        dst[offset..].copy_from_slice(&data.0);
    }

    fn unpack(&self, tensor: &[F], offset: usize) -> SO3Matrix<F> {
        assert!(tensor.len() >= offset + 9);
        let mut data: [F; 9] = [F::default(); 9];
        data.copy_from_slice(&tensor[offset..]);
        SO3Matrix(data)
    }
}

impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> Manifold<CubeBackend<R, F, I>> for SO3 {
    fn dimension(&self) -> usize {
        3
    }

    fn retraction(&self) -> impl DiffOperation<CubeBackend<R, F, I>> {
        SO3Retraction
    }

    fn inverse_retraction(&self) -> impl DiffOperation<CubeBackend<R, F, I>> {
        SO3InverseRetraction
    }
}

impl<F, I> Manifold<NdArrayBackend<F, I>> for SO3
where
    F: NdArrayFloat + Float,
    I: NdArrayInt,
    NdArrayTensor: From<SharedArray<F>> + From<SharedArray<I>>,
{
    fn dimension(&self) -> usize {
        3
    }

    fn retraction(&self) -> impl DiffOperation<NdArrayBackend<F, I>> {
        SO3Retraction
    }

    fn inverse_retraction(&self) -> impl DiffOperation<NdArrayBackend<F, I>> {
        SO3InverseRetraction
    }
}

impl<B: Backend> AssociatedTangentVector<Euclidean3, B> for SO3 where SO3: ContinuousQuantity<B> {}
impl<B: Backend> AssociatedTangentVectorBuilder<Euclidean3, B> for SO3
where
    SO3: ContinuousQuantity<B>,
{
    fn build_tangent(&self) -> Euclidean3 {
        Euclidean::<3>
    }
}
