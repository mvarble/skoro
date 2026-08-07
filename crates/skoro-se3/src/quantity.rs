use num_traits::Float;
use skoro_backend::{
    Backend, CubeBackend, CubeFloat, CubeInt, CubeRuntime, Element, NdArrayBackend, NdArrayFloat,
    NdArrayInt, NdArrayTensor, SharedArray,
};
use skoro_core::pack::{NumComponents, Packable};
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::ops::{AssociatedTangentVector, AssociatedTangentVectorBuilder};
use skoro_operation::diff::DiffOperation;
use skoro_riemann::{Manifold, euclidean::Euclidean};
use skoro_so3::quantity::SO3Matrix;

use crate::ops::{SE3InverseRetraction, SE3Retraction};

#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub struct SE3Pose<F: Element> {
    pub attitude: SO3Matrix<F>,
    pub position: [F; 3],
}

impl<F: Element> SE3Pose<F> {
    pub fn new(attitude: SO3Matrix<F>, position: [F; 3]) -> Self {
        Self { attitude, position }
    }
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SE3;

impl NumComponents for SE3 {
    fn num_components(&self) -> usize {
        12
    }
}

impl<F: Element> Packable<F, SE3Pose<F>> for SE3 {
    fn pack(&self, data: &SE3Pose<F>, offset: usize, dst: &mut [F]) {
        dst[offset..(offset + 9)].copy_from_slice(&data.attitude.0);
        dst[(offset + 9)..(offset + 12)].copy_from_slice(&data.position);
    }

    fn unpack(&self, tensor: &[F], offset: usize) -> SE3Pose<F> {
        assert!(tensor.len() >= offset + 12);
        let mut attitude_data: [F; 9] = [F::default(); 9];
        let mut psn_data: [F; 3] = [F::default(); 3];
        attitude_data.copy_from_slice(&tensor[offset..(offset + 9)]);
        psn_data.copy_from_slice(&tensor[(offset + 9)..(offset + 12)]);
        SE3Pose {
            attitude: SO3Matrix(attitude_data),
            position: psn_data,
        }
    }
}

impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> Manifold<CubeBackend<R, F, I>> for SE3 {
    fn dimension(&self) -> usize {
        6
    }

    fn retraction(&self) -> impl DiffOperation<CubeBackend<R, F, I>> {
        SE3Retraction
    }

    fn inverse_retraction(&self) -> impl DiffOperation<CubeBackend<R, F, I>> {
        SE3InverseRetraction
    }
}

impl<F, I> Manifold<NdArrayBackend<F, I>> for SE3
where
    F: NdArrayFloat + Float,
    I: NdArrayInt,
    NdArrayTensor: From<SharedArray<F>> + From<SharedArray<I>>,
{
    fn dimension(&self) -> usize {
        6
    }

    fn retraction(&self) -> impl DiffOperation<NdArrayBackend<F, I>> {
        SE3Retraction
    }

    fn inverse_retraction(&self) -> impl DiffOperation<NdArrayBackend<F, I>> {
        SE3InverseRetraction
    }
}

impl<B: Backend> AssociatedTangentVector<Euclidean<6>, B> for SE3 where SE3: ContinuousQuantity<B> {}
impl<B: Backend> AssociatedTangentVectorBuilder<Euclidean<6>, B> for SE3
where
    SE3: ContinuousQuantity<B>,
{
    fn build_tangent(&self) -> Euclidean<6> {
        Euclidean::<6>
    }
}
