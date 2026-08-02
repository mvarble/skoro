//! Skoro uses Burn as its backend, so its backend trait is just a subtrait of that from Burn, but
//! only for specific backends we support. Since Skoro supports differentiation up to second-order,
//! none of the usual autodiff backends from Burn are used.
use burn::tensor::TensorPrimitive;

pub use burn::tensor::{
    Element, ElementConversion, Float, Int, Tensor, TensorData, cast::ToElement,
};
pub use burn_cubecl::{
    CubeRuntime, FloatElement as CubeFloat, IntElement as CubeInt, tensor::CubeTensor,
};
pub use burn_ndarray::{
    FloatNdArrayElement as NdArrayFloat, IntNdArrayElement as NdArrayInt, NdArrayTensor,
    SharedArray,
};

mod private {
    pub trait Sealed {}
}

/// Collect the backends we support.
///
/// This is not intended to be extended outside of this crate.
pub trait Backend: burn::prelude::Backend + private::Sealed {}

/// Alias the cubecl backend, fixing bool type we do not use.
pub type CubeBackend<R, F, I> = burn_cubecl::CubeBackend<R, F, I, u8>;

impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> private::Sealed for CubeBackend<R, F, I> {}

/// Support for cubecl backends.
impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> Backend for CubeBackend<R, F, I> {}

/// Alias the ndarray backend, fixing bool type we do not use.
pub type NdArrayBackend<F, I> = burn_ndarray::NdArray<F, I, i8>;

impl<F, I> private::Sealed for NdArrayBackend<F, I>
where
    F: NdArrayFloat + num_traits::Float,
    I: NdArrayInt,
    NdArrayTensor: From<SharedArray<F>> + From<SharedArray<I>>,
{
}

/// Support for ndarray backend.
impl<F, I> Backend for NdArrayBackend<F, I>
where
    F: NdArrayFloat + num_traits::Float,
    I: NdArrayInt,
    NdArrayTensor: From<SharedArray<F>> + From<SharedArray<I>>,
{
}

/// When implementing kernels over the cubecl backends, this utility will help obtain cube
/// primitives.
pub trait ToCubeTensor<R: CubeRuntime> {
    fn to_cube_tensor(self) -> CubeTensor<R>;
}

impl<R: CubeRuntime, F: CubeFloat, I: CubeInt, const D: usize> ToCubeTensor<R>
    for Tensor<CubeBackend<R, F, I>, D>
{
    fn to_cube_tensor(self) -> CubeTensor<R> {
        match self.into_primitive() {
            TensorPrimitive::Float(x) => x,
            TensorPrimitive::QFloat(_) => unreachable!("Skoro will never quantize."),
        }
    }
}

/// Batch of points packed into a tensor.
///
/// The shape convention is `[batch, storage-size]`.
pub type Point<B> = Tensor<B, 2>;

/// Batch of discrete points packed into a tensor.
///
/// The shape convention is `[batch, storage-size]`.
pub type DiscretePoint<B> = Tensor<B, 2, Int>;

/// Batch of tangent vectors in a tangent space of a manifold packed into a tensor.
///
/// The shape convention is `[batch, dimension]`.
pub type TangentVector<B> = Tensor<B, 2>;

/// Batch of cotangent vectors in a tangent space of a manifold packed into a tensor.
///
/// The shape convention is `[batch, dimension]`.
pub type CotangentVector<B> = Tensor<B, 2>;

/// Batch of tangent frames in a tangent space of a manifold packed into a tensor.
///
/// The shape convention is `[batch, vector, dimension]`.
pub type TangentFrame<B> = Tensor<B, 3>;

/// Batch of cotangent frames in a tangent space of a manifold packed into a tensor.
///
/// The shape convention is `[batch, vector, dimension]`.
pub type CotangentFrame<B> = Tensor<B, 3>;
