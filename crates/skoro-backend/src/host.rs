//! Host-side reads and writes of float tensors, at the tensor's own precision.
//!
//! A float tensor's element type is its runtime [`Tensor::dtype`], which need not be
//! `B::FloatElem`: the Flex backend runs at `f64` under a device policy while its `FloatElem`
//! stays `f32`. A kernel that computes on the host therefore picks its precision with
//! [`is_f64`] from one input, reads **every** input with [`read`] at that precision, and writes
//! its output back with [`write`] at the dtype of that same input.

use burn::tensor::{DType, Element, TensorCreationOptions};

use crate::{Backend, Tensor, TensorData};

/// Whether `tensor` holds `f64` elements. Host kernels compute at `f32` otherwise.
pub fn is_f64<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> bool {
    tensor.dtype() == DType::F64
}

/// The row-major elements of `tensor`, converted to `F`.
pub fn read<B: Backend, F: Element, const D: usize>(tensor: Tensor<B, D>) -> Vec<F> {
    tensor
        .into_data()
        .convert::<F>()
        .into_vec::<F>()
        .expect("tensor data converted to an element type is readable as it")
}

/// A tensor of `shape` holding `data` on `device`, stored at `dtype`.
pub fn write<B: Backend, F: Element, const D: usize>(
    data: Vec<F>,
    shape: [usize; D],
    device: &B::Device,
    dtype: DType,
) -> Tensor<B, D> {
    Tensor::from_data(
        TensorData::new(data, shape),
        TensorCreationOptions::new(device.clone()).with_dtype(dtype),
    )
}
