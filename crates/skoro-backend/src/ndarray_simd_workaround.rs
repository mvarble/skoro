//! Hand-rolled workarounds for a defect in burn-ndarray 0.21's SIMD unary kernels.
//!
//! **This module is temporary.** Skoro will move off the ndarray backend in favour of the flex
//! backend; when it does, delete this module and replace each call into it with the plain tensor
//! method it wraps (`ndarray_simd_workaround::abs(t)` becomes `t.abs()`).
//!
//! # The defect
//!
//! burn-ndarray routes some unary kernels through `try_unary_simd`, which accepts any input that
//! is contiguous in memory. When the input is not uniquely owned it takes an "owned" path: it
//! allocates the output in standard (row-major) layout, then fills it by walking the input in
//! *memory order*. For an input whose memory order is not its logical order, every element lands
//! in the wrong position. Two ordinary operations produce such inputs:
//!
//! - `select` along a trailing axis, which ndarray assembles with `concatenate` in column-major
//!   order;
//! - `swap_dims` on a contiguous tensor, which permutes the strides without moving data.
//!
//! The SIMD path is taken once the input has at least 32 elements. A uniquely owned input takes an
//! in-place path that keeps its strides and is correct, so whether the defect shows depends on how
//! many handles to the tensor are alive — in an autodiff pass, where operations keep their inputs
//! for the backward pass, they usually are.
//!
//! The affected float kernels are `abs` (`f32` and `f64`) and `recip` (`f32`). burn-ndarray's
//! scalar-binary and comparison kernels allocate their output with the input's own layout and are
//! not affected. It surfaced as a wrong log-determinant — `log |diag L⁻¹|`, whose `diag` is a
//! `select` — for every `Normal` evaluated at a batch of 16 or more.
//!
//! # The workaround
//!
//! For `abs`, reshape to the same shape before calling the kernel. burn-ndarray's `reshape` copies
//! an array that is not in standard layout into one that is, and returns a view of one that already
//! is, so the kernel's memory order and logical order then agree. On the cube backends a reshape
//! only rewrites metadata. For `recip`, divide instead; see [`recip`].

use burn::tensor::Tensor;

use crate::Backend;

/// `tensor` with its data in standard (row-major) layout on the ndarray backend.
///
/// A copy only when the layout is not already standard; metadata only on the cube backends.
pub fn standard_layout<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    let dims = tensor.dims();
    tensor.reshape(dims)
}

/// Elementwise `|x|`: `tensor.abs()`, correct whatever the layout of `tensor`.
pub fn abs<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    standard_layout(tensor).abs()
}

/// Elementwise `1 / x`: `tensor.recip()`, correct whatever the layout of `tensor`.
///
/// Computed as a division rather than through `recip` at all: besides the layout defect, the
/// `f32` SIMD `recip` kernel is approximate (relative errors around `1e-4` were observed), while
/// burn-ndarray's division kernel checks layout before vectorizing and divides exactly.
pub fn recip<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.ones_like() / tensor
}
