use skoro_backend::{Backend, ElementConversion, Int, Tensor, TensorData, host};

#[inline]
pub(super) fn k_to_n(k: usize) -> usize {
    ((-1 + ((1 + 8 * k) as f32).sqrt() as i32) / 2) as usize
}

pub fn unpack<B: Backend, const D: usize, const D2: usize>(
    mat_packed: Tensor<B, D>,
) -> Tensor<B, D2> {
    // Enforce rank conditions
    assert!(
        D >= 2,
        "Input tensor must have at least 2 dimensions [b, k, ...]"
    );
    assert_eq!(D2, D + 1, "Output rank D2 must be exactly D + 1");

    let device = mat_packed.device();
    let dtype = mat_packed.dtype();
    let dims = mat_packed.dims();

    // Extract the batch size 'b' and vector dimension `k`
    let b = dims[0];
    let k = dims[1];

    // Solve for n: k = n * (n + 1) / 2
    let n = k_to_n(k);

    // 1. Generate indices and mask
    let mut indices = Vec::<B::IntElem>::with_capacity(n * n);
    let mut mask = Vec::<f64>::with_capacity(n * n);

    for i in 0..n {
        for j in 0..n {
            if i >= j {
                indices.push(B::IntElem::from_elem(i * (i + 1) / 2 + j));
                mask.push(1.0);
            } else {
                // Out-of-bounds protection: just point to 0,
                // it will be zeroed out by the mask anyway.
                indices.push(B::IntElem::from_elem(0));
                mask.push(0.0);
            }
        }
    }

    let indices_tensor = Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [n * n]), &device);

    // 2. Create the mask with shape [1, n, n, 1, 1...] for batch/feature broadcasting
    let mut mask_shape = [1; D2];
    mask_shape[1] = n;
    mask_shape[2] = n;

    // TensorData naturally consumes the 1D mask vec and applies it to the n-D shape
    let mask_tensor = host::write::<B, f64, D2>(mask, mask_shape, &device, dtype);

    // 3. Dynamically construct the reshaped dimensions
    // From [b, n * n, f, ...] -> [b, n, n, f, ...]
    let mut new_shape = [0; D2];
    new_shape[0] = b;
    new_shape[1] = n;
    new_shape[2] = n;

    // Copy trailing dimensions (if any exist)
    // E.g., for D=3 (dims = [b, k, f]), dims[2] becomes new_shape[3]
    new_shape[3..(D + 1)].copy_from_slice(&dims[2..D]);

    // 4. Unpack, Reshape, and Mask
    // select(1, ...) transforms [b, k, f] into [b, n * n, f]
    let mat_full = mat_packed.select(1, indices_tensor).reshape(new_shape);

    // [b, n, n, f] * [1, n, n, 1] broadcasts perfectly natively
    mat_full * mask_tensor
}

pub fn pack<B: Backend, const D: usize, const D2: usize>(mat_tril: Tensor<B, D>) -> Tensor<B, D2> {
    // Ensure the ranks are valid
    assert!(
        D >= 3,
        "Input tensor must have at least 3 dimensions [b, n, n, ...]"
    );
    assert_eq!(D2, D - 1, "Output rank D2 must be exactly D - 1");

    let device = mat_tril.device();
    let dims = mat_tril.dims();

    let b = dims[0];
    let n = dims[1];
    assert_eq!(n, dims[2], "Dimensions 1 and 2 must be equal (n x n)");

    // Dynamically construct the reshaped dimensions
    // e.g., [b, n, n, f] -> [b, n * n, f]
    let mut flat_shape = [0; D2];
    flat_shape[0] = b;
    flat_shape[1] = n * n;
    flat_shape[2..(D - 1)].copy_from_slice(&dims[3..D]);

    let mat_flat = mat_tril.reshape(flat_shape);
    let k = n * (n + 1) / 2;
    let mut indices = Vec::with_capacity(k);
    for row in 0..n {
        for col in 0..=row {
            indices.push((row * n + col) as i32);
        }
    }

    let indices_tensor = Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [k]), &device);

    mat_flat.select(1, indices_tensor)
}
