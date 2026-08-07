use skoro_backend::{Backend, Tensor};

// multiply tensors of shape `[b, m, n]` and `[b, n, k, f]`, broadcasting over the `b` and the `f`
// to get one of shape `[b, m, k, f]`.
pub fn left_broadcast_matmul<B: Backend>(x: Tensor<B, 3>, y: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, m, n] = x.dims();
    let [_, _, k, f] = y.dims();
    x.matmul(y.reshape([b, n, k * f])).reshape([b, m, k, f])
}

// multiply tensors of shape `[b, m, n, f]` and `[b, n, k]`, broadcasting over the `b` and the `f`
// to get one of shape `[b, m, k, f]`.
pub fn right_broadcast_matmul<B: Backend>(x: Tensor<B, 4>, y: Tensor<B, 3>) -> Tensor<B, 4> {
    let x_swap = x.swap_dims(2, 3).swap_dims(1, 2); // [b, f, m, n]
    let [b, n, k] = y.dims();
    let y_broad = y.reshape([b, 1, n, k]);
    let prod_swap = x_swap.matmul(y_broad); // [b, f, m, k]
    prod_swap.swap_dims(1, 2).swap_dims(2, 3) // [b, m, k, f]
}

pub fn conjugate<B: Backend>(conj: Tensor<B, 3>, omega: Tensor<B, 4>) -> Tensor<B, 4> {
    right_broadcast_matmul(left_broadcast_matmul(conj.clone(), omega), conj)
}
