use skoro_backend::{Backend, Tensor};

pub fn left_broadcast_matmul<B: Backend>(x: Tensor<B, 3>, y: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, k, m] = x.dims();
    let [_, _, n, f] = y.dims();
    x.matmul(y.reshape([b, m, n * f])).reshape([b, k, n, f])
}

pub fn right_broadcast_matmul<B: Backend>(x: Tensor<B, 4>, y: Tensor<B, 3>) -> Tensor<B, 4> {
    let x_swap = x.swap_dims(2, 3).swap_dims(1, 2); // [b, f, k, m]
    let [b, m, n] = y.dims();
    let y_broad = y.reshape([b, 1, m, n]);
    let prod_swap = x_swap.matmul(y_broad); // [b, f, k, n]
    prod_swap.swap_dims(1, 2).swap_dims(2, 3) // [b, k, n, f]
}

pub fn conjugate<B: Backend>(conj: Tensor<B, 3>, omega: Tensor<B, 4>) -> Tensor<B, 4> {
    right_broadcast_matmul(left_broadcast_matmul(conj.clone(), omega), conj)
}
