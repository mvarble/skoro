use skoro_backend::{
    Backend, CotangentFrame, Element, ElementConversion, Point, Tensor, TensorData,
};
use skoro_core::error::SkoroResult;
use skoro_model::{map::register_map1, q::Q};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

use super::{Tril, k_to_n, pack, unpack};
use crate::matrix::{conjugate, left_broadcast_matmul, right_broadcast_matmul};

/// Inversion of a lower-triangular matrix (with diagonals).
#[derive(Debug, Clone, Copy)]
pub struct TrilInvert;

#[inline]
fn invert_packed_lower_triangular<
    F: Element
        + ElementConversion
        + std::ops::Div<F, Output = F>
        + std::ops::Mul<F, Output = F>
        + std::ops::Add<F, Output = F>
        + std::ops::Sub<F, Output = F>
        + std::ops::Neg<Output = F>
        + std::ops::AddAssign<F>,
>(
    packed_l: &[F],
    batches: usize,
) -> Vec<F> {
    let len = packed_l.len();
    let k = len / batches;
    let n = k_to_n(k);
    assert_eq!(len, batches * n * (n + 1) / 2);
    let mut inv: Vec<F> = vec![F::from_elem(0.0); packed_l.len()];
    for b in 0..batches {
        let mut offset_i = b * k;
        for i in 0..n {
            // 1. Diagonal element: X_{i,i} = 1.0 / L_{i,i}
            let l_ii = packed_l[offset_i + i];
            let inv_l_ii = F::from_elem(1.0) / l_ii;
            inv[offset_i + i] = inv_l_ii;

            // 2. Off-diagonal elements in row i: X_{i,j} for j < i
            for j in 0..i {
                let mut sum = F::from_elem(0.0);

                // To compute the dot product, we need the offset for row k
                // We start k at j, so the initial offset for row j is j*(j+1)/2
                let mut offset_k = j * (j + 1) / 2;

                for k in j..i {
                    // sum += L_{i,k} * X_{k,j}
                    sum += packed_l[offset_i + k] * inv[offset_k + j];

                    // Advance offset_k to the next row (k + 1)
                    offset_k += k + 1;
                }

                inv[offset_i + j] = -sum * inv_l_ii;
            }

            // Advance offset_i to the next row (i + 1)
            offset_i += i + 1;
        }
    }
    inv
}

impl<B: Backend> DiffOperation<B> for TrilInvert
where
    B::FloatElem: std::ops::Div<B::FloatElem, Output = B::FloatElem>
        + std::ops::Mul<B::FloatElem, Output = B::FloatElem>
        + std::ops::Add<B::FloatElem, Output = B::FloatElem>
        + std::ops::Sub<B::FloatElem, Output = B::FloatElem>
        + std::ops::Neg<Output = B::FloatElem>
        + std::ops::AddAssign<B::FloatElem>,
{
    type VjpState = Option<Tensor<B, 2>>; // A^{-1}
    type HvpState = Option<Tensor<B, 2>>; // ΔA
    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::into_float).collect();
        let [mat_packed] = inputs.try_into().unwrap();
        let shape = mat_packed.shape();
        let mat_data = mat_packed.clone().into_data().convert::<B::FloatElem>();
        let mat_elems = mat_data.as_slice::<B::FloatElem>().unwrap();
        let inv_mat_elems = invert_packed_lower_triangular(mat_elems, shape[0]);
        let inv_mat_packed =
            Tensor::from_data(TensorData::new(inv_mat_elems, shape), &mat_packed.device());
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(inv_mat_packed),
            DiffModeState::Jvp(tvecs) => {
                let [dmat_packed] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    inv_mat_packed.clone(),
                    dmat_packed.map(|dmat_packed| {
                        -pack(conjugate(unpack(inv_mat_packed), unpack(dmat_packed)))
                    }),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [is_diff] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    inv_mat_packed.clone(),
                    if is_diff { Some(inv_mat_packed) } else { None },
                )
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dmat_packed] = tvecs.try_into().unwrap();
                DiffOperationForward::hvp(
                    inv_mat_packed.clone(),
                    dmat_packed.as_ref().map(|dmat_packed| {
                        let inv_mat: Tensor<B, 3> = unpack(inv_mat_packed.clone());
                        let dmat: Tensor<B, 3> = unpack(dmat_packed.clone());
                        -pack(inv_mat.clone().matmul(dmat).matmul(inv_mat))
                    }),
                    if diffs[0] { Some(inv_mat_packed) } else { None },
                    dmat_packed,
                )
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        if let Some(inv_mat_packed) = state {
            let inv_mat: Tensor<B, 3> = unpack(inv_mat_packed.clone());
            let omega_tril: Tensor<B, 4> = unpack(omega);
            vec![Some(-pack(conjugate(inv_mat, omega_tril)))]
        } else {
            vec![None]
        }
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        if let (Some(inv_mat_packed), Some(dmat_packed)) = (vjp_state, hvp_state) {
            let inv_mat: Tensor<B, 3> = unpack(inv_mat_packed.clone());
            let omega_mat: Tensor<B, 4> = unpack(omega);
            let conj_omega = conjugate(inv_mat.clone(), omega_mat.clone());

            let dmat = unpack(dmat_packed.clone()); // [b, n, n]
            let conj_dmat = inv_mat.clone().matmul(dmat.clone()).matmul(inv_mat.clone());

            let s0 = -right_broadcast_matmul(
                left_broadcast_matmul(conj_dmat, omega_mat.clone()),
                inv_mat.clone(),
            );
            let s1 = -right_broadcast_matmul(conj_omega, dmat.matmul(inv_mat));
            vec![Some(pack(s0 + s1))]
        } else {
            vec![None]
        }
    }
}

pub fn invert<B: Backend>(tril: &Q<B, Tril>) -> SkoroResult<Q<B, Tril>>
where
    B::FloatElem: std::ops::Div<B::FloatElem, Output = B::FloatElem>
        + std::ops::Mul<B::FloatElem, Output = B::FloatElem>
        + std::ops::Add<B::FloatElem, Output = B::FloatElem>
        + std::ops::Sub<B::FloatElem, Output = B::FloatElem>
        + std::ops::Neg<Output = B::FloatElem>
        + std::ops::AddAssign<B::FloatElem>,
{
    let tril_type = tril.get_quantity_type();
    register_map1(tril_type, TrilInvert, tril)
}
