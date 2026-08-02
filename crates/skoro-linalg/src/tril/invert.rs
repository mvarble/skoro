use skoro_backend::{Backend, CotangentFrame, Element, Point, Tensor, TensorData};
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
fn invert_packed_lower_triangular<F: Element>(packed_l: &[F], batches: usize) -> Vec<F> {
    let len = packed_l.len();
    let k = len / batches;
    let n = k_to_n(k);
    assert_eq!(len, batches * n * (n + 1) / 2);
    let mut inv: Vec<f64> = vec![0.0; packed_l.len()];
    for b in 0..batches {
        let mut offset_i = b * k;
        for i in 0..n {
            // 1. Diagonal element: X_{i,i} = 1.0 / L_{i,i}
            let l_ii = packed_l[offset_i + i].to_f64();
            let inv_l_ii = 1.0 / l_ii;
            inv[offset_i + i] = inv_l_ii;

            // 2. Off-diagonal elements in row i: X_{i,j} for j < i
            for j in 0..i {
                let mut sum = 0.0;

                // To compute the dot product, we need the offset for row k
                // We start k at j, so the initial offset for row j is j*(j+1)/2
                let mut offset_k = j * (j + 1) / 2;

                for k in j..i {
                    // sum += L_{i,k} * X_{k,j}
                    sum += packed_l[offset_i + k].to_f64() * inv[offset_k + j];

                    // Advance offset_k to the next row (k + 1)
                    offset_k += k + 1;
                }

                inv[offset_i + j] = -sum * inv_l_ii;
            }

            // Advance offset_i to the next row (i + 1)
            offset_i += i + 1;
        }
    }

    inv.into_iter().map(F::from_elem).collect()
}

impl<B: Backend> DiffOperation<B> for TrilInvert {
    type VjpState = Option<Tensor<B, 2>>; // A^{-1}
    type HvpState = Option<Tensor<B, 2>>; // ΔA
    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::into_float).collect();
        let [mat_packed] = inputs.try_into().unwrap();
        let mat_elems = mat_packed
            .clone()
            .into_data()
            .to_vec::<B::FloatElem>()
            .unwrap();
        let shape = mat_packed.shape();
        let inv_mat_elems = invert_packed_lower_triangular(&mat_elems, shape[0]);
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
            DiffModeState::Hvp(tvecs) => {
                let [dmat_packed] = tvecs.try_into().unwrap();
                DiffOperationForward::hvp(
                    inv_mat_packed.clone(),
                    dmat_packed.as_ref().map(|dmat_packed| {
                        let inv_mat: Tensor<B, 3> = unpack(inv_mat_packed.clone());
                        let dmat: Tensor<B, 3> = unpack(dmat_packed.clone());
                        -pack(inv_mat.clone().matmul(dmat).matmul(inv_mat))
                    }),
                    if dmat_packed.is_some() {
                        Some(inv_mat_packed)
                    } else {
                        None
                    },
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

pub fn invert<B: Backend>(tril: &Q<B, Tril>) -> Q<B, Tril> {
    let tril_type = tril.get_quantity_type();
    register_map1(tril_type, TrilInvert, tril)
}
