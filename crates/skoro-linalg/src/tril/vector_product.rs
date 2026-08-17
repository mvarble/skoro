use skoro_backend::{Backend, CotangentFrame, Point, Tensor};
use skoro_core::error::{SkoroError, SkoroResult};
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{map::register_map2, q::Q};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    sum_opt_tensors,
};
use skoro_riemann::euclidean::{Euclidean, EuclideanD};

use super::{Tril, pack, unpack};
use crate::matrix::right_broadcast_matmul;

/// Multiplication of a lower-triangular matrix (with diagonals) against a vector.
#[derive(Debug, Clone, Copy)]
pub struct TrilVectorProduct;

impl<B: Backend> DiffOperation<B> for TrilVectorProduct {
    type VjpState = [Option<Tensor<B, 3>>; 2];
    type HvpState = [Option<Tensor<B, 3>>; 2];

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::unwrap_float).collect();
        let [mat_packed, v] = inputs.try_into().unwrap();
        let [b, n] = v.dims();
        let mat_tril = unpack(mat_packed);

        let v = v.unsqueeze_dim(2);
        let output = mat_tril.clone().matmul(v.clone());
        let output = output.reshape([b, n]);

        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::ForwardOnly { output },
            DiffModeState::Jvp(tvecs) => {
                let [dmat_packed, dv] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    output,
                    sum_opt_tensors([
                        dmat_packed.map(|dmat_packed| {
                            right_broadcast_matmul(unpack(dmat_packed), v.clone()).squeeze_dim(2)
                        }),
                        dv.map(|dv| mat_tril.matmul(dv)),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [diff_mat, diff_v] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    output,
                    [
                        if diff_mat { Some(v) } else { None },
                        if diff_v { Some(mat_tril) } else { None },
                    ],
                )
            }
            DiffModeState::Hvp {
                diffs,
                tangents: cvecs,
            } => {
                let [dmat_packed, dv] = cvecs.try_into().unwrap();
                let [diff_mat, diff_v] = diffs.try_into().unwrap();
                DiffOperationForward::hvp(
                    output,
                    sum_opt_tensors([
                        dmat_packed.as_ref().map(|dmat_packed| {
                            unpack(dmat_packed.clone()).matmul(v.clone()).squeeze_dim(2)
                        }),
                        dv.as_ref().map(|dv| {
                            mat_tril
                                .clone()
                                .matmul(dv.clone().unsqueeze_dim(2))
                                .squeeze_dim(2)
                        }),
                    ]),
                    [
                        if diff_mat { Some(v) } else { None },
                        if diff_v { Some(mat_tril) } else { None },
                    ],
                    [
                        dv.map(|dv| dv.unsqueeze_dim(2)),
                        dmat_packed.map(|dmat_packed| unpack(dmat_packed)),
                    ],
                )
            }
        }
    }

    fn vjp(
        &self,
        [v, mat_tril]: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        let [b, n, f] = omega.dims();

        let dmat_tril = v.as_ref().map(|v| {
            let omega_reshaped = omega.clone().reshape([b, n, 1, f]);
            let v_reshaped = v.clone().reshape([b, 1, n, 1]);
            pack(omega_reshaped.mul(v_reshaped))
        });

        let dv = mat_tril.as_ref().map(|mat_tril| {
            let l_t = mat_tril.clone().transpose();
            l_t.matmul(omega.clone())
        });

        vec![dmat_tril, dv]
    }

    fn hvp(
        &self,
        _: &Self::VjpState,
        [dv, dmat_tril]: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        let [b, n, f] = omega.dims();

        let hd_mat_tril = dv.as_ref().map(|dv| {
            let omega_reshaped = omega.clone().reshape([b, n, 1, f]);
            let dv_reshaped = dv.clone().reshape([b, 1, n, 1]);
            pack(omega_reshaped.mul(dv_reshaped))
        });

        let hd_v = dmat_tril.as_ref().map(|dmat_tril| {
            let dl_t = dmat_tril.clone().transpose();
            dl_t.matmul(omega.clone())
        });

        vec![hd_mat_tril, hd_v]
    }
}

pub trait TrilVector<B: Backend>: ContinuousQuantity<B> {
    fn check_compatibility(&self, tril: &Tril) -> bool;
}
impl<B: Backend> TrilVector<B> for EuclideanD {
    fn check_compatibility(&self, tril: &Tril) -> bool {
        self.0 == tril.0
    }
}
impl<B: Backend, const D: usize> TrilVector<B> for Euclidean<D> {
    fn check_compatibility(&self, tril: &Tril) -> bool {
        D == tril.0
    }
}

pub fn vector_product<B: Backend, T: TrilVector<B>>(
    tril: &Q<B, Tril>,
    vector: &Q<B, T>,
) -> SkoroResult<Q<B, T>> {
    let ttype = tril.get_quantity_type();
    let vtype = vector.get_quantity_type();
    if vtype.check_compatibility(&ttype) {
        register_map2(vtype, TrilVectorProduct, (tril, vector))
    } else {
        Err(SkoroError::UnsupportedVectorProduct {
            tril: format!("{ttype:?}"),
            vector: format!("{vtype:?}"),
        })
    }
}
