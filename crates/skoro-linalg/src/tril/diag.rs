use skoro_backend::{Backend, CotangentFrame, ElementConversion, Int, Point, Tensor, TensorData};
use skoro_core::error::SkoroResult;
use skoro_core::pack::NumComponents;
use skoro_model::{map::register_map1, q::Q};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};
use skoro_riemann::euclidean::EuclideanD;

use super::{Tril, k_to_n};

/// Diagonal of a lower-triangular matrix (with diagonal).
#[derive(Debug, Clone, Copy)]
pub struct TrilDiag;

fn diagonal<B: Backend, const D: usize>(tril: Tensor<B, D>) -> Tensor<B, D> {
    let indices = {
        let device = tril.device();
        let k = tril.dims()[1];
        let n = k_to_n(k);
        let mut indices = Vec::with_capacity(n);
        for i in 0..n {
            indices.push(B::IntElem::from_elem(i * (i + 3) / 2));
        }
        Tensor::<B, 1, Int>::from_data(
            TensorData::new(indices, [n]).convert::<B::IntElem>(),
            &device,
        )
    };
    tril.select(1, indices)
}

fn packed_diagonal_mask<B: Backend>(n: usize, device: &B::Device) -> Tensor<B, 1> {
    let k = n * (n + 1) / 2;
    let mut mask = vec![B::FloatElem::from_elem(0.0); k];
    for i in 0..n {
        let diag_idx = i * (i + 3) / 2;
        mask[diag_idx] = B::FloatElem::from_elem(1.0);
    }
    Tensor::<B, 1>::from_data(TensorData::new(mask, [k]).convert::<B::FloatElem>(), device)
}

impl<B: Backend> DiffOperation<B> for TrilDiag {
    type VjpState = bool;
    type HvpState = ();

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::into_float).collect();
        let [mat_packed] = inputs.try_into().unwrap();
        let diag = diagonal(mat_packed.clone());
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(diag),
            DiffModeState::Jvp(tvecs) => {
                let [dmat_packed] = tvecs.try_into().unwrap();
                if let Some(dmat_packed) = dmat_packed {
                    DiffOperationForward::jvp(diag, Some(diagonal(dmat_packed)))
                } else {
                    DiffOperationForward::jvp(diag, None)
                }
            }
            DiffModeState::Vjp(diffs) => DiffOperationForward::vjp(diag, diffs[0]),
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dmat_packed] = tvecs.try_into().unwrap();
                let tangent = dmat_packed.map(|dmat_packed| {
                    let [_, k] = dmat_packed.dims();
                    let n = k_to_n(k);
                    let mask = packed_diagonal_mask(n, &dmat_packed.device()).reshape([1, k]);
                    diagonal(dmat_packed * mask)
                });
                DiffOperationForward::hvp(diag, tangent, diffs[0], ())
            }
        }
    }

    fn vjp(
        &self,
        is_diff: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        if *is_diff {
            let device = omega.device();
            let [_, n, _] = omega.dims();
            let k = n * (n + 1) / 2;
            let mut indices = vec![B::IntElem::from_elem(0); k];
            let mut mask = vec![B::FloatElem::from_elem(0.0); k];
            for i in 0..n {
                let diag_idx = i * (i + 3) / 2;
                indices[diag_idx] = B::IntElem::from_elem(i);
                mask[diag_idx] = B::FloatElem::from_elem(1.0);
            }
            let indices_tensor = Tensor::<B, 1, Int>::from_data(
                TensorData::new(indices, [k]).convert::<B::IntElem>(),
                &device,
            );
            let mask_tensor = Tensor::<B, 3>::from_data(
                TensorData::new(mask, [1, k, 1]).convert::<B::FloatElem>(),
                &device,
            );
            let expanded = omega.select(1, indices_tensor);
            vec![Some(expanded * mask_tensor)]
        } else {
            vec![None]
        }
    }

    fn hvp(
        &self,
        is_diff: &Self::VjpState,
        _: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        if *is_diff {
            let device = omega.device();
            let [b, n, f] = omega.dims();
            let k = n * (n + 1) / 2;
            vec![Some(Tensor::<B, 3>::zeros([b, k, f], &device))]
        } else {
            vec![None]
        }
    }
}

pub fn diag<B: Backend>(tril: &Q<B, Tril>) -> SkoroResult<Q<B, EuclideanD>> {
    let k = tril.get_quantity_type().num_components();
    let n = k_to_n(k);
    register_map1(EuclideanD(n), TrilDiag, tril)
}
