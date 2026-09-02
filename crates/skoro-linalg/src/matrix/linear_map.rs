//! A fixed (non-differentiable) linear map `y = K x`, for turning a host-computed matrix into
//! a graph node without a distribution being able to add constants of its own (`Q.client` is
//! `pub(crate)` — see `crates/skoro-model/src/q.rs`).

use skoro_backend::{Backend, CotangentFrame, Point, Tensor, TensorData};
use skoro_core::error::{SkoroError, SkoroResult};
use skoro_core::kind::Continuous;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{map::register_map1, q::Q};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

/// `K`, shape `[out, in]`, fixed at construction.
#[derive(Debug, Clone)]
pub struct FixedLinearMap<B: Backend> {
    matrix: Tensor<B, 2>,
}

impl<B: Backend> FixedLinearMap<B> {
    pub fn new(matrix: Tensor<B, 2>) -> Self {
        Self { matrix }
    }

    /// Build `K` from row-major host data, through `TensorData` — never `Tensor::from_floats`,
    /// which rounds through `f32` even on an `f64` backend.
    pub fn from_row_major(rows: usize, cols: usize, data: &[f64], device: &B::Device) -> Self {
        let tensor = Tensor::<B, 2>::from_data(
            TensorData::new(data.to_vec(), [rows, cols]).convert::<B::FloatElem>(),
            device,
        );
        Self::new(tensor)
    }

    fn dims(&self) -> [usize; 2] {
        self.matrix.dims()
    }

    /// `K` broadcast to `[batch, out, in]` — `K` itself carries no batch dimension, so every
    /// frame-shaped mode has to repeat it explicitly; burn's `matmul` does not implicitly
    /// broadcast a batch-1 operand against a wider one.
    fn batched(&self, batch: usize) -> Tensor<B, 3> {
        self.matrix
            .clone()
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, batch)
    }

    /// `Kᵀ` broadcast to `[batch, in, out]`.
    fn batched_transpose(&self, batch: usize) -> Tensor<B, 3> {
        self.matrix
            .clone()
            .transpose()
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, batch)
    }
}

impl<B: Backend> DiffOperation<B> for FixedLinearMap<B> {
    type VjpState = bool;
    type HvpState = ();

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::unwrap_float).collect();
        let [x] = inputs.try_into().unwrap();
        let batch = x.dims()[0];
        // Plain rank-2 matmul: x is [batch, in] as a literal matrix, Kᵀ is [in, out] — no
        // batching broadcast needed here, unlike the frame-shaped modes below.
        let y = x.clone().matmul(self.matrix.clone().transpose());
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(y),
            DiffModeState::Jvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                let dy = dx.map(|dx| self.batched(batch).matmul(dx));
                DiffOperationForward::jvp(y, dy)
            }
            DiffModeState::Vjp(diffs) => DiffOperationForward::vjp(y, diffs[0]),
            DiffModeState::Hvp { diffs, tangents } => {
                let [dx] = tangents.try_into().unwrap();
                let dy = dx.map(|dx| {
                    self.batched(batch)
                        .matmul(dx.unsqueeze_dim(2))
                        .squeeze_dim(2)
                });
                DiffOperationForward::hvp(y, dy, diffs[0], ())
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        let batch = omega.dims()[0];
        vec![state.then(|| self.batched_transpose(batch).matmul(omega))]
    }

    fn hvp(
        &self,
        state: &Self::VjpState,
        _: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        // A linear map has zero Hessian, but the cotangent lives on `x`, not `y`: its shape is
        // `[batch, in, frame]`, which only matches `omega`'s `[batch, out, frame]` when
        // `in == out`. Build the zero explicitly at the right shape rather than reusing `omega`'s.
        let [batch, _, frame] = omega.dims();
        let [_, cols] = self.dims();
        vec![state.then(|| Tensor::<B, 3>::zeros([batch, cols, frame], &omega.device()))]
    }
}

/// Register `x` mapped through a fixed linear map, checking `map`'s shape against `x`'s and
/// `out`'s manifold dimensions first.
pub fn linear_map<B, I, O>(x: &Q<B, I>, out: O, map: FixedLinearMap<B>) -> SkoroResult<Q<B, O>>
where
    B: Backend,
    I: ContinuousQuantity<B>,
    O: ContinuousQuantity<B>,
{
    let [rows, cols] = map.dims();
    let out_dim = out.dimension();
    let in_dim = x.get_quantity_type::<Continuous>().dimension();
    if rows != out_dim {
        return Err(SkoroError::ShapeMismatch {
            expected: out_dim,
            found: rows,
        });
    }
    if cols != in_dim {
        return Err(SkoroError::ShapeMismatch {
            expected: in_dim,
            found: cols,
        });
    }
    register_map1(out, map, x)
}
