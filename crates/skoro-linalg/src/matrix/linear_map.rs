//! A fixed linear map `y = K x`, with `K` supplied from the host at construction.
//!
//! This is how a builder that cannot add constants of its own (a distribution only ever sees
//! quantities, never the client) still gets a known matrix into the graph: the matrix lives in
//! the operation, not in a node.

use skoro_backend::{Backend, CotangentFrame, Point, Tensor, TensorData};
use skoro_core::error::{SkoroError, SkoroResult};
use skoro_core::kind::Continuous;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{map::register_map1, q::Q};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

/// The linear map `x ↦ K x` for a fixed `K` of shape `[out, in]`.
#[derive(Debug, Clone)]
pub struct FixedLinearMap<B: Backend> {
    matrix: Tensor<B, 2>,
}

impl<B: Backend> FixedLinearMap<B> {
    /// The map whose matrix is `matrix`, shape `[out, in]`.
    pub fn new(matrix: Tensor<B, 2>) -> Self {
        Self { matrix }
    }

    /// The map whose `rows × cols` matrix is `data`, in row-major order.
    ///
    /// Built through `TensorData` so an `f64` backend keeps full precision.
    pub fn from_row_major(rows: usize, cols: usize, data: &[f64], device: &B::Device) -> Self {
        let data = TensorData::new(data.to_vec(), [rows, cols]).convert::<B::FloatElem>();
        Self::new(Tensor::from_data(data, device))
    }

    /// The matrix's `[out, in]` shape.
    pub fn dims(&self) -> [usize; 2] {
        self.matrix.dims()
    }
}

/// `M · frame` for a `[batch, n, f]` frame and an `[m, n]` matrix, giving `[batch, m, f]`.
///
/// Every column of every batch is a row of one rank-2 product, so nothing relies on the backend
/// broadcasting a batch dimension.
fn left_multiply_frame<B: Backend>(frame: Tensor<B, 3>, matrix: &Tensor<B, 2>) -> Tensor<B, 3> {
    let [batch, n, frames] = frame.dims();
    let rows = matrix.dims()[0];
    frame
        .swap_dims(1, 2)
        .reshape([batch * frames, n])
        .matmul(matrix.clone().transpose())
        .reshape([batch, frames, rows])
        .swap_dims(1, 2)
}

impl<B: Backend> DiffOperation<B> for FixedLinearMap<B> {
    /// Whether the input is differentiable.
    type VjpState = bool;
    type HvpState = ();

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::unwrap_float).collect();
        let [x] = inputs.try_into().unwrap();
        let transpose = self.matrix.clone().transpose();
        let y = x.matmul(transpose.clone());
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(y),
            DiffModeState::Jvp(tangents) => {
                let [dx] = tangents.try_into().unwrap();
                let dy = dx.map(|dx| left_multiply_frame(dx, &self.matrix));
                DiffOperationForward::jvp(y, dy)
            }
            DiffModeState::Vjp(diffs) => DiffOperationForward::vjp(y, diffs[0]),
            DiffModeState::Hvp { diffs, tangents } => {
                let [dx] = tangents.try_into().unwrap();
                let dy = dx.map(|dx| dx.matmul(transpose));
                DiffOperationForward::hvp(y, dy, diffs[0], ())
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        let transpose = self.matrix.clone().transpose();
        vec![state.then(|| left_multiply_frame(omega, &transpose))]
    }

    fn hvp(
        &self,
        state: &Self::VjpState,
        _: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        // Linear, so the second-order term vanishes wherever the first-order one exists. It is
        // shaped like the input's cotangent, `[batch, in, f]`, not like `omega`.
        let [batch, _, frames] = omega.dims();
        let cols = self.matrix.dims()[1];
        vec![state.then(|| Tensor::zeros([batch, cols, frames], &omega.device()))]
    }
}

/// Register `map` applied to `x` as a quantity of type `out`.
///
/// Returns [`SkoroError::ShapeMismatch`] if the matrix's rows disagree with `out`'s dimension
/// (checked first) or its columns with `x`'s manifold dimension.
pub fn linear_map<B, I, O>(x: &Q<B, I>, out: O, map: FixedLinearMap<B>) -> SkoroResult<Q<B, O>>
where
    B: Backend,
    I: ContinuousQuantity<B>,
    O: ContinuousQuantity<B>,
{
    let [rows, cols] = map.dims();
    if rows != out.dimension() {
        return Err(SkoroError::ShapeMismatch {
            expected: out.dimension(),
            found: rows,
        });
    }
    let input_dimension = x.get_quantity_type::<Continuous>().dimension();
    if cols != input_dimension {
        return Err(SkoroError::ShapeMismatch {
            expected: input_dimension,
            found: cols,
        });
    }
    register_map1(out, map, x)
}
