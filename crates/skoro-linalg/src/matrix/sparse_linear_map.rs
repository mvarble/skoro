//! A fixed sparse linear map `y = K x`, with the nonzero entries of `K` supplied from the host.
//!
//! [`FixedLinearMap`](super::FixedLinearMap) stores `K` as a dense backend tensor. This operation
//! stores only the nonzero entries of `K`, as host triplets, and applies them
//! by gathering the entries of `x` they read and scatter-adding the scaled values into `y`. That
//! matters when `K` is large but sparse: the Gaussian Fisher root's map has O(d⁵) entries and only
//! O(d³) nonzeros. The index and coefficient tensors are built on the input's own device at every
//! call, so the map is never tied to a device.

use std::collections::BTreeMap;
use std::sync::Arc;

use burn::tensor::IndexingUpdateOp;
use skoro_backend::{Backend, CotangentFrame, Int, Point, Tensor, TensorData, host};
use skoro_core::error::SkoroResult;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{map::register_map1, q::Q};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

use super::linear_map::check_map_shape;

/// The nonzero entries of `K`: `K[targets[i], sources[i]] = coefficients[i]`, one per position.
#[derive(Debug)]
struct Entries {
    targets: Vec<i64>,
    sources: Vec<i64>,
    coefficients: Vec<f64>,
}

/// The linear map `x ↦ K x` for a fixed sparse `K` of shape `[rows, cols]`.
///
/// Cloning is cheap: the entries are shared.
#[derive(Debug, Clone)]
pub struct SparseLinearMap {
    rows: usize,
    cols: usize,
    entries: Arc<Entries>,
}

impl SparseLinearMap {
    /// The map whose `rows × cols` matrix has the given `(row, col, value)` entries, and zero
    /// everywhere else.
    ///
    /// Entries at the same position are summed, and a position whose entries sum to exactly zero
    /// is dropped.
    ///
    /// # Panics
    ///
    /// If an entry lies outside `rows × cols`.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: impl IntoIterator<Item = (usize, usize, f64)>,
    ) -> Self {
        let mut merged = BTreeMap::new();
        for (row, col, value) in triplets {
            assert!(
                row < rows && col < cols,
                "entry ({row}, {col}) lies outside a {rows} × {cols} map"
            );
            *merged.entry((row, col)).or_insert(0.0) += value;
        }
        let mut entries = Entries {
            targets: Vec::with_capacity(merged.len()),
            sources: Vec::with_capacity(merged.len()),
            coefficients: Vec::with_capacity(merged.len()),
        };
        for ((row, col), value) in merged {
            if value != 0.0 {
                entries.targets.push(row as i64);
                entries.sources.push(col as i64);
                entries.coefficients.push(value);
            }
        }
        Self {
            rows,
            cols,
            entries: Arc::new(entries),
        }
    }

    /// The map's `[rows, cols]` shape.
    pub fn dims(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    /// The number of nonzero entries stored.
    pub fn nnz(&self) -> usize {
        self.entries.coefficients.len()
    }

    /// Gather axis 1 of `x` at `from`, scale by the coefficients, and scatter-add the result into
    /// positions `to` of an axis of length `len`.
    ///
    /// With `from = sources` and `to = targets` this is `K x`; swapping them gives `Kᵀ x`, since
    /// the adjoint reads the same entries in the opposite direction.
    fn gather_scatter<B: Backend, const D: usize>(
        &self,
        x: Tensor<B, D>,
        from: &[i64],
        to: &[i64],
        len: usize,
    ) -> Tensor<B, D> {
        let device = x.device();
        let dtype = x.dtype();
        let mut shape = x.dims();
        shape[1] = len;
        let zeros = Tensor::zeros(shape, (&device, dtype));
        let nnz = from.len();
        if nnz == 0 {
            return zeros;
        }
        let indices = |ids: &[i64]| {
            Tensor::<B, 1, Int>::from_data(
                TensorData::new(ids.to_vec(), [nnz]).convert::<B::IntElem>(),
                &device,
            )
        };
        let mut coefficient_shape = [1; D];
        coefficient_shape[1] = nnz;
        let coefficients =
            host::write::<B, f64, 1>(self.entries.coefficients.clone(), [nnz], &device, dtype)
                .reshape(coefficient_shape);
        // Several entries can write one output position; `Add` sums them.
        zeros.select_assign(
            1,
            indices(to),
            x.select(1, indices(from)) * coefficients,
            IndexingUpdateOp::Add,
        )
    }

    /// `K x` along axis 1: `[batch, cols, ..]` to `[batch, rows, ..]`.
    fn apply<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let entries = &self.entries;
        self.gather_scatter(x, &entries.sources, &entries.targets, self.rows)
    }

    /// `Kᵀ ω` along axis 1: `[batch, rows, ..]` to `[batch, cols, ..]`.
    fn apply_adjoint<B: Backend, const D: usize>(&self, omega: Tensor<B, D>) -> Tensor<B, D> {
        let entries = &self.entries;
        self.gather_scatter(omega, &entries.targets, &entries.sources, self.cols)
    }
}

impl<B: Backend> DiffOperation<B> for SparseLinearMap {
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
        let y = self.apply(x);
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(y),
            DiffModeState::Jvp(tangents) => {
                let [dx] = tangents.try_into().unwrap();
                DiffOperationForward::jvp(y, dx.map(|dx| self.apply(dx)))
            }
            DiffModeState::Vjp(diffs) => DiffOperationForward::vjp(y, diffs[0]),
            DiffModeState::Hvp { diffs, tangents } => {
                let [dx] = tangents.try_into().unwrap();
                DiffOperationForward::hvp(y, dx.map(|dx| self.apply(dx)), diffs[0], ())
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![state.then(|| self.apply_adjoint(omega))]
    }

    fn hvp(
        &self,
        state: &Self::VjpState,
        _: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        // Linear, so the second-order term vanishes wherever the first-order one exists, shaped
        // like the input's cotangent.
        let [batch, _, frames] = omega.dims();
        vec![
            state.then(|| {
                Tensor::zeros([batch, self.cols, frames], (&omega.device(), omega.dtype()))
            }),
        ]
    }
}

/// Register `map` applied to `x` as a quantity of type `out`.
///
/// Returns [`SkoroError::ShapeMismatch`](skoro_core::error::SkoroError::ShapeMismatch) if the map's
/// rows disagree with `out`'s dimension (checked first) or its columns with `x`'s manifold
/// dimension.
pub fn sparse_linear_map<B, I, O>(x: &Q<B, I>, out: O, map: SparseLinearMap) -> SkoroResult<Q<B, O>>
where
    B: Backend,
    I: ContinuousQuantity<B>,
    O: ContinuousQuantity<B>,
{
    check_map_shape(x, &out, map.dims())?;
    register_map1(out, map, x)
}
