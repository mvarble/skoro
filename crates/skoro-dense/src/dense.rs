//! The host-resident matrix the estimators compute on.
//!
//! Everything in this module is `f64`, and a [`Dense`] is always exactly one matrix.
//! [`Dense::from_tensor`] enforces that: a `[b, m, n]` tensor with `b != 1` is
//! [`SkoroError::BatchNotSingleton`], never silently slice 0.
//!
//! That is a statement about *this type*, not about its callers. A batch of `N` independent
//! problems splits into `N` of these ([`Dense::split_tensor`], [`Dense::split_vector`]) and
//! stacks back afterwards ([`Dense::stack_vector`]), which is how the batch axis reaches the
//! factorizations without any of them growing one.
//!
//! Storage is a `faer::Mat<f64>`, so the factorizations borrow it rather than copying it, and
//! [`Dense::as_faer`] / [`Dense::from_faer`] are the only crossings. The tensor boundary is
//! row-major, as a `burn` tensor is, and [`Dense::from_row_major`] /
//! [`Dense::to_row_major`] are where that order is fixed.

use faer::{Mat, MatRef};
use skoro_backend::{Backend, Tensor, TensorData};
use skoro_core::error::{SkoroError, SkoroResult};

/// One dense matrix, host-resident, always `f64`.
#[derive(Debug, Clone, PartialEq)]
pub struct Dense(Mat<f64>);

impl Dense {
    /// An `rows × cols` matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self(Mat::zeros(rows, cols))
    }

    /// The `n × n` identity.
    pub fn identity(n: usize) -> Self {
        Self(Mat::identity(n, n))
    }

    /// `s · I_n`.
    ///
    /// The row block a damped least-squares step augments with is `√λ I`, which is why
    /// this exists rather than a general scalar multiply.
    pub fn scaled_identity(n: usize, s: f64) -> Self {
        Self(Mat::from_fn(n, n, |r, c| if r == c { s } else { 0.0 }))
    }

    /// Element `(r, c)` is `f(r, c)`.
    pub fn from_fn(rows: usize, cols: usize, f: impl FnMut(usize, usize) -> f64) -> Self {
        Self(Mat::from_fn(rows, cols, f))
    }

    /// From row-major data. Errors [`SkoroError::ShapeMismatch`] if the length disagrees.
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<f64>) -> SkoroResult<Self> {
        if data.len() != rows * cols {
            return Err(SkoroError::ShapeMismatch {
                expected: rows * cols,
                found: data.len(),
            });
        }
        Ok(Self::from_fn(rows, cols, |r, c| data[r * cols + c]))
    }

    /// Wrap a `faer` matrix.
    pub fn from_faer(m: Mat<f64>) -> Self {
        Self(m)
    }

    /// Copy a `faer` view.
    pub fn from_faer_ref(m: MatRef<'_, f64>) -> Self {
        Self(m.to_owned())
    }

    /// The `faer` view the factorizations work on.
    pub fn as_faer(&self) -> MatRef<'_, f64> {
        self.0.as_ref()
    }

    /// The `faer` matrix itself.
    pub fn into_faer(self) -> Mat<f64> {
        self.0
    }

    pub fn rows(&self) -> usize {
        self.0.nrows()
    }

    pub fn cols(&self) -> usize {
        self.0.ncols()
    }

    /// The elements in row-major order: `(r, c)` at `r * cols + c`.
    pub fn to_row_major(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.rows() * self.cols());
        for r in 0..self.rows() {
            for c in 0..self.cols() {
                out.push(self.0[(r, c)]);
            }
        }
        out
    }

    /// Element `(row, col)`. Panics out of range — an index error here is a bug in the
    /// caller's shape arithmetic, not a user-recoverable condition.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.0[(row, col)]
    }

    /// Set element `(row, col)`. Panics out of range, as [`Self::get`] does.
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.0[(row, col)] = value;
    }

    /// The `rows × cols` block whose top-left corner is `(row, col)`, copied.
    ///
    /// Panics out of range, as [`Self::get`] does.
    pub fn block(&self, row: usize, col: usize, rows: usize, cols: usize) -> Self {
        Self(self.0.get(row..row + rows, col..col + cols).to_owned())
    }

    /// Write `block` into `self` with its top-left corner at `(row, col)`.
    ///
    /// Panics out of range, as [`Self::get`] does.
    pub fn set_block(&mut self, row: usize, col: usize, block: &Self) {
        self.0
            .get_mut(row..row + block.rows(), col..col + block.cols())
            .copy_from(block.0.as_ref());
    }

    /// From `[1, m, n]`.
    ///
    /// Errors [`SkoroError::BatchNotSingleton`] on any other leading dimension rather than
    /// silently taking slice 0.
    pub fn from_tensor<B: Backend>(t: Tensor<B, 3>) -> SkoroResult<Self> {
        let [batch, rows, cols] = t.dims();
        if batch != 1 {
            return Err(SkoroError::BatchNotSingleton { found: batch });
        }
        Self::from_row_major(rows, cols, host_f64(t.into_data()))
    }

    /// To `[1, m, n]`, stored at the device's float policy.
    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 3> {
        Tensor::from_data(
            TensorData::new(self.to_row_major(), [1, self.rows(), self.cols()]),
            device,
        )
    }

    /// From `[1, n]` — an `n × 1` column.
    ///
    /// **Tangent and residual vectors only**: a `δ`, an `r`, an innovation. A point must
    /// never come through here. [`Point`](skoro_backend::Point),
    /// [`TangentVector`](skoro_backend::TangentVector) and
    /// [`CotangentVector`](skoro_backend::CotangentVector) are all `Tensor<B, 2>`, so the
    /// compiler cannot enforce this; `n` is a manifold *dimension*, not a storage size.
    pub fn from_vector<B: Backend>(t: Tensor<B, 2>) -> SkoroResult<Self> {
        let [batch, n] = t.dims();
        if batch != 1 {
            return Err(SkoroError::BatchNotSingleton { found: batch });
        }
        Self::from_row_major(n, 1, host_f64(t.into_data()))
    }

    /// To `[1, rows * cols]`, which for a column is `[1, n]`.
    pub fn to_vector<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
        Tensor::from_data(
            TensorData::new(self.to_row_major(), [1, self.rows() * self.cols()]),
            device,
        )
    }

    /// One `Dense` per batch row of a `[N, m, n]` tensor.
    ///
    /// This is how a batched consumer reaches the factorizations without any of them
    /// growing a batch axis: `N` independent problems become `N` independent matrices, each
    /// still exactly what [`crate::factor`] expects.
    ///
    /// **One host transfer, then host-side slicing.** The obvious alternative — `narrow` on
    /// dim 0 followed by [`Self::from_tensor`] per row — is `N` separate device reads, and
    /// on a cube backend each one fences the queue.
    pub fn split_tensor<B: Backend>(t: Tensor<B, 3>) -> SkoroResult<Vec<Dense>> {
        let [batch, rows, cols] = t.dims();
        let data = host_f64(t.into_data());
        if rows * cols == 0 {
            return Ok((0..batch).map(|_| Self::zeros(rows, cols)).collect());
        }
        Ok(data
            .chunks_exact(rows * cols)
            .take(batch)
            .map(|chunk| Self::from_fn(rows, cols, |r, c| chunk[r * cols + c]))
            .collect())
    }

    /// One `n × 1` column per batch row of a `[N, n]` tensor.
    ///
    /// The same tangent-and-residual-vectors-only restriction as [`Self::from_vector`]
    /// applies: `n` is a manifold dimension, not a storage size.
    pub fn split_vector<B: Backend>(t: Tensor<B, 2>) -> SkoroResult<Vec<Dense>> {
        let [batch, n] = t.dims();
        let data = host_f64(t.into_data());
        if n == 0 {
            return Ok((0..batch).map(|_| Self::zeros(0, 1)).collect());
        }
        Ok(data
            .chunks_exact(n)
            .take(batch)
            .map(|chunk| Self::from_fn(n, 1, |r, _| chunk[r]))
            .collect())
    }

    /// `N` columns of height `n` back into one `[N, n]` tensor — the inverse of
    /// [`Self::split_vector`].
    ///
    /// Errors [`SkoroError::ShapeMismatch`] on an empty slice, on any element that is not a
    /// single column, or on a ragged set of heights. An empty slice is rejected rather than
    /// producing a zero-row tensor because `n` would be unrecoverable.
    pub fn stack_vector<B: Backend>(
        columns: &[Dense],
        device: &B::Device,
    ) -> SkoroResult<Tensor<B, 2>> {
        let Some(first) = columns.first() else {
            return Err(SkoroError::ShapeMismatch {
                expected: 1,
                found: 0,
            });
        };
        let n = first.rows();
        let mut data = Vec::with_capacity(columns.len() * n);
        for column in columns {
            if column.cols() != 1 {
                return Err(SkoroError::ShapeMismatch {
                    expected: 1,
                    found: column.cols(),
                });
            }
            if column.rows() != n {
                return Err(SkoroError::ShapeMismatch {
                    expected: n,
                    found: column.rows(),
                });
            }
            data.extend((0..n).map(|r| column.get(r, 0)));
        }
        Ok(Tensor::from_data(
            TensorData::new(data, [columns.len(), n]),
            device,
        ))
    }

    /// Stack `[self; other]` — the row augmentation a damped least-squares step needs.
    ///
    /// Errors [`SkoroError::ShapeMismatch`] if the column counts disagree.
    pub fn vstack(&self, other: &Self) -> SkoroResult<Self> {
        Self::vstack_all(&[self, other])
    }

    /// Stack every matrix in `parts` vertically, in order.
    ///
    /// Errors [`SkoroError::ShapeMismatch`] on an empty slice or if the column counts disagree.
    pub fn vstack_all(parts: &[&Self]) -> SkoroResult<Self> {
        let Some(first) = parts.first() else {
            return Err(SkoroError::ShapeMismatch {
                expected: 1,
                found: 0,
            });
        };
        let cols = first.cols();
        let mut rows = 0;
        for part in parts {
            if part.cols() != cols {
                return Err(SkoroError::ShapeMismatch {
                    expected: cols,
                    found: part.cols(),
                });
            }
            rows += part.rows();
        }
        let mut out = Self::zeros(rows, cols);
        let mut at = 0;
        for part in parts {
            out.set_block(at, 0, part);
            at += part.rows();
        }
        Ok(out)
    }

    /// Elementwise negation. `−r` is the right-hand side of every least-squares step.
    pub fn neg(&self) -> Self {
        Self(-&self.0)
    }

    /// `selfᵀ`.
    pub fn transpose(&self) -> Self {
        Self(self.0.transpose().to_owned())
    }

    /// `self · other`. Errors [`SkoroError::ShapeMismatch`] on an inner-dimension
    /// disagreement.
    pub fn matmul(&self, other: &Self) -> SkoroResult<Self> {
        if self.cols() != other.rows() {
            return Err(SkoroError::ShapeMismatch {
                expected: self.cols(),
                found: other.rows(),
            });
        }
        Ok(Self(&self.0 * &other.0))
    }

    /// `self + other`. Errors [`SkoroError::ShapeMismatch`] if the shapes disagree.
    pub fn add(&self, other: &Self) -> SkoroResult<Self> {
        self.same_shape(other)?;
        Ok(Self(&self.0 + &other.0))
    }

    /// `self − other`. Errors [`SkoroError::ShapeMismatch`] if the shapes disagree.
    pub fn sub(&self, other: &Self) -> SkoroResult<Self> {
        self.same_shape(other)?;
        Ok(Self(&self.0 - &other.0))
    }

    /// `s · self`.
    pub fn scale(&self, s: f64) -> Self {
        Self::from_fn(self.rows(), self.cols(), |r, c| s * self.0[(r, c)])
    }

    /// `½‖self‖²` over every element — the least-squares objective when `self` is the
    /// stacked residual column.
    pub fn half_sum_of_squares(&self) -> f64 {
        let n = self.norm();
        0.5 * n * n
    }

    /// The Euclidean (Frobenius) norm over every element.
    pub fn norm(&self) -> f64 {
        self.0.norm_l2()
    }

    /// The largest absolute elementwise difference, for tolerance checks.
    ///
    /// Errors [`SkoroError::ShapeMismatch`] if the shapes disagree.
    pub fn max_abs_diff(&self, other: &Self) -> SkoroResult<f64> {
        self.same_shape(other)?;
        let mut worst: f64 = 0.0;
        for r in 0..self.rows() {
            for c in 0..self.cols() {
                worst = worst.max((self.0[(r, c)] - other.0[(r, c)]).abs());
            }
        }
        Ok(worst)
    }

    fn same_shape(&self, other: &Self) -> SkoroResult<()> {
        if self.rows() != other.rows() {
            return Err(SkoroError::ShapeMismatch {
                expected: self.rows(),
                found: other.rows(),
            });
        }
        if self.cols() != other.cols() {
            return Err(SkoroError::ShapeMismatch {
                expected: self.cols(),
                found: other.cols(),
            });
        }
        Ok(())
    }
}

/// `TensorData` to host `f64`, the inbound half of the boundary.
///
/// Mirrors the round-trip `TrilInvert` already performs
/// (`skoro-linalg/src/tril/invert.rs:87-94`), except that the target element type is
/// `f64` rather than `B::FloatElem`: this side of the boundary is always `f64`, whatever
/// the backend runs in.
fn host_f64(data: TensorData) -> Vec<f64> {
    data.convert::<f64>()
        .to_vec::<f64>()
        .expect("internal: tensor data converted to f64 is readable as f64")
}
