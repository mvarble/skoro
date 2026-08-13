//! The host-resident matrix the estimators compute on.
//!
//! Everything in this module is `f64` and row-major, and a [`Dense`] is always exactly one
//! matrix. [`Dense::from_tensor`] enforces that: a `[b, m, n]` tensor with `b != 1` is
//! [`SkoroError::BatchNotSingleton`], never silently slice 0.
//!
//! That is a statement about *this type*, not about its callers. A batch of `N` independent
//! problems splits into `N` of these ([`Dense::split_tensor`], [`Dense::split_vector`]) and
//! stacks back afterwards ([`Dense::stack_vector`]), which is how the batch axis reaches the
//! factorizations without any of them growing one.

use skoro_backend::{Backend, Tensor, TensorData};
use skoro_core::error::{SkoroError, SkoroResult};

/// One dense matrix, host-resident, always `f64`.
///
/// Row-major: element `(r, c)` is `data[r * cols + c]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Dense {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Dense {
    /// An `rows × cols` matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// The `n × n` identity.
    pub fn identity(n: usize) -> Self {
        Self::scaled_identity(n, 1.0)
    }

    /// `s · I_n`.
    ///
    /// The row block a damped least-squares step augments with is `√λ I`, which is why
    /// this exists rather than a general scalar multiply.
    pub fn scaled_identity(n: usize, s: f64) -> Self {
        let mut out = Self::zeros(n, n);
        for i in 0..n {
            out.data[i * (n + 1)] = s;
        }
        out
    }

    /// From row-major data. Errors [`SkoroError::ShapeMismatch`] if the length disagrees.
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<f64>) -> SkoroResult<Self> {
        if data.len() != rows * cols {
            return Err(SkoroError::ShapeMismatch {
                expected: rows * cols,
                found: data.len(),
            });
        }
        Ok(Self { rows, cols, data })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The row-major backing store.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Element `(row, col)`. Panics out of range — an index error here is a bug in the
    /// caller's shape arithmetic, not a user-recoverable condition.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
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
        Ok(Self {
            rows,
            cols,
            data: host_f64(t.into_data()),
        })
    }

    /// To `[1, m, n]`, converted back to the backend's float element.
    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 3> {
        Tensor::from_data(
            TensorData::new(self.data.clone(), [1, self.rows, self.cols]).convert::<B::FloatElem>(),
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
        Ok(Self {
            rows: n,
            cols: 1,
            data: host_f64(t.into_data()),
        })
    }

    /// To `[1, rows * cols]`, which for a column is `[1, n]`.
    pub fn to_vector<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
        Tensor::from_data(
            TensorData::new(self.data.clone(), [1, self.rows * self.cols])
                .convert::<B::FloatElem>(),
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
        Ok(data
            .chunks_exact(rows * cols)
            .take(batch)
            .map(|chunk| Self {
                rows,
                cols,
                data: chunk.to_vec(),
            })
            .collect())
    }

    /// One `n × 1` column per batch row of a `[N, n]` tensor.
    ///
    /// The same tangent-and-residual-vectors-only restriction as [`Self::from_vector`]
    /// applies: `n` is a manifold dimension, not a storage size.
    pub fn split_vector<B: Backend>(t: Tensor<B, 2>) -> SkoroResult<Vec<Dense>> {
        let [batch, n] = t.dims();
        let data = host_f64(t.into_data());
        Ok(data
            .chunks_exact(n)
            .take(batch)
            .map(|chunk| Self {
                rows: n,
                cols: 1,
                data: chunk.to_vec(),
            })
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
        let n = first.rows;
        let mut data = Vec::with_capacity(columns.len() * n);
        for column in columns {
            if column.cols != 1 {
                return Err(SkoroError::ShapeMismatch {
                    expected: 1,
                    found: column.cols,
                });
            }
            if column.rows != n {
                return Err(SkoroError::ShapeMismatch {
                    expected: n,
                    found: column.rows,
                });
            }
            data.extend_from_slice(&column.data);
        }
        Ok(Tensor::from_data(
            TensorData::new(data, [columns.len(), n]).convert::<B::FloatElem>(),
            device,
        ))
    }

    /// Stack `[self; other]` — the row augmentation a damped least-squares step needs.
    ///
    /// Errors [`SkoroError::ShapeMismatch`] if the column counts disagree.
    pub fn vstack(&self, other: &Self) -> SkoroResult<Self> {
        if self.cols != other.cols {
            return Err(SkoroError::ShapeMismatch {
                expected: self.cols,
                found: other.cols,
            });
        }
        let mut data = Vec::with_capacity(self.data.len() + other.data.len());
        data.extend_from_slice(&self.data);
        data.extend_from_slice(&other.data);
        Ok(Self {
            rows: self.rows + other.rows,
            cols: self.cols,
            data,
        })
    }

    /// Elementwise negation. `−r` is the right-hand side of every least-squares step.
    pub fn neg(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|v| -v).collect(),
        }
    }

    /// `selfᵀ`.
    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        out
    }

    /// `self · other`. Errors [`SkoroError::ShapeMismatch`] on an inner-dimension
    /// disagreement.
    pub fn matmul(&self, other: &Self) -> SkoroResult<Self> {
        if self.cols != other.rows {
            return Err(SkoroError::ShapeMismatch {
                expected: self.cols,
                found: other.rows,
            });
        }
        let mut out = Self::zeros(self.rows, other.cols);
        for r in 0..self.rows {
            for k in 0..self.cols {
                let a = self.data[r * self.cols + k];
                if a == 0.0 {
                    continue;
                }
                for c in 0..other.cols {
                    out.data[r * other.cols + c] += a * other.data[k * other.cols + c];
                }
            }
        }
        Ok(out)
    }

    /// `½‖self‖²` over every element — the least-squares objective when `self` is the
    /// stacked residual column.
    pub fn half_sum_of_squares(&self) -> f64 {
        0.5 * self.data.iter().map(|v| v * v).sum::<f64>()
    }

    /// The Euclidean norm over every element.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
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
