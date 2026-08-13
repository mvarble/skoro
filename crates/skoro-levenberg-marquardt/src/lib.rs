//! # skoro-levenberg-marquardt
//!
//! Minimize `½Σ‖rᵢ(x)‖²` over the quantities named by a [`TangentLayout`], where the `rᵢ`
//! are the whitened residuals a [`NlsStructure`] model's certificate recorded.
//!
//! Everything this crate needs was built elsewhere, which is the measure of whether the
//! foundations are right: [`step`] is four calls and no mathematics of its own —
//! `jacobian_vjp` for `J`, `evaluate_ids` for `r`, [`Dense::vstack`] to augment with
//! `√λ I`, and [`lstsq`] for `δ`.
//!
//! ## The batch axis is N independent problems
//!
//! A `PointMap` at batch `N` is `N` separate optimizations sharing one graph — the shape a
//! multi-start search has, where several starting points are run at once to see which local
//! optima they find. Each row carries **its own `λ`, its own cost, its own accept/reject and
//! its own termination**; row 3 may converge while row 5 is still stepping.
//!
//! The split of work follows from that. One `jacobian_vjp` pass produces every row's
//! Jacobian, because frames and batch are different axes — so the expensive half is shared.
//! Only the small dense solves loop, one matrix at a time, which is exactly what
//! `skoro-dense` is for.
//!
//! **What the batch axis cannot do is carry different *data* per row.** Constants and
//! observations are rank-1 storage that the compiler expands to the running batch, so every
//! row sees the same dataset. "One dataset, N starting points" works; "N datasets" does not.
//! And because `Parameter` values are *checked* against the running batch rather than
//! expanded, **every parameter in `points` must be at batch N** — free or not.
//!
//! ## The two things that are easy to get wrong
//!
//! **The objective is only the model's negative log-density under a precondition.**
//! `−log p = ½Σ‖rᵢ‖² + Σnᵢ(θ)`, so dropping the normalizers is valid exactly when they do
//! not vary with what is being optimized. [`solve`] checks this before its first iteration
//! and fails there rather than optimizing the wrong function. See [`precondition`].
//!
//! **Steps are retractions, not vector addition.** `x ← retract(x, δ)`, which is why `free`
//! is a layout of manifold *dimensions* — 3 for `SO3`, 6 for `SE3`, never 9 and 12 — and
//! why points never flatten.
//!
//! ## Where the arithmetic runs
//!
//! `J` and `δ` are the only tensor-typed values. Everything between them is host `f64`,
//! one matrix at a time, through `skoro-dense`. The damped step is solved by column-pivoted
//! QR on the row-augmented system rather than through `JᵀJ + λI`, so it runs at condition
//! `κ` rather than `κ²` — at *every* damping value, not only at `λ = 0`.

use std::collections::HashMap;

use skoro_backend::{Backend, Tensor};
use skoro_core::error::{SkoroError, SkoroResult};
use skoro_core::id::Id;
use skoro_dense::dense::Dense;
use skoro_dense::factor::lstsq;
use skoro_graph::point_map::PointMap;
use skoro_model::jacobian::jacobian_vjp;
use skoro_model::{NlsStructure, TangentLayout};

pub mod precondition;

pub use precondition::check_normalizers;

/// How the iteration is damped and when it stops.
///
/// Every field applies to each batch row independently; there is no shared budget except
/// [`max_iterations`](Self::max_iterations), which counts sweeps over the whole batch.
#[derive(Debug, Clone, PartialEq)]
pub struct LmOptions {
    /// Initial damping, the same for every row. `0.0` is an undamped step, exact on a
    /// linear problem.
    ///
    /// If an undamped step is ever *rejected*, that row's damping is introduced at
    /// [`lambda_increase`](Self::lambda_increase) — the only scale the caller supplied —
    /// since scaling zero would leave the row stuck.
    pub lambda: f64,
    /// Factor applied to a row's `λ` when its step fails to decrease the cost.
    pub lambda_increase: f64,
    /// Factor applied to a row's `λ` when its step succeeds.
    pub lambda_decrease: f64,
    /// Sweeps over the batch. A sweep steps every row that has not yet terminated.
    pub max_iterations: usize,
    /// A row stops once its `‖δ‖` falls to this.
    pub step_tolerance: f64,
    /// A row stops once an accepted step improves its cost by no more than this.
    pub cost_tolerance: f64,
}

impl Default for LmOptions {
    fn default() -> Self {
        Self {
            lambda: 1.0e-3,
            lambda_increase: 10.0,
            lambda_decrease: 0.1,
            max_iterations: 100,
            step_tolerance: 1.0e-10,
            cost_tolerance: 1.0e-12,
        }
    }
}

/// What happened to the batch, and to each row in it.
#[derive(Debug, Clone, PartialEq)]
pub struct LmReport {
    /// Sweeps performed. A sweep steps every row still active, so this is the *longest*
    /// row's iteration count, not the sum.
    pub iterations: usize,
    /// One entry per batch row, in batch order.
    pub batches: Vec<BatchReport>,
}

/// One row's outcome. At batch 1 this is the whole report.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchReport {
    /// Sweeps during which this row was still active.
    pub iterations: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub final_lambda: f64,
    pub termination: Termination,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    CostTolerance,
    StepTolerance,
    MaxIterations,
}

/// Minimize `½Σ‖rᵢ(x)‖²` over the quantities named by `free`, independently per batch row.
///
/// `points` is updated in place **through the manifold retraction**, never by vector
/// addition. It must hold a value for every `Parameter` / `Stochastic` leaf the residuals
/// reach, free or not, each at the batch size being optimized; observed quantities are held
/// by the graph and do not belong in it. Everything outside `free` is left untouched.
///
/// Calls [`check_normalizers`] before the first iteration and fails there rather than
/// silently optimizing the wrong objective.
///
/// # Errors
///
/// [`SkoroError::NormalizerNotConstant`] if the precondition fails, and whatever evaluation,
/// differentiation or the least-squares solve report. There is no *per-row* failure to
/// recover from: `lstsq` is column-pivoted QR, whose only error path is a shape
/// disagreement, and shapes are identical across rows — so an error here fires for every row
/// or for none.
pub fn solve<B, M>(
    model: &M,
    points: &mut PointMap<B>,
    free: &TangentLayout,
    options: &LmOptions,
) -> SkoroResult<LmReport>
where
    B: Backend,
    M: NlsStructure<B>,
{
    check_normalizers(model, free)?;
    let residuals = model.residual_layout()?;
    let device = device_of(points);
    let batch = batch_of(points);
    let width = free.total();

    let initial_cost = costs(model, points, &residuals)?;
    let mut cost = initial_cost.clone();
    let mut lambda = vec![options.lambda; batch];
    let mut active = vec![true; batch];
    let mut termination = vec![Termination::MaxIterations; batch];
    let mut row_iterations = vec![0usize; batch];
    let mut sweeps = 0;

    while sweeps < options.max_iterations && active.iter().any(|a| *a) {
        let mut columns = step_columns(model, points, free, &residuals, &lambda)?;

        // A vanishing step means that row has converged, whatever its cost did. Checking it
        // before spending a sweep is what lets an exact linear solve report one iteration
        // rather than a second one rediscovering `δ = 0`.
        for row in 0..batch {
            if !active[row] {
                columns[row] = Dense::zeros(width, 1);
            } else if columns[row].norm() <= options.step_tolerance {
                active[row] = false;
                termination[row] = Termination::StepTolerance;
                columns[row] = Dense::zeros(width, 1);
            }
        }
        if !active.iter().any(|a| *a) {
            break;
        }

        sweeps += 1;
        let trial = points.retract_flat(free, Dense::stack_vector::<B>(&columns, &device)?)?;
        let trial_cost = costs(model, &trial, &residuals)?;

        // Accept per row. Inactive rows already carry a zero step, so they are trivially
        // "accepted" and the whole-batch fast path below stays available.
        let mut accepted = vec![true; batch];
        for row in 0..batch {
            if active[row] {
                row_iterations[row] += 1;
                accepted[row] = trial_cost[row] < cost[row];
            }
        }

        if accepted.iter().all(|a| *a) {
            *points = trial;
        } else {
            // Roll the rejected rows back by zeroing their step and retracting again from
            // the *original* points. Two things make this the right shape:
            //
            // - a zero tangent is an exact no-op on every manifold — `Exp(0)` reaches the
            //   Taylor branch of the rotation coefficients, so there is no `0/0` and no
            //   approximation — which is what lets one retraction serve both outcomes;
            // - the selection happens **here, on the host, by replacing whole columns**,
            //   never by multiplying a tangent by a 0/1 mask. `0.0 * NaN` is `NaN`, and
            //   nothing downstream sanitizes it: a NaN `ω` would take the same Taylor
            //   branch and turn that row's rotation block to NaN. Multiplying would poison
            //   precisely the row being skipped.
            for row in 0..batch {
                if !accepted[row] {
                    columns[row] = Dense::zeros(width, 1);
                }
            }
            *points = points.retract_flat(free, Dense::stack_vector::<B>(&columns, &device)?)?;
        }

        for row in 0..batch {
            if !active[row] {
                continue;
            }
            if accepted[row] {
                let improvement = cost[row] - trial_cost[row];
                cost[row] = trial_cost[row];
                lambda[row] *= options.lambda_decrease;
                if improvement <= options.cost_tolerance {
                    active[row] = false;
                    termination[row] = Termination::CostTolerance;
                }
            } else {
                lambda[row] = if lambda[row] > 0.0 {
                    lambda[row] * options.lambda_increase
                } else {
                    options.lambda_increase
                };
            }
        }
    }

    Ok(LmReport {
        iterations: sweeps,
        batches: (0..batch)
            .map(|row| BatchReport {
                iterations: row_iterations[row],
                initial_cost: initial_cost[row],
                final_cost: cost[row],
                final_lambda: lambda[row],
                termination: termination[row],
            })
            .collect(),
    })
}

/// One step for every batch row, for callers driving their own loop.
///
/// Returns `δ` as `[batch, free.total()]`, ready for [`PointMap::retract_flat`]. It is *not*
/// applied: a caller deciding its own accept / reject rule needs the step before the point
/// moves.
///
/// `lambda` is per row. A slice of length 1 broadcasts to every row; a slice of length
/// `batch` is used as given; anything else is [`SkoroError::ShapeMismatch`]. `λ = 0` is the
/// undamped Gauss-Newton step, exact on a linear problem; `λ > 0` augments that row's system
/// with `√λ I` rather than forming `JᵀJ + λI`, which is the same step at half the condition
/// number.
pub fn step<B, M>(
    model: &M,
    points: &PointMap<B>,
    free: &TangentLayout,
    residuals: &TangentLayout,
    lambda: &[f64],
) -> SkoroResult<Tensor<B, 2>>
where
    B: Backend,
    M: NlsStructure<B>,
{
    let columns = step_columns(model, points, free, residuals, lambda)?;
    Dense::stack_vector::<B>(&columns, &device_of(points))
}

/// [`step`], before the per-row columns are stacked back into a tensor.
///
/// `solve` works in this form because its accept / reject rule is per row, and rebuilding
/// the tensor only to split it again would be the same work twice.
fn step_columns<B, M>(
    model: &M,
    points: &PointMap<B>,
    free: &TangentLayout,
    residuals: &TangentLayout,
    lambda: &[f64],
) -> SkoroResult<Vec<Dense>>
where
    B: Backend,
    M: NlsStructure<B>,
{
    let batch = batch_of(points);
    let width = free.total();
    let lambda = per_row(lambda, batch)?;

    // `D_out ≪ D_in` is the usual shape of a least-squares problem, so the reverse pass is
    // the cheaper one: it costs `D_out` frames against the forward pass's `D_in`. One pass
    // covers the whole batch — frames and batch are different axes.
    let jacobians = Dense::split_tensor(jacobian_vjp(model, points.as_ref(), free, residuals)?)?;
    let residual_columns = Dense::split_vector(residual_vector(model, points, residuals)?)?;

    let mut out = Vec::with_capacity(batch);
    for row in 0..batch {
        let (j, r) = (&jacobians[row], &residual_columns[row]);
        let (lhs, rhs) = if lambda[row] > 0.0 {
            (
                j.vstack(&Dense::scaled_identity(width, lambda[row].sqrt()))?,
                r.neg().vstack(&Dense::zeros(width, 1))?,
            )
        } else {
            (j.clone(), r.neg())
        };
        out.push(lstsq(&lhs, &rhs)?);
    }
    Ok(out)
}

/// Broadcast a length-1 `lambda` to the batch, or check a per-row one.
fn per_row(lambda: &[f64], batch: usize) -> SkoroResult<Vec<f64>> {
    match lambda.len() {
        1 => Ok(vec![lambda[0]; batch]),
        n if n == batch => Ok(lambda.to_vec()),
        // `ShapeMismatch` rather than `BatchSizeMismatch`: the latter names an `Id`, and a
        // slice of damping values names no node.
        n => Err(SkoroError::ShapeMismatch {
            expected: batch,
            found: n,
        }),
    }
}

/// `½Σ‖rᵢ(x)‖²` at `points`, per batch row.
///
/// Per row, not summed across rows: the rows are independent problems, and pooling their
/// costs would make one row's progress hide another's divergence.
fn costs<B, M>(model: &M, points: &PointMap<B>, residuals: &TangentLayout) -> SkoroResult<Vec<f64>>
where
    B: Backend,
    M: NlsStructure<B>,
{
    Ok(
        Dense::split_vector(residual_vector(model, points, residuals)?)?
            .iter()
            .map(Dense::half_sum_of_squares)
            .collect(),
    )
}

/// Every residual, concatenated in layout order — `[batch, residuals.total()]`.
fn residual_vector<B, M>(
    model: &M,
    points: &PointMap<B>,
    residuals: &TangentLayout,
) -> SkoroResult<Tensor<B, 2>>
where
    B: Backend,
    M: NlsStructure<B>,
{
    let ids: Vec<Id> = residuals.ids().collect();
    let evaluated = model.evaluate_ids(points.as_ref(), &ids)?;

    let mut values = HashMap::with_capacity(ids.len());
    for id in ids {
        let (value, _) = evaluated
            .evaluations
            .get(&id)
            .expect("internal: every residual is an evaluation target of this pass");
        values.insert(id, value.as_float().clone());
    }
    residuals.flatten_vectors(&values, batch_of(points), &device_of(points))
}

/// The batch size the model's points are supplied at.
///
/// Read from an arbitrary entry, which is safe because the compiler *checks* every
/// `Parameter` against the running batch rather than broadcasting it — so either they all
/// agree or the evaluation fails and says so.
fn batch_of<B: Backend>(points: &PointMap<B>) -> usize {
    points
        .iter()
        .next()
        .map(|(_, input, _)| input.dims()[0])
        .unwrap_or(1)
}

/// The device the model's points live on.
fn device_of<B: Backend>(points: &PointMap<B>) -> B::Device {
    points
        .iter()
        .next()
        .map(|(_, input, _)| input.device())
        .unwrap_or_default()
}
