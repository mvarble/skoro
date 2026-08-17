//! End-to-end checks on the optimizer and the two things that gate it.
//!
//! Nothing here compares against a table of numbers taken from the implementation. The
//! least-squares fixture is linear, so it has a closed-form solution computed in the test
//! from the data; the Jacobian check is a central difference taken *through the manifold
//! retraction*; the precondition check is the same model twice, differing only in which
//! variables are declared free.

use skoro_backend::{Backend, ElementConversion, Tensor, TensorData};
use skoro_core::{error::SkoroError, id::Id, pack::Packable};
use skoro_dense::dense::Dense;
use skoro_graph::point_map::PointMap;
use skoro_levenberg_marquardt::{LmOptions, Termination, check_normalizers, solve, step};
use skoro_linalg::tril::Tril;
use skoro_linalg::vector::LinalgVectorOps;
use skoro_model::{
    ModelClientOps, NlsStructure, TangentLayout,
    hash_map_builders::point_map,
    jacobian::{jacobian_jvp, jacobian_vjp},
    model::ModelClient,
    q::Q,
};
use skoro_normal::facade::{
    GaussianModelClient, GaussianModelClientOps, constant_covariance_diagonal,
};
use skoro_riemann::real::Real;
use skoro_se3::quantity::{SE3, SE3Pose};
use skoro_so3::quantity::SO3Matrix;

/// The six `FloatElem` operator bounds the Gaussian machinery restates at every impl,
/// bundled once so the fixtures below read as fixtures.
///
/// These have to be **supertraits**, not a `where` clause on the trait: a `where` clause on
/// a trait is an obligation on its implementors, not an implied bound for its users, so
/// bundling them that way buys nothing at a call site. Supertraits are elaborated, so
/// `E: LmElem` really does give you `E: Mul<E, Output = E>`.
pub trait LmElem:
    Sized
    + std::ops::Div<Self, Output = Self>
    + std::ops::Mul<Self, Output = Self>
    + std::ops::Add<Self, Output = Self>
    + std::ops::Sub<Self, Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign<Self>
{
}

impl<E> LmElem for E where
    E: Sized
        + std::ops::Div<E, Output = E>
        + std::ops::Mul<E, Output = E>
        + std::ops::Add<E, Output = E>
        + std::ops::Sub<E, Output = E>
        + std::ops::Neg<Output = E>
        + std::ops::AddAssign<E>
{
}

fn tensor<B: Backend>(values: &[f64]) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(values.to_vec(), [1, values.len()]).convert::<B::FloatElem>(),
        &B::Device::default(),
    )
}

fn into_f64<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f64> {
    t.into_data().convert::<f64>().to_vec::<f64>().unwrap()
}

/// `N` rows of a one-dimensional quantity — `[N, 1]`.
fn column<B: Backend>(values: &[f64]) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(values.to_vec(), [values.len(), 1]).convert::<B::FloatElem>(),
        &B::Device::default(),
    )
}

// --------------------------------------------------------------------------------------
// A weighted linear least-squares problem, with its closed form.
// --------------------------------------------------------------------------------------

/// `(t, y, σ)`. Not collinear, so the fit is a genuine compromise rather than an
/// interpolation, and the `σ` differ so the whitening actually weights the rows.
const DATA: [(f64, f64, f64); 4] = [
    (-2.0, -3.25, 0.5),
    (0.5, 1.75, 2.0),
    (1.0, 2.5, 1.0),
    (3.5, 8.25, 0.25),
];

/// The weighted least-squares line through [`DATA`], by the normal equations solved in
/// closed form on a 2×2 system.
///
/// This is the answer the optimizer must find, computed here from the data alone.
fn analytic_fit() -> (f64, f64) {
    analytic_fit_weighted(&DATA.map(|(_, _, sigma)| 1.0 / (sigma * sigma)))
}

/// The same closed form at arbitrary weights. Equal weights give the unweighted line, which
/// is what a single shared covariance produces.
fn analytic_fit_weighted(weights: &[f64; 4]) -> (f64, f64) {
    let (mut sw, mut swt, mut swtt, mut swy, mut swty) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for ((t, y, _), &w) in DATA.iter().zip(weights) {
        let (t, y) = (*t, *y);
        sw += w;
        swt += w * t;
        swtt += w * t * t;
        swy += w * y;
        swty += w * t * y;
    }
    let slope = (sw * swty - swt * swy) / (sw * swtt - swt * swt);
    let intercept = (swy - slope * swt) / sw;
    (slope, intercept)
}

/// The same line fit, but sharing **one free `Tril` covariance** across every factor.
///
/// This is the configuration `check_normalizers` exists to *permit*: a noise level that is a
/// `Parameter` — fixed at run time rather than at compile time — and simply not optimized.
struct FreeCovarianceFit<B: Backend> {
    model: GaussianModelClient<B>,
    slope: Q<B, Real>,
    intercept: Q<B, Real>,
    covariance: Q<B, Tril>,
}

impl<B: Backend> FreeCovarianceFit<B>
where
    B::FloatElem: LmElem,
    Real: Packable<B::FloatElem, B::FloatElem>,
{
    fn new() -> Self {
        let mut model = GaussianModelClient::<B>::default();
        let slope = model.add_parameter(Real);
        let intercept = model.add_parameter(Real);
        let covariance = model.add_parameter(Tril(1));

        for (t, y, _) in DATA {
            let mean = ((slope.clone() * t).unwrap() + intercept.clone()).unwrap();
            let observation = model.add_gaussian_quantity(&mean, &covariance).unwrap();
            model
                .observe(&observation, &B::FloatElem::from_elem(y))
                .unwrap();
        }

        Self {
            model,
            slope,
            intercept,
            covariance,
        }
    }
}

/// `y_i ~ Normal(slope · t_i + intercept, σ_i)`, observed at the data.
///
/// The free variables are `Parameter` leaves and the residuals are mapped nodes, which is
/// the structural constraint both Jacobian paths impose: cotangents are only collected at
/// leaves, and tangents only emitted at operations.
struct LineFit<B: Backend> {
    model: GaussianModelClient<B>,
    slope: Q<B, Real>,
    intercept: Q<B, Real>,
}

impl<B: Backend> LineFit<B>
where
    B::FloatElem: LmElem,
    Real: Packable<B::FloatElem, B::FloatElem>,
{
    fn new() -> Self {
        let mut model = GaussianModelClient::<B>::default();
        let slope = model.add_parameter(Real);
        let intercept = model.add_parameter(Real);

        for (t, y, sigma) in DATA {
            let mean = ((slope.clone() * t).unwrap() + intercept.clone()).unwrap();
            let cov = constant_covariance_diagonal(&model, &[sigma]);
            let observation = model.add_gaussian_quantity(&mean, &cov).unwrap();
            model
                .observe(&observation, &B::FloatElem::from_elem(y))
                .unwrap();
        }

        Self {
            model,
            slope,
            intercept,
        }
    }

    fn free(&self) -> TangentLayout {
        self.model
            .tangent_layout(&[self.slope.id(), self.intercept.id()])
            .unwrap()
    }

    fn points(&self, slope: f64, intercept: f64) -> PointMap<B> {
        point_map()
            .with_quantity(&self.slope, tensor::<B>(&[slope]))
            .with_quantity(&self.intercept, tensor::<B>(&[intercept]))
            .build()
    }

    /// The same fixture at batch `starts.len()`, one starting point per row.
    fn points_batched(&self, starts: &[(f64, f64)]) -> PointMap<B> {
        let slopes: Vec<f64> = starts.iter().map(|(s, _)| *s).collect();
        let intercepts: Vec<f64> = starts.iter().map(|(_, i)| *i).collect();
        point_map()
            .with_quantity(&self.slope, column::<B>(&slopes))
            .with_quantity(&self.intercept, column::<B>(&intercepts))
            .build()
    }

    /// Every row's `(slope, intercept)`.
    fn read_batched(&self, points: &PointMap<B>) -> Vec<(f64, f64)> {
        let get = |q: &Q<B, Real>| {
            let (value, _) = points.get(&q.id()).unwrap();
            into_f64(value.unwrap_float_ref().clone())
        };
        get(&self.slope)
            .into_iter()
            .zip(get(&self.intercept))
            .collect()
    }

    fn read(&self, points: &PointMap<B>) -> (f64, f64) {
        let get = |q: &Q<B, Real>| {
            let (value, _) = points.get(&q.id()).unwrap();
            into_f64(value.unwrap_float_ref().clone())[0]
        };
        (get(&self.slope), get(&self.intercept))
    }
}

/// **The undamped step is exact on a linear problem**, so anything other than one iteration
/// is a bug in `J` or in the solve — not slow convergence.
pub fn undamped_lm_solves_a_linear_problem_in_one_iteration_impl<B: Backend>(epsilon: f64)
where
    B::FloatElem: LmElem,
    Real: Packable<B::FloatElem, B::FloatElem>,
{
    let fit = LineFit::<B>::new();
    let free = fit.free();
    // Deliberately far from the answer: an exact step does not care.
    let mut points = fit.points(-4.0, 6.0);

    let options = LmOptions {
        lambda: 0.0,
        // The default `step_tolerance` is `f64`-scaled. On an `f32` backend `δ` never falls
        // below it — the round-off floor is higher — so the run would spend every iteration
        // rediscovering a step it cannot shrink. Tying it to the suite's own tolerance is
        // what makes "one iteration" a claim about the algorithm rather than about `f64`.
        step_tolerance: epsilon,
        ..Default::default()
    };
    let report = solve(&fit.model, &mut points, &free, &options).unwrap();

    let (slope, intercept) = fit.read(&points);
    let (expected_slope, expected_intercept) = analytic_fit();
    approx::assert_abs_diff_eq!(slope, expected_slope, epsilon = epsilon);
    approx::assert_abs_diff_eq!(intercept, expected_intercept, epsilon = epsilon);

    assert_eq!(
        report.iterations, 1,
        "the undamped step is exact on a linear problem"
    );
    let row = &report.batches[0];
    assert_eq!(report.batches.len(), 1, "batch 1 is one report");
    assert_eq!(row.iterations, 1);
    assert_eq!(row.termination, Termination::StepTolerance);
    assert!(row.final_cost < row.initial_cost);
}

/// Damping rescales the step; it must not move the optimum.
///
/// Same problem, same start, `λ > 0`: more iterations, the *same* point.
pub fn damped_lm_reaches_the_same_point_impl<B: Backend>(epsilon: f64)
where
    B::FloatElem: LmElem,
    Real: Packable<B::FloatElem, B::FloatElem>,
{
    let fit = LineFit::<B>::new();
    let free = fit.free();
    let mut points = fit.points(-4.0, 6.0);

    let options = LmOptions {
        lambda: 10.0,
        step_tolerance: epsilon,
        ..Default::default()
    };
    let report = solve(&fit.model, &mut points, &free, &options).unwrap();

    let (slope, intercept) = fit.read(&points);
    let (expected_slope, expected_intercept) = analytic_fit();
    approx::assert_abs_diff_eq!(slope, expected_slope, epsilon = epsilon);
    approx::assert_abs_diff_eq!(intercept, expected_intercept, epsilon = epsilon);

    assert!(
        report.iterations > 1,
        "a damped start must take more than the exact step"
    );
    assert!(
        report.batches[0].final_lambda < options.lambda,
        "successive successes must relax the damping"
    );
}

/// `step` at `λ = 0` from the optimum is zero, and from anywhere else lands on it.
pub fn a_single_step_is_the_gauss_newton_step_impl<B: Backend>(epsilon: f64)
where
    B::FloatElem: LmElem,
    Real: Packable<B::FloatElem, B::FloatElem>,
{
    let fit = LineFit::<B>::new();
    let free = fit.free();
    let residuals = fit.model.residual_layout().unwrap();

    // Four scalar residuals over two scalar parameters: tall, which is why the reverse pass
    // is the cheaper one.
    assert_eq!((residuals.total(), free.total()), (4, 2));

    let points = fit.points(-4.0, 6.0);
    let delta = step(&fit.model, &points, &free, &residuals, &[0.0]).unwrap();
    assert_eq!(delta.dims(), [1, 2]);

    let moved = points.retract_flat(&free, delta).unwrap();
    let (slope, intercept) = fit.read(&moved);
    let (expected_slope, expected_intercept) = analytic_fit();
    approx::assert_abs_diff_eq!(slope, expected_slope, epsilon = epsilon);
    approx::assert_abs_diff_eq!(intercept, expected_intercept, epsilon = epsilon);

    // From the optimum, the step is zero.
    let at_optimum = fit.points(expected_slope, expected_intercept);
    let delta = step(&fit.model, &at_optimum, &free, &residuals, &[0.0]).unwrap();
    for component in into_f64(delta) {
        approx::assert_abs_diff_eq!(component, 0.0, epsilon = epsilon);
    }
}

// --------------------------------------------------------------------------------------
// The normalizer precondition, both directions.
// --------------------------------------------------------------------------------------

/// **The same model twice.** A free `Tril` covariance is rejected when it is a column of
/// `free` and accepted when it is not.
///
/// The second half is the one that matters: an implementation that simply refused free
/// covariances would pass the first and fail this, and "a covariance that is a `Parameter`
/// but is not being optimized" is an ordinary modelling case — a noise level fixed at run
/// time rather than at compile time.
pub fn the_normalizer_precondition_holds_both_ways_impl<B: Backend>()
where
    B::FloatElem: LmElem,
    Real: Packable<B::FloatElem, B::FloatElem>,
{
    let mut model = GaussianModelClient::<B>::default();
    let mean = model.add_parameter(Real);
    // Free, not constant: `Tril` is a manifold, so the Gaussian facade will hand one out.
    let covariance = model.add_parameter(Tril(1));
    let observation = model.add_gaussian_quantity(&mean, &covariance).unwrap();
    model
        .observe(&observation, &B::FloatElem::from_elem(1.25f64))
        .unwrap();

    let density = model.certificate().factors()[0].density;

    let with_covariance = model.tangent_layout(&[mean.id(), covariance.id()]).unwrap();
    assert_eq!(
        check_normalizers(&model, &with_covariance),
        Err(SkoroError::NormalizerNotConstant {
            density,
            parent: covariance.id(),
        }),
        "optimizing the covariance changes `log|L⁻¹|`, so the sum of squares is not the \
         negative log-density"
    );

    let without_covariance = model.tangent_layout(&[mean.id()]).unwrap();
    assert_eq!(
        check_normalizers(&model, &without_covariance),
        Ok(()),
        "the same free covariance is fine as long as it is not being optimized"
    );
}

/// `solve` refuses before its first iteration rather than converging to the wrong thing.
pub fn solve_refuses_a_varying_normalizer_impl<B: Backend>()
where
    B::FloatElem: LmElem,
    Real: Packable<B::FloatElem, B::FloatElem>,
{
    let mut model = GaussianModelClient::<B>::default();
    let mean = model.add_parameter(Real);
    let covariance = model.add_parameter(Tril(1));
    let observation = model.add_gaussian_quantity(&mean, &covariance).unwrap();
    model
        .observe(&observation, &B::FloatElem::from_elem(1.25f64))
        .unwrap();

    let free = model.tangent_layout(&[mean.id(), covariance.id()]).unwrap();
    let mut points = point_map()
        .with_quantity(&mean, tensor::<B>(&[0.0]))
        .with_quantity(&covariance, tensor::<B>(&[1.0]))
        .build();

    assert!(matches!(
        solve(&model, &mut points, &free, &LmOptions::default()),
        Err(SkoroError::NormalizerNotConstant { .. })
    ));
}

/// **The permissive half of the precondition, end to end.**
///
/// The covariance is a free `Parameter`, so a naive "covariances must be constant" rule would
/// reject this model. It is not in `free`, so the normalizer is constant over what is being
/// optimized and the sum of squares really is the objective — `solve` must run, reach the
/// unweighted least-squares line, and leave the covariance exactly where it was.
///
/// That last part is also the only check that `solve` honours its contract to touch nothing
/// outside `free`: every other fixture here puts only free variables in `points`.
pub fn a_free_but_unoptimized_covariance_is_allowed_impl<B: Backend>(epsilon: f64)
where
    B::FloatElem: LmElem,
    Real: Packable<B::FloatElem, B::FloatElem>,
{
    let fit = FreeCovarianceFit::<B>::new();
    let free = fit
        .model
        .tangent_layout(&[fit.slope.id(), fit.intercept.id()])
        .unwrap();

    // A covariance of 2.5, present in `points` because the residuals need it, absent from
    // `free` because it is not being inferred.
    let mut points = point_map()
        .with_quantity(&fit.slope, tensor::<B>(&[-4.0]))
        .with_quantity(&fit.intercept, tensor::<B>(&[6.0]))
        .with_quantity(&fit.covariance, tensor::<B>(&[2.5]))
        .build();

    let options = LmOptions {
        lambda: 0.0,
        step_tolerance: epsilon,
        ..Default::default()
    };
    let report = solve(&fit.model, &mut points, &free, &options).unwrap();

    // One shared covariance means equal weights, so the answer is the unweighted line.
    let (expected_slope, expected_intercept) = analytic_fit_weighted(&[1.0; 4]);
    let get = |id| {
        let (value, _) = points.get(&id).unwrap();
        into_f64(value.unwrap_float_ref().clone())[0]
    };
    approx::assert_abs_diff_eq!(get(fit.slope.id()), expected_slope, epsilon = epsilon);
    approx::assert_abs_diff_eq!(
        get(fit.intercept.id()),
        expected_intercept,
        epsilon = epsilon
    );
    assert_eq!(report.iterations, 1);

    assert_eq!(
        get(fit.covariance.id()),
        2.5,
        "`solve` must leave every quantity outside `free` untouched"
    );
}

// --------------------------------------------------------------------------------------
// The batch axis: N independent optimizations.
// --------------------------------------------------------------------------------------

/// `y ~ Normal(x², σ)` observed at `4.0` — a residual with **two** exact zeros, at `x = ±2`.
///
/// The linear fixture above cannot show what a batch is for: its objective is convex, so
/// every starting point reaches the same answer and a per-row step is indistinguishable
/// from a shared one. This one has two basins, so the rows must disagree.
struct SquareFit<B: Backend> {
    model: GaussianModelClient<B>,
    x: Q<B, Real>,
}

impl<B: Backend> SquareFit<B>
where
    B::FloatElem: LmElem,
    Real: Packable<B::FloatElem, B::FloatElem>,
{
    fn new() -> Self {
        let mut model = GaussianModelClient::<B>::default();
        let x = model.add_parameter(Real);
        let mean = (x.clone() * x.clone()).unwrap();
        let cov = constant_covariance_diagonal(&model, &[0.5]);
        let observation = model.add_gaussian_quantity(&mean, &cov).unwrap();
        model
            .observe(&observation, &B::FloatElem::from_elem(4.0))
            .unwrap();

        Self { model, x }
    }

    fn free(&self) -> TangentLayout {
        self.model.tangent_layout(&[self.x.id()]).unwrap()
    }

    fn points(&self, starts: &[f64]) -> PointMap<B> {
        point_map()
            .with_quantity(&self.x, column::<B>(starts))
            .build()
    }

    fn read(&self, points: &PointMap<B>) -> Vec<f64> {
        let (value, _) = points.get(&self.x.id()).unwrap();
        into_f64(value.unwrap_float_ref().clone())
    }
}

/// **Batch N is exactly N separate runs.** This is the test that pins the semantics.
///
/// The same linear fit from three starting points, once as a batch and once as three
/// independent batch-1 `solve` calls. Points must agree, and so must every row's
/// termination and iteration count — if anything couples the rows, one of those diverges.
///
/// Compared to `$epsilon`, not for equality: the batched path seeds through
/// `identity_frames(3, ..)` and a batched `matmul`, and on a GPU a batched gemm is a
/// different kernel with a different reduction order than the batch-1 one. Exact equality
/// would fail there for a reason that is not a bug.
pub fn a_batch_is_n_independent_runs_impl<B: Backend>(epsilon: f64)
where
    B::FloatElem: LmElem,
    Real: Packable<B::FloatElem, B::FloatElem>,
{
    const STARTS: [(f64, f64); 3] = [(-4.0, 6.0), (0.0, 0.0), (10.0, -7.5)];

    let options = LmOptions {
        lambda: 0.5,
        step_tolerance: epsilon,
        ..Default::default()
    };

    let batched = LineFit::<B>::new();
    let free = batched.free();
    let mut points = batched.points_batched(&STARTS);
    let batched_report = solve(&batched.model, &mut points, &free, &options).unwrap();
    let batched_points = batched.read_batched(&points);

    assert_eq!(batched_report.batches.len(), 3);

    for (row, start) in STARTS.iter().enumerate() {
        let single = LineFit::<B>::new();
        let free = single.free();
        let mut points = single.points_batched(&[*start]);
        let report = solve(&single.model, &mut points, &free, &options).unwrap();

        let (slope, intercept) = single.read_batched(&points)[0];
        approx::assert_abs_diff_eq!(batched_points[row].0, slope, epsilon = epsilon);
        approx::assert_abs_diff_eq!(batched_points[row].1, intercept, epsilon = epsilon);

        let (batched_row, single_row) = (&batched_report.batches[row], &report.batches[0]);
        assert_eq!(
            batched_row.iterations, single_row.iterations,
            "row {row} took a different number of iterations in the batch"
        );
        assert_eq!(
            batched_row.termination, single_row.termination,
            "row {row} terminated differently in the batch"
        );
        approx::assert_abs_diff_eq!(
            batched_row.final_cost,
            single_row.final_cost,
            epsilon = epsilon
        );
    }
}

/// **Two local minima, found in one pass.** The motivating case.
///
/// `y ~ Normal(x², σ)` observed at `4.0` has residual `(x² − 4)/σ` and therefore two exact
/// minima, at `x = ±2`. Starting at `−3` and `+3`, the two rows must find *different*
/// answers — which is only possible if each row carries its own step.
///
/// This fixture is genuinely nonlinear, so there is no one-iteration claim to make here:
/// what is asserted is where it converges and that the residual really reaches zero.
pub fn two_starting_points_find_two_minima_impl<B: Backend>(epsilon: f64)
where
    B::FloatElem: LmElem,
    Real: Packable<B::FloatElem, B::FloatElem>,
{
    let fit = SquareFit::<B>::new();
    let free = fit.free();
    let mut points = fit.points(&[-3.0, 3.0]);

    let report = solve(&fit.model, &mut points, &free, &LmOptions::default()).unwrap();
    let found = fit.read(&points);

    approx::assert_abs_diff_eq!(found[0], -2.0, epsilon = epsilon);
    approx::assert_abs_diff_eq!(found[1], 2.0, epsilon = epsilon);
    for row in &report.batches {
        assert!(
            row.final_cost < epsilon,
            "each row reaches an exact zero of the residual, got {}",
            row.final_cost
        );
        assert!(row.final_cost < row.initial_cost);
    }
}

/// Rows stop when *they* are done, not when the batch is.
///
/// Row 0 starts at a minimum and terminates on the very first step it is offered; row 1
/// starts far away and is still working when the sweep budget runs out. Different
/// iteration counts *and* different terminations.
pub fn rows_terminate_independently_impl<B: Backend>()
where
    B::FloatElem: LmElem,
    Real: Packable<B::FloatElem, B::FloatElem>,
{
    let fit = SquareFit::<B>::new();
    let free = fit.free();
    let mut points = fit.points(&[2.0, -0.1]);

    let options = LmOptions {
        max_iterations: 2,
        ..Default::default()
    };
    let report = solve(&fit.model, &mut points, &free, &options).unwrap();

    assert_eq!(report.batches[0].iterations, 0, "row 0 began converged");
    assert_eq!(report.batches[0].termination, Termination::StepTolerance);
    assert!(
        report.batches[1].iterations > 0,
        "row 1 began far from any minimum"
    );
    assert_eq!(report.batches[1].termination, Termination::MaxIterations);
    assert_eq!(
        report.iterations, 2,
        "sweeps are the longest row, not the sum"
    );
}

/// A rejected row is left exactly where it was, and so is a row that has already finished.
///
/// Started just off the saddle at `x = 0.1`, the undamped Gauss-Newton step divides by a
/// near-zero Jacobian and overshoots badly, so the first step *increases* the cost and must
/// be rolled back. Row 0 meanwhile starts at a minimum and is inactive from the first
/// sweep. Neither point may move.
pub fn a_rejected_row_does_not_move_impl<B: Backend>(epsilon: f64)
where
    B::FloatElem: LmElem,
    Real: Packable<B::FloatElem, B::FloatElem>,
{
    let fit = SquareFit::<B>::new();
    let free = fit.free();
    let start = [2.0, 0.1];
    let mut points = fit.points(&start);

    let options = LmOptions {
        lambda: 0.0,
        max_iterations: 1,
        ..Default::default()
    };
    let report = solve(&fit.model, &mut points, &free, &options).unwrap();

    assert!(
        report.batches[1].final_cost >= report.batches[1].initial_cost,
        "the fixture must actually produce a rejection, or this proves nothing"
    );
    for (row, expected) in start.iter().enumerate() {
        approx::assert_abs_diff_eq!(fit.read(&points)[row], expected, epsilon = epsilon);
    }
}

/// A row that has terminated stays put while the rest of the batch keeps working.
///
/// The coarse `cost_tolerance` is the point: it stops each row while it is still *visibly*
/// short of the minimum, so a row that kept stepping after termination would end up
/// somewhere measurably different. With the default tolerance every row terminates within
/// rounding of the optimum and the bug hides.
///
/// The invariant asserted is the one that defines the whole feature — each row equals its
/// own solo run — and the two starts are chosen to terminate after different numbers of
/// sweeps, so at least one row is idle while another is active.
pub fn a_terminated_row_stops_moving_impl<B: Backend>(epsilon: f64)
where
    B::FloatElem: LmElem,
    Real: Packable<B::FloatElem, B::FloatElem>,
{
    const STARTS: [f64; 2] = [-3.0, 5.0];
    let options = LmOptions {
        cost_tolerance: 10.0,
        ..Default::default()
    };

    let batched = SquareFit::<B>::new();
    let free = batched.free();
    let mut points = batched.points(&STARTS);
    let report = solve(&batched.model, &mut points, &free, &options).unwrap();
    let batched_points = batched.read(&points);

    assert!(
        report
            .batches
            .iter()
            .any(|r| r.termination == Termination::CostTolerance),
        "the fixture must actually stop a row on cost, or this proves nothing"
    );
    assert_ne!(
        report.batches[0].iterations, report.batches[1].iterations,
        "the rows must finish at different times for idleness to be exercised"
    );

    for (row, start) in STARTS.iter().enumerate() {
        let single = SquareFit::<B>::new();
        let free = single.free();
        let mut points = single.points(&[*start]);
        let solo = solve(&single.model, &mut points, &free, &options).unwrap();

        approx::assert_abs_diff_eq!(
            batched_points[row],
            single.read(&points)[0],
            epsilon = epsilon
        );
        assert_eq!(report.batches[row].iterations, solo.batches[0].iterations);
        assert_eq!(report.batches[row].termination, solo.batches[0].termination);
    }
}

/// A `lambda` slice that is neither one value nor one per row is rejected.
pub fn a_mismatched_lambda_is_rejected_impl<B: Backend>()
where
    B::FloatElem: LmElem,
    Real: Packable<B::FloatElem, B::FloatElem>,
{
    let fit = SquareFit::<B>::new();
    let free = fit.free();
    let residuals = fit.model.residual_layout().unwrap();
    let points = fit.points(&[-3.0, 3.0, 1.0]);

    assert_eq!(
        step(&fit.model, &points, &free, &residuals, &[0.0, 1.0]).map(|d| d.dims()),
        Err(SkoroError::ShapeMismatch {
            expected: 3,
            found: 2
        })
    );
    // One value broadcasts, and so does one-per-row.
    assert_eq!(
        step(&fit.model, &points, &free, &residuals, &[0.5])
            .unwrap()
            .dims(),
        [3, 1]
    );
    assert_eq!(
        step(&fit.model, &points, &free, &residuals, &[0.5, 1.0, 2.0])
            .unwrap()
            .dims(),
        [3, 1]
    );
}

// --------------------------------------------------------------------------------------
// Jacobians against central differences through the retraction.
// --------------------------------------------------------------------------------------

/// `x` is a free `SE3` pose; the rows are `a ⊖ x` (6 wide) and its squared norm (1 wide).
///
/// `SE3` is the point: dimension 6 over 12 components, so a Jacobian sized by storage is
/// visible here and nowhere in a Euclidean fixture. The row layout is 7 wide against 6
/// columns, so the shape is not square either.
struct Se3Model<B: Backend> {
    client: ModelClient<B>,
    x: Q<B, SE3>,
    rows: Vec<Id>,
}

impl<B: Backend> Se3Model<B>
where
    SE3: skoro_riemann::Manifold<B>,
{
    fn new() -> Self {
        let client = ModelClient::<B>::default();
        // Rotation about z by 0.4 rad, translated — far from the identity, where a
        // transposed frame or a missing `J⁻¹(ω)` would be invisible.
        let (c, s) = (0.4f64.cos(), 0.4f64.sin());
        let e = |v: f64| B::FloatElem::from_elem(v);
        let anchor = client.add_constant(
            SE3,
            &SE3Pose::new(
                SO3Matrix([
                    e(c),
                    e(-s),
                    e(0.0),
                    e(s),
                    e(c),
                    e(0.0),
                    e(0.0),
                    e(0.0),
                    e(1.0),
                ]),
                [e(1.5), e(-2.25), e(0.75)],
            ),
        );
        let x = client.add_parameter(SE3);
        let difference: Q<B, skoro_riemann::euclidean::EuclideanD> =
            anchor.inverse_retract(&x).unwrap();
        let squared = difference.norm_squared().unwrap();
        let rows = vec![difference.id(), squared.id()];

        Self { client, x, rows }
    }

    /// Rotation about z by 0.9 rad, translated — a different pose from the anchor.
    fn points(&self) -> PointMap<B> {
        let (c, s) = (0.9f64.cos(), 0.9f64.sin());
        point_map()
            .with_quantity(
                &self.x,
                tensor::<B>(&[c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0, 2.0, -1.0, 1.25]),
            )
            .build()
    }
}

/// Both Jacobian paths against a central difference taken **through the retraction**.
///
/// The perturbation is `retract_flat(x, ±ε eᵢ)`, not `x ± ε eᵢ`: on `SE3` those are
/// different points, and only the first is what `J` is the derivative with respect to.
pub fn se3_jacobians_match_central_differences_impl<B: Backend>(delta: f64, epsilon: f64)
where
    SE3: skoro_riemann::Manifold<B>,
{
    let model = Se3Model::<B>::new();
    let points = model.points();
    let free = model.client.tangent_layout(&[model.x.id()]).unwrap();
    let rows = model.client.tangent_layout(&model.rows).unwrap();

    assert_eq!(
        (rows.total(), free.total()),
        (7, 6),
        "an SE3 column is 6 wide — a dimension, not the 12-wide storage — and the shape is \
         not square"
    );

    let value_map = points.as_ref();
    let jvp =
        Dense::from_tensor(jacobian_jvp(&model.client, value_map, &free, &rows).unwrap()).unwrap();
    let vjp =
        Dense::from_tensor(jacobian_vjp(&model.client, value_map, &free, &rows).unwrap()).unwrap();

    for column in 0..free.total() {
        let mut seed = vec![0.0; free.total()];
        seed[column] = delta;
        let plus = evaluate_rows(
            &model,
            &points.retract_flat(&free, tensor::<B>(&seed)).unwrap(),
        );
        seed[column] = -delta;
        let minus = evaluate_rows(
            &model,
            &points.retract_flat(&free, tensor::<B>(&seed)).unwrap(),
        );

        for row in 0..rows.total() {
            let numerical = (plus[row] - minus[row]) / (2.0 * delta);
            approx::assert_abs_diff_eq!(jvp.get(row, column), numerical, epsilon = epsilon);
            approx::assert_abs_diff_eq!(vjp.get(row, column), numerical, epsilon = epsilon);
        }
    }
}

/// The row values, concatenated in layout order.
fn evaluate_rows<B: Backend>(model: &Se3Model<B>, points: &PointMap<B>) -> Vec<f64>
where
    SE3: skoro_riemann::Manifold<B>,
{
    let evaluated = model
        .client
        .evaluate_ids(points.as_ref(), &model.rows)
        .unwrap();
    model
        .rows
        .iter()
        .flat_map(|id| {
            let (value, _) = evaluated.evaluations.get(id).unwrap();
            into_f64(value.unwrap_float_ref().clone())
        })
        .collect()
}

#[macro_export]
macro_rules! tests {
    ($B:ty, $delta:literal, $epsilon:literal) => {
        #[test]
        fn undamped_lm_solves_a_linear_problem_in_one_iteration() {
            common::undamped_lm_solves_a_linear_problem_in_one_iteration_impl::<$B>($epsilon);
        }

        #[test]
        fn damped_lm_reaches_the_same_point() {
            common::damped_lm_reaches_the_same_point_impl::<$B>($epsilon);
        }

        #[test]
        fn a_single_step_is_the_gauss_newton_step() {
            common::a_single_step_is_the_gauss_newton_step_impl::<$B>($epsilon);
        }

        #[test]
        fn a_free_but_unoptimized_covariance_is_allowed() {
            common::a_free_but_unoptimized_covariance_is_allowed_impl::<$B>($epsilon);
        }

        #[test]
        fn the_normalizer_precondition_holds_both_ways() {
            common::the_normalizer_precondition_holds_both_ways_impl::<$B>();
        }

        #[test]
        fn solve_refuses_a_varying_normalizer() {
            common::solve_refuses_a_varying_normalizer_impl::<$B>();
        }

        #[test]
        fn a_batch_is_n_independent_runs() {
            common::a_batch_is_n_independent_runs_impl::<$B>($epsilon);
        }

        #[test]
        fn two_starting_points_find_two_minima() {
            common::two_starting_points_find_two_minima_impl::<$B>($epsilon);
        }

        #[test]
        fn rows_terminate_independently() {
            common::rows_terminate_independently_impl::<$B>();
        }

        #[test]
        fn a_rejected_row_does_not_move() {
            common::a_rejected_row_does_not_move_impl::<$B>($epsilon);
        }

        #[test]
        fn a_terminated_row_stops_moving() {
            common::a_terminated_row_stops_moving_impl::<$B>($epsilon);
        }

        #[test]
        fn a_mismatched_lambda_is_rejected() {
            common::a_mismatched_lambda_is_rejected_impl::<$B>();
        }

        #[test]
        fn se3_jacobians_match_central_differences() {
            common::se3_jacobians_match_central_differences_impl::<$B>($delta, $epsilon);
        }
    };
}
