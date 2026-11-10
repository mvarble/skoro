//! A pose graph on SE(3), estimated in a batch, with the calibration of its Laplace marginals.
//!
//! A robot drives a closed circuit, measuring its motion by odometry and the landmarks it passes
//! by a range sensor. A hundred datasets are drawn from the model, each with its own truth, and
//! estimated together: one Levenberg–Marquardt run per row, sharing the passes that linearize
//! them. The example then checks that the Laplace marginals at the mode are calibrated, that their
//! credible ellipsoids contain the truth as often as they claim to.

mod common;

use std::collections::BTreeMap;
use std::time::Instant;

use skoro::{Q, Query, Target, Vector};
use skoro_linearize::{Assembly, LevenbergMarquardt, Linearizer, decrement};
use skoro_spatial::Pose;

use common::batch::{Dataset, dataset};
use common::laplace::{CHI2_3_90, CHI2_6_90, SLOPE_3, SLOPE_6, calibration, normalized_errors};
use common::report::{self, Result, calibrated, check, converged, mean, stationary};
use common::scenario::Scenario;

/// The number of datasets, each a row of the batch.
const N: usize = 100;

/// The seed of the datasets, so that every run of the example is the same run.
const SEED: u64 = 20_261_109;

/// The largest rotational standard deviation the check's tolerance was designed for.
const SIGMA_PHI: f64 = 0.05;

fn main() -> Result {
    let started = Instant::now();
    let target = Target::host();
    let scenario = Scenario::circuit();

    // 1. The model, every tick in one build; a hundred datasets drawn from it; the query that
    // observes their measurements; and the start of each.
    let Dataset {
        snapshot,
        ticks,
        truth,
        given,
        start,
        ..
    } = dataset(&scenario, N, SEED, &target)?;
    let poses: Vec<Q<Pose>> = ticks.iter().map(|t| t.pose).collect();
    let landmarks: BTreeMap<usize, Q<Vector<3>>> = ticks
        .iter()
        .flat_map(|t| t.new_landmarks.iter().copied())
        .collect();
    let query = Query::log_density().given(given);

    // 2. The mode of every row.
    let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Residual)?;
    let schedule = linearizer.schedule()?;
    let method = LevenbergMarquardt::default();
    let minimized = method.minimize(&linearizer, &schedule, &start)?;
    converged(&minimized.rows, "the batch")?;
    let mode = minimized.point;

    // 3. Stationarity, by the undamped decrement.
    let bound = 10.0 * method.tolerance;
    stationary(
        &decrement(&linearizer, &schedule, &mode)?,
        bound,
        "the batch",
    )?;
    let solved = started.elapsed();

    // 4. The normalized errors of every pose and landmark, at the mode and, through the mode's
    // marginals, at the start; and the bound on the rotational variance.
    let chart = linearizer.at_in_chart(&mode, linearizer.layout().quantities())?;
    let mut pose_errors = Vec::new();
    let mut start_pose_errors = Vec::new();
    let mut variance: f64 = 0.0;
    for &pose in &poses {
        let (errors, spread) = normalized_errors(&linearizer, &chart, &mode, &truth, pose.id())?;
        let (at_start, _) = normalized_errors(&linearizer, &chart, &start, &truth, pose.id())?;
        variance = spread.into_iter().flatten().fold(variance, f64::max);
        pose_errors.push(errors);
        start_pose_errors.push(at_start);
    }
    let mut landmark_errors = Vec::new();
    let mut start_landmark_errors = Vec::new();
    for &landmark in landmarks.values() {
        let id = landmark.id();
        landmark_errors.push(normalized_errors(&linearizer, &chart, &mode, &truth, id)?.0);
        start_landmark_errors.push(normalized_errors(&linearizer, &chart, &start, &truth, id)?.0);
    }
    let marginals = started.elapsed() - solved;

    // 6. The report, printed before the checks so that a failure shows its figures.
    let objective = linearizer.objective(&mode)?;
    println!(
        "pose graph: {} poses, {} landmarks, {} sightings, {N} datasets",
        poses.len(),
        landmarks.len(),
        scenario.sightings.len()
    );
    for (r, outcome) in minimized.rows.iter().enumerate().take(5) {
        println!(
            "row {r}: {} iterations, f = {:.6}",
            outcome.iterations, objective[r]
        );
    }
    let iterations: Vec<f64> = minimized.rows.iter().map(|o| o.iterations as f64).collect();
    println!("mean iterations over the rows: {:.2}", mean(&iterations));
    report::print_history(0, &minimized.rows[0]);
    let mean_of = |e: &[Vec<f64>]| mean(&e.iter().map(|q| mean(q)).collect::<Vec<_>>());
    println!(
        "mean normalized error of a pose: {:.3} at the start, {:.3} at the mode (6 if calibrated)",
        mean_of(&start_pose_errors),
        mean_of(&pose_errors)
    );
    println!(
        "mean normalized error of a landmark: {:.3} at the start, {:.3} at the mode (3 if calibrated)",
        mean_of(&start_landmark_errors),
        mean_of(&landmark_errors)
    );
    println!(
        "largest rotational standard deviation: {:.4} rad (at most {SIGMA_PHI})",
        variance.sqrt()
    );
    println!(
        "solved in {:.1} s, marginals in {:.1} s",
        solved.as_secs_f64(),
        marginals.as_secs_f64()
    );

    // 4, checked: the scales are in the regime the tolerance was designed for.
    check(variance <= SIGMA_PHI * SIGMA_PHI, || {
        format!(
            "the largest rotational standard deviation {:.4} exceeds {SIGMA_PHI}: the scales \
             left the regime the check's tolerance was designed for",
            variance.sqrt()
        )
    })?;
    // 5. The calibrations, each against its tolerance.
    let kappa = variance;
    calibrated(
        "poses",
        &calibration(&pose_errors, CHI2_6_90),
        SLOPE_6 * kappa,
    )?;
    calibrated(
        "landmarks",
        &calibration(&landmark_errors, CHI2_3_90),
        SLOPE_3 * kappa,
    )?;
    println!("every check passed");
    Ok(())
}
