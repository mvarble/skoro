//! A sliding-window filter over an open path, each retired tick replaced by its linearized factor.
//!
//! A robot drives an open path, and the model it keeps is a window: the last few poses, the
//! landmarks they see, and one factor standing in for everything before. At every tick the window
//! appends the new pose and its measurements, finds its mode, reports the newest pose's marginal,
//! and retires the oldest pose with the landmarks only it still sees, replacing them by the
//! linearization of their terms at the window's mode. The window's calibration and its gap to the
//! batch on the same data are reported, not checked; what is checked is that every solve stopped
//! at a stationary point.

mod common;

use std::time::Instant;

use skoro::{Observations, QDyn, Query, Target, Values, Writer};
use skoro_linearize::{Assembly, LevenbergMarquardt, Linearizer, Replacing, decrement};

use common::batch::{Dataset, dataset, value};
use common::laplace::{CHI2_6_90, SLOPE_6, calibration, normalized_errors};
use common::live::Live;
use common::model::tick;
use common::report::{Result, converged, mean, print_calibration, stationary};
use common::scenario::Scenario;

/// The number of datasets, each a row of the batch and of the window.
const N: usize = 100;

/// The seed of the datasets, so that every run of the example is the same run.
const SEED: u64 = 20_261_110;

/// `w`, the poses the window keeps.
const W: usize = 5;

fn main() -> Result {
    let started = Instant::now();
    let target = Target::host();
    let scenario = Scenario::open_path()?;
    let method = LevenbergMarquardt::default();
    let bound = 10.0 * method.tolerance;

    // The truth and the measurements, from the open path's whole model.
    let whole: Dataset = dataset(&scenario, N, SEED, &target)?;

    // The window, a second model built tick by tick.
    let mut window = Writer::new();
    let mut live = Live::default();
    let mut filtering: Vec<Vec<f64>> = Vec::new();
    let mut variance: f64 = 0.0;
    let mut iterations: Vec<f64> = Vec::new();
    for k in 0..=scenario.last() {
        // 1. Append, and start what was appended.
        let previous = live.newest();
        let landmarks = &mut live.landmarks;
        let appended = window.build(|| tick(&scenario, k, previous, landmarks))?;
        live.append(k, &appended, &whole.ticks[k], &whole.truth)?;
        live.start(&scenario, k, &appended, N, &target)?;

        // 2. Solve, from the previous mode.
        let snapshot = window.snapshot();
        let query = Query::log_density().given(live.observations());
        let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Residual)?;
        let schedule = linearizer.schedule()?;
        let minimized = method.minimize(&linearizer, &schedule, &live.point)?;
        let context = format!("the window at tick {k}");
        converged(&minimized.rows, &context)?;
        stationary(
            &decrement(&linearizer, &schedule, &minimized.point)?,
            bound,
            &context,
        )?;
        iterations.extend(minimized.rows.iter().map(|o| o.iterations as f64));
        live.point = minimized.point;

        // 3. Report the newest pose's marginal: the window's filtering law of it.
        let truth = Values::new().with(
            appended.pose,
            value(&whole.truth, whole.ticks[k].pose)?.clone(),
        );
        let quantities = linearizer.layout().quantities().to_vec();
        let chart = linearizer.at_in_chart(&live.point, &quantities)?;
        let (errors, spread) =
            normalized_errors(&linearizer, &chart, &live.point, &truth, appended.pose.id())?;
        variance = spread.into_iter().flatten().fold(variance, f64::max);
        filtering.push(errors);

        // 4. Retire the oldest pose once the window holds more than W, with the landmarks only it
        // still sees, replacing their terms by their linearization at the window's mode.
        if live.len() > W {
            let a = k - W;
            let (addresses, observed) = live.retiring(a);
            let retirement = window.retire(addresses, observed)?;
            let keep: Vec<QDyn> = retirement.plan().blanket.iter().map(|(q, ..)| *q).collect();
            let replacing = Replacing::new(&retirement, &keep, &Observations::new(), &target)?;
            let replacement = replacing.at(&live.point)?;
            retirement.replace(|_| replacement.build())?;
            live.retire(a);
        }
    }
    let windowed = started.elapsed();

    // The batch on the same path and data, for the gap at the last tick.
    let last = scenario.last();
    let query = Query::log_density().given(whole.given.clone());
    let linearizer = Linearizer::new(&whole.snapshot, &query, &target, Assembly::Residual)?;
    let schedule = linearizer.schedule()?;
    let batch = method.minimize(&linearizer, &schedule, &whole.start)?;
    converged(&batch.rows, "the batch on the open path")?;
    let pose = whole.ticks[last].pose;
    let newest = live.newest().ok_or("an empty window")?;
    let windowed_last = Values::new().with(pose, value(&live.point, newest)?.clone());
    let chart = linearizer.at_in_chart(&batch.point, linearizer.layout().quantities())?;
    let (gaps, _) =
        normalized_errors(&linearizer, &chart, &batch.point, &windowed_last, pose.id())?;
    let gaps: Vec<f64> = gaps.into_iter().map(f64::sqrt).collect();

    // The report: the window's calibration beside the batch's tolerance, and its gap.
    println!(
        "sliding window: {} poses, {} landmarks, {} sightings, w = {W}, {N} datasets",
        scenario.nominal.len(),
        scenario.landmarks.len(),
        scenario.sightings.len()
    );
    println!(
        "every tick of every row converged, in {:.2} iterations on average",
        mean(&iterations)
    );
    print_calibration(
        "filtering",
        &calibration(&filtering, CHI2_6_90),
        SLOPE_6 * variance,
    );
    println!(
        "largest rotational standard deviation of a filtering marginal: {:.4} rad",
        variance.sqrt()
    );
    let largest = gaps.iter().copied().fold(0.0, f64::max);
    println!(
        "gap to the batch at the last tick, in its standard deviations: mean {:.4}, largest {:.4}",
        mean(&gaps),
        largest
    );
    println!(
        "window in {:.1} s, batch and gap in {:.1} s",
        windowed.as_secs_f64(),
        (started.elapsed() - windowed).as_secs_f64()
    );
    println!("every check passed");
    Ok(())
}
