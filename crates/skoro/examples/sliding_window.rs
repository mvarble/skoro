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

use common::batch::{Dataset, dataset, rows, value};
use common::figures::{self, Json};
use common::laplace::{CHI2_6_90, SLOPE_6, calibration, normalized_errors};
use common::live::Live;
use common::model::tick;
use common::report::{Result, check, converged, mean, print_calibration, stationary};
use common::scenario::Scenario;

/// The number of datasets, each a row of the batch and of the window.
const N: usize = 100;

/// The seed of the datasets, so that every run of the example is the same run.
const SEED: u64 = 20_261_110;

/// `w`, the poses the window keeps.
const W: usize = 5;

/// The row whose path the figure draws.
const FIGURE_ROW: usize = 0;

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
    let mut filtered: Vec<Json> = Vec::new();
    // The window's number of nodes after each tick.
    let mut nodes: Vec<usize> = Vec::new();
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
        let marginal =
            normalized_errors(&linearizer, &chart, &live.point, &truth, appended.pose.id())?;
        variance = marginal
            .spread
            .into_iter()
            .flatten()
            .fold(variance, f64::max);
        filtering.push(marginal.errors);
        if figures::wanted() {
            let at = rows(&live.point, appended.pose)?;
            filtered.push(Json::Object(vec![
                ("mode", figures::position(&at, 7, FIGURE_ROW)),
                ("covariance", Json::numbers(&marginal.position[FIGURE_ROW])),
            ]));
        }

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
        nodes.push(window.snapshot().len());
    }
    // A retirement frees what it leaves needed by nothing, so the window stops growing once it is
    // full: its count after tick 2w stays within 1.5 times its largest over the first full
    // windows, the spread of the landmarks in view.
    let largest = |ticks: &[usize]| ticks.iter().copied().max().unwrap_or(0);
    let (first, later) = (largest(&nodes[W + 1..=2 * W]), largest(&nodes[2 * W + 1..]));
    check(2 * later <= 3 * first, || {
        format!("the window grew from {first} nodes to {later}")
    })?;
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
    let gaps = normalized_errors(&linearizer, &chart, &batch.point, &windowed_last, pose.id())?;
    let gaps: Vec<f64> = gaps.errors.into_iter().map(f64::sqrt).collect();
    let filtering_calibration = calibration(&filtering, CHI2_6_90);

    // The figures' data: one row's path, truth, dead reckoning, the window's filtered poses and
    // the batch's smoothed ones, and every filtering error.
    if figures::wanted() {
        let path = |values: &Values, storage: usize, ids: Vec<skoro::NodeId>| {
            ids.into_iter()
                .map(|id| Ok(figures::position(&rows(values, id)?, storage, FIGURE_ROW)))
                .collect::<Result<Vec<Json>>>()
                .map(Json::Array)
        };
        let poses: Vec<skoro::NodeId> = whole.ticks.iter().map(|t| t.pose.id()).collect();
        let landmarks: Vec<skoro::NodeId> = whole
            .ticks
            .iter()
            .flat_map(|t| t.new_landmarks.iter().map(|(_, m)| m.id()))
            .collect();
        figures::write(
            "sliding_window",
            &Json::Object(vec![
                ("rows", Json::Number(N as f64)),
                ("window", Json::Number(W as f64)),
                ("truth", path(&whole.truth, 7, poses.clone())?),
                ("start", path(&whole.start, 7, poses.clone())?),
                ("batch", path(&batch.point, 7, poses)?),
                ("filtered", Json::Array(std::mem::take(&mut filtered))),
                ("landmark_truth", path(&whole.truth, 3, landmarks.clone())?),
                ("landmark_batch", path(&batch.point, 3, landmarks)?),
                ("filtering_errors", Json::numbers(&filtering.concat())),
                (
                    "filtering_calibration",
                    Json::Object(vec![
                        ("coverage", Json::Number(filtering_calibration.mean)),
                        (
                            "tolerance",
                            Json::Number(filtering_calibration.sampling + SLOPE_6 * variance),
                        ),
                    ]),
                ),
                ("gaps", Json::numbers(&gaps)),
            ]),
        )?;
    }

    // The report: the window's calibration beside the batch's tolerance, and its gap.
    println!(
        "sliding window: {} poses, {} landmarks, {} sightings, w = {W}, {N} datasets",
        scenario.nominal.len(),
        scenario.landmarks.len(),
        scenario.sightings.len()
    );
    println!(
        "the window held at most {} nodes, {} at the last tick",
        nodes.iter().copied().max().unwrap_or(0),
        nodes.last().copied().unwrap_or(0)
    );
    println!(
        "every tick of every row converged, in {:.2} iterations on average",
        mean(&iterations)
    );
    print_calibration("filtering", &filtering_calibration, SLOPE_6 * variance);
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
