//! The report: what the examples print, and their checks as errors that name the figures.

use skoro_linearize::{RowOutcome, Status};

use super::laplace::Calibration;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The level of the credible ellipsoids.
pub const ALPHA: f64 = 0.9;

/// Fails, naming `what`, unless `holds`.
pub fn check(holds: bool, what: impl FnOnce() -> String) -> Result {
    if holds { Ok(()) } else { Err(what().into()) }
}

/// Fails unless every row converged, naming the first that did not.
pub fn converged(rows: &[RowOutcome], context: &str) -> Result {
    match rows
        .iter()
        .enumerate()
        .find(|(_, r)| r.status != Status::Converged)
    {
        None => Ok(()),
        Some((r, outcome)) => Err(format!(
            "{context}: row {r} ended {:?} after {} iterations",
            outcome.status, outcome.iterations
        )
        .into()),
    }
}

/// Fails unless every row's decrement is at most `bound`, naming the largest.
pub fn stationary(decrements: &[f64], bound: f64, context: &str) -> Result {
    let (r, largest) = decrements
        .iter()
        .copied()
        .enumerate()
        .fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
    check(largest <= bound, || {
        format!("{context}: row {r}'s decrement {largest:.3e} exceeds {bound:.1e}")
    })
}

/// The check `|c̄ - α| ≤ 4 s*/√N + δ` of a set of quantities, printed and then enforced.
pub fn calibrated(name: &str, calibration: &Calibration, delta: f64) -> Result {
    let tolerance = calibration.sampling + delta;
    print_calibration(name, calibration, delta);
    check((calibration.mean - ALPHA).abs() <= tolerance, || {
        format!(
            "the {name}' coverage {:.4} is off {ALPHA} by more than {tolerance:.4}",
            calibration.mean
        )
    })
}

/// A calibration beside `α`, its sampling part and `δ`.
pub fn print_calibration(name: &str, calibration: &Calibration, delta: f64) {
    println!(
        "{name:>10}: coverage {:.4} against {ALPHA}, tolerance {:.4} = {:.4} sampling + {:.1e} δ",
        calibration.mean,
        calibration.sampling + delta,
        calibration.sampling,
        delta
    );
}

/// One row's iterations, step by step.
pub fn print_history(row: usize, outcome: &RowOutcome) {
    println!(
        "row {row}: {:?} after {} iterations",
        outcome.status, outcome.iterations
    );
    println!(
        "  {:>4}  {:>16}  {:>10}  {:>10}  {:>8}  accepted",
        "step", "f", "λ", "promised", "gain"
    );
    for (i, it) in outcome.history.iter().enumerate() {
        println!(
            "  {i:>4}  {:>16.9}  {:>10.3e}  {:>10.3e}  {:>8.4}  {}",
            it.objective, it.damping, it.promised, it.gain, it.accepted
        );
    }
}

/// The mean of some values.
pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}
