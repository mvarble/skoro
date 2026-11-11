//! The Levenberg–Marquardt method: Marquardt's diagonal, Nielsen's schedule of the damping, and a
//! stop on the decrease a step promises, run on each row of a batch separately.

use skoro_core::{Layout, Values};
use skoro_elimination::{Damping, Problem, Schedule};
use skoro_tensor::{Points, Target, Value};

use crate::solved::undetermined;
use crate::{Assembly, Error, Linearizer};

/// The Levenberg–Marquardt method of the examples' mathematics: Marquardt's diagonal, Nielsen's
/// schedule, and a stop on the promised decrease, run on each row of a batch separately.
#[derive(Clone, Debug)]
pub struct LevenbergMarquardt {
    /// `λ₀`: 1e-3.
    pub initial_damping: f64,
    /// `δ`, the floor of `D`: 1e-12.
    pub floor: f64,
    /// `ε`, in nats: 1e-9. Must stay well above `u |f|` at the query's precision.
    pub tolerance: f64,
    /// `λ_max`: 1e10. A row whose damping passes it has not converged.
    pub max_damping: f64,
    /// The bound on iterations, accepted and rejected alike: 100.
    pub max_iterations: usize,
}

impl Default for LevenbergMarquardt {
    fn default() -> LevenbergMarquardt {
        LevenbergMarquardt {
            initial_damping: 1e-3,
            floor: 1e-12,
            tolerance: 1e-9,
            max_damping: 1e10,
            max_iterations: 100,
        }
    }
}

/// How a row's minimization ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    /// A step, accepted or not, promised at most `tolerance`.
    Converged,
    /// `λ` passed `max_damping`: no step decreased `f`.
    DampingExceeded,
    /// `max_iterations` passed first.
    IterationsExceeded,
}

/// One iteration of one row.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Iterate {
    /// `f` at the point the step was taken from.
    pub objective: f64,
    /// `λ` the step was taken with.
    pub damping: f64,
    /// `Δ(τ)`, the gain's denominator.
    pub promised: f64,
    /// `ρ_λ`: the obtained decrease over the promised one, `NaN` for a step promising nothing.
    pub gain: f64,
    pub accepted: bool,
}

/// How one row's minimization went.
#[derive(Clone, Debug)]
pub struct RowOutcome {
    pub status: Status,
    pub iterations: usize,
    pub history: Vec<Iterate>,
}

/// The method's result: the final point of every row, and how each row ended.
#[derive(Clone, Debug)]
pub struct Minimized {
    /// The final point of every row, at the linearizer's target.
    pub point: Values,
    pub rows: Vec<RowOutcome>,
}

/// One row's state between iterations.
struct Row {
    lambda: f64,
    beta: f64,
    done: Option<Status>,
    history: Vec<Iterate>,
}

impl Row {
    fn outcome(self) -> RowOutcome {
        RowOutcome {
            status: self.done.unwrap_or(Status::IterationsExceeded),
            iterations: self.history.len(),
            history: self.history,
        }
    }
}

impl LevenbergMarquardt {
    /// Minimizes the linearizer's `f` from `start`, row by row, eliminating along `schedule`
    /// (`Shape` of a schedule of another structure). Refuses the Fisher assembly
    /// (`NotAnIntegral`), whose rows are not the curvature of `f`, and what the linearizer and
    /// the elimination refuse. A row that does not converge is not an error: its outcome says so.
    pub fn minimize(
        &self,
        linearizer: &Linearizer,
        schedule: &Schedule,
        start: &Values,
    ) -> Result<Minimized, Error> {
        if linearizer.assembly() == Assembly::Fisher {
            return Err(Error::NotAnIntegral);
        }
        let (layout, target) = (linearizer.layout(), linearizer.target());
        let d = layout.dimension();
        let mut at = start.clone();
        let mut state: Vec<Row> = Vec::new();
        loop {
            let linearized = linearizer.at(&at)?;
            let problems = linearized.problems();
            let rows = problems.len();
            if state.is_empty() {
                state = (0..rows)
                    .map(|_| Row {
                        lambda: self.initial_damping,
                        beta: 2.0,
                        done: None,
                        history: Vec::new(),
                    })
                    .collect();
            }
            let mut steps = vec![0.0; rows * d];
            let mut promised = vec![0.0; rows];
            for (r, problem) in problems.iter().enumerate() {
                let row = &mut state[r];
                if row.done.is_some() {
                    continue;
                }
                if row.history.len() >= self.max_iterations {
                    row.done = Some(Status::IterationsExceeded);
                    continue;
                }
                if row.lambda > self.max_damping {
                    row.done = Some(Status::DampingExceeded);
                    continue;
                }
                let damping = Damping {
                    lambda: row.lambda,
                    diagonal: marquardt(problem, self.floor),
                };
                let step = problem
                    .eliminate(schedule, Some(&damping))
                    .and_then(|e| e.minimizer())
                    .map_err(|e| undetermined(e, layout))?;
                promised[r] = problem.promised(&step, &damping);
                steps[r * d..(r + 1) * d].copy_from_slice(&step);
            }
            if state.iter().all(|r| r.done.is_some()) {
                break;
            }
            let trial = layout.retract(&at, &Points::from_host(&steps, rows, d, target), target)?;
            let zero = vec![0.0; d];
            let after = linearizer.objective(&trial)?;
            let mut accepted = vec![false; rows];
            for (r, row) in state.iter_mut().enumerate() {
                if row.done.is_some() {
                    continue;
                }
                // `f` before the step is the problem's value at zero, by its constant.
                let before = problems[r].value(&zero);
                let gain = (before - after[r]) / promised[r];
                let accept = gain > 0.0;
                row.history.push(Iterate {
                    objective: before,
                    damping: row.lambda,
                    promised: promised[r],
                    gain,
                    accepted: accept,
                });
                // Taken if it decreases f, converged or not.
                accepted[r] = accept;
                if promised[r] <= self.tolerance {
                    row.done = Some(Status::Converged);
                    continue;
                }
                nielsen(&mut row.lambda, &mut row.beta, gain);
            }
            at = select_rows(&at, &trial, &accepted, layout, target)?;
        }
        Ok(Minimized {
            point: at,
            rows: state.into_iter().map(Row::outcome).collect(),
        })
    }
}

/// `½|J τ₀|²` per row at `at`: the Gauss–Newton decrement, with `τ₀` the undamped step, zero
/// exactly where the query's chart gradient is, the rows being of full rank.
pub fn decrement(
    linearizer: &Linearizer,
    schedule: &Schedule,
    at: &Values,
) -> Result<Vec<f64>, Error> {
    let linearized = linearizer.at(at)?;
    let layout = linearizer.layout();
    let zero = Damping {
        lambda: 0.0,
        diagonal: vec![0.0; layout.dimension()],
    };
    linearized
        .problems()
        .iter()
        .map(|problem| {
            let step = problem
                .eliminate(schedule, None)
                .and_then(|e| e.minimizer())
                .map_err(|e| undetermined(e, layout))?;
            Ok(problem.promised(&step, &zero))
        })
        .collect()
}

/// `D_ii = max(|J_{:,i}|², δ)`: the squared norm of each coordinate's column over every factor's
/// rows.
fn marquardt(problem: &Problem, floor: f64) -> Vec<f64> {
    let structure = problem.structure();
    let mut diagonal = vec![0.0; structure.dimension()];
    for (rows, blocks) in problem.factors().iter().zip(structure.factors()) {
        let mut column = 0;
        for &k in blocks {
            for i in structure.range(k) {
                diagonal[i] += (0..rows.a.rows())
                    .map(|r| rows.a.get(r, column).powi(2))
                    .sum::<f64>();
                column += 1;
            }
        }
    }
    diagonal.into_iter().map(|v| v.max(floor)).collect()
}

/// Nielsen's schedule on one row's `λ` and `β`: on a positive gain `λ` shrinks by
/// `max(1/3, 1 - (2ρ - 1)³)` and `β` resets to 2; otherwise `λ` grows by `β`, which doubles.
fn nielsen(lambda: &mut f64, beta: &mut f64, gain: f64) {
    if gain > 0.0 {
        *lambda *= (1.0 - (2.0 * gain - 1.0).powi(3)).max(1.0 / 3.0);
        *beta = 2.0;
    } else {
        *lambda *= *beta;
        *beta *= 2.0;
    }
}

/// Each quantity of `layout` from `trial` in the accepted rows and from `at` in the others,
/// through the host; every other value of `at` as it is.
fn select_rows(
    at: &Values,
    trial: &Values,
    accepted: &[bool],
    layout: &Layout,
    target: &Target,
) -> Result<Values, Error> {
    if !accepted.iter().any(|&a| a) {
        return Ok(at.clone());
    }
    let mut out = at.clone();
    let rows = accepted.len();
    for &id in layout.quantities() {
        let host = |values: &Values| match values.get(id) {
            Some(Value::Continuous(points)) => Ok(points.to_host()),
            _ => Err(Error::Core(skoro_core::Error::MissingValue { id })),
        };
        let (old, new) = (host(at)?, host(trial)?);
        let storage = new.len() / rows;
        // A value of `at` shared by every row is read as one row and expanded.
        let old_row = |r: usize| {
            if old.len() == storage {
                &old[..]
            } else {
                &old[r * storage..(r + 1) * storage]
            }
        };
        let data: Vec<f64> = (0..rows)
            .flat_map(|r| {
                if accepted[r] {
                    new[r * storage..(r + 1) * storage].to_vec()
                } else {
                    old_row(r).to_vec()
                }
            })
            .collect();
        out.insert(
            id,
            Value::Continuous(Points::from_host(&data, rows, storage, target)),
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::sync::Arc;

    use skoro_elimination::{Matrix, Problem, Rows, Structure};

    use super::{marquardt, nielsen};

    #[test]
    fn the_marquardt_diagonal_is_the_column_norms() {
        // Two blocks of dimensions 2 and 1; a factor on both and a factor on the second alone.
        let structure = Arc::new(Structure::new(vec![2, 1], vec![vec![0, 1], vec![1]]).unwrap());
        let a0 = Matrix::new(2, 3, vec![1.0, 2.0, 0.0, 3.0, -1.0, 0.0]).unwrap();
        let a1 = Matrix::new(1, 1, vec![0.0]).unwrap();
        let problem = Problem::new(
            structure,
            vec![
                Rows {
                    a: a0,
                    b: vec![0.0; 2],
                },
                Rows {
                    a: a1,
                    b: vec![0.0],
                },
            ],
            None,
            0.0,
            f64::EPSILON / 2.0,
        )
        .unwrap();
        // The third coordinate's column is zero in every factor: the floor takes it.
        assert_eq!(marquardt(&problem, 1e-12), [10.0, 5.0, 1e-12]);
    }

    #[test]
    fn the_schedule_is_nielsens() {
        let step = |gain: f64, beta: f64| {
            let (mut lambda, mut beta) = (1.0, beta);
            nielsen(&mut lambda, &mut beta, gain);
            (lambda, beta)
        };
        let close = |a: f64, b: f64| (a - b).abs() < 1e-15;
        let (l, b) = step(1.0, 8.0);
        assert!(close(l, 1.0 / 3.0) && b == 2.0, "{l} {b}");
        let (l, _) = step(0.5, 2.0);
        assert!(close(l, 1.0), "{l}");
        let (l, _) = step(0.9, 2.0);
        assert!(close(l, 1.0 - 0.8f64.powi(3)), "{l}");
        assert_eq!(step(0.0, 2.0), (2.0, 4.0));
        assert_eq!(step(-1.0, 4.0), (4.0, 8.0));
        assert_eq!(step(f64::NAN, 2.0), (2.0, 4.0));
    }
}
