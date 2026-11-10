//! A batch of datasets on one scenario: the whole model, every tick in one build; the datasets
//! drawn from it; the observations of their measurements; and the start of each.

use std::collections::BTreeMap;

use skoro::tensor::{Points, Value};
use skoro::{Key, NodeId, Observations, QDyn, Snapshot, Target, Values, Writer};
use skoro_spatial::host;

use super::model::{Tick, tick};
use super::report::Result;
use super::scenario::Scenario;

/// The model of a whole scenario and `rows` datasets drawn from it.
pub struct Dataset {
    pub snapshot: Snapshot,
    pub ticks: Vec<Tick>,
    /// The drawn values of every quantity: the truth, and the measurements.
    pub truth: Values,
    /// The measurements, observed row by row.
    pub given: Observations,
    /// The start of every row.
    pub start: Values,
    pub rows: usize,
}

/// Builds the scenario's model, draws `rows` datasets from `seed` with nothing observed, observes
/// every odometry and sighting at the drawn rows, and starts every row.
pub fn dataset(scenario: &Scenario, rows: usize, seed: u64, target: &Target) -> Result<Dataset> {
    let mut writer = Writer::new();
    let ticks: Vec<Tick> = writer.build(|| {
        let mut landmarks = BTreeMap::new();
        let mut previous = None;
        (0..=scenario.last())
            .map(|k| {
                let t = tick(scenario, k, previous, &mut landmarks);
                previous = Some(t.pose);
                t
            })
            .collect()
    })?;
    let snapshot = writer.snapshot();
    let truth = snapshot
        .generate(&Observations::new(), &Key::from_seed(seed), rows, target)?
        .values;
    let mut given = Observations::new();
    for t in &ticks {
        for q in measurements(t) {
            given = given.observe_value(q, value(&truth, q)?);
        }
    }
    let start = start(scenario, &ticks, &truth, rows, target)?;
    Ok(Dataset {
        snapshot,
        ticks,
        truth,
        given,
        start,
        rows,
    })
}

/// A tick's measurements: its odometry, if any, and its sightings.
pub fn measurements(t: &Tick) -> Vec<QDyn> {
    t.odometry
        .map(QDyn::from)
        .into_iter()
        .chain(t.sightings.iter().map(|&(_, z)| QDyn::from(z)))
        .collect()
}

/// The value of `q` in `values`.
pub fn value(values: &Values, q: impl Into<NodeId>) -> Result<&Value> {
    let id = q.into();
    values
        .get(id)
        .ok_or_else(|| format!("no value for {id:?}").into())
}

/// The rows of a continuous value, on the host.
pub fn rows(values: &Values, q: impl Into<NodeId>) -> Result<Vec<f64>> {
    match value(values, q)? {
        Value::Continuous(points) => Ok(points.to_host()),
        Value::Discrete(_) => Err("a continuous value".into()),
    }
}

/// Row `r` of a value stored `N` to a row.
pub fn row<const N: usize>(data: &[f64], r: usize) -> Result<[f64; N]> {
    Ok(data[N * r..N * r + N].try_into()?)
}

/// `rows` values of `N` each, laid end to end, as a continuous value.
pub fn load<const N: usize>(data: &[[f64; N]], target: &Target) -> Value {
    Value::Continuous(Points::from_host(&data.concat(), data.len(), N, target))
}

/// `x̂ ∘ y` row by row: the dead-reckoned pose from `x̂` along the odometry `y`.
pub fn dead_reckon(from: &[[f64; 7]], odometry: &[f64]) -> Result<Vec<[f64; 7]>> {
    (0..from.len())
        .map(|r| Ok(host::pose_compose(from[r], row::<7>(odometry, r)?)))
        .collect()
}

/// `x̂ · z` row by row: the landmark at its sighting `z` from `x̂`.
pub fn sighted(from: &[[f64; 7]], sighting: &[f64]) -> Result<Vec<[f64; 3]>> {
    (0..from.len())
        .map(|r| Ok(host::pose_transform(from[r], row::<3>(sighting, r)?)))
        .collect()
}

/// The start of every row: the poses at dead reckoning on the odometry, the landmarks at their
/// first sighting.
fn start(
    scenario: &Scenario,
    ticks: &[Tick],
    truth: &Values,
    rows: usize,
    target: &Target,
) -> Result<Values> {
    let mut start = Values::new();
    let mut previous: Vec<[f64; 7]> = vec![scenario.nominal[0]; rows];
    for t in ticks {
        if let Some(odometry) = t.odometry {
            previous = dead_reckon(&previous, &self::rows(truth, odometry)?)?;
        }
        start.insert(t.pose, load(&previous, target));
        for &(j, m) in &t.new_landmarks {
            let z = t
                .sightings
                .iter()
                .find(|&&(l, _)| l == j)
                .map(|&(_, z)| z)
                .ok_or("a new landmark not sighted at its tick")?;
            start.insert(
                m,
                load(&sighted(&previous, &self::rows(truth, z)?)?, target),
            );
        }
    }
    Ok(start)
}
