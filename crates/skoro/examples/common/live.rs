//! The window's live part: the handles of the quantities it holds, the observed values of its
//! measurements, the landmarks' sightings among its poses, and its current mode.

use std::collections::BTreeMap;

use skoro::tensor::Value;
use skoro::{Addresses, NodeId, Observations, Q, QDyn, Target, Values, Vector};
use skoro_spatial::Pose;

use super::batch::{dead_reckon, load, rows, sighted};
use super::model::Tick;
use super::report::Result;
use super::scenario::Scenario;

/// What the window holds, by tick and landmark.
#[derive(Default)]
pub struct Live {
    /// The window's poses, by tick.
    pub poses: BTreeMap<usize, Q<Pose>>,
    /// The odometry from tick `a` to `a + 1`, by `a`.
    odometry: BTreeMap<usize, Q<Pose>>,
    /// The sightings, by tick and landmark.
    sightings: BTreeMap<(usize, usize), Q<Vector<3>>>,
    /// The landmarks in the window, by index: the map the model builder extends.
    pub landmarks: BTreeMap<usize, Q<Vector<3>>>,
    /// The observed value of each measurement in the window.
    observed: BTreeMap<NodeId, (QDyn, Value)>,
    /// The window's current point: its mode, and the start of what was just appended.
    pub point: Values,
}

impl Live {
    /// The newest pose, none before the first tick.
    pub fn newest(&self) -> Option<Q<Pose>> {
        self.poses.values().next_back().copied()
    }

    /// The number of poses held.
    pub fn len(&self) -> usize {
        self.poses.len()
    }

    /// Records tick `k`, appended to the window as `tick`, its measurements observed at the values
    /// `truth` holds for the same tick of the whole model, `whole`.
    pub fn append(&mut self, k: usize, tick: &Tick, whole: &Tick, truth: &Values) -> Result {
        self.poses.insert(k, tick.pose);
        if let (Some(window), Some(drawn)) = (tick.odometry, whole.odometry) {
            self.odometry.insert(k - 1, window);
            let value = super::batch::value(truth, drawn)?.clone();
            self.observed.insert(window.id(), (window.into(), value));
        }
        for (&(j, window), &(_, drawn)) in tick.sightings.iter().zip(&whole.sightings) {
            self.sightings.insert((k, j), window);
            let value = super::batch::value(truth, drawn)?.clone();
            self.observed.insert(window.id(), (window.into(), value));
        }
        Ok(())
    }

    /// The observations of every measurement in the window.
    pub fn observations(&self) -> Observations {
        self.observed
            .values()
            .fold(Observations::new(), |o, (q, value)| {
                o.observe_value(*q, value)
            })
    }

    /// Starts what tick `k` appended: the pose dead-reckoned from the previous mode along the
    /// odometry, or the nominal first pose, and each new landmark at its sighting.
    pub fn start(
        &mut self,
        scenario: &Scenario,
        k: usize,
        tick: &Tick,
        rows: usize,
        target: &Target,
    ) -> Result {
        let pose: Vec<[f64; 7]> = match (k.checked_sub(1), tick.odometry) {
            (Some(previous), Some(odometry)) => {
                let from = self.poses.get(&previous).ok_or("no previous pose")?;
                let data = super::batch::rows(&self.point, *from)?;
                let from: Vec<[f64; 7]> = (0..rows)
                    .map(|r| super::batch::row::<7>(&data, r))
                    .collect::<Result<_>>()?;
                let y = self.observed_rows(odometry)?;
                dead_reckon(&from, &y)?
            }
            _ => vec![scenario.nominal[0]; rows],
        };
        for &(j, m) in &tick.new_landmarks {
            let z = self
                .sightings
                .get(&(k, j))
                .ok_or("a new landmark not sighted at its tick")?;
            let at = sighted(&pose, &self.observed_rows(*z)?)?;
            self.point.insert(m, load(&at, target));
        }
        self.point.insert(tick.pose, load(&pose, target));
        Ok(())
    }

    fn observed_rows(&self, q: impl Into<NodeId>) -> Result<Vec<f64>> {
        let id = q.into();
        let values = Values::new().with(
            id,
            self.observed
                .get(&id)
                .ok_or("an unobserved measurement")?
                .1
                .clone(),
        );
        rows(&values, id)
    }

    /// The landmarks that pose `a` alone still sees among the window's poses.
    fn seen_only_by(&self, a: usize) -> Vec<usize> {
        self.sightings
            .keys()
            .filter(|&&(k, _)| k == a)
            .map(|&(_, j)| j)
            .filter(|&j| !self.sightings.keys().any(|&(k, l)| l == j && k != a))
            .collect()
    }

    /// The retirement of pose `a`: its addresses (the pose, the odometry from it, its sightings,
    /// and each landmark no other pose of the window sees) and the observed values of the
    /// measurements among them.
    pub fn retiring(&self, a: usize) -> (Addresses, Observations) {
        let mut addresses = Addresses::new()
            .at(format!("pose/{a}"))
            .under(format!("sighting/{a}"));
        let mut observed = Observations::new();
        if let Some(y) = self.odometry.get(&a) {
            addresses = addresses.at(format!("odometry/{a}"));
            if let Some((_, value)) = self.observed.get(&y.id()) {
                observed = observed.observe_value(*y, value);
            }
        }
        for (&(k, _), z) in &self.sightings {
            if k == a
                && let Some((_, value)) = self.observed.get(&z.id())
            {
                observed = observed.observe_value(*z, value);
            }
        }
        for j in self.seen_only_by(a) {
            addresses = addresses.at(format!("landmark/{j}"));
        }
        (addresses, observed)
    }

    /// Drops what the retirement of pose `a` removed from the model.
    pub fn retire(&mut self, a: usize) {
        let mut gone: Vec<NodeId> = Vec::new();
        for j in self.seen_only_by(a) {
            if let Some(m) = self.landmarks.remove(&j) {
                gone.push(m.id());
            }
        }
        if let Some(pose) = self.poses.remove(&a) {
            gone.push(pose.id());
        }
        if let Some(y) = self.odometry.remove(&a) {
            gone.push(y.id());
        }
        let sightings: Vec<(usize, usize)> = self
            .sightings
            .keys()
            .copied()
            .filter(|&(k, _)| k == a)
            .collect();
        for key in sightings {
            if let Some(z) = self.sightings.remove(&key) {
                gone.push(z.id());
            }
        }
        for id in gone {
            self.point.remove(id);
            self.observed.remove(&id);
        }
    }
}
