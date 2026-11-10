//! The model, tick by tick: one builder for the batch, which calls it for every tick in one
//! `build`, and for the window, which calls it once per tick in a `build` of its own.

use std::collections::BTreeMap;

use skoro::scope::{constant, quantity};
use skoro::{Q, Vector};
use skoro_normal::{Normal, Scale};
use skoro_spatial::{Pose, PoseMaps};

use super::scenario::Scenario;

/// The quantities of one tick, as their handles.
pub struct Tick {
    pub pose: Q<Pose>,
    /// The odometry from the previous pose, none at tick 0.
    pub odometry: Option<Q<Pose>>,
    /// The landmarks first sighted at this tick, with their indices.
    pub new_landmarks: Vec<(usize, Q<Vector<3>>)>,
    /// This tick's sightings, with their landmarks' indices.
    pub sightings: Vec<(usize, Q<Vector<3>>)>,
}

/// Adds tick `k` inside the open scope: the pose `pose/k`, scored by the prior at `k = 0` and by
/// the motion term from `previous` after; the odometry `odometry/{k-1}` from `previous`; each
/// landmark `landmark/j` first sighted at `k`, scored by its prior; each sighting `sighting/k/j`.
/// `landmarks` holds every landmark already in the model, by index, and gains the new ones.
pub fn tick(
    scenario: &Scenario,
    k: usize,
    previous: Option<Q<Pose>>,
    landmarks: &mut BTreeMap<usize, Q<Vector<3>>>,
) -> Tick {
    let scales = &scenario.scales;
    let pose = quantity(format!("pose/{k}"), Pose);
    let odometry = match previous {
        None => {
            let first = constant(Pose, scenario.nominal[0]);
            Normal::on(Pose, first, Scale::sds(&scales.first)).score(pose);
            None
        }
        Some(previous) => {
            // The command u_{k-1}, the nominal motion between the ticks, as a mapped value of
            // constants.
            let command = constant(Pose, scenario.nominal[k - 1])
                .between(constant(Pose, scenario.nominal[k]));
            Normal::on(Pose, previous * command, Scale::sds(&scales.motion)).score(pose);
            let odometry = quantity(format!("odometry/{}", k - 1), Pose);
            Normal::on(Pose, previous.between(pose), Scale::sds(&scales.odometry)).score(odometry);
            Some(odometry)
        }
    };
    let mut new_landmarks = Vec::new();
    let mut sightings = Vec::new();
    for &(_, j) in scenario.sightings.iter().filter(|&&(at, _)| at == k) {
        let landmark = *landmarks.entry(j).or_insert_with(|| {
            let m = quantity(format!("landmark/{j}"), Vector::<3>);
            let nominal = constant(Vector::<3>, scenario.landmarks[j]);
            Normal::on(Vector::<3>, nominal, Scale::sd(scales.landmark)).score(m);
            new_landmarks.push((j, m));
            m
        });
        let z = quantity(format!("sighting/{k}/{j}"), Vector::<3>);
        Normal::on(
            Vector::<3>,
            pose.untransform(landmark),
            Scale::sd(scales.sighting),
        )
        .score(z);
        sightings.push((j, z));
    }
    Tick {
        pose,
        odometry,
        new_landmarks,
        sightings,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::collections::BTreeMap;

    use skoro::{Key, Observations, Target, Writer};

    use super::super::scenario::Scenario;
    use super::tick;

    #[test]
    fn the_circuit_builds_and_generates() {
        let scenario = Scenario::circuit();
        let mut writer = Writer::new();
        let ticks = writer
            .build(|| {
                let mut landmarks = BTreeMap::new();
                let mut previous = None;
                let mut ticks = Vec::new();
                for k in 0..=scenario.last() {
                    let t = tick(&scenario, k, previous, &mut landmarks);
                    previous = Some(t.pose);
                    ticks.push(t);
                }
                ticks
            })
            .unwrap();
        let landmarks: usize = ticks.iter().map(|t| t.new_landmarks.len()).sum();
        let sightings: usize = ticks.iter().map(|t| t.sightings.len()).sum();
        assert_eq!(landmarks, scenario.landmarks.len());
        assert_eq!(sightings, scenario.sightings.len());
        let generated = writer
            .snapshot()
            .generate(&Observations::new(), &Key::from_seed(1), 2, &Target::host())
            .unwrap();
        assert!(generated.values.get(ticks[3].pose).is_some());
    }
}
