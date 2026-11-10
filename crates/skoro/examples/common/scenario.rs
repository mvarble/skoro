//! The scenarios: a nominal trajectory, the landmarks' nominal positions, the sightings chosen from
//! them within a sensor's range, and the scales, all fixed before anything is drawn.

use std::f64::consts::PI;

use skoro_spatial::host;

/// The constants of one scenario.
pub struct Scenario {
    /// `x̄₀ … x̄_K`, storage order: the quaternion `(w, x, y, z)`, then the position.
    pub nominal: Vec<[f64; 7]>,
    /// `m̄₁ … m̄_J`.
    pub landmarks: Vec<[f64; 3]>,
    /// `V`, sorted by tick then landmark.
    pub sightings: Vec<(usize, usize)>,
    pub scales: Scales,
}

/// The standard deviations of the terms, the same for both scenarios.
pub struct Scales {
    /// `Σ₀`, `Σ_u`, `Σ_y`: standard deviations of `(φ, ρ)`, rotation first.
    pub first: [f64; 6],
    pub motion: [f64; 6],
    pub odometry: [f64; 6],
    /// `Σ_m`, `Σ_z`: isotropic standard deviations in metres.
    pub landmark: f64,
    pub sighting: f64,
}

impl Scales {
    fn standard() -> Scales {
        let pair = |rotation: f64, position: f64| {
            [rotation, rotation, rotation, position, position, position]
        };
        Scales {
            first: pair(0.01, 0.01),
            motion: pair(0.02, 0.05),
            odometry: pair(0.005, 0.02),
            landmark: 2.0,
            sighting: 0.05,
        }
    }
}

/// The pose at `position` whose forward axis points along `direction`, its up axis in the
/// vertical plane through it: the yaw of the direction, then its pitch.
fn heading(position: [f64; 3], direction: [f64; 3]) -> [f64; 7] {
    let yaw = direction[1].atan2(direction[0]);
    let level = direction[0].hypot(direction[1]);
    let pitch = direction[2].atan2(level);
    // About y, a positive angle turns the forward axis down; the path's climb is the negative.
    let q = host::compose(host::exp([0.0, 0.0, yaw]), host::exp([0.0, -pitch, 0.0]));
    [
        q[0],
        q[1],
        q[2],
        q[3],
        position[0],
        position[1],
        position[2],
    ]
}

/// The pairs `(k, j)` within `range` of each other, sorted by tick then landmark.
fn sightings(nominal: &[[f64; 7]], landmarks: &[[f64; 3]], range: f64) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (k, pose) in nominal.iter().enumerate() {
        for (j, m) in landmarks.iter().enumerate() {
            let d2: f64 = (0..3).map(|i| (pose[4 + i] - m[i]).powi(2)).sum();
            if d2.sqrt() <= range {
                pairs.push((k, j));
            }
        }
    }
    pairs
}

impl Scenario {
    /// The closed circuit: 73 poses on a circle of radius 10 m traversed 1.2 times, heading along
    /// the path, the height `0.5 sin 2θ` and the pitch the path's climb,
    /// `atan(0.1 cos 2θ)`; 24 landmarks on rings
    /// of radii 7 and 13 m at alternating heights ±1 m; sighted within `h = 6` m.
    pub fn circuit() -> Scenario {
        let nominal: Vec<[f64; 7]> = (0..73)
            .map(|k| {
                let theta = 1.2 * 2.0 * PI * k as f64 / 72.0;
                let position = [
                    10.0 * theta.cos(),
                    10.0 * theta.sin(),
                    0.5 * (2.0 * theta).sin(),
                ];
                let direction = [-theta.sin(), theta.cos(), 0.1 * (2.0 * theta).cos()];
                heading(position, direction)
            })
            .collect();
        let landmarks: Vec<[f64; 3]> = (0..24)
            .map(|j| {
                let (ring, i) = (j % 2, j / 2);
                let radius = if ring == 0 { 7.0 } else { 13.0 };
                let angle = 2.0 * PI * (i as f64 + 0.5 * ring as f64) / 12.0;
                let height = if i % 2 == 0 { 1.0 } else { -1.0 };
                [radius * angle.cos(), radius * angle.sin(), height]
            })
            .collect();
        let sightings = sightings(&nominal, &landmarks, 6.0);
        Scenario {
            nominal,
            landmarks,
            sightings,
            scales: Scales::standard(),
        }
    }

    /// The open path: 80 poses 1 m apart along `(k, 3 sin(k/10), 0.3 sin(k/7))`, heading along it;
    /// 40 landmarks at `(2j, ±5, 0)`, alternating sides; sighted within `h = 8.5` m. Refuses itself
    /// unless every landmark's sightings are an interval of ticks, as the window requires.
    pub fn open_path() -> Result<Scenario, Box<dyn std::error::Error>> {
        let nominal: Vec<[f64; 7]> = (0..80)
            .map(|k| {
                let s = k as f64;
                let position = [s, 3.0 * (s / 10.0).sin(), 0.3 * (s / 7.0).sin()];
                let direction = [1.0, 0.3 * (s / 10.0).cos(), 0.3 / 7.0 * (s / 7.0).cos()];
                heading(position, direction)
            })
            .collect();
        let landmarks: Vec<[f64; 3]> = (0..40)
            .map(|j| {
                let side = if j % 2 == 0 { 5.0 } else { -5.0 };
                [2.0 * j as f64, side, 0.0]
            })
            .collect();
        let sightings = sightings(&nominal, &landmarks, 8.5);
        let scenario = Scenario {
            nominal,
            landmarks,
            sightings,
            scales: Scales::standard(),
        };
        for j in 0..scenario.landmarks.len() {
            let ticks: Vec<usize> = scenario.ticks_sighting(j);
            if ticks.windows(2).any(|w| w[1] != w[0] + 1) {
                return Err(
                    format!("landmark {j} is sighted at ticks {ticks:?}, not an interval").into(),
                );
            }
        }
        Ok(scenario)
    }

    /// `K`, the last tick.
    pub fn last(&self) -> usize {
        self.nominal.len() - 1
    }

    /// The ticks at which landmark `j` is sighted, increasing.
    pub fn ticks_sighting(&self, j: usize) -> Vec<usize> {
        self.sightings
            .iter()
            .filter(|&&(_, l)| l == j)
            .map(|&(k, _)| k)
            .collect()
    }

    /// `k(j)`, the first tick each landmark is sighted at; `Err` for a landmark never sighted.
    pub fn first_sightings(&self) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        (0..self.landmarks.len())
            .map(|j| {
                self.ticks_sighting(j)
                    .first()
                    .copied()
                    .ok_or_else(|| format!("landmark {j} is never sighted").into())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::{Scenario, heading};
    use skoro_spatial::host;

    #[test]
    fn a_pose_heads_along_its_direction() {
        for direction in [[1.0, 0.0, 0.0], [0.3, -2.0, 0.5], [-1.0, 1.0, -0.4]] {
            let pose = heading([0.0; 3], direction);
            let r = host::matrix([pose[0], pose[1], pose[2], pose[3]]);
            let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
            for i in 0..3 {
                assert!(
                    (r[i][0] - direction[i] / norm).abs() < 1e-12,
                    "{direction:?}"
                );
            }
        }
    }

    #[test]
    fn the_scenarios_are_sighted() {
        // `open_path` refuses itself unless every landmark's sightings are an interval of ticks.
        for scenario in [Scenario::circuit(), Scenario::open_path().unwrap()] {
            let first = scenario.first_sightings().unwrap();
            assert_eq!(first.len(), scenario.landmarks.len());
        }
    }
}
