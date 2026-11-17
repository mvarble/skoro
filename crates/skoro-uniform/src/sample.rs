//! The uniform's sampler.

use skoro_core::{Key, Param, Sample, Space};
use skoro_tensor::{Integers, Points, Target, Value};

/// Draws a uniform point on the host, from the variates of each row's key: as many as the space
/// has dimensions, or one for a categorical space.
#[derive(Debug)]
pub struct UniformSample {
    space: Space,
}

impl UniformSample {
    /// The sampler of the uniform distribution on `space`, which `Uniform::score` gives only to
    /// a term on a space of finite mass.
    pub(crate) fn new(space: Space) -> UniformSample {
        UniformSample { space }
    }
}

impl Sample for UniformSample {
    fn name(&self) -> &'static str {
        "skoro.uniform"
    }

    fn params(&self) -> Vec<Param> {
        vec![Param::Space(self.space.clone())]
    }

    fn draw(&self, _: &[Value], keys: &[Key], target: &Target) -> Value {
        let (rows, n) = (keys.len(), self.space.storage());
        match self.space.cardinality() {
            Some(n) => {
                let categories: Vec<i64> = keys
                    .iter()
                    .map(|key| category(n, key.uniforms(1)[0]))
                    .collect();
                match Integers::from_host(&categories, rows, 1, target) {
                    Ok(integers) => Value::Discrete(integers),
                    Err(_) => {
                        unreachable!("a category is below n, which the target's integers hold")
                    }
                }
            }
            None => {
                let d = self.space.dimension().unwrap_or(0);
                let mut points = vec![0.0; rows * n];
                // A space of no storage has no rows to fill.
                for (key, point) in keys.iter().zip(points.chunks_mut(n.max(1))) {
                    if self.space.draw_uniform(&key.uniforms(d), point).is_err() {
                        unreachable!("a term is added on a space of finite mass alone")
                    }
                }
                Value::Continuous(Points::from_host(&points, rows, n, target))
            }
        }
    }
}

/// `⌊n u⌋`, never `n`: the category of a variate `u` in (0, 1), for `n` of one or more. The
/// product is rounded to a double, and the minimum keeps that rounding from ever returning `n`.
pub(crate) fn category(n: usize, u: f64) -> i64 {
    ((n as f64 * u) as usize).min(n - 1) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_never_returns_n() {
        // The largest and the smallest variates a key gives.
        let (largest, smallest) = (1.0 - f64::EPSILON / 2.0, f64::EPSILON / 2.0);
        assert!(largest < 1.0 && smallest > 0.0);
        for n in [1, 2, 3, 1000, 1 << 52] {
            assert_eq!(category(n, largest), n as i64 - 1, "{n}");
            assert_eq!(category(n, smallest), 0, "{n}");
        }
        assert_eq!(category(4, 0.5), 2);
        assert_eq!(category(4, 0.49), 1);
    }
}
