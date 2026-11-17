//! The uniform's sampler.

use skoro_core::{Key, Param, Sample, Space};
use skoro_tensor::{Points, Target, Value};

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
            Some(_) => unreachable!("a categorical draw is not yet written"),
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
