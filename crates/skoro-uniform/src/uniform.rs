//! The distribution: a constant term, its exact sampler, and a residual of no components.

use std::sync::Arc;

use skoro_core::scope::{constant, score};
use skoro_core::{
    Composite, Declares, Distribution, Error, Param, QDyn, Real, Sampler, Space, TermHandle, Typed,
    VectorDyn,
};
use skoro_normal::roles;

use crate::UniformSample;
use crate::refuse::refuse;

/// The marker of a scored quantity of any space: the uniform reads the space from `Uniform`
/// itself.
#[derive(Clone, Copy, Debug)]
struct Scored;

impl Typed for Scored {
    fn admits(_: &Space) -> bool {
        true
    }
}

/// The uniform distribution on a space of finite mass: the law `ν / ν(𝕏)`, with no parameters.
#[derive(Clone, Debug)]
pub struct Uniform {
    space: Space,
}

impl Uniform {
    /// The uniform distribution on the space `marker` declares.
    pub fn on<T: Declares>(marker: T) -> Uniform {
        Uniform::in_space(marker.space())
    }

    /// The uniform distribution on `space`, for a product or any space without a marker.
    pub fn in_space(space: Space) -> Uniform {
        Uniform { space }
    }

    pub fn space(&self) -> &Space {
        &self.space
    }

    /// `log ν(𝕏)` (`NoMass` for a space without a finite mass).
    pub fn log_mass(&self) -> Result<f64, Error> {
        self.space.log_mass()
    }

    /// Scores `x`: the constant term `-log ν(𝕏)`, its exact sampler, and a residual of no
    /// components. Refuses a space without a finite mass (`NoMass`) and an `x` of another space
    /// (`SpaceMismatch` of `skoro.uniform.log_density`), recording the refusal at the caller's
    /// line. [`sample`](skoro_core::scope::sample) declares a quantity and scores it by a uniform
    /// in one call.
    ///
    /// The term is a call of a composite of `x` that does not read it: a mapped node, so a
    /// log-density, and one of its own for each quantity scored; refused by the core's check of
    /// an operation's inputs when `x` is of another space; and of zero derivative maps. The
    /// residual is what a linearization reads of a continuous quantity, a factor with no rows.
    #[track_caller]
    pub fn score(&self, x: impl Into<QDyn>) -> TermHandle {
        let x = x.into();
        let term = self.log_mass().and_then(|log_mass| {
            let space = self.space.clone();
            Composite::define(
                "skoro.uniform.log_density",
                vec![Param::Space(space.clone())],
                &[space],
                |_| (-constant(Real, log_mass)).untyped(),
            )
            .map_err(|failure| failure.error)
        });
        let log_density = match term {
            Ok(composite) => composite.call(&[x]),
            Err(error) => refuse(error, &[x]),
        };
        score(x.narrow::<Scored>(), log_density.narrow())
            .sampler(Sampler {
                sample: Arc::new(UniformSample::new(self.space.clone())),
                inputs: Vec::new(),
                log_density: None,
            })
            .bind(&roles::RESIDUAL, constant(VectorDyn(0), &[] as &[f64]))
    }
}

impl Distribution for Uniform {
    #[track_caller]
    fn score(&self, x: QDyn) -> TermHandle {
        Uniform::score(self, x)
    }
}
