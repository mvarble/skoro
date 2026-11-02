//! The normal: a term scoring a quantity, or its Gaussian part as a likelihood factor.

use skoro_core::scope::factor;
use skoro_core::{Declares, Error, Q, QDyn, Real, Space, TermHandle, VectorDyn};

use crate::composites::{normalizer, whiten};
use crate::refuse::refuse;
use crate::{Scale, roles};

/// The normal `𝒩(μ, Σ)` on a space, by its mean and scale: a term to add, or a factor.
#[derive(Clone, Debug)]
pub struct Normal {
    space: Space,
    mean: QDyn,
    scale: Scale,
}

/// The nodes of a normal's term, each a node of the model.
#[derive(Clone, Copy, Debug)]
pub struct Parts {
    /// `x ⊖ μ`.
    pub tangent: Q<VectorDyn>,
    /// `W (x ⊖ μ)`.
    pub residual: Q<VectorDyn>,
    /// `λ_μ(x ⊖ μ)`, `None` where it is identically zero.
    pub log_volume: Option<Q<Real>>,
    /// The term: the Gaussian log-density of `x ⊖ μ`, less the log-volume.
    pub log_density: Q<Real>,
}

impl Normal {
    /// The normal on the space `marker` declares.
    pub fn on<T: Declares>(marker: T, mean: Q<T>, scale: Scale) -> Normal {
        Normal::in_space(marker.space(), mean, scale)
    }

    /// The normal on `space`, for a product or any handle without a declaring marker.
    pub fn in_space(space: Space, mean: impl Into<QDyn>, scale: Scale) -> Normal {
        Normal {
            space,
            mean: mean.into(),
            scale,
        }
    }

    pub fn space(&self) -> &Space {
        &self.space
    }

    pub fn mean(&self) -> QDyn {
        self.mean
    }

    pub fn scale(&self) -> Scale {
        self.scale
    }

    /// The term's nodes at `x`, added to the model with no term: what `score` scores.
    #[track_caller]
    pub fn log_density(&self, x: impl Into<QDyn>) -> Parts {
        let x = x.into();
        let (tangent, residual, gaussian) = self.gaussian(x);
        let log_volume = (!euclidean(&self.space)).then(|| self.mean.log_volume(tangent));
        let log_density = match log_volume {
            Some(lambda) => gaussian - lambda,
            None => gaussian,
        };
        Parts {
            tangent: tangent.narrow(),
            residual: residual.narrow(),
            log_volume: log_volume.map(QDyn::narrow),
            log_density: log_density.narrow(),
        }
    }

    /// The Gaussian part at `z` as a likelihood factor, scoring nothing, with the residual, mean
    /// and scale bound: `-½|W (z ⊖ μ)|² + Σ log W_ii - (d/2) log 2π`. It needs no measure and no
    /// global chart. With a fixed scale its normalizer is a constant of every query; with a free
    /// one it is the normalizer of no law, since the law whose likelihood the factor is has none
    /// in closed form.
    #[track_caller]
    pub fn factor_at(&self, z: impl Into<QDyn>) -> TermHandle {
        let (_, residual, gaussian) = self.gaussian(z.into());
        factor(gaussian.narrow())
            .bind(&roles::RESIDUAL, residual)
            .bind(&roles::MEAN, self.mean)
            .bind(roles::scale(self.scale.form()), self.scale.value())
    }

    /// `x ⊖ μ`, `W (x ⊖ μ)` and the Gaussian log-density of `x ⊖ μ`, in the open scope.
    #[track_caller]
    fn gaussian(&self, x: QDyn) -> (QDyn, QDyn, QDyn) {
        let Some(d) = self.space.dimension() else {
            let failed = refuse(
                Error::NotContinuous {
                    space: self.space.clone(),
                },
                &[x],
            );
            return (failed, failed, failed);
        };
        let form = self.scale.form();
        let composites = whiten(form, d).and_then(|w| Ok((w, normalizer(form, d)?)));
        let (whiten, normalizer) = match composites {
            Ok(composites) => composites,
            Err(error) => {
                let failed = refuse(error, &[x]);
                return (failed, failed, failed);
            }
        };
        let tangent = x.local(self.mean);
        let scale = self.scale.value();
        let residual = whiten.call(&[tangent, scale]);
        let gaussian = (residual * residual).sum() * -0.5 + normalizer.call(&[scale]);
        (tangent, residual, gaussian)
    }
}

/// Whether the space's log-volume is identically zero: `Real`, `Vector` and `Matrix`, and
/// products of them alone.
fn euclidean(space: &Space) -> bool {
    match space {
        Space::Real | Space::Vector(_) | Space::Matrix(..) => true,
        Space::Product(factors) => factors.iter().all(euclidean),
        _ => false,
    }
}
