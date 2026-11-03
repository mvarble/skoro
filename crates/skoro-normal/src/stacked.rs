//! The stacked tangent of a retirement's blanket at constant anchors, and the Gaussian
//! replacement on it.

use skoro_core::scope::{concatenate, constant, factor};
use skoro_core::{Declares, IntoHostValue, Q, QDyn, Real, Space, TermHandle, VectorDyn};

use crate::{Normal, Scale, roles};

/// The stacked tangent `δ = (x₁ ⊖ p₁; …; x_k ⊖ p_k)`, block by block, each at a constant anchor
/// the builder makes itself, so that no anchor moves with the quantities it is read at.
#[derive(Clone, Debug, Default)]
pub struct Stacked {
    blocks: Vec<QDyn>,
    dimension: usize,
}

impl Stacked {
    pub fn new() -> Stacked {
        Stacked::default()
    }

    /// Appends `x ⊖ p`, `p` the constant `anchor` of `marker`'s space. A discrete space is
    /// refused (`NotContinuous`, by the core's inverse retraction). A quantity of a product type
    /// is given factor by factor, `x.factor(i)` with each factor's marker.
    #[track_caller]
    pub fn block<T: Declares>(
        mut self,
        x: Q<T>,
        marker: T,
        anchor: impl IntoHostValue<T>,
    ) -> Stacked {
        let dimension = marker.space().dimension().unwrap_or(0);
        let anchor = constant(marker, anchor);
        self.blocks.push(x.untyped().local(anchor.untyped()));
        self.dimension += dimension;
        self
    }

    /// The dimension `D` of the stacked tangent.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// `δ`, a `Vector(D)`.
    #[track_caller]
    pub fn tangent(&self) -> Q<VectorDyn> {
        concatenate(&self.blocks, Space::Vector(self.dimension)).narrow()
    }

    /// `ℓ_N(δ; m, W) + c`, as a log-density with no term: what a `Replacement` holds.
    ///
    /// A replacement written with this binds no roles by itself, while the same factor added with
    /// [`factor`](Stacked::factor) binds its residual, mean and scale. A caller binds them on the
    /// replacement with `Replacement::bind`, which the core checks and binds to the factor it
    /// adds.
    #[track_caller]
    pub fn log_density(&self, mean: &[f64], scale: Scale, constant: f64) -> Q<Real> {
        self.gaussian(mean, scale, constant).0
    }

    /// `ℓ_N(δ; m, W) + c`, added as a likelihood factor with its residual, mean and scale bound.
    #[track_caller]
    pub fn factor(&self, mean: &[f64], scale: Scale, constant: f64) -> TermHandle {
        let (log_density, residual, mean) = self.gaussian(mean, scale, constant);
        factor(log_density)
            .bind(&roles::RESIDUAL, residual)
            .bind(&roles::MEAN, mean)
            .bind(roles::scale(scale.form()), scale.value())
    }

    /// The log-density, the residual `W (δ - m)` and the mean's node, in the open scope.
    #[track_caller]
    fn gaussian(&self, mean: &[f64], scale: Scale, plus: f64) -> (Q<Real>, QDyn, QDyn) {
        let space = Space::Vector(self.dimension);
        let mean = constant(VectorDyn(mean.len()), mean).untyped();
        // On ℝ^D the log-volume is zero, and the term is the Gaussian part alone.
        let parts = Normal::in_space(space, mean, scale).log_density(self.tangent());
        let log_density = (parts.log_density.untyped() + plus).narrow();
        (log_density, parts.residual.untyped(), mean)
    }
}
