//! The stacked tangent of a retirement's blanket at constant anchors, and the Gaussian
//! replacement on it.

use skoro_core::scope::{concatenate, constant, factor};
use skoro_core::{
    Declares, IntoHostValue, Multiplicity, Q, QDyn, Real, Replacement, Role, Space, TermHandle,
    VectorDyn,
};

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

    /// `ℓ_N(δ; m, W) + c`, as a log-density alone, which binds no roles. A retirement's
    /// replacement with its roles bound is [`replacement`](Stacked::replacement).
    #[track_caller]
    pub fn log_density(&self, mean: &[f64], scale: Scale, constant: f64) -> Q<Real> {
        self.gaussian(mean, scale, constant).0
    }

    /// `ℓ_N(δ; m, W) + c` as a retirement's `Replacement`, with no proposals and its residual,
    /// mean and scale bound as [`factor`](Stacked::factor) binds them, which the retirement checks
    /// and binds to the factor it adds. A caller adds proposals, or further bindings, to it.
    #[track_caller]
    pub fn replacement(&self, mean: &[f64], scale: Scale, constant: f64) -> Replacement {
        let (log_density, residual, mean) = self.gaussian(mean, scale, constant);
        let replacement = Replacement {
            log_density,
            proposals: Vec::new(),
            bindings: Vec::new(),
        };
        with_roles(replacement, residual, mean, scale)
    }

    /// `ℓ_N(δ; m, W) + c`, added as a likelihood factor with its residual, mean and scale bound.
    #[track_caller]
    pub fn factor(&self, mean: &[f64], scale: Scale, constant: f64) -> TermHandle {
        let (log_density, residual, mean) = self.gaussian(mean, scale, constant);
        with_roles(factor(log_density), residual, mean, scale)
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

/// What the normal's roles are bound on: a term in a scope, or a retirement's replacement.
trait Binds: Sized {
    fn bind_role<M: Multiplicity>(self, role: &Role<M>, node: QDyn) -> Self;
}

impl Binds for TermHandle {
    #[track_caller]
    fn bind_role<M: Multiplicity>(self, role: &Role<M>, node: QDyn) -> TermHandle {
        self.bind(role, node)
    }
}

impl Binds for Replacement {
    fn bind_role<M: Multiplicity>(self, role: &Role<M>, node: QDyn) -> Replacement {
        self.bind(role, node)
    }
}

/// `target` with the residual, the mean and the scale bound under the normal's roles, the same for
/// a factor and a replacement.
#[track_caller]
fn with_roles<B: Binds>(target: B, residual: QDyn, mean: QDyn, scale: Scale) -> B {
    target
        .bind_role(&roles::RESIDUAL, residual)
        .bind_role(&roles::MEAN, mean)
        .bind_role(roles::scale(scale.form()), scale.value())
}
