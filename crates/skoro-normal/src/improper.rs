//! The improper Gaussian factor `-½|A δ - b|² - ηᵀδ + c` over a stacked tangent at constant
//! anchors: a sum of squared affine rows of any number, a linear part and a constant, which no
//! normalization need make a density of `δ`.

use skoro_core::scope::{constant, factor};
use skoro_core::{IntoHostValue, MatrixDyn, Q, Real, Replacement, TermHandle, VectorDyn};

use crate::stacked::Binds;
use crate::{Stacked, roles};

/// An improper Gaussian factor over a stacked tangent: its nodes, made in the open scope by
/// [`Stacked::improper`]. Its value is `-½|A δ - b|² - ηᵀδ + c`; it binds the residual `A δ - b`
/// and, when a linear part is given, `ηᵀδ` under the log-volume role.
#[derive(Clone, Copy, Debug)]
pub struct Improper {
    tangent: Q<VectorDyn>,
    dimension: usize,
    /// `A δ - b`, a `Vector(r)`.
    residual: Q<VectorDyn>,
    /// `ηᵀδ`, when a linear part is given.
    linear: Option<Q<Real>>,
    constant: Q<Real>,
}

impl Stacked {
    /// The improper Gaussian factor over `δ` with `rows` rows: `a` the `rows × D` matrix `A`,
    /// **column by column** as a `MatrixDyn` is stored, `b` the right-hand side, a `Vector(rows)`,
    /// and `c` the constant, and no linear part. Each may be shared or given per row of a batch
    /// with `Rows`. Any number of rows is a factor, none included; the rows are kept as given, and
    /// none is added or removed.
    ///
    /// **The order of the blocks is load-bearing.** The columns of `A`, and the entries of `η`,
    /// follow the order of the `block` calls, factor by factor, each over its quantity's tangent
    /// coordinates. Nothing checks that correspondence, because nothing can: a matrix of the right
    /// width is the rows of a factor in any order.
    ///
    /// Numbers that do not fit their spaces are refused with `ValueShape` at the caller's line,
    /// and a stacked tangent of no blocks by the core's concatenation (`SpaceMismatch`).
    #[track_caller]
    pub fn improper(
        &self,
        rows: usize,
        a: impl IntoHostValue<MatrixDyn>,
        b: impl IntoHostValue<VectorDyn>,
        c: impl IntoHostValue<Real>,
    ) -> Improper {
        let tangent = self.tangent();
        let a = constant(MatrixDyn(rows, self.dimension()), a);
        let b = constant(VectorDyn(rows), b);
        Improper {
            tangent,
            dimension: self.dimension(),
            residual: a.matmul(tangent) - b,
            linear: None,
            constant: constant(Real, c),
        }
    }
}

impl Improper {
    /// The same factor with the linear part `ηᵀδ`, `eta` a `Vector(D)`, shared or per row.
    /// Whether a factor has one is a matter of its shape, the same for every row of a batch: an
    /// `eta` of zeros is still bound.
    #[track_caller]
    pub fn linear(self, eta: impl IntoHostValue<VectorDyn>) -> Improper {
        let eta = constant(VectorDyn(self.dimension), eta);
        Improper {
            linear: Some((eta * self.tangent).sum()),
            ..self
        }
    }

    /// `δ`.
    pub fn tangent(&self) -> Q<VectorDyn> {
        self.tangent
    }

    /// `A δ - b`.
    pub fn residual(&self) -> Q<VectorDyn> {
        self.residual
    }

    /// `ηᵀδ`, when a linear part was given.
    pub fn linear_part(&self) -> Option<Q<Real>> {
        self.linear
    }

    /// `-½|A δ - b|² - ηᵀδ + c`, as a log-density alone, which binds no roles. A retirement's
    /// replacement with its roles bound is [`replacement`](Improper::replacement).
    #[track_caller]
    pub fn log_density(&self) -> Q<Real> {
        let value = (self.residual * self.residual).sum() * -0.5 + self.constant;
        match self.linear {
            Some(linear) => value - linear,
            None => value,
        }
    }

    /// The factor as a retirement's `Replacement`, with no proposals and its roles bound as
    /// [`factor`](Improper::factor) binds them. A blanket quantity it leaves undetermined needs a
    /// proposal from the caller to be generated.
    #[track_caller]
    pub fn replacement(&self) -> Replacement {
        let replacement = Replacement {
            log_density: self.log_density(),
            proposals: Vec::new(),
            bindings: Vec::new(),
        };
        self.with_roles(replacement)
    }

    /// The factor added as a likelihood factor, scoring nothing, with its roles bound.
    #[track_caller]
    pub fn factor(&self) -> TermHandle {
        self.with_roles(factor(self.log_density()))
    }

    /// `target` with the residual bound, and the linear part under the log-volume role when there
    /// is one: the same for a factor and a replacement.
    #[track_caller]
    fn with_roles<B: Binds>(&self, target: B) -> B {
        let target = target.bind_role(&roles::RESIDUAL, self.residual.untyped());
        match self.linear {
            Some(linear) => target.bind_role(&roles::LOG_VOLUME, linear.untyped()),
            None => target,
        }
    }
}
