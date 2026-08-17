use std::fmt::Debug;

use skoro_backend::{Backend, CotangentFrame, Point, TangentFrame, TangentVector};

use crate::Input;

/// Controls which derivative computations are actually performed, and provides tangent vectors if
/// relevant.
#[derive(Debug)]
pub enum DiffModeState<B: Backend> {
    /// Only compute the forward map `F(p0, ..., pn)` on point `(p0, ..., pn)` without derivatives.
    ///
    /// There is no state, because no additional context is needed.
    ForwardOnly,

    /// Compute the forward map `F(p0, ..., pn)` on point `p = (p0, ..., pn)` and perform forward
    /// propagation `dF_p(v)` of tangent frame `v = (v0, ..., vn)`.
    ///
    /// The state consists of _optional_ sub-blocks of tangent frames `v0, ..., vn`, where `None`
    /// means the respective component `pj` of the input point was constant.
    Jvp(Vec<Option<TangentFrame<B>>>),

    /// Compute the forward map `F(p0, ..., pn)` on point `p = (p0, ..., pn)` and memoize relevant
    /// state for cotangent map `dF^*_p`. This is the "forward" step of backpropagation, so we
    /// cannot eagerly compute `dF^*_p(α)`, as the cotangent vector `α` does not yet exist.
    ///
    /// The state consists of booleans indicating if the input at respective index is constant.
    /// This is useful for prepping VJP state with context on the respective input indices the
    /// cotangent vector is pulled back.
    Vjp(Vec<bool>),

    /// Compute the forward map `F(p0, ..., pn)` on point `p = (p0, ..., pn)`, the tangent map
    /// `dF_p(v)` on tangent vector `v = (v0, ..., vn)`, and memoize relevant states for cotangent
    /// maps `dF^*_p` and `∇_v dF^*_p`.
    ///
    /// The backward pass of an HVP invokes [`vjp`](DiffOperation::vjp) as well as
    /// [`hvp`](DiffOperation::hvp), so this variant carries **both** signals, which mean different
    /// things and must not be conflated:
    ///
    /// - `diffs`: booleans indicating whether the input at the respective index traces back to a
    ///   free variable, exactly as in [`Vjp`](DiffModeState::Vjp). This is what gates the
    ///   [`VjpState`](DiffOperation::VjpState), so that `vjp` still short-circuits inputs rooted
    ///   only in constants.
    /// - `tangents`: _optional_ sub-blocks of tangent vectors `v0, ..., vn`. `None` means no
    ///   tangent was seeded at that input, which is a strictly weaker statement than the input
    ///   being constant — a differentiable input with no seeded tangent is `None` here too.
    ///
    /// Gating the `VjpState` on `tangents[j].is_some()` is therefore a bug: it silently drops the
    /// order-1 cotangent of any differentiable input that happened not to be seeded. Conversely,
    /// state needed only by `hvp` belongs in [`HvpState`](DiffOperation::HvpState), which the
    /// tangents already gate correctly (a constant input never receives a tangent).
    Hvp {
        diffs: Vec<bool>,
        tangents: Vec<Option<TangentVector<B>>>,
    },
}

/// State returned during forward operation of [`DiffOperation`]; the enum output should match the
/// enum input from [`DiffModeState`].
pub enum DiffOperationForward<B: Backend, VjpState, HvpState> {
    /// Output point `y = F(p0, ..., pn)`.
    ForwardOnly { output: Point<B> },

    /// Output point `y = F(p0, ..., pn)` and tangent frame `w = dF_p(v)`.
    Jvp {
        output: Point<B>,
        tangent: Option<TangentFrame<B>>,
    },

    /// Output point `y = F(p0, ..., pn)` and memoized state for cotangent map `dF^*_p`.
    Vjp {
        output: Point<B>,
        vjp_state: VjpState,
    },

    /// Output point `y = F(p0, ..., pn)`, tangent vector `w = dF_p(v)`, and memoized states for
    /// cotangent map `dF^*_p` and its sensitivity `∇_v dF^*_p`.
    Hvp {
        output: Point<B>,
        tangent: Option<TangentVector<B>>,
        vjp_state: VjpState,
        hvp_state: HvpState,
    },
}

impl<B: Backend, VjpState, HvpState> DiffOperationForward<B, VjpState, HvpState> {
    /// Constructor of [`ForwardOnly`](DiffOperationForward::ForwardOnly) variant.
    pub fn forward_only(output: Point<B>) -> Self {
        Self::ForwardOnly { output }
    }

    /// Constructor of [`Jvp`](DiffOperationForward::Jvp) variant.
    pub fn jvp(output: Point<B>, tangent: Option<TangentFrame<B>>) -> Self {
        Self::Jvp { output, tangent }
    }

    /// Constructor of [`Vjp`](DiffOperationForward::Vjp) variant.
    pub fn vjp(output: Point<B>, vjp_state: VjpState) -> Self {
        Self::Vjp { output, vjp_state }
    }

    /// Constructor of [`Hvp`](DiffOperationForward::Hvp) variant.
    pub fn hvp(
        output: Point<B>,
        tangent: Option<TangentVector<B>>,
        vjp_state: VjpState,
        hvp_state: HvpState,
    ) -> Self {
        Self::Hvp {
            output,
            tangent,
            vjp_state,
            hvp_state,
        }
    }
}

/// A stateless trait defining the operations required for different modes of differentiation.
/// This trait isolates mathematical implementation from the tape management.
///
/// An implementation of this trait should be stateless; any state should be delegated to the
/// associated types, and managed in [`forward`](DiffOperation::forward).
pub trait DiffOperation<B: Backend>: Debug + Clone + Send + Sync + 'static {
    type VjpState: Send + Sync + 'static;
    type HvpState: Send + Sync + 'static;

    /// The forward operation `F` which returns state [`DiffOperationForward`] necessary for backward
    /// steps, conditionally on [`DiffModeState`]. The enum variants match accordingly.
    ///
    /// Inputs are [`Input`] instances, so a continuous operation may read discrete (integer)
    /// inputs. Continuous operations that only consume continuous inputs typically convert up front
    /// with [`Input::unwrap_float`].
    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState>;

    /// The backward VJP operation on a cotangent vector.
    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>>;

    /// The backward HVP operation on a cotangent vector.
    ///
    /// This method is not responsible for performing Leibniz rule `ω ↦  hvp(α) + jvp(β)`. It only
    /// serves as the `hvp` map.
    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>>;
}
