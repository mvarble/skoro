//! Explicit bridges between the discrete and continuous domains.
//!
//! - **Embedding** (`disc → cont`): a [`DiffOperation`] that reads a discrete input
//!   and produces a continuous output. The gradient to the discrete parent is
//!   always `None`.
//! - **Discretization** (`cont → disc`): a [`DiscreteOperation`] that reads a
//!   continuous input and produces a discrete output.

use skoro_backend::{Backend, CotangentFrame, DiscretePoint, Point};
use skoro_core::discrete::{Boolean, Integer};
use skoro_core::error::SkoroResult;
use skoro_graph::operation::ForwardOnly;
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    discrete::DiscreteOperation,
};
use skoro_riemann::real::Real;

use crate::map::register_map1;
use crate::q::Q;

// ----- Embedding: discrete -> continuous (DiffOperation reading an Int input) -----

/// Cast a discrete integer value to a continuous real value.
///
/// The output is continuous; the discrete input is non-differentiable, so all
/// cotangents are `None`.
#[derive(Debug, Clone, Copy)]
pub struct IntToReal;

impl<B: Backend> DiffOperation<B> for IntToReal {
    type VjpState = ();
    type HvpState = ();

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, (), ()> {
        let [x] = inputs.try_into().unwrap();
        let out: Point<B> = x.unwrap_int().float();
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(out),
            // The single input is discrete (non-differentiable), so no tangents.
            DiffModeState::Jvp(_) => DiffOperationForward::jvp(out, None),
            DiffModeState::Vjp(_) => DiffOperationForward::vjp(out, ()),
            DiffModeState::Hvp { .. } => DiffOperationForward::hvp(out, None, (), ()),
        }
    }

    fn vjp(&self, _: &(), _: CotangentFrame<B>) -> Vec<Option<CotangentFrame<B>>> {
        vec![None]
    }

    fn hvp(&self, _: &(), _: &(), _: CotangentFrame<B>) -> Vec<Option<CotangentFrame<B>>> {
        vec![None]
    }
}

/// Embedding bridges from discrete quantities into continuous space.
pub trait DiscreteEmbed<B: Backend> {
    /// Cast this discrete quantity to a continuous real value.
    fn to_real(&self) -> SkoroResult<Q<B, Real>>;
}

impl<B: Backend> DiscreteEmbed<B> for Q<B, Integer> {
    fn to_real(&self) -> SkoroResult<Q<B, Real>> {
        register_map1(Real, IntToReal, self)
    }
}

impl<B: Backend> DiscreteEmbed<B> for Q<B, Boolean> {
    fn to_real(&self) -> SkoroResult<Q<B, Real>> {
        register_map1(Real, IntToReal, self)
    }
}

// ----- Discretization: continuous -> discrete (DiscreteOperation reading a Float) -----

/// Threshold a continuous value at zero, producing a boolean (`x >= 0`).
#[derive(Debug, Clone, Copy)]
pub struct Threshold;

impl<B: Backend> DiscreteOperation<B> for Threshold {
    fn forward(&self, inputs: Vec<Input<B>>) -> DiscretePoint<B> {
        let [x] = inputs.try_into().unwrap();
        x.unwrap_float().greater_equal_elem(0.0).int()
    }
}

/// Discretization bridges from continuous quantities into discrete space.
pub trait Discretize<B: Backend> {
    /// Produce a boolean for whether the (real) value is non-negative.
    fn is_non_negative(&self) -> SkoroResult<Q<B, Boolean>>;
}

impl<B: Backend> Discretize<B> for Q<B, Real> {
    fn is_non_negative(&self) -> SkoroResult<Q<B, Boolean>> {
        register_map1(Boolean, ForwardOnly(Threshold), self)
    }
}
