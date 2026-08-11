//! Regression guards for what `DiffModeState::Hvp` gates on.
//!
//! The HVP backward pass runs `vjp` as well as `hvp`, so a `DiffOperation` needs two *different*
//! signals during `DiffModeState::Hvp`, and conflating them breaks in opposite directions:
//!
//! - Gating the `VjpState` on `tangents[j].is_some()` drops the order-1 cotangent of a
//!   differentiable input that simply was not seeded ([`unseeded_differentiable_input_*`]).
//! - Gating it on nothing at all (always memoizing) stops `vjp` short-circuiting subtrees rooted
//!   only in constants, so cotangents get computed and surfaced for them ([`constant_input_*`]).

use std::sync::{Arc, Mutex};

use skoro_backend::{CotangentFrame, NdArrayBackend, Point, Tensor};
use skoro_model::{
    hash_map_builders::{cotangent_frames, points, tangents},
    map::register_map2,
    model::ModelClient,
};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};
use skoro_riemann::real::Real;

type B = NdArrayBackend<f64, i32>;

/// A binary op that records the `diffs` it is handed during `DiffModeState::Hvp`, so a test can
/// assert on the signal the framework actually delivers rather than on a downstream side effect.
#[derive(Debug, Clone)]
struct DiffSpy(Arc<Mutex<Option<Vec<bool>>>>);

impl DiffOperation<B> for DiffSpy {
    type VjpState = ();
    type HvpState = ();

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::into_float).collect();
        let [x, y] = inputs.try_into().unwrap();
        let out = x * y;
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(out),
            DiffModeState::Jvp(_) => DiffOperationForward::jvp(out, None),
            DiffModeState::Vjp(_) => DiffOperationForward::vjp(out, ()),
            DiffModeState::Hvp { diffs, .. } => {
                *self.0.lock().unwrap() = Some(diffs);
                DiffOperationForward::hvp(out, None, (), ())
            }
        }
    }

    fn vjp(&self, _: &(), _: CotangentFrame<B>) -> Vec<Option<CotangentFrame<B>>> {
        vec![None, None]
    }

    fn hvp(&self, _: &(), _: &(), _: CotangentFrame<B>) -> Vec<Option<CotangentFrame<B>>> {
        vec![None, None]
    }
}

/// A `[1, 1, 1]` identity cotangent frame over a single scalar quantity.
fn unit_frame() -> Tensor<B, 3> {
    Tensor::<B, 1>::from_floats([1.0f64].as_slice(), &Default::default()).reshape([1, 1, 1])
}

fn scalar(x: f64) -> Tensor<B, 2> {
    Tensor::<B, 1>::from_floats([x].as_slice(), &Default::default()).reshape([1, 1])
}

/// `DiffModeState::Hvp` must carry the same per-input differentiability flags that
/// `DiffModeState::Vjp` does, so that an operation can keep `vjp` short-circuiting subtrees rooted
/// only in constants during an HVP pass.
///
/// The effect is asserted on the signal itself rather than on the returned cotangents: the
/// collection pass only surfaces `Parameter`/`Sampler` slots, so a cotangent wrongly computed for a
/// constant is discarded before it becomes visible, and the wasted work would go unnoticed.
#[test]
fn hvp_mode_reports_per_input_differentiability() {
    let seen = Arc::new(Mutex::new(None));
    let model = ModelClient::<B>::default();
    let param = model.add_parameter(Real);
    let konst = model.add_constant(Real, &3.0f64);
    let out = register_map2(Real, DiffSpy(seen.clone()), (&param, &konst)).unwrap();

    let point = points().with_quantity(&param, scalar(2.0)).build();
    let ts = tangents().with_quantity(&param, scalar(1.0));
    let cots1 = cotangent_frames().with_quantity(&out, unit_frame());
    let cots2 =
        cotangent_frames().with_quantity(&out, Tensor::zeros([1, 1, 1], &Default::default()));
    out.evaluate_hvp(&point, &ts, &cots1, &cots2).unwrap();

    assert_eq!(
        seen.lock().unwrap().clone(),
        Some(vec![true, false]),
        "the parameter is differentiable and the constant is not; `Hvp` must say so"
    );
}

/// The mirror image: an input that *is* differentiable but happens to carry no seeded tangent is
/// still entitled to its order-1 cotangent. `None` in `tangents` is a strictly weaker statement
/// than the input being constant.
#[test]
fn unseeded_differentiable_input_still_gets_cotangents() {
    let model = ModelClient::<B>::default();
    let seeded = model.add_parameter(Real);
    let unseeded = model.add_parameter(Real);
    let out = (seeded.clone() * unseeded.clone()).unwrap();

    let point = points()
        .with_quantity(&seeded, scalar(2.0))
        .with_quantity(&unseeded, scalar(5.0))
        .build();
    // Only `seeded` gets a tangent; `unseeded` arrives at the operation as `None`.
    let ts = tangents().with_quantity(&seeded, scalar(1.0));
    let cots1 = cotangent_frames().with_quantity(&out, unit_frame());
    let cots2 =
        cotangent_frames().with_quantity(&out, Tensor::zeros([1, 1, 1], &Default::default()));

    let eval = out.evaluate_hvp(&point, &ts, &cots1, &cots2).unwrap();

    for id in [seeded.id(), unseeded.id()] {
        assert!(
            eval.ord1_cotangents.contains_key(&id),
            "every differentiable input needs an order-1 cotangent, seeded or not"
        );
    }
    // `∂(x*y)/∂y = x`, whose sensitivity along `dx = 1` is `1` — so the unseeded operand carries
    // a genuinely non-zero second-order cotangent, which a tangent-gated `VjpState` would drop.
    let hvp = eval
        .ord2_cotangents
        .get(&unseeded.id())
        .expect("the unseeded operand must receive an order-2 cotangent");
    let hvp: Vec<f64> = hvp.clone().into_data().to_vec().unwrap();
    assert_eq!(hvp, vec![1.0]);
}
