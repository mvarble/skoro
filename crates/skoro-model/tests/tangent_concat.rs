//! [`TangentConcat`]: stacking tangent blocks into one flat residual.
//!
//! This is the operation that lets a factor read *several* quantities without a product manifold.
//! Each quantity contributes its own tangent block, and this stacks them; the result is an
//! ordinary `EuclideanD(Σdᵢ)`, so everything downstream — whitening, `norm_squared`, the Jacobian
//! row layout — is unchanged.
//!
//! What is worth testing here beyond "the numbers match" is that blocks land in **their own
//! slots**: `vjp` slices the cotangent frame by width, and an off-by-one there is a well-shaped,
//! wrong answer whenever two blocks happen to have the same width.
//!
//! The operation's `VjpState`/`HvpState` gating is *not* tested here, because it is not observable
//! from a model — the collection pass discards a cotangent computed for a constant before anything
//! could assert on it. That lives in `skoro-operation/tests/ops.rs`, against the returned `Vec`s.

use skoro_backend::{NdArrayBackend, Tensor};
use skoro_model::{
    hash_map_builders::{cotangent_frames, points},
    map::register_map_slice,
    model::ModelClient,
    q::Q,
    test_utils::{
        into_vec, test_hvp_cross_numerically, test_jvp_numerically, test_vjp_numerically,
    },
};
use skoro_operation::ops::TangentConcat;
use skoro_riemann::euclidean::{EuclideanD, VectorD};
use skoro_riemann::real::Real;

type B = NdArrayBackend<f64, i32>;

const DELTA: f64 = 1e-6;
const EPSILON: f64 = 1e-5;

fn concat(blocks: &[&Q<B, EuclideanD>]) -> Q<B, EuclideanD> {
    let widths: Vec<usize> = blocks
        .iter()
        .map(|q| q.get_quantity_type::<skoro_core::kind::Continuous>().0)
        .collect();
    let total = widths.iter().sum();
    register_map_slice(EuclideanD(total), TangentConcat(widths), blocks).unwrap()
}

fn row(values: &[f64]) -> Tensor<B, 2> {
    Tensor::<B, 1>::from_floats(values, &Default::default()).reshape([1, values.len()])
}

/// Blocks are stacked in the order given, at their own widths.
///
/// The widths deliberately differ (2, 3, 1) so an implementation that assumed a uniform block
/// size, or that reversed the order, cannot pass.
#[test]
fn blocks_are_stacked_in_order() {
    let model = ModelClient::<B>::default();
    let a: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
    let b: Q<B, EuclideanD> = model.add_parameter(EuclideanD(3));
    let c: Q<B, EuclideanD> = model.add_parameter(EuclideanD(1));
    let stacked = concat(&[&a, &b, &c]);

    assert_eq!(
        stacked
            .get_quantity_type::<skoro_core::kind::Continuous>()
            .0,
        6
    );

    let values = points()
        .with_quantity(&a, row(&[1.0, 2.0]))
        .with_quantity(&b, row(&[3.0, 4.0, 5.0]))
        .with_quantity(&c, row(&[6.0]))
        .build();

    assert_eq!(
        into_vec(model.evaluate(&values, &stacked).unwrap()),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

/// A cotangent seeded on one output component reaches only the block that owns it.
///
/// This is the slot test. All three blocks below are width 2, so a `vjp` that sliced from the
/// wrong offset would still return correctly-shaped cotangents — and every downstream shape check
/// would pass while the Jacobian silently mixed up its columns.
#[test]
fn a_cotangent_reaches_only_the_block_that_owns_it() {
    let model = ModelClient::<B>::default();
    let a: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
    let b: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
    let c: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
    let stacked = concat(&[&a, &b, &c]);

    let values = points()
        .with_quantity(&a, row(&[0.0, 0.0]))
        .with_quantity(&b, row(&[0.0, 0.0]))
        .with_quantity(&c, row(&[0.0, 0.0]))
        .build();

    // One frame, seeding output component 3 — the second component of block `b`.
    let seed = Tensor::<B, 1>::from_floats(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0].as_slice(),
        &Default::default(),
    )
    .reshape([1, 6, 1]);
    let eval = stacked
        .evaluate_vjp(&values, &cotangent_frames().with_quantity(&stacked, seed))
        .unwrap();

    assert_eq!(into_vec(eval.cotangents[&a.id()].clone()), vec![0.0, 0.0]);
    assert_eq!(into_vec(eval.cotangents[&b.id()].clone()), vec![0.0, 1.0]);
    assert_eq!(into_vec(eval.cotangents[&c.id()].clone()), vec![0.0, 0.0]);
}

/// A constant block is a legal input, and the model reports a cotangent only for the free one.
///
/// Note what this does **not** test: whether the operation gates its `VjpState` correctly. The
/// collection pass surfaces only `Parameter`/`Sampler` slots, so a cotangent wrongly computed for
/// a constant is discarded before it could be observed here — this assertion passes either way.
/// The gating itself is pinned at the operation, in `skoro-operation/tests/ops.rs`, which is the
/// only place the difference is visible. What this does pin is that a mixed free/constant input
/// list evaluates and differentiates at all.
#[test]
fn a_constant_block_gets_no_cotangent() {
    let model = ModelClient::<B>::default();
    let free: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
    let fixed = model.add_constant(EuclideanD(2), &VectorD::new(&[9.0, 9.0]));
    let stacked = concat(&[&free, &fixed]);

    let values = points().with_quantity(&free, row(&[1.0, 2.0])).build();
    let seed = Tensor::<B, 1>::from_floats([1.0, 1.0, 1.0, 1.0].as_slice(), &Default::default())
        .reshape([1, 4, 1]);
    let eval = stacked
        .evaluate_vjp(&values, &cotangent_frames().with_quantity(&stacked, seed))
        .unwrap();

    assert!(eval.cotangents.contains_key(&free.id()));
    assert!(
        !eval.cotangents.contains_key(&fixed.id()),
        "a constant block must not collect a cotangent"
    );
}

/// The pushforward and the pullback both agree with central differences, per block.
///
/// Run for each input separately, because a block that is *not* the one under test arrives with
/// no tangent — which is the ordinary case, and the one where an implementation that skipped
/// absent blocks instead of zero-filling them would produce a short vector.
#[test]
fn concat_derivatives_match_central_differences() {
    let model = ModelClient::<B>::default();
    let a: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
    let b: Q<B, EuclideanD> = model.add_parameter(EuclideanD(3));
    let stacked = concat(&[&a, &b]);

    let values = points()
        .with_quantity(&a, row(&[1.0, -2.0]))
        .with_quantity(&b, row(&[0.5, 3.0, -1.5]))
        .build();

    test_jvp_numerically(&a, &stacked, values.clone(), DELTA, EPSILON);
    test_jvp_numerically(&b, &stacked, values.clone(), DELTA, EPSILON);
    test_vjp_numerically(&a, &stacked, values.clone(), DELTA, EPSILON);
    test_vjp_numerically(&b, &stacked, values, DELTA, EPSILON);
}

/// Second order, through something that actually has a mixed Hessian.
///
/// `TangentConcat` is linear, so its own second-order term is zero and testing it alone proves
/// little. What has to hold is that it routes the second-order pullback correctly for an operation
/// above it. `s · [a; b]` is bilinear, so `∂²/∂s∂a` is non-zero while `∂²/∂a∂b` is zero — and
/// confusing the slots gets one of those wrong in a way a single diagonal check would not see.
#[test]
fn second_order_routes_through_the_stack() {
    let model = ModelClient::<B>::default();
    let a: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
    let b: Q<B, EuclideanD> = model.add_parameter(EuclideanD(3));
    let s: Q<B, Real> = model.add_parameter(Real);
    let stacked = concat(&[&a, &b]);
    let scaled = (s.clone() * stacked).unwrap();

    let values = points()
        .with_quantity(&a, row(&[1.0, -2.0]))
        .with_quantity(&b, row(&[0.5, 3.0, -1.5]))
        .with_quantity(&s, row(&[1.5]))
        .build();

    // Mixed with the scalar: non-zero, and it has to reach the right block.
    test_hvp_cross_numerically(&s, &a, &scaled, values.clone(), DELTA, EPSILON);
    test_hvp_cross_numerically(&s, &b, &scaled, values.clone(), DELTA, EPSILON);
    test_hvp_cross_numerically(&a, &s, &scaled, values.clone(), DELTA, EPSILON);
    // Between two blocks, and within one: zero, but a correctly-shaped zero.
    test_hvp_cross_numerically(&a, &b, &scaled, values.clone(), DELTA, EPSILON);
    test_hvp_cross_numerically(&a, &a, &scaled, values, DELTA, EPSILON);
}

/// The same quantity may appear in more than one block.
///
/// A runtime-variadic input list makes this reachable for the first time: the `register_mapN`
/// family takes distinct arguments, but a slice can hold the same `Q` twice. The mathematics is
/// unambiguous — `d/da [a; a]` applied to `ω` is `ω[0..2] + ω[2..4]` — so the question is whether
/// the reverse pass *accumulates* the two contributions into one slot or lets the second
/// overwrite the first. Asserting `[1, 1]` rather than `[1, 0]` or `[0, 1]` is what separates
/// those.
#[test]
fn a_quantity_may_appear_in_two_blocks() {
    let model = ModelClient::<B>::default();
    let a: Q<B, EuclideanD> = model.add_parameter(EuclideanD(2));
    let stacked = concat(&[&a, &a]);

    let values = points().with_quantity(&a, row(&[1.0, 2.0])).build();
    assert_eq!(
        into_vec(model.evaluate(&values, &stacked).unwrap()),
        vec![1.0, 2.0, 1.0, 2.0]
    );

    // Seed the first component of the first block and the second of the second block.
    let seed = Tensor::<B, 1>::from_floats([1.0, 0.0, 0.0, 1.0].as_slice(), &Default::default())
        .reshape([1, 4, 1]);
    let eval = stacked
        .evaluate_vjp(&values, &cotangent_frames().with_quantity(&stacked, seed))
        .unwrap();
    assert_eq!(into_vec(eval.cotangents[&a.id()].clone()), vec![1.0, 1.0]);

    // The duplicated parent leaves the graph's derived indices consistent.
    model
        .server_call(|server| server.graph().validate_indices())
        .unwrap();
}
