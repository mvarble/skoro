//! Operations with a genuine second derivative: products and quotients.
//!
//! Unlike the linear family, these have a real `hvp`, and a wrong one is invisible to every
//! first-order check. The numerical comparison here is a forward difference *of the VJP*, which
//! is the definition the `hvp` contract states.

mod common;

use common::*;
use skoro_operation::{
    diff::DiffOperation,
    ops::{ScalarDivision, ScalarDivisionVjpState, ScalarMultiplication},
};

/// `(α, v) ↦ α v`: bilinear, so both inputs are differentiable and the second-order term is the
/// cross term between them. Every gate combination is run, because which inputs are free decides
/// which states the operation may return -- and the derivatives are checked against forward
/// differences rather than against the closed form the implementation already used.
mod scalar_multiplication {
    use super::*;

    /// Input convention is `(scalar, vector)`, and the scalar broadcasts across components.
    #[test]
    fn scales_a_vector_by_a_scalar() {
        let alpha = scalar(3.0);
        let x = row(vec![1.0, -2.0, 4.0]);
        assert_eq!(
            into_vec(forward_only(&ScalarMultiplication, floats(&[alpha, x]))),
            vec![3.0, -6.0, 12.0]
        );
    }

    /// All three derivatives, in both slots and both mixed blocks.
    ///
    /// The mixed block `∂²(αx)/∂α∂x = I` is the one that matters: it is the whole second-order
    /// content of a bilinear map, and a `hvp` that returned zeros would still pass every JVP and
    /// VJP check.
    #[test]
    fn derivatives_match_forward_differences() {
        check_all_derivatives(
            &ScalarMultiplication,
            &[scalar(3.0), row(vec![1.0, -2.0, 4.0])],
        );
    }

    #[test]
    fn derivatives_are_computed_per_row() {
        check_all_derivatives(
            &ScalarMultiplication,
            &[
                tensor2(vec![2.0, -1.0], 2, 1),
                tensor2(vec![1.0, 2.0, 3.0, 4.0], 2, 2),
            ],
        );
    }

    /// Each slot's `VjpState` is gated on its own differentiability flag.
    #[test]
    fn every_gate_combination_is_honoured() {
        let alpha = scalar(3.0);
        let x = row(vec![1.0, -2.0]);
        let omega = identity(2, 1);

        for gate in [[true, true], [true, false], [false, true], [false, false]] {
            let (_, state) = vjp_state(
                &ScalarMultiplication,
                floats(&[alpha.clone(), x.clone()]),
                gate.to_vec(),
            );
            let cotangents = ScalarMultiplication.vjp(&state, omega.clone());
            for slot in 0..2 {
                assert_eq!(
                    cotangents[slot].is_some(),
                    gate[slot],
                    "slot {slot} under {gate:?}"
                );
            }
        }
    }

    /// The second-order term is gated on the *tangents*, not on differentiability.
    ///
    /// That is the correct gate for `HvpState` specifically: `∇_v` of anything is zero when `v`
    /// is, so an unseeded input owes no sensitivity. It is the opposite of the rule for
    /// `VjpState`, which must follow `diffs` — conflating the two is what silently drops the
    /// order-1 cotangent of a differentiable-but-unseeded input.
    #[test]
    fn the_second_order_term_follows_the_seeded_tangents() {
        let alpha = scalar(3.0);
        let x = row(vec![1.0, -2.0]);
        let omega = identity(2, 1);

        // Differentiable everywhere, but only the scalar is seeded.
        let (_, _, vjp_st, hvp_st) = hvp_state::<ScalarMultiplication>(
            &ScalarMultiplication,
            floats(&[alpha.clone(), x.clone()]),
            vec![true, true],
            vec![Some(scalar(1.0)), None],
        );

        // The order-1 cotangent still reaches both inputs...
        let first = ScalarMultiplication.vjp(&vjp_st, omega.clone());
        assert!(first[0].is_some() && first[1].is_some());

        // ...while the sensitivity only reaches the one the seed can move.
        let second = ScalarMultiplication.hvp(&vjp_st, &hvp_st, omega);
        assert!(
            second[0].is_none(),
            "no tangent on the vector, so no ∂²/∂α∂x contribution to the scalar slot"
        );
        assert!(
            second[1].is_some(),
            "the seeded scalar drives the vector slot"
        );
    }
}

/// `(α, v) ↦ v/α`: bilinear in the same slots but genuinely nonlinear in the scalar, so the scalar
/// keeps its curvature even when the vector is constant. That asymmetry is what most of these
/// cases pin -- a constant vector must not flatten the scalar, while a constant scalar leaves a
/// linear map behind.
mod scalar_division {
    use super::*;

    /// Input convention is `(scalar, vector)`, computing `x / α`.
    #[test]
    fn divides_a_vector_by_a_scalar() {
        let alpha = scalar(2.0);
        let x = row(vec![1.0, -3.0, 8.0]);
        assert_eq!(
            into_vec(forward_only(&ScalarDivision, floats(&[alpha, x]))),
            vec![0.5, -1.5, 4.0]
        );
    }

    /// The full derivative surface, including the `2x/α³` term that only the diagonal
    /// `∂²/∂α²` block contains.
    #[test]
    fn derivatives_match_forward_differences() {
        check_all_derivatives(&ScalarDivision, &[scalar(2.0), row(vec![1.0, -3.0, 8.0])]);
    }

    #[test]
    fn derivatives_are_computed_per_row() {
        check_all_derivatives(
            &ScalarDivision,
            &[
                tensor2(vec![2.0, 4.0], 2, 1),
                tensor2(vec![1.0, 2.0, 3.0, 4.0], 2, 2),
            ],
        );
    }

    /// The four `VjpState` variants correspond exactly to the four gate combinations, and each
    /// carries only the operands its own pullback needs.
    #[test]
    fn the_vjp_state_variant_follows_the_gates() {
        let alpha = scalar(2.0);
        let x = row(vec![1.0, -3.0]);

        for (gate, expected) in [
            ([true, true], "both"),
            ([true, false], "left"),
            ([false, true], "right"),
            ([false, false], "neither"),
        ] {
            let (_, state) = vjp_state(
                &ScalarDivision,
                floats(&[alpha.clone(), x.clone()]),
                gate.to_vec(),
            );
            let actual = match state {
                ScalarDivisionVjpState::Both(_) => "both",
                ScalarDivisionVjpState::Left(_) => "left",
                ScalarDivisionVjpState::Right(_) => "right",
                ScalarDivisionVjpState::Neither => "neither",
            };
            assert_eq!(actual, expected, "under {gate:?}");
        }
    }

    /// With the vector held constant, the scalar still has a second derivative.
    ///
    /// `∂²(x/α)/∂α² = 2x/α³`, and it does not depend on `x` being free. This is the `Left` state
    /// -- reachable whenever a residual is divided by a free scale factor, which is what a
    /// learned observation weight is -- and the seeded `dα` is what drives it.
    #[test]
    fn the_scalar_keeps_its_curvature_when_the_vector_is_constant() {
        let alpha = scalar(2.0);
        let x = row(vec![1.0, -3.0]);
        let omega = identity(2, 1);

        let (_, _, vjp_st, hvp_st) = hvp_state::<ScalarDivision>(
            &ScalarDivision,
            floats(&[alpha.clone(), x.clone()]),
            vec![true, false],
            vec![Some(scalar(1.0)), None],
        );
        assert!(matches!(vjp_st, ScalarDivisionVjpState::Left(_)));

        let second = ScalarDivision.hvp(&vjp_st, &hvp_st, omega.clone());
        // The constant vector gets nothing, at either order.
        assert!(second[1].is_none());

        // ∂²/∂α² (x/α) · dα = 2x/α³ = 2·[1, -3]/8 = [0.25, -0.75];
        // contracted with the identity cotangent frame that is the row itself.
        let expected = vec![0.25, -0.75];
        approx::assert_abs_diff_eq!(
            into_vec(
                second[0]
                    .clone()
                    .expect("the scalar is differentiable and seeded")
            )
            .as_slice(),
            expected.as_slice(),
            epsilon = EPSILON
        );
    }

    /// The same, checked numerically rather than against a hand-derived constant.
    #[test]
    fn a_constant_vector_does_not_flatten_the_scalar_numerically() {
        let inputs = [scalar(2.0), row(vec![1.0, -3.0])];
        let omega = identity(2, 1);

        let (_, _, vjp_st, hvp_st) = hvp_state::<ScalarDivision>(
            &ScalarDivision,
            floats(&inputs),
            vec![true, false],
            vec![Some(scalar(1.0)), None],
        );
        let exact = ScalarDivision.hvp(&vjp_st, &hvp_st, omega.clone())[0]
            .clone()
            .unwrap();

        // Forward difference of the first-order pullback along the same direction.
        let read = |a: f64| {
            let stepped = [scalar(a), inputs[1].clone()];
            let (_, st) = vjp_state(&ScalarDivision, floats(&stepped), vec![true, false]);
            ScalarDivision.vjp(&st, omega.clone())[0].clone().unwrap()
        };
        let expected = into_vec((read(2.0 + DELTA) - read(2.0)).div_scalar(DELTA));

        approx::assert_abs_diff_eq!(
            into_vec(exact).as_slice(),
            expected.as_slice(),
            epsilon = EPSILON
        );
    }

    /// With the scalar constant, only the vector's slot survives -- and it is flat, since
    /// `x ↦ x/α` is linear for fixed `α`.
    #[test]
    fn a_constant_scalar_leaves_a_linear_map() {
        let omega = identity(2, 1);
        let (_, _, vjp_st, hvp_st) = hvp_state::<ScalarDivision>(
            &ScalarDivision,
            floats(&[scalar(2.0), row(vec![1.0, -3.0])]),
            vec![false, true],
            vec![None, Some(row(vec![1.0, 1.0]))],
        );
        assert!(matches!(vjp_st, ScalarDivisionVjpState::Right(_)));

        let second = ScalarDivision.hvp(&vjp_st, &hvp_st, omega);
        assert!(second[0].is_none(), "the constant scalar gets nothing");
        assert_eq!(into_vec(second[1].clone().unwrap()), vec![0.0; 4]);
    }

    /// The `Right` pullback itself: `ω ↦ ω/α`, checked numerically at two rows.
    ///
    /// `Right` is the only state carrying `alpha` alone, so it is computed by an arm that the
    /// all-differentiable check never reaches. It is also the arm a wrong broadcast axis hides
    /// in -- `omega / alpha.unsqueeze_dim(2)` divides a `[b, m, f]` frame by a `[b, 1, 1]`
    /// scalar, and with one row and one frame every axis looks alike. Two rows with *different*
    /// scale factors separate them.
    #[test]
    fn the_constant_scalar_pullback_divides_row_by_row() {
        test_vjp_numerically_gated(
            &ScalarDivision,
            &[
                tensor2(vec![2.0, 4.0], 2, 1),
                tensor2(vec![1.0, 2.0, 3.0, 4.0], 2, 2),
            ],
            1,
            vec![false, true],
            DELTA,
            EPSILON,
        );
    }

    /// Neither input free: nothing at either order.
    #[test]
    fn two_constants_produce_no_cotangents() {
        let omega = identity(2, 1);
        let (_, _, vjp_st, hvp_st) = hvp_state::<ScalarDivision>(
            &ScalarDivision,
            floats(&[scalar(2.0), row(vec![1.0, -3.0])]),
            vec![false, false],
            vec![None, None],
        );
        assert!(
            ScalarDivision
                .vjp(&vjp_st, omega.clone())
                .iter()
                .all(Option::is_none)
        );
        assert!(
            ScalarDivision
                .hvp(&vjp_st, &hvp_st, omega)
                .iter()
                .all(Option::is_none)
        );
    }

    /// Differentiable in both, but nothing seeded: the sensitivity is a present zero.
    #[test]
    fn an_unseeded_pass_yields_zero_sensitivity() {
        let omega = identity(2, 1);
        let (_, tangent, vjp_st, hvp_st) = hvp_state::<ScalarDivision>(
            &ScalarDivision,
            floats(&[scalar(2.0), row(vec![1.0, -3.0])]),
            vec![true, true],
            vec![None, None],
        );
        assert!(tangent.is_none());

        let second = ScalarDivision.hvp(&vjp_st, &hvp_st, omega);
        assert_eq!(into_vec(second[0].clone().unwrap()), vec![0.0, 0.0]);
        assert_eq!(into_vec(second[1].clone().unwrap()), vec![0.0; 4]);
    }

    /// Only the vector seeded: the scalar slot picks up the `-dx/α²` cross term.
    #[test]
    fn seeding_only_the_vector_still_moves_the_scalar_slot() {
        check_seeded_cross(&[None, Some(row(vec![1.0, 1.0]))]);
    }

    /// Only the scalar seeded.
    #[test]
    fn seeding_only_the_scalar_moves_both_slots() {
        check_seeded_cross(&[Some(scalar(1.0)), None]);
    }

    /// Compare `hvp` against a forward difference of the VJP along whatever was seeded.
    fn check_seeded_cross(seeds: &[Option<skoro_backend::Tensor<B, 2>>]) {
        let inputs = [scalar(2.0), row(vec![1.0, -3.0])];
        let omega = identity(2, 1);
        let diffs = vec![true, true];

        let (_, _, vjp_st, hvp_st) = hvp_state::<ScalarDivision>(
            &ScalarDivision,
            floats(&inputs),
            diffs.clone(),
            seeds.to_vec(),
        );
        let exact = ScalarDivision.hvp(&vjp_st, &hvp_st, omega.clone());

        // Step every input along its own seed at once, which is what `∇_v` means for a
        // multi-input operation.
        let stepped: Vec<_> = inputs
            .iter()
            .zip(seeds)
            .map(|(x, seed)| match seed {
                Some(v) => x.clone() + v.clone().mul_scalar(DELTA),
                None => x.clone(),
            })
            .collect();

        let (_, base) = vjp_state(&ScalarDivision, floats(&inputs), diffs.clone());
        let (_, moved) = vjp_state(&ScalarDivision, floats(&stepped), diffs);
        let vjp0 = ScalarDivision.vjp(&base, omega.clone());
        let vjp1 = ScalarDivision.vjp(&moved, omega);

        for slot in 0..2 {
            let expected = into_vec(
                (vjp1[slot].clone().unwrap() - vjp0[slot].clone().unwrap()).div_scalar(DELTA),
            );
            approx::assert_abs_diff_eq!(
                into_vec(exact[slot].clone().unwrap()).as_slice(),
                expected.as_slice(),
                epsilon = EPSILON
            );
        }
    }
}

/// `α ↦ c/α`: one differentiable input, but genuinely nonlinear, so it belongs here rather than
/// with the literals.
mod literal_scalar_division_left {
    use super::*;
    use skoro_operation::ops::LiteralScalarDivisionLeft;

    #[test]
    fn inverts_and_scales() {
        let op = LiteralScalarDivisionLeft::<B>::new(6.0);
        assert_eq!(
            into_vec(forward_only(&op, floats(&[row(vec![2.0, 3.0])]))),
            vec![3.0, 2.0]
        );
    }

    /// The full derivative surface at `α = 2`, `c = 1`: `f = 0.5`, `f' = -0.25`, `f'' = 0.25`.
    ///
    /// The `VjpState` here stores `1/α` rather than `α`, unlike its sibling `ScalarDivision`. A
    /// second-order term that reuses the sibling's formula unchanged divides by `1/α³` where it
    /// should multiply -- off by `α⁶`, which is a factor of 64 at this point.
    #[test]
    fn derivatives_match_forward_differences() {
        let op = LiteralScalarDivisionLeft::<B>::new(1.0);
        check_all_derivatives(&op, &[scalar(2.0)]);
    }

    /// Two more points, so the check cannot pass by an accident of `α = 2`.
    ///
    /// Kept away from zero on purpose. The third derivative `-6c/α⁴` blows up there, and a
    /// forward difference's `O(δ)` truncation error grows with it: below about `α = 1` the
    /// method's own error exceeds `EPSILON` and the test would fail for reasons that have
    /// nothing to do with the operation.
    #[test]
    fn derivatives_match_forward_differences_away_from_two() {
        let op = LiteralScalarDivisionLeft::<B>::new(-3.0);
        check_all_derivatives(&op, &[scalar(1.5)]);
        check_all_derivatives(&op, &[tensor2(vec![1.5, 4.0], 2, 1)]);
    }

    /// The curvature, stated as a number rather than as a difference quotient.
    #[test]
    fn the_second_derivative_is_two_c_over_alpha_cubed() {
        let op = LiteralScalarDivisionLeft::<B>::new(1.0);
        let omega = identity(1, 1);
        let (_, _, vjp_st, hvp_st) = hvp_state::<LiteralScalarDivisionLeft<B>>(
            &op,
            floats(&[scalar(2.0)]),
            vec![true],
            vec![Some(scalar(1.0))],
        );
        // f''(2) · dα = 2·1/2³ = 0.25
        approx::assert_abs_diff_eq!(
            into_vec(op.hvp(&vjp_st, &hvp_st, omega)[0].clone().unwrap()).as_slice(),
            [0.25].as_slice(),
            epsilon = EPSILON
        );
    }

    /// A constant input gets nothing; a differentiable but unseeded one gets a present zero.
    #[test]
    fn gating_follows_differentiability_and_seeding_separately() {
        let op = LiteralScalarDivisionLeft::<B>::new(1.0);
        let omega = identity(1, 1);

        let (_, state) = vjp_state(&op, floats(&[scalar(2.0)]), vec![false]);
        assert!(op.vjp(&state, omega.clone())[0].is_none());
        assert!(op.hvp(&state, &None, omega.clone())[0].is_none());

        let (_, _, vjp_st, hvp_st) = hvp_state::<LiteralScalarDivisionLeft<B>>(
            &op,
            floats(&[scalar(2.0)]),
            vec![true],
            vec![None],
        );
        assert!(op.vjp(&vjp_st, omega.clone())[0].is_some());
        assert_eq!(
            into_vec(op.hvp(&vjp_st, &hvp_st, omega)[0].clone().unwrap()),
            vec![0.0]
        );
    }
}
