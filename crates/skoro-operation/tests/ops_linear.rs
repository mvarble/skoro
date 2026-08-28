//! Operations that are linear (or affine) in every differentiable input.
//!
//! Their second derivative vanishes, which makes them the wrong place to look for a wrong
//! Hessian and the right place to check the *gating*: which slots get a cotangent at all. A
//! linear operation that returns a blanket `None` from `hvp` and one that returns a correctly
//! gated zero produce the same numbers and differ in whether a downstream accumulation ever
//! sees the slot.
// A one-input operation still reads better as `&[x.clone()]`, matching the two-input call
// sites beside it, than as `std::slice::from_ref(&x)`.
#![allow(clippy::cloned_ref_to_slice_refs)]

mod common;

use common::*;
use skoro_operation::{
    Input,
    diff::DiffOperation,
    ops::{
        Addition, LiteralAddition, LiteralScalarDivisionRight, LiteralScalarMultiplication,
        LiteralSubtractionLeft, LiteralSubtractionRight, Negation, Subtraction,
    },
};

/// Every combination of differentiability flags for a two-input operation.
const GATES: [[bool; 2]; 4] = [[true, true], [true, false], [false, true], [false, false]];

/// `v ↦ -v`: one input, linear, so its second-order term is a *present* zero rather than an absent
/// one. An operation that returned nothing there would look identical until something downstream
/// tried to accumulate into the slot.
mod negation {
    use super::*;

    #[test]
    fn negates_and_differentiates() {
        let x = row(vec![1.0, -2.0, 3.0]);
        assert_eq!(
            into_vec(forward_only(&Negation, floats(&[x.clone()]))),
            vec![-1.0, 2.0, -3.0]
        );
        check_all_derivatives(&Negation, &[x]);
    }

    /// A constant input receives nothing, at either order.
    #[test]
    fn a_constant_input_is_gated_out() {
        let x = row(vec![1.0, -2.0]);
        let (_, state) = vjp_state(&Negation, floats(&[x.clone()]), vec![false]);
        assert!(Negation.vjp(&state, identity(2, 1))[0].is_none());
        assert!(Negation.hvp(&state, &(), identity(2, 1))[0].is_none());

        // And with no tangent seeded, the pushforward is absent rather than zero.
        let (_, tangent) = jvp(&Negation, floats(&[x]), vec![None]);
        assert!(tangent.is_none());
    }

    /// Linear, so the second-order term is a *present* zero wherever the first-order one is.
    #[test]
    fn the_second_order_term_is_a_present_zero() {
        let x = row(vec![1.0, -2.0]);
        let (_, state) = vjp_state(&Negation, floats(&[x]), vec![true]);
        let second = Negation.hvp(&state, &(), identity(2, 1));
        assert_eq!(into_vec(second[0].clone().unwrap()), vec![0.0; 4]);
    }
}

/// Addition and subtraction over two inputs. Subtraction negates its right operand and nothing
/// else, every gate combination is honoured, and a pushforward seeded on only one input uses only
/// what it was given rather than assuming a zero tangent for the other.
mod additive {
    use super::*;

    #[test]
    fn addition_and_subtraction_differentiate() {
        let x = row(vec![1.0, 2.0]);
        let y = row(vec![10.0, -4.0]);

        assert_eq!(
            into_vec(forward_only(&Addition, floats(&[x.clone(), y.clone()]))),
            vec![11.0, -2.0]
        );
        assert_eq!(
            into_vec(forward_only(&Subtraction, floats(&[x.clone(), y.clone()]))),
            vec![-9.0, 6.0]
        );

        check_all_derivatives(&Addition, &[x.clone(), y.clone()]);
        check_all_derivatives(&Subtraction, &[x, y]);
    }

    /// Both operations gate each slot independently, at both orders.
    ///
    /// `[false, false]` is the case worth naming: an operation whose inputs are all constant
    /// must return `None` everywhere rather than a zero, so the tape stops rather than
    /// accumulating into quantities that have no gradient.
    #[test]
    fn every_gate_combination_is_honoured() {
        let x = row(vec![1.0, 2.0]);
        let y = row(vec![3.0, 4.0]);
        let omega = identity(2, 1);

        for gate in GATES {
            for (name, first, second) in [
                (
                    "addition",
                    Addition.vjp(
                        &vjp_state(&Addition, floats(&[x.clone(), y.clone()]), gate.to_vec()).1,
                        omega.clone(),
                    ),
                    Addition.hvp(
                        &vjp_state(&Addition, floats(&[x.clone(), y.clone()]), gate.to_vec()).1,
                        &(),
                        omega.clone(),
                    ),
                ),
                (
                    "subtraction",
                    Subtraction.vjp(
                        &vjp_state(&Subtraction, floats(&[x.clone(), y.clone()]), gate.to_vec()).1,
                        omega.clone(),
                    ),
                    Subtraction.hvp(
                        &vjp_state(&Subtraction, floats(&[x.clone(), y.clone()]), gate.to_vec()).1,
                        &(),
                        omega.clone(),
                    ),
                ),
            ] {
                for slot in 0..2 {
                    assert_eq!(
                        first[slot].is_some(),
                        gate[slot],
                        "{name}: order-1 slot {slot} under {gate:?}"
                    );
                    // The invariant: no first-order cotangent means no second-order one either.
                    assert_eq!(
                        second[slot].is_some(),
                        gate[slot],
                        "{name}: order-2 slot {slot} under {gate:?}"
                    );
                    if let Some(term) = &second[slot] {
                        assert_eq!(into_vec(term.clone()), vec![0.0; 4], "{name}: linear");
                    }
                }
            }
        }
    }

    /// Subtraction is not symmetric: the right operand's cotangent is negated.
    #[test]
    fn subtraction_negates_its_right_operand() {
        let x = row(vec![5.0]);
        let y = row(vec![2.0]);
        let (_, state) = vjp_state(&Subtraction, floats(&[x, y]), vec![true, true]);
        let cotangents = Subtraction.vjp(&state, tensor3(vec![3.0], [1, 1, 1]));
        assert_eq!(into_vec(cotangents[0].clone().unwrap()), vec![3.0]);
        assert_eq!(into_vec(cotangents[1].clone().unwrap()), vec![-3.0]);
    }

    /// With only one side seeded, the pushforward is that side's contribution alone.
    #[test]
    fn a_partially_seeded_pushforward_uses_only_what_was_seeded() {
        let x = row(vec![1.0]);
        let y = row(vec![2.0]);
        let seed = Some(tensor3(vec![1.0], [1, 1, 1]));

        let (_, tangent) = jvp(
            &Subtraction,
            floats(&[x.clone(), y.clone()]),
            vec![seed.clone(), None],
        );
        assert_eq!(into_vec(tangent.unwrap()), vec![1.0]);

        let (_, tangent) = jvp(
            &Subtraction,
            floats(&[x.clone(), y.clone()]),
            vec![None, seed],
        );
        assert_eq!(into_vec(tangent.unwrap()), vec![-1.0]);

        // Nothing seeded at all: no tangent, not a zero one.
        let (_, tangent) = jvp(&Subtraction, floats(&[x, y]), vec![None, None]);
        assert!(tangent.is_none());
    }
}

/// The literal-valued operations: one differentiable input and a baked-in constant.
mod literals {
    use super::*;

    #[test]
    fn literal_addition_and_subtraction() {
        let x = row(vec![1.0, 4.0]);

        let add = LiteralAddition::<B>::new(2.5);
        assert_eq!(
            into_vec(forward_only(&add, floats(&[x.clone()]))),
            vec![3.5, 6.5]
        );
        check_all_derivatives(&add, &[x.clone()]);

        // `x - c`
        let right = LiteralSubtractionRight::<B>::new(2.5);
        assert_eq!(
            into_vec(forward_only(&right, floats(&[x.clone()]))),
            vec![-1.5, 1.5]
        );
        check_all_derivatives(&right, &[x.clone()]);

        // `c - x`, which flips the sign of the derivative.
        let left = LiteralSubtractionLeft::<B>::new(2.5);
        assert_eq!(
            into_vec(forward_only(&left, floats(&[x.clone()]))),
            vec![1.5, -1.5]
        );
        check_all_derivatives(&left, &[x]);
    }

    #[test]
    fn literal_scaling_and_division() {
        let x = row(vec![3.0, -6.0]);

        let scale = LiteralScalarMultiplication::<B>::new(2.0);
        assert_eq!(
            into_vec(forward_only(&scale, floats(&[x.clone()]))),
            vec![6.0, -12.0]
        );
        check_all_derivatives(&scale, &[x.clone()]);

        let divide = LiteralScalarDivisionRight::<B>::new(4.0);
        assert_eq!(
            into_vec(forward_only(&divide, floats(&[x.clone()]))),
            vec![0.75, -1.5]
        );
        check_all_derivatives(&divide, &[x]);
    }

    /// All five gate their single input the same way, and all five are linear.
    #[test]
    fn every_literal_gates_its_input_and_vanishes_at_second_order() {
        let x = row(vec![3.0, -6.0]);
        let omega = identity(2, 1);

        macro_rules! check {
            ($op:expr) => {{
                let op = $op;
                // Differentiable: present at both orders, zero at the second.
                let (_, state) = vjp_state(&op, floats(&[x.clone()]), vec![true]);
                assert!(op.vjp(&state, omega.clone())[0].is_some());
                let second = op.hvp(&state, &(), omega.clone());
                assert_eq!(into_vec(second[0].clone().unwrap()), vec![0.0; 4]);

                // Constant: absent at both.
                let (_, state) = vjp_state(&op, floats(&[x.clone()]), vec![false]);
                assert!(op.vjp(&state, omega.clone())[0].is_none());
                assert!(op.hvp(&state, &(), omega.clone())[0].is_none());

                // Unseeded pushforward is absent rather than zero.
                let (_, tangent) = jvp(&op, floats(&[x.clone()]), vec![None]);
                assert!(tangent.is_none());
            }};
        }

        check!(LiteralAddition::<B>::new(2.5));
        check!(LiteralSubtractionRight::<B>::new(2.5));
        check!(LiteralSubtractionLeft::<B>::new(2.5));
        check!(LiteralScalarMultiplication::<B>::new(2.0));
        check!(LiteralScalarDivisionRight::<B>::new(4.0));
    }

    /// A literal operation is `Copy` and prints, so it can be stored in a graph node.
    #[test]
    fn literals_are_copy_and_debug() {
        let op = LiteralAddition::<B>::new(1.0);
        let copy = op;
        assert!(!format!("{copy:?}").is_empty());
        assert!(!format!("{:?}", Negation).is_empty());
        assert!(!format!("{:?}", Addition).is_empty());
    }
}

/// Batched inputs: every operation has to work row-wise, not just at batch 1.
mod batching {
    use super::*;

    #[test]
    fn derivatives_are_computed_per_row() {
        let x = tensor2(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let y = tensor2(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        check_all_derivatives(&Addition, &[x.clone(), y.clone()]);
        check_all_derivatives(&Subtraction, &[x.clone(), y]);
        check_all_derivatives(&LiteralScalarMultiplication::<B>::new(3.0), &[x]);
    }
}

/// `Input` accessors, exercised through the operations that rely on them.
mod inputs {
    use super::*;
    use skoro_backend::{Int, Tensor};

    #[test]
    fn a_discrete_input_to_a_continuous_operation_is_a_hard_error() {
        let discrete = Input::Int(Tensor::<B, 2, Int>::from_data(
            skoro_backend::TensorData::new(vec![1i32], [1, 1]),
            &Default::default(),
        ));
        assert!(discrete.is_discrete());
        assert_eq!(discrete.dims(), [1, 1]);
    }

    #[test]
    #[should_panic(expected = "expected a continuous (float) tensor")]
    fn unwrapping_a_discrete_input_as_float_panics() {
        let discrete = Input::<B>::Int(Tensor::<B, 2, Int>::from_data(
            skoro_backend::TensorData::new(vec![1i32], [1, 1]),
            &Default::default(),
        ));
        let _ = forward_only(&Negation, vec![discrete]);
    }
}
