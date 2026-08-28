//! The `Manifold` / `DynManifold` surface for the flat manifolds this crate provides.
//!
//! `DynManifold` is the dyn-compatible half, reached by a blanket implementation, and it is what
//! the graph stores. Its `retract` / `inverse_retract` run the underlying operations in
//! forward-only mode, which is the path a `PointMap` takes on every step of an optimizer.
//!
//! Everything here is flat, so retraction *is* addition. What the tests pin down is not the
//! arithmetic but the interface contract: dimension versus storage size, the round trip, and that
//! the erased form agrees with the typed one.

mod common;

use common::*;
use skoro_core::pack::NumComponents;
use skoro_operation::diff::DiffOperation;
use skoro_riemann::{
    DynManifold, Manifold,
    euclidean::{Euclidean, Euclidean3, EuclideanD, EuclideanInverseRetraction},
    matrix::{EuclideanMatrix, EuclideanMatrix23, EuclideanMatrixD},
    real::Real,
};

// =============================================================================================
mod dimensions {
    use super::*;

    /// Dimension is a property of the manifold, and for these it coincides with storage size.
    ///
    /// It does not in general: `SO3` occupies 3 tangent slots and 9 storage components. Every
    /// layout downstream is sized by `dimension()`, so the two must not be conflated even where
    /// they happen to agree.
    #[test]
    fn each_manifold_reports_its_dimension() {
        assert_eq!(<Real as Manifold<B>>::dimension(&Real), 1);
        assert_eq!(<EuclideanD as Manifold<B>>::dimension(&EuclideanD(5)), 5);
        assert_eq!(<Euclidean3 as Manifold<B>>::dimension(&Euclidean), 3);
        assert_eq!(
            <EuclideanMatrixD as Manifold<B>>::dimension(&EuclideanMatrixD(2, 3)),
            6
        );
        assert_eq!(
            <EuclideanMatrix23 as Manifold<B>>::dimension(&EuclideanMatrix),
            6
        );
    }

    /// Storage size is the other half of that pair, and `NumComponents` is where it is declared.
    ///
    /// For these manifolds the two coincide, so the claim worth asserting is the coincidence
    /// itself rather than either number on its own: a packer sized by `num_components` and a
    /// tangent layout sized by `dimension` have to agree, and here they do.
    #[test]
    fn storage_size_agrees_with_dimension() {
        for (components, dimension) in [
            (
                Real.num_components(),
                <Real as Manifold<B>>::dimension(&Real),
            ),
            (
                EuclideanD(5).num_components(),
                <EuclideanD as Manifold<B>>::dimension(&EuclideanD(5)),
            ),
            (
                NumComponents::num_components(&Euclidean::<3>),
                <Euclidean3 as Manifold<B>>::dimension(&Euclidean),
            ),
            (
                EuclideanMatrixD(2, 3).num_components(),
                <EuclideanMatrixD as Manifold<B>>::dimension(&EuclideanMatrixD(2, 3)),
            ),
            (
                NumComponents::num_components(&EuclideanMatrix::<2, 3>),
                <EuclideanMatrix23 as Manifold<B>>::dimension(&EuclideanMatrix),
            ),
        ] {
            assert_eq!(components, dimension);
        }
    }

    /// The erased form reports the same dimension as the typed one.
    #[test]
    fn erasure_preserves_the_dimension() {
        let erased: &dyn DynManifold<B> = &EuclideanD(4);
        assert_eq!(erased.dimension(), 4);

        let erased: &dyn DynManifold<B> = &EuclideanMatrix::<3, 2>;
        assert_eq!(erased.dimension(), 6);
    }
}

// =============================================================================================
mod round_trips {
    use super::*;

    #[test]
    fn real_retracts_and_inverts() {
        check_retraction_round_trip(&Real, row(vec![2.5]), row(vec![-0.75]));
    }

    #[test]
    fn runtime_dimensioned_euclidean_retracts_and_inverts() {
        check_retraction_round_trip(
            &EuclideanD(3),
            row(vec![1.0, -2.0, 3.0]),
            row(vec![0.5, 0.25, -1.5]),
        );
    }

    #[test]
    fn statically_dimensioned_euclidean_retracts_and_inverts() {
        check_retraction_round_trip(
            &Euclidean::<4>,
            row(vec![1.0, 2.0, 3.0, 4.0]),
            row(vec![0.1, 0.2, 0.3, 0.4]),
        );
    }

    #[test]
    fn matrix_manifolds_retract_and_invert() {
        check_retraction_round_trip(
            &EuclideanMatrixD(2, 3),
            row(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            row(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        );
        check_retraction_round_trip(
            &EuclideanMatrix::<2, 2>,
            row(vec![1.0, 2.0, 3.0, 4.0]),
            row(vec![-1.0, -2.0, -3.0, -4.0]),
        );
    }

    /// Batched: every row is stepped by its own tangent, independently.
    ///
    /// This is what makes a batch of `N` independent problems work -- a retraction that
    /// broadcast one row across all of them would move every problem by the first one's step.
    #[test]
    fn retraction_is_row_wise() {
        let points = tensor2(vec![1.0, 2.0, 10.0, 20.0], 2, 2);
        let tangents = tensor2(vec![0.5, 0.5, -1.0, -2.0], 2, 2);
        let moved = DynManifold::<B>::retract(&EuclideanD(2), points.clone(), tangents.clone());
        assert_eq!(into_vec(moved.clone()), vec![1.5, 2.5, 9.0, 18.0]);

        let recovered = DynManifold::<B>::inverse_retract(&EuclideanD(2), points, moved);
        assert_eq!(into_vec(recovered), into_vec(tangents));
    }

    /// A zero tangent is the identity, and two points inverse-retract to their displacement.
    #[test]
    fn a_zero_step_stays_put() {
        let point = row(vec![3.0, -4.0]);
        let zero = row(vec![0.0, 0.0]);
        assert_eq!(
            into_vec(DynManifold::<B>::retract(
                &EuclideanD(2),
                point.clone(),
                zero
            )),
            into_vec(point.clone())
        );
        assert_eq!(
            into_vec(DynManifold::<B>::inverse_retract(
                &EuclideanD(2),
                point.clone(),
                point
            )),
            vec![0.0, 0.0]
        );
    }
}

// =============================================================================================
/// `EuclideanInverseRetraction` is `(y, x) ↦ x - y`: subtraction with the operands reversed.
///
/// The reversal is the point. `inverse_retract(point, other)` yields `other ⊖ point`, so the
/// *anchor* comes first and the result is the displacement away from it. Getting the order wrong
/// negates every residual in the model while leaving all the shapes correct.
mod inverse_retraction_operation {
    use super::*;

    #[test]
    fn it_subtracts_the_first_operand_from_the_second() {
        let anchor = row(vec![1.0, 2.0]);
        let other = row(vec![4.0, 10.0]);
        assert_eq!(
            into_vec(forward_only(
                &EuclideanInverseRetraction,
                floats(&[anchor, other])
            )),
            vec![3.0, 8.0],
            "other minus anchor, not the reverse"
        );
    }

    /// It agrees with the manifold method that wraps it.
    #[test]
    fn it_agrees_with_the_manifold_method() {
        let anchor = row(vec![1.0, 2.0, 3.0]);
        let other = row(vec![0.0, 5.0, -3.0]);
        assert_eq!(
            into_vec(forward_only(
                &EuclideanInverseRetraction,
                floats(&[anchor.clone(), other.clone()])
            )),
            into_vec(DynManifold::<B>::inverse_retract(
                &EuclideanD(3),
                anchor,
                other
            ))
        );
    }

    #[test]
    fn derivatives_match_forward_differences() {
        check_all_derivatives(
            &EuclideanInverseRetraction,
            &[row(vec![1.0, 2.0]), row(vec![4.0, 10.0])],
        );
    }

    #[test]
    fn derivatives_are_computed_per_row() {
        check_all_derivatives(
            &EuclideanInverseRetraction,
            &[
                tensor2(vec![1.0, 2.0, 3.0, 4.0], 2, 2),
                tensor2(vec![5.0, 6.0, 7.0, 8.0], 2, 2),
            ],
        );
    }

    /// The anchor's cotangent is negated; the moving point's is not.
    #[test]
    fn the_anchor_receives_the_negated_cotangent() {
        let (_, state) = vjp_state(
            &EuclideanInverseRetraction,
            floats(&[row(vec![1.0]), row(vec![2.0])]),
            vec![true, true],
        );
        let cotangents = DiffOperation::<B>::vjp(
            &EuclideanInverseRetraction,
            &state,
            tensor3(vec![3.0], [1, 1, 1]),
        );
        assert_eq!(into_vec(cotangents[0].clone().unwrap()), vec![-3.0]);
        assert_eq!(into_vec(cotangents[1].clone().unwrap()), vec![3.0]);
    }

    /// Each slot is gated independently, at both orders, and the map is linear so the
    /// second-order term is a present zero.
    ///
    /// The anchor of a linearized factor is a frozen constant, so `[false, true]` is the case
    /// that actually runs in a model -- the residual is differentiable in the live quantity and
    /// not in the snapshot it is measured from.
    #[test]
    fn every_gate_combination_is_honoured() {
        let inputs = [row(vec![1.0, 2.0]), row(vec![4.0, 10.0])];
        let omega = identity(2, 1);

        for gate in [[true, true], [true, false], [false, true], [false, false]] {
            let (_, state) = vjp_state(&EuclideanInverseRetraction, floats(&inputs), gate.to_vec());
            let first = DiffOperation::<B>::vjp(&EuclideanInverseRetraction, &state, omega.clone());
            let second =
                DiffOperation::<B>::hvp(&EuclideanInverseRetraction, &state, &(), omega.clone());
            for slot in 0..2 {
                assert_eq!(first[slot].is_some(), gate[slot], "order 1, slot {slot}");
                assert_eq!(second[slot].is_some(), gate[slot], "order 2, slot {slot}");
                if let Some(term) = &second[slot] {
                    assert_eq!(into_vec(term.clone()), vec![0.0; 4], "linear");
                }
            }
        }
    }

    /// With nothing seeded there is no pushforward, and with one side seeded only that side
    /// contributes -- negated for the anchor.
    #[test]
    fn the_pushforward_follows_what_was_seeded() {
        let inputs = [row(vec![1.0]), row(vec![4.0])];
        let seed = Some(tensor3(vec![1.0], [1, 1, 1]));

        let (_, tangent) = jvp(
            &EuclideanInverseRetraction,
            floats(&inputs),
            vec![seed.clone(), None],
        );
        assert_eq!(
            into_vec(tangent.unwrap()),
            vec![-1.0],
            "the anchor moves it backwards"
        );

        let (_, tangent) = jvp(
            &EuclideanInverseRetraction,
            floats(&inputs),
            vec![None, seed],
        );
        assert_eq!(into_vec(tangent.unwrap()), vec![1.0]);

        let (_, tangent) = jvp(
            &EuclideanInverseRetraction,
            floats(&inputs),
            vec![None, None],
        );
        assert!(tangent.is_none());
    }

    #[test]
    fn it_is_copy_and_debug() {
        let op = EuclideanInverseRetraction;
        let copy = op;
        assert!(format!("{copy:?}").contains("EuclideanInverseRetraction"));
    }
}

// =============================================================================================
/// Retraction on a flat manifold *is* `Addition`, and it differentiates like it.
mod retraction_operation {
    use super::*;

    #[test]
    fn derivatives_match_forward_differences() {
        let op = <EuclideanD as Manifold<B>>::retraction(&EuclideanD(2));
        check_all_derivatives(&op, &[row(vec![1.0, 2.0]), row(vec![0.5, -0.5])]);
    }

    #[test]
    fn every_manifold_supplies_one() {
        assert_eq!(
            into_vec(forward_only(
                &<Real as Manifold<B>>::retraction(&Real),
                floats(&[row(vec![1.0]), row(vec![2.0])])
            )),
            vec![3.0]
        );
        assert_eq!(
            into_vec(forward_only(
                &<EuclideanMatrixD as Manifold<B>>::retraction(&EuclideanMatrixD(1, 2)),
                floats(&[row(vec![1.0, 2.0]), row(vec![10.0, 20.0])])
            )),
            vec![11.0, 22.0]
        );
        assert_eq!(
            into_vec(forward_only(
                &<EuclideanMatrix23 as Manifold<B>>::retraction(&EuclideanMatrix),
                floats(&[row(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), row(vec![1.0; 6])])
            )),
            vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
        assert_eq!(
            into_vec(forward_only(
                &<Euclidean3 as Manifold<B>>::retraction(&Euclidean),
                floats(&[row(vec![1.0, 2.0, 3.0]), row(vec![1.0, 1.0, 1.0])])
            )),
            vec![2.0, 3.0, 4.0]
        );
    }
}
