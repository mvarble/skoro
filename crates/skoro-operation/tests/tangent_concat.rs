//! `TangentConcat`: stacking per-quantity tangent blocks into one flat residual.
//!
//! Linear, so its numbers are easy. What is not easy is its bookkeeping: the operation holds the
//! per-block widths because the cotangent frame does not carry them, and a missing block occupies
//! real columns of the output that have to be filled rather than skipped.
// A one-input operation still reads better as `&[x.clone()]`, matching the two-input call
// sites beside it, than as `std::slice::from_ref(&x)`.
#![allow(clippy::cloned_ref_to_slice_refs)]

mod common;

use common::*;
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    ops::TangentConcat,
};

/// Two blocks of width 2 and 3.
fn op() -> TangentConcat {
    TangentConcat(vec![2, 3])
}

fn blocks() -> [skoro_backend::Tensor<B, 2>; 2] {
    [row(vec![1.0, 2.0]), row(vec![3.0, 4.0, 5.0])]
}

#[test]
fn concatenates_in_the_declared_order() {
    let [a, b] = blocks();
    assert_eq!(
        into_vec(forward_only(&op(), floats(&[a, b]))),
        vec![1.0, 2.0, 3.0, 4.0, 5.0]
    );
}

#[test]
fn derivatives_match_forward_differences() {
    let [a, b] = blocks();
    check_all_derivatives(&op(), &[a, b]);
}

#[test]
fn derivatives_are_computed_per_row() {
    check_all_derivatives(
        &op(),
        &[
            tensor2(vec![1.0, 2.0, 6.0, 7.0], 2, 2),
            tensor2(vec![3.0, 4.0, 5.0, 8.0, 9.0, 10.0], 2, 3),
        ],
    );
}

/// A block with no seeded tangent still occupies its columns, filled with zeros of its own width.
///
/// This is `cat`, never a sum: the blocks are *different components* of one vector, not
/// contributions to the same one. Skipping an absent block would silently shorten the output and
/// shift every block after it.
#[test]
fn an_unseeded_block_becomes_zeros_of_its_own_width() {
    let [a, b] = blocks();
    let seed = tensor3(vec![1.0, 1.0], [1, 2, 1]);

    let (_, tangent) = jvp(
        &op(),
        floats(&[a.clone(), b.clone()]),
        vec![Some(seed), None],
    );
    let tangent = tangent.expect("one seeded block is enough to produce a tangent");
    assert_eq!(tangent.dims(), [1, 5, 1]);
    assert_eq!(into_vec(tangent), vec![1.0, 1.0, 0.0, 0.0, 0.0]);

    // The mirror case: the second block seeded and the first absent.
    let seed = tensor3(vec![2.0, 2.0, 2.0], [1, 3, 1]);
    let (_, tangent) = jvp(&op(), floats(&[a, b]), vec![None, Some(seed)]);
    assert_eq!(into_vec(tangent.unwrap()), vec![0.0, 0.0, 2.0, 2.0, 2.0]);
}

/// With nothing seeded at all there is no tangent, rather than a zero one.
#[test]
fn nothing_seeded_yields_no_tangent() {
    let [a, b] = blocks();
    let (output, tangent) = jvp(&op(), floats(&[a, b]), vec![None, None]);
    assert_eq!(output.dims(), [1, 5]);
    assert!(tangent.is_none());

    // The same in a second-order pass.
    let (_, tangent, _, _) =
        hvp_state::<TangentConcat>(&op(), floats(&blocks()), vec![true, true], vec![None, None]);
    assert!(tangent.is_none());
}

/// The declared block count is the operation's contract with its caller.
#[test]
#[should_panic(expected = "TangentConcat: 1 inputs for 2 declared blocks")]
fn a_wrong_number_of_inputs_is_rejected() {
    let [a, _] = blocks();
    let _ = forward_only(&op(), floats(&[a]));
}

/// `TangentConcat` gates its `VjpState` on `diffs`, and its `HvpState` slots on that same state.
///
/// Tested here, at the operation, rather than through a model. The collection pass only surfaces
/// `Parameter`/`Sampler` cotangents, so a cotangent wrongly *computed* for a constant input is
/// discarded before any model-level assertion could see it — an ungated `VjpState` costs wasted
/// work and allocation, not a wrong answer, and is invisible downstream. The only place the
/// distinction is observable is the `Vec` these two methods return.
#[test]
fn gates_on_differentiability() {
    let op = op();
    let [a, b] = blocks();
    let omega = tensor3(vec![1.0, 2.0, 3.0, 4.0, 5.0], [1, 5, 1]);

    // Second input constant: it gets no cotangent, at either order.
    let DiffOperationForward::Vjp { output, vjp_state } = op.forward(
        vec![Input::Float(a.clone()), Input::Float(b.clone())],
        DiffModeState::Vjp(vec![true, false]),
    ) else {
        panic!("Vjp mode must return the Vjp variant")
    };
    assert_eq!(into_vec(output), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    let cotangents = op.vjp(&vjp_state, omega.clone());
    assert_eq!(cotangents.len(), 2);
    assert_eq!(
        into_vec(cotangents[0].clone().expect("differentiable input")),
        vec![1.0, 2.0]
    );
    assert!(
        cotangents[1].is_none(),
        "a constant input must not receive a cotangent"
    );

    // In `Hvp` mode the two signals differ: `diffs` says both inputs are free, while only the
    // first was seeded. The `VjpState` must follow `diffs` — gating it on the tangents instead
    // would silently drop the second input's order-1 cotangent.
    let DiffOperationForward::Hvp {
        vjp_state,
        hvp_state,
        ..
    } = op.forward(
        vec![Input::Float(a), Input::Float(b)],
        DiffModeState::Hvp {
            diffs: vec![true, true],
            tangents: vec![Some(row(vec![0.5, 0.5])), None],
        },
    )
    else {
        panic!("Hvp mode must return the Hvp variant")
    };

    let cotangents = op.vjp(&vjp_state, omega.clone());
    assert!(
        cotangents.iter().all(Option::is_some),
        "an unseeded but differentiable input still owes an order-1 cotangent"
    );

    // The map is linear, so every second-order slot is zero — but present wherever the
    // first-order one is, and shaped to its own block.
    let second = op.hvp(&vjp_state, &hvp_state, omega.clone());
    assert_eq!(
        into_vec(second[0].clone().expect("gated on VjpState[0]")),
        vec![0.0, 0.0]
    );
    assert_eq!(
        into_vec(second[1].clone().expect("gated on VjpState[1]")),
        vec![0.0, 0.0, 0.0]
    );

    // And with a constant input in the mix, the second-order slot follows the first-order one
    // into `None`. The invariant is that `∂f/∂xⱼ` being absent forces `∂²f/∂y∂xⱼ` absent too, so
    // this needs a case where the two `Hvp` signals disagree in the *other* direction from
    // above: input 1 is constant, hence neither differentiable nor seeded.
    let DiffOperationForward::Hvp {
        vjp_state,
        hvp_state,
        ..
    } = op.forward(
        floats(&blocks()),
        DiffModeState::Hvp {
            diffs: vec![true, false],
            tangents: vec![Some(row(vec![0.5, 0.5])), None],
        },
    )
    else {
        panic!("Hvp mode must return the Hvp variant")
    };

    assert!(op.vjp(&vjp_state, omega.clone())[1].is_none());
    let second = op.hvp(&vjp_state, &hvp_state, omega);
    assert!(second[0].is_some());
    assert!(
        second[1].is_none(),
        "a slot with no order-1 cotangent must have no order-2 one either"
    );
}

/// A single block is a degenerate but legal concatenation, and the widths are public data.
#[test]
fn a_single_block_is_legal_and_the_widths_are_readable() {
    let op = TangentConcat(vec![3]);
    assert_eq!(op.0, vec![3]);
    assert_eq!(op, TangentConcat(vec![3]));
    assert!(format!("{op:?}").contains("TangentConcat"));

    let x = row(vec![1.0, 2.0, 3.0]);
    assert_eq!(
        into_vec(forward_only(&op, floats(&[x.clone()]))),
        vec![1.0, 2.0, 3.0]
    );
    check_all_derivatives(&op, &[x]);
}
