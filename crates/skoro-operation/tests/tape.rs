//! The tape: a flat list of backward steps, consumed in reverse.
//!
//! The tape owns the accumulation recurrence, which is the part that is easy to get subtly wrong.
//! Fan-in (two children contributing to one parent) has to *sum* rather than overwrite, and
//! fan-out (one step feeding several parents) has to reach every one of them. Both are tested
//! here against hand-built steps rather than through an operation, so a failure names the tape.

mod common;

use common::*;
use skoro_backend::{CotangentFrame, Tensor};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward, HvpStep, VjpStep},
    ops::{Addition, ScalarMultiplication},
    step::{BoxDynHvpStep, BoxDynVjpStep, DynHvpStep},
    tape::Tape,
};

/// A `VjpStep` for `Addition`, which fans one cotangent out to both parents.
fn addition_step(inputs: [usize; 2], output: usize) -> BoxDynVjpStep<B> {
    Box::new(VjpStep {
        operation: Addition,
        state: [true, true],
        input_indices: inputs.to_vec(),
        output_index: output,
    })
}

fn hvp_addition_step(inputs: [usize; 2], output: usize) -> BoxDynHvpStep<B> {
    Box::new(HvpStep {
        operation: Addition,
        vjp_state: [true, true],
        hvp_state: (),
        input_indices: inputs.to_vec(),
        output_index: output,
    })
}

fn seed(n: usize, at: usize, value: f64) -> Vec<Option<CotangentFrame<B>>> {
    let mut out = vec![None; n];
    out[at] = Some(tensor3(vec![value], [1, 1, 1]));
    out
}

/// The tape as a container: empty until pushed to, growing as steps arrive, and each step
/// reporting the slots it reads and writes. Nothing here runs a backward pass; a failure names the
/// tape rather than an operation.
mod bookkeeping {
    use super::*;

    #[test]
    fn an_empty_tape_reports_itself_empty() {
        let tape: Tape<BoxDynVjpStep<B>> = Tape::new();
        assert!(tape.is_empty());
        assert_eq!(tape.len(), 0);

        let tape: Tape<BoxDynVjpStep<B>> = Tape::default();
        assert!(tape.is_empty());

        // Capacity is a hint, not content.
        let tape: Tape<BoxDynVjpStep<B>> = Tape::with_capacity(8);
        assert!(tape.is_empty());
        assert_eq!(tape.len(), 0);
    }

    #[test]
    fn pushing_grows_the_tape() {
        let mut tape: Tape<BoxDynVjpStep<B>> = Tape::with_capacity(2);
        tape.push(addition_step([0, 1], 2));
        assert_eq!(tape.len(), 1);
        assert!(!tape.is_empty());
        tape.push(addition_step([2, 0], 3));
        assert_eq!(tape.len(), 2);
    }

    /// A step reports the positions it was built with, which is how the tape finds its slots.
    #[test]
    fn a_step_reports_its_indices() {
        let step = addition_step([4, 5], 6);
        assert_eq!(step.input_indices(), &[4, 5]);
        assert_eq!(step.output_index(), 6);

        let step = hvp_addition_step([4, 5], 6);
        assert_eq!(DynHvpStep::input_indices(step.as_ref()), &[4, 5]);
        assert_eq!(DynHvpStep::output_index(step.as_ref()), 6);
    }
}

/// The accumulation recurrence, against hand-built steps rather than through an operation.
/// Fan-out has to reach every parent and fan-in has to *sum* into a shared one rather than
/// overwrite it -- the second is the one that silently halves a gradient. `backward` leaves the
/// tape intact and `consume_backward` must agree with it.
mod first_order {
    use super::*;

    /// One step, both parents reached.
    #[test]
    fn a_step_fans_its_cotangent_out_to_every_parent() {
        let mut tape: Tape<BoxDynVjpStep<B>> = Tape::new();
        tape.push(addition_step([0, 1], 2));

        let mut cotangents = seed(3, 2, 1.0);
        tape.backward(&mut cotangents);

        assert_eq!(into_vec(cotangents[0].clone().unwrap()), vec![1.0]);
        assert_eq!(into_vec(cotangents[1].clone().unwrap()), vec![1.0]);
        // The output's cotangent was consumed on the way through.
        assert!(cotangents[2].is_none());
    }

    /// The step itself is total: handed no cotangent at its output, it writes nothing.
    ///
    /// `Tape::backward` panics before ever reaching this branch, and deliberately -- an unseeded
    /// output means the pass was set up wrong, and silently reporting zeros there would look like
    /// a real gradient. Called on its own, though, the step must be a no-op rather than an index
    /// panic, which is what lets a caller drive a step outside the tape's ordering.
    #[test]
    fn a_step_without_an_output_cotangent_writes_nothing() {
        let step = addition_step([0, 1], 2);
        let mut cotangents: Vec<Option<CotangentFrame<B>>> = vec![None; 3];
        skoro_operation::step::DynVjpStep::step(step.as_ref(), &mut cotangents);
        assert!(cotangents.iter().all(Option::is_none));
    }

    /// Two steps feeding one slot must *accumulate*, not overwrite.
    ///
    /// This is the case that separates a correct tape from one that happens to work on trees:
    /// with `y = x + x` reached through two different steps, an overwriting tape reports `1`
    /// where the answer is `2`.
    #[test]
    fn two_children_accumulate_into_a_shared_parent() {
        // slots: 0 = x, 1 = x + x (left), 2 = x + x (right), 3 = sum of both
        let mut tape: Tape<BoxDynVjpStep<B>> = Tape::new();
        tape.push(addition_step([0, 0], 1));
        tape.push(addition_step([0, 0], 2));
        tape.push(addition_step([1, 2], 3));

        let mut cotangents = seed(4, 3, 1.0);
        tape.backward(&mut cotangents);

        // Each of the two middle nodes contributes 2, so x sees 4.
        assert_eq!(into_vec(cotangents[0].clone().unwrap()), vec![4.0]);
    }

    /// `backward` leaves the tape intact, so it can be replayed with a different seed.
    #[test]
    fn backward_does_not_consume_the_tape() {
        let mut tape: Tape<BoxDynVjpStep<B>> = Tape::new();
        tape.push(addition_step([0, 1], 2));

        let mut first = seed(3, 2, 1.0);
        tape.backward(&mut first);
        assert_eq!(tape.len(), 1, "the tape survives a traversal");

        let mut second = seed(3, 2, 5.0);
        tape.backward(&mut second);
        assert_eq!(into_vec(second[0].clone().unwrap()), vec![5.0]);
    }

    /// `consume_backward` gives the same answer and takes the tape with it.
    #[test]
    fn consume_backward_matches_backward() {
        let mut replayable: Tape<BoxDynVjpStep<B>> = Tape::new();
        replayable.push(addition_step([0, 1], 2));
        let mut expected = seed(3, 2, 3.0);
        replayable.backward(&mut expected);

        let mut consumed: Tape<BoxDynVjpStep<B>> = Tape::new();
        consumed.push(addition_step([0, 1], 2));
        let mut actual = seed(3, 2, 3.0);
        consumed.consume_backward(&mut actual);

        for slot in 0..3 {
            assert_eq!(
                actual[slot].clone().map(into_vec),
                expected[slot].clone().map(into_vec),
                "slot {slot}"
            );
        }
    }

    /// A step whose output slot never received a cotangent means the pass was seeded wrong.
    #[test]
    #[should_panic(expected = "Cotangent vector not present at index 2")]
    fn an_unseeded_output_panics() {
        let mut tape: Tape<BoxDynVjpStep<B>> = Tape::new();
        tape.push(addition_step([0, 1], 2));
        let mut cotangents: Vec<Option<CotangentFrame<B>>> = vec![None; 3];
        tape.backward(&mut cotangents);
    }

    #[test]
    #[should_panic(expected = "Cotangent vector not present at index 2")]
    fn an_unseeded_output_panics_when_consuming() {
        let mut tape: Tape<BoxDynVjpStep<B>> = Tape::new();
        tape.push(addition_step([0, 1], 2));
        let mut cotangents: Vec<Option<CotangentFrame<B>>> = vec![None; 3];
        tape.consume_backward(&mut cotangents);
    }

    /// A step gated entirely on constants writes nothing.
    #[test]
    fn a_fully_constant_step_contributes_nothing() {
        let mut tape: Tape<BoxDynVjpStep<B>> = Tape::new();
        tape.push(Box::new(VjpStep {
            operation: Addition,
            state: [false, false],
            input_indices: vec![0, 1],
            output_index: 2,
        }));

        let mut cotangents = seed(3, 2, 1.0);
        tape.backward(&mut cotangents);
        assert!(cotangents[0].is_none());
        assert!(cotangents[1].is_none());
    }
}

/// Both orders travel down the tape together. A step with neither order seeded is skipped, one with
/// only one of them panics rather than guessing, and a curved step contributes to the second order
/// where a linear one does not. Two children accumulate at both orders, not just the first.
mod second_order {
    use super::*;

    #[test]
    fn both_orders_propagate_together() {
        let mut tape: Tape<BoxDynHvpStep<B>> = Tape::new();
        tape.push(hvp_addition_step([0, 1], 2));

        let mut ord1 = seed(3, 2, 1.0);
        let mut ord2 = seed(3, 2, 0.0);
        tape.backward(&mut ord1, &mut ord2);

        assert_eq!(into_vec(ord1[0].clone().unwrap()), vec![1.0]);
        // Addition is linear, so the second order is a present zero.
        assert_eq!(into_vec(ord2[0].clone().unwrap()), vec![0.0]);
    }

    /// The order-2 accumulation is `hvp(α) + vjp(β)`, and the tape owns that sum.
    ///
    /// Seeding a non-zero `β` is what shows the `vjp(β)` half is really there: for a linear
    /// operation `hvp(α)` is zero, so whatever arrives is `β` alone.
    #[test]
    fn the_second_order_seed_is_pulled_back_as_well() {
        let mut tape: Tape<BoxDynHvpStep<B>> = Tape::new();
        tape.push(hvp_addition_step([0, 1], 2));

        let mut ord1 = seed(3, 2, 1.0);
        let mut ord2 = seed(3, 2, 7.0);
        tape.backward(&mut ord1, &mut ord2);

        assert_eq!(into_vec(ord2[0].clone().unwrap()), vec![7.0]);
        assert_eq!(into_vec(ord2[1].clone().unwrap()), vec![7.0]);
    }

    /// A genuinely curved step contributes to the second order through `hvp`.
    #[test]
    fn a_curved_step_contributes_to_the_second_order() {
        // `y = α · x`, seeded with `dα = 1`, so `∂²y/∂α∂x · dα = 1`.
        let alpha = scalar(3.0);
        let x = scalar(5.0);
        let DiffOperationForward::Hvp {
            vjp_state,
            hvp_state,
            ..
        } = ScalarMultiplication.forward(
            vec![Input::Float(alpha), Input::Float(x)],
            DiffModeState::Hvp {
                diffs: vec![true, true],
                tangents: vec![Some(scalar(1.0)), None],
            },
        )
        else {
            panic!("Hvp mode must return the Hvp variant")
        };

        let mut tape: Tape<BoxDynHvpStep<B>> = Tape::new();
        tape.push(Box::new(HvpStep {
            operation: ScalarMultiplication,
            vjp_state,
            hvp_state,
            input_indices: vec![0, 1],
            output_index: 2,
        }));

        let mut ord1 = seed(3, 2, 1.0);
        let mut ord2 = seed(3, 2, 0.0);
        tape.backward(&mut ord1, &mut ord2);

        assert_eq!(into_vec(ord1[0].clone().unwrap()), vec![5.0]); // ∂y/∂α = x
        assert_eq!(into_vec(ord1[1].clone().unwrap()), vec![3.0]); // ∂y/∂x = α
        assert_eq!(into_vec(ord2[1].clone().unwrap()), vec![1.0]); // ∂²y/∂α∂x · dα
    }

    #[test]
    fn consume_backward_matches_backward() {
        let mut replayable: Tape<BoxDynHvpStep<B>> = Tape::new();
        replayable.push(hvp_addition_step([0, 1], 2));
        let (mut e1, mut e2) = (seed(3, 2, 2.0), seed(3, 2, 3.0));
        replayable.backward(&mut e1, &mut e2);
        assert_eq!(replayable.len(), 1);

        let mut consumed: Tape<BoxDynHvpStep<B>> = Tape::new();
        consumed.push(hvp_addition_step([0, 1], 2));
        let (mut a1, mut a2) = (seed(3, 2, 2.0), seed(3, 2, 3.0));
        consumed.consume_backward(&mut a1, &mut a2);

        assert_eq!(a1[0].clone().map(into_vec), e1[0].clone().map(into_vec));
        assert_eq!(a2[0].clone().map(into_vec), e2[0].clone().map(into_vec));
    }

    /// A step whose output has neither order seeded is skipped, not an error.
    ///
    /// This is what lets a tape carry steps outside the cone of whatever is being differentiated
    /// -- a second target compiled into the same sequence, say.
    #[test]
    fn a_step_with_neither_order_seeded_is_skipped() {
        let step = hvp_addition_step([0, 1], 2);
        let mut ord1: Vec<Option<CotangentFrame<B>>> = vec![None; 3];
        let mut ord2: Vec<Option<CotangentFrame<B>>> = vec![None; 3];
        // Calling the step directly, since `Tape::backward` demands both are present.
        step.step(&mut ord1, &mut ord2);
        assert!(ord1.iter().all(Option::is_none));
        assert!(ord2.iter().all(Option::is_none));
    }

    /// Seeding one order and not the other is a caller error with a specific complaint.
    #[test]
    #[should_panic(expected = "Order-2 cotangent vector must be seeded if Order-1 is seeded")]
    fn seeding_only_the_first_order_panics() {
        let step = hvp_addition_step([0, 1], 2);
        let mut ord1 = seed(3, 2, 1.0);
        let mut ord2: Vec<Option<CotangentFrame<B>>> = vec![None; 3];
        step.step(&mut ord1, &mut ord2);
    }

    #[test]
    #[should_panic(expected = "Order-1 cotangent vector must be seeded if Order-2 is seeded")]
    fn seeding_only_the_second_order_panics() {
        let step = hvp_addition_step([0, 1], 2);
        let mut ord1: Vec<Option<CotangentFrame<B>>> = vec![None; 3];
        let mut ord2 = seed(3, 2, 1.0);
        step.step(&mut ord1, &mut ord2);
    }

    #[test]
    #[should_panic(expected = "Cotangent vector not present at index 2")]
    fn an_unseeded_output_panics() {
        let mut tape: Tape<BoxDynHvpStep<B>> = Tape::new();
        tape.push(hvp_addition_step([0, 1], 2));
        let mut ord1: Vec<Option<CotangentFrame<B>>> = vec![None; 3];
        let mut ord2: Vec<Option<CotangentFrame<B>>> = vec![None; 3];
        tape.backward(&mut ord1, &mut ord2);
    }

    #[test]
    #[should_panic(expected = "Cotangent vector not present at index 2")]
    fn an_unseeded_output_panics_when_consuming() {
        let mut tape: Tape<BoxDynHvpStep<B>> = Tape::new();
        tape.push(hvp_addition_step([0, 1], 2));
        let mut ord1: Vec<Option<CotangentFrame<B>>> = vec![None; 3];
        let mut ord2 = seed(3, 2, 1.0);
        tape.consume_backward(&mut ord1, &mut ord2);
    }

    /// Fan-in accumulates at both orders.
    #[test]
    fn two_children_accumulate_at_both_orders() {
        let mut tape: Tape<BoxDynHvpStep<B>> = Tape::new();
        tape.push(hvp_addition_step([0, 0], 1));
        tape.push(hvp_addition_step([0, 0], 2));
        tape.push(hvp_addition_step([1, 2], 3));

        let mut ord1 = seed(4, 3, 1.0);
        let mut ord2 = seed(4, 3, 1.0);
        tape.backward(&mut ord1, &mut ord2);

        assert_eq!(into_vec(ord1[0].clone().unwrap()), vec![4.0]);
        assert_eq!(into_vec(ord2[0].clone().unwrap()), vec![4.0]);
    }
}

/// `sum_opt_tensors`, the small fold every operation uses to combine optional tangent blocks.
mod summing {
    use super::*;
    use skoro_operation::sum_opt_tensors;

    #[test]
    fn it_folds_over_whatever_is_present() {
        let a = scalar(1.0);
        let b = scalar(2.0);

        assert_eq!(
            into_vec(sum_opt_tensors([Some(a.clone()), Some(b.clone())]).unwrap()),
            vec![3.0]
        );
        assert_eq!(
            into_vec(sum_opt_tensors([Some(a.clone()), None]).unwrap()),
            vec![1.0]
        );
        assert_eq!(
            into_vec(sum_opt_tensors([None, Some(b)]).unwrap()),
            vec![2.0]
        );
        // Nothing present is `None`, not a zero -- the caller decides what an absent tangent
        // means, and for most operations it means "do not report one".
        let nothing: [Option<Tensor<B, 2>>; 2] = [None, None];
        assert!(sum_opt_tensors(nothing).is_none());
    }
}
