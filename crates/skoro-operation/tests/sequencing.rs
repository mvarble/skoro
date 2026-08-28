//! `SequencedDiffOperation`: a `DiffOperation` positioned in an index-based value sequence.
//!
//! The operation does the mathematics; this adapter decides *where* its inputs come from and
//! where its outputs go, and enforces the contract that a mode must return its own variant.
//! Composing several of them by hand is also the cheapest way to check that the tape and the
//! per-operation derivatives agree end to end, without a graph in the way.

mod common;

use common::*;
use skoro_backend::{CotangentFrame, Point, TangentFrame, TangentVector};
use skoro_core::sequence::SequenceMutator;
use skoro_operation::{
    DynSequenceOperation, Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward, SequencedDiffOperation},
    ops::{Addition, ScalarMultiplication},
    step::{BoxDynHvpStep, BoxDynVjpStep},
    tape::Tape,
};

/// `x ↦ x²`, componentwise. Small enough to differentiate by inspection.
#[derive(Debug, Clone, Copy)]
struct Square;

impl DiffOperation<B> for Square {
    type VjpState = Option<Point<B>>;
    type HvpState = Option<TangentVector<B>>;

    fn forward(
        &self,
        points: Vec<Input<B>>,
        mode: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let points: Vec<Point<B>> = points.into_iter().map(Input::unwrap_float).collect();
        let [x] = points.try_into().unwrap();
        let output = x.clone() * x.clone();
        match mode {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(output),
            DiffModeState::Jvp(tvecs) => {
                let [v] = tvecs.try_into().unwrap();
                let t = v.map(|v| v * (x.unsqueeze_dim(1) * 2.0));
                DiffOperationForward::jvp(output, t)
            }
            DiffModeState::Vjp(diffs) => {
                let [diff] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(output, if diff { Some(x) } else { None })
            }
            DiffModeState::Hvp { diffs, tangents } => {
                let [diff] = diffs.try_into().unwrap();
                let [v] = tangents.try_into().unwrap();
                let t = v.as_ref().map(|v| v.clone() * (x.clone() * 2.0));
                DiffOperationForward::hvp(output, t, diff.then_some(x), v.clone())
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![
            state
                .as_ref()
                .map(|x| omega * (x.clone().unsqueeze_dim(1) * 2.0)),
        ]
    }

    fn hvp(
        &self,
        _state: &Self::VjpState,
        v: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![
            v.as_ref()
                .map(|v| v.clone().unsqueeze_dim(1) * (omega * 2.0)),
        ]
    }
}

fn get_scalar<const D: usize>(t: &skoro_backend::Tensor<B, D>) -> f64 {
    t.to_data().to_vec::<f64>().unwrap()[0]
}

fn t2(v: f64) -> skoro_backend::Tensor<B, 2> {
    scalar(v)
}

fn t3(v: f64) -> skoro_backend::Tensor<B, 3> {
    tensor3(vec![v], [1, 1, 1])
}

// Running a hand-built sequence in each of the four modes.

fn run_forward(
    ops: &[Box<dyn DynSequenceOperation<B>>],
    initial_points: Vec<Option<Input<B>>>,
    output_index: usize,
) -> Point<B> {
    let mut points = initial_points;
    for op in ops {
        op.forward(&mut points);
    }
    points[output_index].take().unwrap().unwrap_float()
}

fn run_jvp(
    ops: &[Box<dyn DynSequenceOperation<B>>],
    initial_points: Vec<Option<Input<B>>>,
    initial_tangents: Vec<Option<TangentFrame<B>>>,
    output_index: usize,
) -> (Point<B>, TangentFrame<B>) {
    let mut points = initial_points;
    let mut tangents = initial_tangents;
    for op in ops {
        op.jvp_forward(&mut points, &mut tangents);
    }
    (
        points[output_index].take().unwrap().unwrap_float(),
        tangents[output_index].take().unwrap(),
    )
}

fn run_vjp(
    ops: &[Box<dyn DynSequenceOperation<B>>],
    initial_points: Vec<Option<Input<B>>>,
    output_index: usize,
    seed: CotangentFrame<B>,
) -> Vec<Option<CotangentFrame<B>>> {
    let n = initial_points.len();
    let mut points = initial_points;
    let mut tape: Tape<BoxDynVjpStep<B>> = Tape::new();
    for op in ops {
        op.vjp_forward(&mut points, &mut tape);
    }
    let mut cotangents = vec![None; n];
    cotangents[output_index] = Some(seed);
    tape.backward(&mut cotangents);
    cotangents
}

fn run_hvp(
    ops: &[Box<dyn DynSequenceOperation<B>>],
    initial_points: Vec<Option<Input<B>>>,
    initial_tangents: Vec<Option<TangentVector<B>>>,
    output_index: usize,
    ord1_seed: CotangentFrame<B>,
    ord2_seed: CotangentFrame<B>,
) -> [Vec<Option<CotangentFrame<B>>>; 2] {
    let n = initial_points.len();
    let mut points = initial_points;
    let mut tangents = initial_tangents;
    let mut tape: Tape<BoxDynHvpStep<B>> = Tape::new();
    for op in ops {
        op.hvp_forward(&mut points, &mut tangents, &mut tape);
    }
    let mut ord1_seeds = vec![None; n];
    ord1_seeds[output_index] = Some(ord1_seed);
    let mut ord2_seeds = vec![None; n];
    ord2_seeds[output_index] = Some(ord2_seed);
    tape.backward(&mut ord1_seeds, &mut ord2_seeds);
    [ord1_seeds, ord2_seeds]
}

/// Whole programs rather than single operations: a unary chain, a composition, and a fan-out whose
/// pullback has to reach both branches. This is where a step that worked alone shows whether it
/// composes.
mod chains {
    use super::*;

    /// Unary chain: F(x) = (3 * x)^2 = 9x^2
    ///   F(2) = 36,  F'(2) = 36, F''(2) = 18
    #[test]
    fn unary_chain() {
        let c_index = 0;
        let x_index = 1;
        let mid_index = 2;
        let out_index = 3;

        let ops: Vec<Box<dyn DynSequenceOperation<B>>> = vec![
            Box::new(SequencedDiffOperation::new(
                vec![c_index, x_index],
                vec![false, true],
                mid_index,
                ScalarMultiplication,
            )),
            Box::new(SequencedDiffOperation::new(
                vec![mid_index],
                vec![true],
                out_index,
                Square,
            )),
        ];

        let mut points = vec![None; 4];
        points[c_index] = Some(Input::Float(t2(3.0)));
        points[x_index] = Some(Input::Float(t2(2.0)));

        let out = run_forward(&ops, points.clone(), out_index);
        approx::assert_abs_diff_eq!(get_scalar(&out), 36.0, epsilon = 1e-5);

        let mut tangents = vec![None; 4];
        tangents[x_index] = Some(t3(1.0));

        let (out, jvp) = run_jvp(&ops, points.clone(), tangents.clone(), out_index);

        approx::assert_abs_diff_eq!(get_scalar(&out), 36.0, epsilon = 1e-5);
        approx::assert_abs_diff_eq!(get_scalar(&jvp), 36.0, epsilon = 1e-5);

        let mut cotangents = run_vjp(&ops, points.clone(), out_index, t3(1.0));
        let dx = cotangents[x_index].take().unwrap();
        approx::assert_abs_diff_eq!(get_scalar(&dx), 36.0, epsilon = 1e-5);

        let mut tangents = vec![None; 4];
        tangents[x_index] = Some(t2(1.0));

        let [mut ord1, mut ord2] = run_hvp(
            &ops,
            points.clone(),
            tangents.clone(),
            out_index,
            t3(1.0),
            t3(0.0),
        );

        let dx = ord1[x_index].take().unwrap();
        approx::assert_abs_diff_eq!(get_scalar(&dx), 36.0, epsilon = 1e-5);
        let dxdx = ord2[x_index].take().unwrap();
        approx::assert_abs_diff_eq!(get_scalar(&dxdx), 18.0, epsilon = 1e-5);

        // The same function written as a product of one node with itself, so the second
        // operation reads the *same* slot twice. Fan-out through a single step rather than
        // through two, which the accumulation has to handle identically.
        let ops: Vec<Box<dyn DynSequenceOperation<B>>> = vec![
            Box::new(SequencedDiffOperation::new(
                vec![c_index, x_index],
                vec![false, true],
                mid_index,
                ScalarMultiplication,
            )),
            Box::new(SequencedDiffOperation::new(
                vec![mid_index, mid_index],
                vec![true, true],
                out_index,
                ScalarMultiplication,
            )),
        ];

        let out = run_forward(&ops, points.clone(), out_index);
        approx::assert_abs_diff_eq!(get_scalar(&out), 36.0, epsilon = 1e-5);

        let [mut ord1, mut ord2] = run_hvp(&ops, points, tangents, out_index, t3(1.0), t3(0.0));

        let dx = ord1[x_index].take().unwrap();
        approx::assert_abs_diff_eq!(get_scalar(&dx), 36.0, epsilon = 1e-5);
        let dxdx = ord2[x_index].take().unwrap();
        approx::assert_abs_diff_eq!(get_scalar(&dxdx), 18.0, epsilon = 1e-5);
    }

    /// Composed: F(x, y) = (x + y)^2
    ///   F(2, 3) = 25, dF/dx = dF/dy = 10, d^2F/dxdy = 2
    #[test]
    fn composed_add_square() {
        let x_index = 0;
        let y_index = 1;
        let sum_index = 2;
        let out_index = 3;

        let ops: Vec<Box<dyn DynSequenceOperation<B>>> = vec![
            Box::new(SequencedDiffOperation::new(
                vec![x_index, y_index],
                vec![true, true],
                sum_index,
                Addition,
            )),
            Box::new(SequencedDiffOperation::new(
                vec![sum_index],
                vec![true],
                out_index,
                Square,
            )),
        ];

        let mut points = vec![None; 4];
        points[x_index] = Some(Input::Float(t2(2.0)));
        points[y_index] = Some(Input::Float(t2(3.0)));

        let out = run_forward(&ops, points.clone(), out_index);
        approx::assert_abs_diff_eq!(get_scalar(&out), 25.0, epsilon = 1e-5);

        let mut cotangents = run_vjp(&ops, points.clone(), out_index, t3(1.0));
        let dx = cotangents[x_index].take().unwrap();
        let dy = cotangents[y_index].take().unwrap();
        approx::assert_abs_diff_eq!(get_scalar(&dx), 10.0, epsilon = 1e-5);
        approx::assert_abs_diff_eq!(get_scalar(&dy), 10.0, epsilon = 1e-5);

        let mut tangents = vec![None; 4];
        tangents[x_index] = Some(t2(1.0));
        tangents[y_index] = Some(t2(0.0));

        let [mut cotangents0, mut cotangents1] =
            run_hvp(&ops, points, tangents, out_index, t3(1.0), t3(0.0));

        let dx = cotangents0[x_index].take().unwrap();
        let dy = cotangents0[y_index].take().unwrap();
        let dxdy = cotangents1[y_index].take().unwrap();

        approx::assert_abs_diff_eq!(get_scalar(&dx), 10.0, epsilon = 1e-5);
        approx::assert_abs_diff_eq!(get_scalar(&dy), 10.0, epsilon = 1e-5);
        approx::assert_abs_diff_eq!(get_scalar(&dxdy), 2.0, epsilon = 1e-5);
    }

    /// Fan-out: F(x) = x^2 + x
    ///   F(3) = 12, F'(3) = 7, F''(3) = 2
    #[test]
    fn fan_out_vjp() {
        let x_index = 0;
        let sq_index = 1;
        let out_index = 2;

        let ops: Vec<Box<dyn DynSequenceOperation<B>>> = vec![
            Box::new(SequencedDiffOperation::new(
                vec![x_index],
                vec![true],
                sq_index,
                Square,
            )),
            Box::new(SequencedDiffOperation::new(
                vec![sq_index, x_index],
                vec![true, true],
                out_index,
                Addition,
            )),
        ];

        let mut points = vec![None; 3];
        points[x_index] = Some(Input::Float(t2(3.0)));

        let out = run_forward(&ops, points.clone(), out_index);
        approx::assert_abs_diff_eq!(get_scalar(&out), 12.0, epsilon = 1e-5);

        let mut cotangents = run_vjp(&ops, points.clone(), out_index, t3(1.0));
        let dx = cotangents[x_index].take().unwrap();
        approx::assert_abs_diff_eq!(get_scalar(&dx), 7.0, epsilon = 1e-5);

        let mut tangents = vec![None; 3];
        tangents[x_index] = Some(t2(1.0));
        let [mut cotangents0, mut cotangents1] =
            run_hvp(&ops, points, tangents, out_index, t3(1.0), t3(0.0));

        let dx = cotangents0[x_index].take().unwrap();
        let dxdx = cotangents1[x_index].take().unwrap();

        approx::assert_abs_diff_eq!(get_scalar(&dx), 7.0, epsilon = 1e-5);
        approx::assert_abs_diff_eq!(get_scalar(&dxdx), 2.0, epsilon = 1e-5);
    }
}

/// The adapter's own bookkeeping, separate from the mathematics it delegates.
mod adapter {
    use super::*;

    fn op() -> SequencedDiffOperation<B, Addition> {
        SequencedDiffOperation::new(vec![0, 1], vec![true, false], 2, Addition)
    }

    #[test]
    fn it_reports_the_positions_it_was_built_with() {
        let op = op();
        assert_eq!(op.input_indices(), &[0, 1]);
        assert_eq!(op.output_index(), 2);
        assert_eq!(op.input_diffs(), &[true, false]);
    }

    #[test]
    fn it_names_the_operation_it_wraps_when_printed() {
        let text = format!("{:?}", op());
        assert!(text.contains("SequenceOperation"));
        assert!(text.contains("input_diffs"));
        assert!(text.contains("Addition"));
    }

    /// A non-differentiable input is never handed a tangent, even when one sits in its slot.
    ///
    /// `get_tangents` gates on `input_diffs`, so a constant's slot reads as `None` regardless of
    /// what is stored there. Without that, a leftover tangent from an unrelated quantity at the
    /// same index would leak into the pushforward.
    #[test]
    fn a_constant_input_is_never_handed_a_tangent() {
        let op = op();
        let mut points = vec![
            Some(Input::Float(t2(1.0))),
            Some(Input::Float(t2(2.0))),
            None,
        ];
        // Both slots carry a tangent, but only slot 0 is declared differentiable.
        let mut tangents = vec![Some(t3(1.0)), Some(t3(100.0)), None];
        op.jvp_forward(&mut points, &mut tangents);

        // `Addition`'s pushforward is the sum of the tangents it receives -- so 1, not 101.
        approx::assert_abs_diff_eq!(
            get_scalar(&tangents[2].take().unwrap()),
            1.0,
            epsilon = 1e-5
        );
    }

    /// A parent that has not been realized is a sequencing bug, and says which input it was.
    #[test]
    #[should_panic(expected = "requires parent at (relative) input index 1")]
    fn an_unrealized_parent_panics() {
        let op = op();
        let mut points = vec![Some(Input::Float(t2(1.0))), None, None];
        op.forward(&mut points);
    }

    /// An operation that returns the wrong variant for its mode is a contract violation, and the
    /// adapter names both the operation and the mode it was asked for.
    ///
    /// All four modes are checked, because the failure is a copy-pasted `match` arm and it would
    /// otherwise only be caught in whichever mode happened to be exercised.
    mod mode_contract {
        use super::*;

        /// Always returns `ForwardOnly`, whatever it was asked for.
        #[derive(Debug, Clone, Copy)]
        struct AlwaysForwardOnly;

        impl DiffOperation<B> for AlwaysForwardOnly {
            type VjpState = ();
            type HvpState = ();
            fn forward(
                &self,
                points: Vec<Input<B>>,
                _: DiffModeState<B>,
            ) -> DiffOperationForward<B, (), ()> {
                DiffOperationForward::forward_only(points[0].clone().unwrap_float())
            }
            fn vjp(&self, _: &(), _: CotangentFrame<B>) -> Vec<Option<CotangentFrame<B>>> {
                unimplemented!()
            }
            fn hvp(&self, _: &(), _: &(), _: CotangentFrame<B>) -> Vec<Option<CotangentFrame<B>>> {
                unimplemented!()
            }
        }

        fn broken() -> SequencedDiffOperation<B, AlwaysForwardOnly> {
            SequencedDiffOperation::new(vec![0], vec![true], 1, AlwaysForwardOnly)
        }

        fn points() -> Vec<Option<Input<B>>> {
            vec![Some(Input::Float(t2(1.0))), None]
        }

        #[test]
        #[should_panic(expected = "must pass `DiffModeState::Jvp`")]
        fn jvp() {
            let mut tangents = vec![None; 2];
            broken().jvp_forward(&mut points(), &mut tangents);
        }

        #[test]
        #[should_panic(expected = "must pass `DiffModeState::Vjp`")]
        fn vjp() {
            let mut tape: Tape<BoxDynVjpStep<B>> = Tape::new();
            broken().vjp_forward(&mut points(), &mut tape);
        }

        #[test]
        #[should_panic(expected = "must pass `DiffModeState::Hvp`")]
        fn hvp() {
            let mut tangents = vec![None; 2];
            let mut tape: Tape<BoxDynHvpStep<B>> = Tape::new();
            broken().hvp_forward(&mut points(), &mut tangents, &mut tape);
        }

        /// And the mirror: an operation that never returns `ForwardOnly`.
        #[derive(Debug, Clone, Copy)]
        struct NeverForwardOnly;

        impl DiffOperation<B> for NeverForwardOnly {
            type VjpState = ();
            type HvpState = ();
            fn forward(
                &self,
                points: Vec<Input<B>>,
                _: DiffModeState<B>,
            ) -> DiffOperationForward<B, (), ()> {
                DiffOperationForward::vjp(points[0].clone().unwrap_float(), ())
            }
            fn vjp(&self, _: &(), _: CotangentFrame<B>) -> Vec<Option<CotangentFrame<B>>> {
                unimplemented!()
            }
            fn hvp(&self, _: &(), _: &(), _: CotangentFrame<B>) -> Vec<Option<CotangentFrame<B>>> {
                unimplemented!()
            }
        }

        #[test]
        #[should_panic(expected = "must pass `DiffModeState::ForwardOnly`")]
        fn forward_only() {
            let op = SequencedDiffOperation::new(vec![0], vec![true], 1, NeverForwardOnly);
            op.forward(&mut points());
        }
    }
}

/// The `DiffOperationForward` constructors, which every operation returns through.
mod forward_variants {
    use super::*;

    #[test]
    fn each_constructor_builds_its_own_variant() {
        let y = t2(1.0);
        assert!(matches!(
            DiffOperationForward::<B, (), ()>::forward_only(y.clone()),
            DiffOperationForward::ForwardOnly { .. }
        ));
        assert!(matches!(
            DiffOperationForward::<B, (), ()>::jvp(y.clone(), None),
            DiffOperationForward::Jvp { tangent: None, .. }
        ));
        assert!(matches!(
            DiffOperationForward::<B, (), ()>::vjp(y.clone(), ()),
            DiffOperationForward::Vjp { .. }
        ));
        assert!(matches!(
            DiffOperationForward::<B, (), ()>::hvp(y, None, (), ()),
            DiffOperationForward::Hvp { tangent: None, .. }
        ));
    }

    /// The mode enum prints, which is what makes a contract-violation panic readable.
    #[test]
    fn the_mode_state_is_debug() {
        assert!(format!("{:?}", DiffModeState::<B>::ForwardOnly).contains("ForwardOnly"));
        assert!(format!("{:?}", DiffModeState::<B>::Vjp(vec![true])).contains("Vjp"));
        assert!(format!("{:?}", DiffModeState::<B>::Jvp(vec![None])).contains("Jvp"));
        let hvp = DiffModeState::<B>::Hvp {
            diffs: vec![true],
            tangents: vec![None],
        };
        assert!(format!("{hvp:?}").contains("Hvp"));
    }
}
