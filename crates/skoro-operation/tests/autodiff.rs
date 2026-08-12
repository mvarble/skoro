use skoro_backend::{CotangentFrame, NdArrayBackend, Point, TangentFrame, TangentVector};

use skoro_operation::{
    DynSequenceOperation, Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward, SequencedDiffOperation},
    ops::{Addition, ScalarMultiplication},
    step::{BoxDynHvpStep, BoxDynVjpStep},
    tape::Tape,
};

type B = NdArrayBackend<f32, i32>;

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
        let points: Vec<Point<B>> = points.into_iter().map(Input::into_float).collect();
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

fn get_scalar<const D: usize>(t: &skoro_backend::Tensor<B, D>) -> f32 {
    t.to_data().to_vec::<f32>().unwrap()[0]
}

fn tensor2(v: f32) -> skoro_backend::Tensor<B, 2> {
    skoro_backend::Tensor::from_floats([[v]], &Default::default())
}

fn tensor3(v: f32) -> skoro_backend::Tensor<B, 3> {
    skoro_backend::Tensor::from_floats([[[v]]], &Default::default())
}

fn run_forward(
    ops: &[Box<dyn DynSequenceOperation<B>>],
    initial_points: Vec<Option<Input<B>>>,
    output_index: usize,
) -> Point<B> {
    let mut points = initial_points;
    for op in ops {
        op.forward(&mut points);
    }
    points[output_index].take().unwrap().into_float()
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
        points[output_index].take().unwrap().into_float(),
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
    points[c_index] = Some(Input::Float(tensor2(3.0)));
    points[x_index] = Some(Input::Float(tensor2(2.0)));

    let out = run_forward(&ops, points.clone(), out_index);
    approx::assert_abs_diff_eq!(get_scalar(&out), 36.0, epsilon = 1e-5);

    let mut tangents = vec![None; 4];
    tangents[x_index] = Some(tensor3(1.0));

    let (out, jvp) = run_jvp(&ops, points.clone(), tangents.clone(), out_index);

    approx::assert_abs_diff_eq!(get_scalar(&out), 36.0, epsilon = 1e-5);
    approx::assert_abs_diff_eq!(get_scalar(&jvp), 36.0, epsilon = 1e-5);

    let mut cotangents = run_vjp(&ops, points.clone(), out_index, tensor3(1.0));
    let dx = cotangents[x_index].take().unwrap();
    approx::assert_abs_diff_eq!(get_scalar(&dx), 36.0, epsilon = 1e-5);

    let mut tangents = vec![None; 4];
    tangents[x_index] = Some(tensor2(1.0));

    let [mut ord1, mut ord2] = run_hvp(
        &ops,
        points.clone(),
        tangents.clone(),
        out_index,
        tensor3(1.0),
        tensor3(0.0),
    );

    let dx = ord1[x_index].take().unwrap();
    approx::assert_abs_diff_eq!(get_scalar(&dx), 36.0, epsilon = 1e-5);
    let dxdx = ord2[x_index].take().unwrap();
    approx::assert_abs_diff_eq!(get_scalar(&dxdx), 18.0, epsilon = 1e-5);

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

    let [mut ord1, mut ord2] = run_hvp(
        &ops,
        points,
        tangents,
        out_index,
        tensor3(1.0),
        tensor3(0.0),
    );

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
    points[x_index] = Some(Input::Float(tensor2(2.0)));
    points[y_index] = Some(Input::Float(tensor2(3.0)));

    let out = run_forward(&ops, points.clone(), out_index);
    approx::assert_abs_diff_eq!(get_scalar(&out), 25.0, epsilon = 1e-5);

    let mut cotangents = run_vjp(&ops, points.clone(), out_index, tensor3(1.0));
    let dx = cotangents[x_index].take().unwrap();
    let dy = cotangents[y_index].take().unwrap();
    approx::assert_abs_diff_eq!(get_scalar(&dx), 10.0, epsilon = 1e-5);
    approx::assert_abs_diff_eq!(get_scalar(&dy), 10.0, epsilon = 1e-5);

    let mut tangents = vec![None; 4];
    tangents[x_index] = Some(tensor2(1.0));
    tangents[y_index] = Some(tensor2(0.0));

    let [mut cotangents0, mut cotangents1] = run_hvp(
        &ops,
        points,
        tangents,
        out_index,
        tensor3(1.0),
        tensor3(0.0),
    );

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
    points[x_index] = Some(Input::Float(tensor2(3.0)));

    let out = run_forward(&ops, points.clone(), out_index);
    approx::assert_abs_diff_eq!(get_scalar(&out), 12.0, epsilon = 1e-5);

    let mut cotangents = run_vjp(&ops, points.clone(), out_index, tensor3(1.0));
    let dx = cotangents[x_index].take().unwrap();
    approx::assert_abs_diff_eq!(get_scalar(&dx), 7.0, epsilon = 1e-5);

    let mut tangents = vec![None; 3];
    tangents[x_index] = Some(tensor2(1.0));
    let [mut cotangents0, mut cotangents1] = run_hvp(
        &ops,
        points,
        tangents,
        out_index,
        tensor3(1.0),
        tensor3(0.0),
    );

    let dx = cotangents0[x_index].take().unwrap();
    let dxdx = cotangents1[x_index].take().unwrap();

    approx::assert_abs_diff_eq!(get_scalar(&dx), 7.0, epsilon = 1e-5);
    approx::assert_abs_diff_eq!(get_scalar(&dxdx), 2.0, epsilon = 1e-5);
}
