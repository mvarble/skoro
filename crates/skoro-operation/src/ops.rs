use skoro_backend::{Backend, CotangentFrame, Point, TangentVector, Tensor};

use crate::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    sum_opt_tensors,
};

#[derive(Debug, Clone, Copy)]
pub struct Addition;

impl<B: Backend> DiffOperation<B> for Addition {
    type VjpState = [bool; 2];
    type HvpState = ();

    fn forward(
        &self,
        points: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let points: Vec<Point<B>> = points.into_iter().map(Input::into_float).collect();
        let [x, y] = points.try_into().unwrap();
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(x + y),
            DiffModeState::Jvp(tvecs) => DiffOperationForward::jvp(x + y, sum_opt_tensors(tvecs)),
            DiffModeState::Vjp(diffs) => {
                DiffOperationForward::vjp(x + y, diffs.try_into().unwrap())
            }
            DiffModeState::Hvp(tvecs) => {
                let vjp_state = [tvecs[0].is_some(), tvecs[1].is_some()];
                DiffOperationForward::hvp(x + y, sum_opt_tensors(tvecs), vjp_state, ())
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        match state {
            [true, true] => vec![Some(omega.clone()), Some(omega)],
            [true, false] => vec![Some(omega), None],
            [false, true] => vec![None, Some(omega)],
            [false, false] => vec![None, None],
        }
    }

    fn hvp(
        &self,
        state: &Self::VjpState,
        _: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        if state[0] || state[1] {
            let zeros = Tensor::zeros_like(&omega);
            state
                .map(|b| if b { Some(zeros.clone()) } else { None })
                .to_vec()
        } else {
            vec![None, None]
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Subtraction;

impl<B: Backend> DiffOperation<B> for Subtraction {
    type VjpState = [bool; 2];
    type HvpState = ();

    fn forward(
        &self,
        points: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let points: Vec<Point<B>> = points.into_iter().map(Input::into_float).collect();
        let [x, y] = points.try_into().unwrap();
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(x - y),
            DiffModeState::Jvp(tvecs) => {
                let [dx, dy] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(x - y, sum_opt_tensors([dx, dy.map(|dy| -dy)]))
            }
            DiffModeState::Vjp(diffs) => {
                DiffOperationForward::vjp(x - y, diffs.try_into().unwrap())
            }
            DiffModeState::Hvp(tvecs) => {
                let [dx, dy] = tvecs.try_into().unwrap();
                let vjp_state = [dx.is_some(), dy.is_some()];
                DiffOperationForward::hvp(
                    x - y,
                    sum_opt_tensors([dx, dy.map(|dy| -dy)]),
                    vjp_state,
                    (),
                )
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        match state {
            [true, true] => vec![Some(omega.clone()), Some(-omega)],
            [true, false] => vec![Some(omega), None],
            [false, true] => vec![None, Some(-omega)],
            [false, false] => vec![None, None],
        }
    }

    fn hvp(
        &self,
        state: &Self::VjpState,
        _: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        if state[0] || state[1] {
            let zeros = Tensor::zeros_like(&omega);
            state
                .map(|b| if b { Some(zeros.clone()) } else { None })
                .to_vec()
        } else {
            vec![None, None]
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PointwiseMul;

impl<B: Backend> DiffOperation<B> for PointwiseMul {
    type VjpState = [Option<Point<B>>; 2];
    type HvpState = [Option<TangentVector<B>>; 2];

    fn forward(
        &self,
        points: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let points: Vec<Point<B>> = points.into_iter().map(Input::into_float).collect();
        let [x, y] = points.try_into().unwrap();
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(x * y),
            DiffModeState::Jvp(tvecs) => {
                let [dx, dy] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    x.clone() * y.clone(),
                    sum_opt_tensors([
                        dx.map(|dx| dx * y.clone().unsqueeze_dim(1)),
                        dy.map(|dy| x.clone().unsqueeze_dim(1) * dy),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [dx, dy] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    x.clone() * y.clone(),
                    [
                        if dx { Some(y.clone()) } else { None },
                        if dy { Some(x.clone()) } else { None },
                    ],
                )
            }
            DiffModeState::Hvp(cvecs) => {
                let [dx, dy] = cvecs.try_into().unwrap();
                DiffOperationForward::hvp(
                    x.clone() * y.clone(),
                    sum_opt_tensors([
                        dx.as_ref().map(|dx| dx.clone() * y.clone()),
                        dy.as_ref().map(|dy| x.clone() * dy.clone()),
                    ]),
                    [
                        if dx.is_some() { Some(y) } else { None },
                        if dy.is_some() { Some(x) } else { None },
                    ],
                    [dy, dx],
                )
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        state
            .each_ref()
            .map(|x| {
                x.as_ref()
                    .map(|x| omega.clone() * x.clone().unsqueeze_dim(1))
            })
            .to_vec()
    }

    fn hvp(
        &self,
        _: &Self::VjpState,
        state: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        state
            .each_ref()
            .map(|d| {
                d.as_ref()
                    .map(|d| d.clone().unsqueeze_dim(1) * omega.clone())
            })
            .to_vec()
    }
}
#[derive(Debug, Clone, Copy)]
pub struct LiteralPointwiseAdd<B: Backend>(B::FloatElem);

impl<B: Backend> LiteralPointwiseAdd<B> {
    pub fn new(val: B::FloatElem) -> Self {
        Self(val)
    }
}

impl<B: Backend> DiffOperation<B> for LiteralPointwiseAdd<B> {
    type VjpState = bool;
    type HvpState = ();

    fn forward(
        &self,
        points: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let points: Vec<Point<B>> = points.into_iter().map(Input::into_float).collect();
        let [x] = points.try_into().unwrap();
        let y = self.0;
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(x + y),
            DiffModeState::Jvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(x + y, dx)
            }
            DiffModeState::Vjp(diffs) => DiffOperationForward::vjp(x + y, diffs[0]),
            DiffModeState::Hvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                let diff = dx.is_some();
                DiffOperationForward::hvp(x + y, dx, diff, ())
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![if *state { Some(omega) } else { None }]
    }

    fn hvp(
        &self,
        state: &Self::VjpState,
        _: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![if *state {
            Some(omega.zeros_like())
        } else {
            None
        }]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LiteralPointwiseSub<B: Backend>(B::FloatElem);

impl<B: Backend> LiteralPointwiseSub<B> {
    pub fn new(val: B::FloatElem) -> Self {
        Self(val)
    }
}

impl<B: Backend> DiffOperation<B> for LiteralPointwiseSub<B> {
    type VjpState = bool;
    type HvpState = ();

    fn forward(
        &self,
        points: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let points: Vec<Point<B>> = points.into_iter().map(Input::into_float).collect();
        let [x] = points.try_into().unwrap();
        let y = self.0;
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(x - y),
            DiffModeState::Jvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(x - y, dx)
            }
            DiffModeState::Vjp(diffs) => DiffOperationForward::vjp(x - y, diffs[0]),
            DiffModeState::Hvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                let diff = dx.is_some();
                DiffOperationForward::hvp(x - y, dx, diff, ())
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![if *state { Some(omega) } else { None }]
    }

    fn hvp(
        &self,
        state: &Self::VjpState,
        _: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![if *state {
            Some(omega.zeros_like())
        } else {
            None
        }]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LiteralPointwiseMul<B: Backend>(B::FloatElem);

impl<B: Backend> LiteralPointwiseMul<B> {
    pub fn new(val: B::FloatElem) -> Self {
        Self(val)
    }
}

impl<B: Backend> DiffOperation<B> for LiteralPointwiseMul<B> {
    type VjpState = bool;
    type HvpState = ();

    fn forward(
        &self,
        points: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let points: Vec<Point<B>> = points.into_iter().map(Input::into_float).collect();
        let [x] = points.try_into().unwrap();
        let y = self.0;
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(x * y),
            DiffModeState::Jvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(x * y, sum_opt_tensors([dx.map(|dx| dx * y)]))
            }
            DiffModeState::Vjp(diffs) => DiffOperationForward::vjp(x * y, diffs[0]),
            DiffModeState::Hvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::hvp(
                    x * y,
                    sum_opt_tensors([dx.as_ref().map(|dx| dx.clone() * y)]),
                    dx.is_some(),
                    (),
                )
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![if *state { Some(omega * self.0) } else { None }]
    }

    fn hvp(
        &self,
        state: &Self::VjpState,
        _: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![if *state {
            Some(omega.zeros_like())
        } else {
            None
        }]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PointwiseDiv;

pub enum PointwiseDivVjpState {
    X,
    Y,
    Both,
}

impl<B: Backend> DiffOperation<B> for PointwiseDiv {
    type VjpState = Option<(PointwiseDivVjpState, [Point<B>; 2])>;
    type HvpState = [Option<TangentVector<B>>; 2];

    fn forward(
        &self,
        points: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let points: Vec<Point<B>> = points.into_iter().map(Input::into_float).collect();
        let [x, y] = points.try_into().unwrap();
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(x / y),
            DiffModeState::Jvp(tvecs) => {
                let [dx, dy] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    x.clone() / y.clone(),
                    sum_opt_tensors([
                        dx.map(|dx| dx / y.clone().unsqueeze_dim(1)),
                        dy.map(|dy| -x.clone().unsqueeze_dim(1) / dy.square()),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [xdiff, ydiff] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    x.clone() / y.clone(),
                    match (xdiff, ydiff) {
                        (false, false) => None,
                        (true, false) => Some((PointwiseDivVjpState::X, [x, y])),
                        (false, true) => Some((PointwiseDivVjpState::Y, [x, y])),
                        (true, true) => Some((PointwiseDivVjpState::Both, [x, y])),
                    },
                )
            }
            DiffModeState::Hvp(cvecs) => {
                let [dx, dy] = cvecs.try_into().unwrap();
                DiffOperationForward::hvp(
                    x.clone() / y.clone(),
                    sum_opt_tensors([
                        dx.as_ref().map(|dx| dx.clone() / y.clone()),
                        dy.as_ref().map(|dy| x.clone() / dy.clone().square()),
                    ]),
                    match (dx.is_some(), dy.is_some()) {
                        (false, false) => None,
                        (true, false) => Some((PointwiseDivVjpState::X, [x, y])),
                        (false, true) => Some((PointwiseDivVjpState::Y, [x, y])),
                        (true, true) => Some((PointwiseDivVjpState::Both, [x, y])),
                    },
                    [dx, dy],
                )
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        match state {
            None => vec![None, None],
            Some((PointwiseDivVjpState::X, [_, y])) => {
                vec![Some(omega / y.clone().unsqueeze_dim::<3>(2)), None]
            }
            Some((PointwiseDivVjpState::Y, [x, y])) => {
                vec![
                    None,
                    Some(-omega * (x.clone() / y.clone().square()).unsqueeze_dim::<3>(2)),
                ]
            }
            Some((PointwiseDivVjpState::Both, [x, y])) => vec![
                Some(omega.clone() / y.clone().unsqueeze_dim::<3>(2)),
                Some(-omega * (x.clone() / y.clone().square()).unsqueeze_dim::<3>(2)),
            ],
        }
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        match hvp_state {
            [None, None] => vec![None, None],
            [Some(_), None] => {
                let Some((PointwiseDivVjpState::X, [_, _])) = vjp_state else {
                    unreachable!()
                };
                vec![Some(omega.zeros_like()), None]
            }
            [None, Some(dy)] => {
                let Some((PointwiseDivVjpState::Y, [x, y])) = vjp_state else {
                    unreachable!()
                };
                let (x, y, dy) = (x.clone(), y.clone(), dy.clone());
                vec![
                    None,
                    Some(2.0 * omega * (x * dy / y.powf_scalar(3.0)).unsqueeze_dim::<3>(2)),
                ]
            }
            [Some(dx), Some(dy)] => {
                let Some((PointwiseDivVjpState::Both, [x, y])) = vjp_state else {
                    unreachable!()
                };
                let (x, y, dx, dy) = (x.clone(), y.clone(), dx.clone(), dy.clone());

                let dxomega =
                    -omega.clone() * (dy.clone() / y.clone().square()).unsqueeze_dim::<3>(2);

                let dyomega = {
                    let num = 2.0 * x * dy - y.clone() * dx;
                    let dem = y.powf_scalar(3.0);
                    let quot: Tensor<B, 2> = num / dem;
                    let quot = quot.unsqueeze_dim::<3>(2);
                    omega * quot
                };

                vec![Some(dxomega), Some(dyomega)]
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LiteralPointwiseDiv<B: Backend>(B::FloatElem);

impl<B: Backend> LiteralPointwiseDiv<B> {
    pub fn new(val: B::FloatElem) -> Self {
        Self(val)
    }
}

impl<B: Backend> DiffOperation<B> for LiteralPointwiseDiv<B> {
    type VjpState = bool;
    type HvpState = ();

    fn forward(
        &self,
        points: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let points: Vec<Point<B>> = points.into_iter().map(Input::into_float).collect();
        let [x] = points.try_into().unwrap();
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(x / self.0),
            DiffModeState::Jvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(x / self.0, dx.map(|dx| dx / self.0))
            }
            DiffModeState::Vjp(diffs) => DiffOperationForward::vjp(x / self.0, diffs[0]),
            DiffModeState::Hvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::hvp(
                    x / self.0,
                    dx.as_ref().map(|dx| dx.clone() / self.0),
                    dx.is_some(),
                    (),
                )
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![if *state { Some(omega / self.0) } else { None }]
    }

    fn hvp(
        &self,
        state: &Self::VjpState,
        _: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![if *state {
            Some(omega.zeros_like())
        } else {
            None
        }]
    }
}
