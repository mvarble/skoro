use skoro_backend::{Backend, CotangentFrame, Point, TangentVector, Tensor};

use crate::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    sum_opt_tensors,
};

/// Negation is scalar multiplication by -1 on a vector space.
#[derive(Debug, Clone, Copy)]
pub struct Negation;

impl<B: Backend> DiffOperation<B> for Negation {
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
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(-x),
            DiffModeState::Jvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(-x, dx.map(|dx| -dx))
            }
            DiffModeState::Vjp(diffs) => {
                let [xdiff] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(-x, xdiff)
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dx] = tvecs.try_into().unwrap();
                let [xdiff] = diffs.try_into().unwrap();
                DiffOperationForward::hvp(-x, dx.map(|dx| -dx), xdiff, ())
            }
        }
    }

    fn vjp(
        &self,
        isdiff: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![if *isdiff { Some(-omega) } else { None }]
    }

    fn hvp(
        &self,
        isdiff: &Self::VjpState,
        _: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![if *isdiff {
            Some(omega.zeros_like())
        } else {
            None
        }]
    }
}

/// Addition operation on manifolds which are vector spaces.
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
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let vjp_state = diffs.try_into().unwrap();
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

/// Subtraction operation on manifolds which are vector spaces.
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
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dx, dy] = tvecs.try_into().unwrap();
                let vjp_state = diffs.try_into().unwrap();
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

/// Scalar multiplication on a vector space.
///
/// Input convention: (scalar, vector)
#[derive(Debug, Clone, Copy)]
pub struct ScalarMultiplication;

impl<B: Backend> DiffOperation<B> for ScalarMultiplication {
    type VjpState = [Option<Point<B>>; 2];
    type HvpState = [Option<TangentVector<B>>; 2];

    fn forward(
        &self,
        points: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let points: Vec<Point<B>> = points.into_iter().map(Input::into_float).collect();
        let [alpha, x] = points.try_into().unwrap();
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(alpha * x),
            DiffModeState::Jvp(tvecs) => {
                let [dalpha, dx] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    alpha.clone() * x.clone(),
                    sum_opt_tensors([
                        dalpha.map(|dalpha| dalpha * x.clone().unsqueeze_dim(2)),
                        dx.map(|dx| alpha.clone().unsqueeze_dim(2) * dx),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [dalpha, dx] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    alpha.clone() * x.clone(),
                    [
                        if dalpha { Some(x.clone()) } else { None },
                        if dx { Some(alpha.clone()) } else { None },
                    ],
                )
            }
            DiffModeState::Hvp {
                diffs,
                tangents: cvecs,
            } => {
                let [dalpha, dx] = cvecs.try_into().unwrap();
                let [alphadiff, xdiff] = diffs.try_into().unwrap();
                DiffOperationForward::hvp(
                    alpha.clone() * x.clone(),
                    sum_opt_tensors([
                        dalpha.as_ref().map(|dalpha| dalpha.clone() * x.clone()),
                        dx.as_ref().map(|dx| alpha.clone() * dx.clone()),
                    ]),
                    [
                        if alphadiff { Some(x) } else { None },
                        if xdiff { Some(alpha) } else { None },
                    ],
                    [dx, dalpha],
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
            [Some(x), Some(alpha)] => vec![
                Some(x.clone().unsqueeze_dim(1).matmul(omega.clone())),
                Some(alpha.clone().unsqueeze_dim(2) * omega),
            ],
            [None, Some(alpha)] => vec![None, Some(alpha.clone().unsqueeze_dim(2) * omega)],
            [Some(x), None] => vec![Some(x.clone().unsqueeze_dim(1).matmul(omega)), None],
            [None, None] => vec![None, None],
        }
    }

    fn hvp(
        &self,
        _: &Self::VjpState,
        state: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        self.vjp(state, omega)
    }
}

/// Division of a vectorspace element by a scalar.
///
/// Input convention: (scalar, vector)
#[derive(Debug, Clone, Copy)]
pub struct ScalarDivision;

pub enum ScalarDivisionVjpState<B: Backend> {
    Neither,
    Left([Tensor<B, 2>; 2]),
    Right(Tensor<B, 2>),
    Both([Tensor<B, 2>; 2]),
}

impl<B: Backend> DiffOperation<B> for ScalarDivision {
    type VjpState = ScalarDivisionVjpState<B>;
    type HvpState = [Option<Tensor<B, 2>>; 2];

    fn forward(
        &self,
        points: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let points: Vec<Point<B>> = points.into_iter().map(Input::into_float).collect();
        let [alpha, x] = points.try_into().unwrap();
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(x / alpha),
            DiffModeState::Jvp(tvecs) => {
                let [dalpha, dx] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    x.clone() / alpha.clone(),
                    sum_opt_tensors([
                        dalpha.map(|dalpha| {
                            -dalpha * (x.clone() / alpha.clone().square()).unsqueeze_dim(2)
                        }),
                        dx.map(|dx| dx / alpha.unsqueeze_dim(2)),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [alphadiff, xdiff] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    x.clone() / alpha.clone(),
                    match (alphadiff, xdiff) {
                        (false, false) => ScalarDivisionVjpState::Neither,
                        (true, false) => ScalarDivisionVjpState::Left([alpha, x]),
                        (false, true) => ScalarDivisionVjpState::Right(alpha),
                        (true, true) => ScalarDivisionVjpState::Both([alpha, x]),
                    },
                )
            }
            DiffModeState::Hvp {
                diffs,
                tangents: cvecs,
            } => {
                let [dalpha, dx] = cvecs.try_into().unwrap();
                let [alphadiff, xdiff] = diffs.try_into().unwrap();
                DiffOperationForward::hvp(
                    x.clone() / alpha.clone(),
                    sum_opt_tensors([
                        dalpha
                            .as_ref()
                            .map(|dalpha| -dalpha.clone() * (x.clone() / alpha.clone().square())),
                        dx.as_ref().map(|dx| dx.clone() / alpha.clone()),
                    ]),
                    match (alphadiff, xdiff) {
                        (false, false) => ScalarDivisionVjpState::Neither,
                        (true, false) => ScalarDivisionVjpState::Left([alpha, x]),
                        (false, true) => ScalarDivisionVjpState::Right(alpha),
                        (true, true) => ScalarDivisionVjpState::Both([alpha, x]),
                    },
                    [dalpha, dx],
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
            ScalarDivisionVjpState::Neither => vec![None, None],
            ScalarDivisionVjpState::Left([alpha, x]) => vec![
                Some(
                    -(x.clone() / alpha.clone().square())
                        .unsqueeze_dim(1)
                        .matmul(omega),
                ),
                None,
            ],
            ScalarDivisionVjpState::Right(alpha) => {
                vec![None, Some(omega / alpha.clone().unsqueeze_dim(2))]
            }
            ScalarDivisionVjpState::Both([alpha, x]) => vec![
                Some(
                    -(x.clone() / alpha.clone().square())
                        .unsqueeze_dim(1)
                        .matmul(omega.clone()),
                ),
                Some(omega / alpha.clone().unsqueeze_dim(2)),
            ],
        }
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        match (hvp_state, vjp_state) {
            ([Some(dalpha), Some(dx)], ScalarDivisionVjpState::Both([alpha, x])) => {
                let alpha2 = alpha.clone().square();
                vec![
                    Some(
                        (dalpha.clone().mul_scalar(2.0) * x.clone()
                            / (alpha.clone() * alpha2.clone())
                            - dx.clone() / alpha.clone().square())
                        .unsqueeze_dim(1)
                        .matmul(omega.clone()),
                    ),
                    Some(-(dalpha.clone() / alpha2).unsqueeze_dim(2) * omega),
                ]
            }
            ([Some(dalpha), None], ScalarDivisionVjpState::Both([alpha, x])) => {
                let alpha2 = alpha.clone().square();
                vec![
                    Some(
                        (dalpha.clone().mul_scalar(2.0) * x.clone()
                            / (alpha.clone() * alpha2.clone()))
                        .unsqueeze_dim(1)
                        .matmul(omega.clone()),
                    ),
                    Some(-(dalpha.clone() / alpha2).unsqueeze_dim(2) * omega),
                ]
            }
            ([None, Some(dx)], ScalarDivisionVjpState::Both([alpha, _])) => {
                vec![
                    Some(
                        (-dx.clone() / alpha.clone().square())
                            .unsqueeze_dim(1)
                            .matmul(omega.clone()),
                    ),
                    Some(omega.zeros_like()),
                ]
            }
            (_, ScalarDivisionVjpState::Both(_)) => {
                let [batches, _, frames] = omega.dims();
                vec![
                    Some(Tensor::<B, 3>::zeros([batches, 1, frames], &omega.device())),
                    Some(omega.zeros_like()),
                ]
            }
            (_, ScalarDivisionVjpState::Left(_)) => {
                let [batches, _, frames] = omega.dims();
                vec![
                    Some(Tensor::<B, 3>::zeros([batches, 1, frames], &omega.device())),
                    None,
                ]
            }
            (_, ScalarDivisionVjpState::Right(_)) => {
                vec![None, Some(omega.zeros_like())]
            }
            (_, ScalarDivisionVjpState::Neither) => vec![None, None],
        }
    }
}

/// Literal addition of a float on a (scalar) quantity.
#[derive(Debug, Clone, Copy)]
pub struct LiteralAddition<B: Backend>(B::FloatElem);

impl<B: Backend> LiteralAddition<B> {
    pub fn new(val: B::FloatElem) -> Self {
        Self(val)
    }
}

impl<B: Backend> DiffOperation<B> for LiteralAddition<B> {
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
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::hvp(x + y, dx, diffs[0], ())
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

/// Literal subtraction of a float from a (scalar) quantity.
#[derive(Debug, Clone, Copy)]
pub struct LiteralSubtractionRight<B: Backend>(B::FloatElem);

impl<B: Backend> LiteralSubtractionRight<B> {
    pub fn new(val: B::FloatElem) -> Self {
        Self(val)
    }
}

impl<B: Backend> DiffOperation<B> for LiteralSubtractionRight<B> {
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
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::hvp(x - y, dx, diffs[0], ())
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

/// Literal subtraction of (scalar) quantity from a float.
#[derive(Debug, Clone, Copy)]
pub struct LiteralSubtractionLeft<B: Backend>(B::FloatElem);

impl<B: Backend> LiteralSubtractionLeft<B> {
    pub fn new(val: B::FloatElem) -> Self {
        Self(val)
    }
}

impl<B: Backend> DiffOperation<B> for LiteralSubtractionLeft<B> {
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
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(-(x - y)),
            DiffModeState::Jvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(-(x - y), dx.map(|dx| -dx))
            }
            DiffModeState::Vjp(diffs) => DiffOperationForward::vjp(-(x - y), diffs[0]),
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::hvp(-(x - y), dx.map(|dx| -dx), diffs[0], ())
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![if *state { Some(-omega) } else { None }]
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

/// Literal scalar multiplication on a vector space.
#[derive(Debug, Clone, Copy)]
pub struct LiteralScalarMultiplication<B: Backend>(B::FloatElem);

impl<B: Backend> LiteralScalarMultiplication<B> {
    pub fn new(val: B::FloatElem) -> Self {
        Self(val)
    }
}

impl<B: Backend> DiffOperation<B> for LiteralScalarMultiplication<B> {
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
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::hvp(
                    x * y,
                    sum_opt_tensors([dx.as_ref().map(|dx| dx.clone() * y)]),
                    diffs[0],
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

/// Literal scalar division on a vector space.
#[derive(Debug, Clone, Copy)]
pub struct LiteralScalarDivisionRight<B: Backend>(B::FloatElem);

impl<B: Backend> LiteralScalarDivisionRight<B> {
    pub fn new(val: B::FloatElem) -> Self {
        Self(val)
    }
}

impl<B: Backend> DiffOperation<B> for LiteralScalarDivisionRight<B> {
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
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::hvp(
                    x / self.0,
                    dx.as_ref().map(|dx| dx.clone() / self.0),
                    diffs[0],
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

/// Division of a scalar quantity from a literal value.
#[derive(Debug, Clone, Copy)]
pub struct LiteralScalarDivisionLeft<B: Backend>(B::FloatElem);

impl<B: Backend> LiteralScalarDivisionLeft<B> {
    pub fn new(val: B::FloatElem) -> Self {
        Self(val)
    }
}

impl<B: Backend> DiffOperation<B> for LiteralScalarDivisionLeft<B> {
    type VjpState = Option<Tensor<B, 2>>;
    type HvpState = Option<Tensor<B, 2>>;

    fn forward(
        &self,
        points: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let points: Vec<Point<B>> = points.into_iter().map(Input::into_float).collect();
        let [alpha] = points.try_into().unwrap();
        match diff_state {
            DiffModeState::ForwardOnly => {
                DiffOperationForward::forward_only(alpha.recip() * self.0)
            }
            DiffModeState::Jvp(tvecs) => {
                let [dalpha] = tvecs.try_into().unwrap();
                let invalpha = alpha.recip();
                DiffOperationForward::jvp(
                    invalpha.clone() * self.0,
                    dalpha.map(|dalpha| -(dalpha * self.0) * invalpha.square().unsqueeze_dim(2)),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [alphadiff] = diffs.try_into().unwrap();
                let invalpha = alpha.recip();
                DiffOperationForward::vjp(
                    invalpha.clone() * self.0,
                    if alphadiff { Some(invalpha) } else { None },
                )
            }
            DiffModeState::Hvp {
                diffs,
                tangents: cvecs,
            } => {
                let [dalpha] = cvecs.try_into().unwrap();
                let [alphadiff] = diffs.try_into().unwrap();
                let invalpha = alpha.recip();
                DiffOperationForward::hvp(
                    invalpha.clone() * self.0,
                    dalpha
                        .as_ref()
                        .map(|dalpha| -(dalpha.clone() * self.0) * invalpha.clone().square()),
                    if alphadiff { Some(invalpha) } else { None },
                    dalpha,
                )
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
                .map(|invalpha| -(omega * self.0) * invalpha.clone().square().unsqueeze_dim(2)),
        ]
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        match (hvp_state, vjp_state) {
            (Some(dalpha), Some(invalpha)) => vec![Some(
                (dalpha.clone().mul_scalar(2.0) * self.0
                    / (invalpha.clone() * invalpha.clone().square()))
                .unsqueeze_dim(2)
                    * omega,
            )],
            (None, Some(_)) => vec![Some(omega.zeros_like())],
            (_, None) => vec![None],
        }
    }
}

/// Concatenation of tangent blocks along the component axis.
///
/// `[b, d₀] … [b, dₙ] ↦ [b, Σdᵢ]`, in the order the inputs are given. This is what expresses a
/// joint over several quantities without a product manifold: each quantity's own
/// inverse-retraction produces its tangent block, and this stacks them into one flat residual
/// whose manifold is an ordinary `EuclideanD(Σdᵢ)`.
///
/// The op holds the per-block widths because the pullback needs them and the cotangent frame
/// alone does not carry them: `vjp` slices `ω` back into the same blocks. The map is linear, so
/// the second-order term is zero -- but each slot is still gated on its own `VjpState` slot
/// rather than returned as a blanket `None`, so a block rooted only in constants stays `None`
/// while a differentiable one gets its zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TangentConcat(pub Vec<usize>);

impl TangentConcat {
    /// The half-open component range block `i` occupies in the output.
    fn block(&self, i: usize) -> std::ops::Range<usize> {
        let offset: usize = self.0[..i].iter().sum();
        offset..offset + self.0[i]
    }

    /// Assemble the output from per-input blocks, zero-filling the ones that are absent.
    ///
    /// A missing block cannot be skipped: it occupies real columns of the output, so it has to
    /// become zeros of its own width. This is `cat`, never `sum_opt_tensors` -- these are
    /// *different blocks* of one vector, not contributions to the same one.
    fn assemble<B: Backend, const D: usize>(
        &self,
        blocks: Vec<Option<Tensor<B, D>>>,
        reference: &Tensor<B, D>,
    ) -> Tensor<B, D> {
        let mut dims = reference.dims();
        let parts: Vec<Tensor<B, D>> = blocks
            .into_iter()
            .zip(&self.0)
            .map(|(block, &width)| {
                block.unwrap_or_else(|| {
                    dims[1] = width;
                    Tensor::zeros(dims, &reference.device())
                })
            })
            .collect();
        Tensor::cat(parts, 1)
    }
}

impl<B: Backend> DiffOperation<B> for TangentConcat {
    /// One flag per input, gated on `diffs` -- never on whether a tangent was seeded.
    type VjpState = Vec<bool>;
    type HvpState = ();

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::into_float).collect();
        assert_eq!(
            inputs.len(),
            self.0.len(),
            "TangentConcat: {} inputs for {} declared blocks",
            inputs.len(),
            self.0.len()
        );
        let output = Tensor::cat(inputs.clone(), 1);
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(output),
            DiffModeState::Jvp(tangents) => {
                // Concatenation is its own pushforward: the tangent of a stacked vector is the
                // stack of the tangents. Absent blocks contribute zeros of their own width.
                let tangent = tangents
                    .iter()
                    .flatten()
                    .next()
                    .map(|reference| self.assemble(tangents.clone(), reference));
                DiffOperationForward::jvp(output, tangent)
            }
            DiffModeState::Vjp(diffs) => DiffOperationForward::vjp(output, diffs),
            DiffModeState::Hvp { diffs, tangents } => {
                let tangent = tangents
                    .iter()
                    .flatten()
                    .next()
                    .map(|reference| self.assemble(tangents.clone(), reference));
                DiffOperationForward::hvp(output, tangent, diffs, ())
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        state
            .iter()
            .enumerate()
            .map(|(i, &diff)| diff.then(|| omega.clone().slice_dim(1, self.block(i))))
            .collect()
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        _: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        // Linear, so `∇_v dF*_p` vanishes. Each slot is still gated on the corresponding
        // `VjpState` slot: if the first-order cotangent of an input is `None`, its second-order
        // one must be too.
        vjp_state
            .iter()
            .enumerate()
            .map(|(i, &diff)| {
                diff.then(|| Tensor::zeros_like(&omega.clone().slice_dim(1, self.block(i))))
            })
            .collect()
    }
}
