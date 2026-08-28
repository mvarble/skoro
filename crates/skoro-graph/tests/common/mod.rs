//! Shared fixtures for the `skoro-graph` integration tests.
//!
//! One backend for the whole crate's tests: `NdArrayBackend<f64, i32>`. The f64 is not incidental
//! -- the forward-difference checks in `skoro_graph::test_utils` need the headroom, and mixing an
//! f32 and an f64 NdArray backend in one test binary silently demotes the DType for both.
//! Tensors are built through `TensorData` for the same reason: `Tensor::from_floats` rounds
//! through f32 even on an f64 backend.

#![allow(dead_code)]

use std::collections::HashMap;

use skoro_backend::{Int, NdArrayBackend, Tensor, TensorData};
use skoro_core::{id::Id, kind::Continuous, kind::Discrete, sampler::Sampler};
use skoro_graph::{graph::Graph, operation::DiffGraphOperation, point_map::PointMap};
use skoro_graph::{operation::DiscreteGraphOperation, quantity::QuantityType};
use skoro_operation::{
    Input, ValueMap,
    discrete::DiscreteOperation,
    ops::{Addition, ScalarMultiplication},
};
use skoro_riemann::{euclidean::EuclideanD, real::Real};

pub type B = NdArrayBackend<f64, i32>;

pub const DELTA: f64 = 1e-6;
pub const EPSILON: f64 = 1e-5;

// Tensor helpers

/// A `[batch, components]` float tensor from row-major data.
pub fn tensor2(data: Vec<f64>, batch: usize, components: usize) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(data, [batch, components]),
        &Default::default(),
    )
}

/// A batch-1 scalar, the shape most of these fixtures work in.
pub fn scalar(v: f64) -> Tensor<B, 2> {
    tensor2(vec![v], 1, 1)
}

/// A `[batch, components, frames]` float tensor from row-major data.
pub fn tensor3(data: Vec<f64>, dims: [usize; 3]) -> Tensor<B, 3> {
    Tensor::from_data(TensorData::new(data, dims), &Default::default())
}

pub fn int2(data: Vec<i32>, batch: usize, components: usize) -> Tensor<B, 2, Int> {
    Tensor::from_data(
        TensorData::new(data, [batch, components]),
        &Default::default(),
    )
}

pub fn into_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f64> {
    t.into_data().to_vec::<f64>().unwrap()
}

/// A `ValueMap` from `(id, scalar)` pairs.
pub fn points(pairs: impl IntoIterator<Item = (Id, f64)>) -> ValueMap<B> {
    pairs
        .into_iter()
        .map(|(id, v)| (id, Input::Float(scalar(v))))
        .collect()
}

pub fn frames(pairs: impl IntoIterator<Item = (Id, Tensor<B, 3>)>) -> HashMap<Id, Tensor<B, 3>> {
    pairs.into_iter().collect()
}

pub fn vectors(pairs: impl IntoIterator<Item = (Id, Tensor<B, 2>)>) -> HashMap<Id, Tensor<B, 2>> {
    pairs.into_iter().collect()
}

/// The erased quantity type for `Real`, which several fixtures need by hand.
pub fn real_dyn() -> std::sync::Arc<dyn skoro_graph::quantity::DynQuantityType<B>> {
    <Real as QuantityType<B, Continuous>>::into_dyn(Real)
}

// Operations

/// A sampler that ignores its parents and returns a fixed value.
///
/// Deterministic on purpose: `generate` is checked for *what it routes where*, not for its
/// randomness, and a fixture that returned noise could not be asserted on at all.
#[derive(Debug, Clone)]
pub struct ConstantSampler(pub f64);

impl Sampler<B, Continuous> for ConstantSampler {
    fn sample(&self, batch_size: usize, _parameters: Vec<Input<B>>) -> Tensor<B, 2> {
        tensor2(vec![self.0; batch_size], batch_size, 1)
    }
}

/// A sampler that reads its (single) continuous parent and shifts it.
///
/// The only fixture that proves a sampler actually receives its parents' values.
#[derive(Debug, Clone)]
pub struct ShiftSampler(pub f64);

impl Sampler<B, Continuous> for ShiftSampler {
    fn sample(&self, _batch_size: usize, parameters: Vec<Input<B>>) -> Tensor<B, 2> {
        let [parent] = <[Input<B>; 1]>::try_from(parameters).expect("one parent");
        parent.unwrap_float().add_scalar(self.0)
    }
}

/// A discrete sampler, for the discrete half of the sampler surface.
#[derive(Debug, Clone)]
pub struct ConstantIntSampler(pub i32);

impl Sampler<B, Discrete> for ConstantIntSampler {
    fn sample(&self, batch_size: usize, _parameters: Vec<Input<B>>) -> Tensor<B, 2, Int> {
        int2(vec![self.0; batch_size], batch_size, 1)
    }
}

/// Rounds a continuous input to a discrete one.
///
/// The "discretization bridge" the `DiscreteOperation` docs describe: continuous in, integer out,
/// and never differentiable.
#[derive(Debug, Clone)]
pub struct RoundToInt;

impl DiscreteOperation<B> for RoundToInt {
    fn forward(&self, inputs: Vec<Input<B>>) -> Tensor<B, 2, Int> {
        let [x] = <[Input<B>; 1]>::try_from(inputs).expect("one input");
        x.unwrap_float().round().int()
    }
}

/// A differentiable operation reading one continuous and one discrete input: `(x, n) ↦ x + n`.
///
/// The mixed-kind case the `DiffOperation` docs describe. Most continuous operations convert
/// their whole input list with `unwrap_float` up front and would panic on a discrete parent; this
/// one unwraps each input according to its own kind, which is what a discretization-aware
/// operation has to do. The derivative with respect to `n` does not exist -- `n` has no tangent
/// space -- so that slot is always `None` rather than zero.
#[derive(Debug, Clone, Copy)]
pub struct ShiftByInt;

impl skoro_operation::diff::DiffOperation<B> for ShiftByInt {
    /// Whether the continuous input is differentiable. The discrete one never is.
    type VjpState = bool;
    type HvpState = ();

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: skoro_operation::diff::DiffModeState<B>,
    ) -> skoro_operation::diff::DiffOperationForward<B, bool, ()> {
        use skoro_operation::diff::{DiffModeState, DiffOperationForward};

        let [x, n] = <[Input<B>; 2]>::try_from(inputs).expect("two inputs");
        let out = x.unwrap_float() + n.unwrap_int().float();
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(out),
            DiffModeState::Jvp(tangents) => {
                let [dx, _] = <[_; 2]>::try_from(tangents).unwrap();
                DiffOperationForward::jvp(out, dx)
            }
            DiffModeState::Vjp(diffs) => DiffOperationForward::vjp(out, diffs[0]),
            DiffModeState::Hvp { diffs, tangents } => {
                let [dx, _] = <[_; 2]>::try_from(tangents).unwrap();
                DiffOperationForward::hvp(out, dx, diffs[0], ())
            }
        }
    }

    fn vjp(&self, xdiff: &bool, omega: Tensor<B, 3>) -> Vec<Option<Tensor<B, 3>>> {
        vec![xdiff.then_some(omega), None]
    }

    fn hvp(&self, xdiff: &bool, _: &(), omega: Tensor<B, 3>) -> Vec<Option<Tensor<B, 3>>> {
        // Linear in `x`, so the second-order term vanishes -- but it is still *present* wherever
        // the first-order one is.
        vec![xdiff.then(|| omega.zeros_like()), None]
    }
}

/// A differentiable operation whose modes are all `unimplemented!()`.
///
/// For topology tests, which never evaluate. Reaching a method on this is itself the bug.
#[derive(Debug, Clone)]
pub struct InertOperation;

impl skoro_operation::diff::DiffOperation<B> for InertOperation {
    type VjpState = ();
    type HvpState = ();
    fn forward(
        &self,
        _: Vec<Input<B>>,
        _: skoro_operation::diff::DiffModeState<B>,
    ) -> skoro_operation::diff::DiffOperationForward<B, (), ()> {
        unimplemented!("InertOperation is for topology fixtures only")
    }
    fn vjp(&self, _: &(), _: Tensor<B, 3>) -> Vec<Option<Tensor<B, 3>>> {
        unimplemented!()
    }
    fn hvp(&self, _: &(), _: &(), _: Tensor<B, 3>) -> Vec<Option<Tensor<B, 3>>> {
        unimplemented!()
    }
}

/// Sums its inputs. Evaluates for real, so constant folding can be exercised.
#[derive(Debug, Clone)]
pub struct SumOperation;

impl skoro_operation::diff::DiffOperation<B> for SumOperation {
    type VjpState = ();
    type HvpState = ();
    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: skoro_operation::diff::DiffModeState<B>,
    ) -> skoro_operation::diff::DiffOperationForward<B, (), ()> {
        let sum = inputs
            .into_iter()
            .map(Input::unwrap_float)
            .reduce(|a, b| a + b)
            .expect("at least one input");
        match diff_state {
            skoro_operation::diff::DiffModeState::ForwardOnly => {
                skoro_operation::diff::DiffOperationForward::forward_only(sum)
            }
            _ => unimplemented!("constant folding only evaluates forward-only"),
        }
    }
    fn vjp(&self, _: &(), _: Tensor<B, 3>) -> Vec<Option<Tensor<B, 3>>> {
        unimplemented!()
    }
    fn hvp(&self, _: &(), _: &(), _: Tensor<B, 3>) -> Vec<Option<Tensor<B, 3>>> {
        unimplemented!()
    }
}

// Models

/// `a`, `b` free; `sum = a + b`; `prod = a * b`; `sq = sum * sum`.
///
/// Small enough to differentiate by hand and wide enough to have a non-symmetric Hessian:
/// `∂²prod/∂a∂b = 1` while `∂²prod/∂a² = 0`, and `sq` puts a nonlinearity two levels deep.
pub struct Arithmetic {
    pub graph: Graph<B>,
    pub a: Id,
    pub b: Id,
    pub sum: Id,
    pub prod: Id,
    pub sq: Id,
}

impl Arithmetic {
    pub fn new() -> Self {
        let mut graph = Graph::<B>::new();
        let a = graph.add_parameter(Real);
        let b = graph.add_parameter(Real);
        let sum = graph
            .add_mapped(Real, DiffGraphOperation(Addition), vec![a, b])
            .unwrap();
        let prod = graph
            .add_mapped(Real, DiffGraphOperation(ScalarMultiplication), vec![a, b])
            .unwrap();
        let sq = graph
            .add_mapped(
                Real,
                DiffGraphOperation(ScalarMultiplication),
                vec![sum, sum],
            )
            .unwrap();
        Self {
            graph,
            a,
            b,
            sum,
            prod,
            sq,
        }
    }

    /// The canonical evaluation point, `a = 3`, `b = 5`.
    pub fn point(&self) -> ValueMap<B> {
        points([(self.a, 3.0), (self.b, 5.0)])
    }
}

/// A `EuclideanD(3)` parameter scaled by a `Real` one: `y = s · v`.
///
/// The only fixture where the input and output tangent spaces have different, non-unit
/// dimensions, which is what makes a transposed Jacobian visible.
pub struct Scaling {
    pub graph: Graph<B>,
    pub s: Id,
    pub v: Id,
    pub y: Id,
}

impl Scaling {
    pub fn new() -> Self {
        let mut graph = Graph::<B>::new();
        let s = graph.add_parameter(Real);
        let v = graph.add_parameter(EuclideanD(3));
        let y = graph
            .add_mapped(
                EuclideanD(3),
                DiffGraphOperation(ScalarMultiplication),
                vec![s, v],
            )
            .unwrap();
        Self { graph, s, v, y }
    }

    pub fn point(&self) -> ValueMap<B> {
        let mut map = ValueMap::<B>::new();
        map.insert(self.s, Input::Float(scalar(2.0)));
        map.insert(self.v, Input::Float(tensor2(vec![1.0, -2.0, 4.0], 1, 3)));
        map
    }
}

/// A `PointMap` holding `(id, scalar)` pairs, all of `Real` type.
pub fn real_point_map(pairs: impl IntoIterator<Item = (Id, f64)>) -> PointMap<B> {
    let mut out = PointMap::<B>::default();
    for (id, v) in pairs {
        out.insert(id, Input::Float(scalar(v)), real_dyn());
    }
    out
}

/// Marker so the fixtures above are reachable from a test that only wants the discrete adapter.
pub fn discrete_round_op() -> DiscreteGraphOperation<RoundToInt> {
    DiscreteGraphOperation(RoundToInt)
}
