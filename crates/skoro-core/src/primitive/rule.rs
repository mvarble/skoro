//! A primitive as a rule over burn tensors, and the one kernel that runs every rule.
//!
//! A rule gives the forward map and, input by input, the tangent, cotangent and second-order
//! maps of the tables of primitives. The kernel applies the two gates: it pulls back only into
//! the inputs marked differentiable, and a missing tangent or velocity counts as zero.

use std::sync::Arc;

use skoro_tensor::burn::Tensor;
use skoro_tensor::{Frame, Points, Target, Value};

use crate::{FirstOrder, Forward, Kernel, Pullback, SecondOrder, SecondPullback};

/// A primitive's maps on tensors: values `[N, storage]`, frames `[N, dimension, K]`. A discrete
/// input, such as a selection's index, comes as a float tensor at the target's precision; it is
/// never differentiable, so `push`, `pull` and `second` are never asked of it.
pub(crate) trait Rule: Send + Sync + 'static {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2>;
    /// `∂_j f t`: the tangent frame `t` of input `j`, pushed to the output.
    fn push(&self, x: &[Tensor<2>], j: usize, t: Tensor<3>) -> Tensor<3>;
    /// `∂_j fᵀ α`.
    fn pull(&self, x: &[Tensor<2>], j: usize, alpha: Tensor<3>) -> Tensor<3>;
    /// `hvp_j(α)` at the velocities `v`, one per input and `None` for zero, with `α` of one
    /// column; `None` when it is zero, as it is for every linear rule.
    fn second(
        &self,
        _x: &[Tensor<2>],
        _j: usize,
        _v: &[Option<Tensor<3>>],
        _alpha: Tensor<3>,
    ) -> Option<Tensor<3>> {
        None
    }
}

/// The kernel of order two that runs a rule on one target.
pub(crate) struct RuleKernel<R> {
    rule: Arc<R>,
    target: Target,
}

impl<R: Rule> RuleKernel<R> {
    pub(crate) fn kernel(rule: R, target: &Target) -> Kernel {
        Kernel::Two(Arc::new(RuleKernel {
            rule: Arc::new(rule),
            target: target.clone(),
        }))
    }

    fn value(&self, x: &[Tensor<2>]) -> Value {
        value(self.rule.value(x), &self.target)
    }

    fn tensors(&self, inputs: &[Value]) -> Vec<Tensor<2>> {
        inputs
            .iter()
            .map(|input| match input {
                Value::Continuous(points) => points.clone().into_burn(),
                Value::Discrete(integers) => integers.to_points(&self.target).into_burn(),
            })
            .collect()
    }

    fn push_all(&self, x: &[Tensor<2>], tangents: &[Option<Frame>]) -> Option<Frame> {
        let pushed = tangents
            .iter()
            .enumerate()
            .filter_map(|(j, t)| t.as_ref().map(|t| self.rule.push(x, j, burn(t))))
            .reduce(|a, b| a + b)?;
        Some(frame(pushed, &self.target))
    }

    fn pullback(&self, x: Vec<Tensor<2>>, differentiable: &[bool]) -> RulePullback<R> {
        RulePullback {
            rule: self.rule.clone(),
            target: self.target.clone(),
            x,
            differentiable: differentiable.to_vec(),
            velocities: Vec::new(),
        }
    }
}

impl<R: Rule> Forward for RuleKernel<R> {
    fn forward(&self, inputs: &[Value]) -> Value {
        self.value(&self.tensors(inputs))
    }
}

impl<R: Rule> FirstOrder for RuleKernel<R> {
    fn tangent(&self, inputs: &[Value], tangents: &[Option<Frame>]) -> (Value, Option<Frame>) {
        let x = self.tensors(inputs);
        (self.value(&x), self.push_all(&x, tangents))
    }

    fn record(&self, inputs: &[Value], differentiable: &[bool]) -> (Value, Box<dyn Pullback>) {
        let x = self.tensors(inputs);
        (self.value(&x), Box::new(self.pullback(x, differentiable)))
    }
}

impl<R: Rule> SecondOrder for RuleKernel<R> {
    fn record_second(
        &self,
        inputs: &[Value],
        differentiable: &[bool],
        velocities: &[Option<Frame>],
    ) -> (Value, Option<Frame>, Box<dyn SecondPullback>) {
        let x = self.tensors(inputs);
        let velocity = self.push_all(&x, velocities);
        let value = self.value(&x);
        let mut pullback = self.pullback(x, differentiable);
        pullback.velocities = velocities.iter().map(|v| v.as_ref().map(burn)).collect();
        (value, velocity, Box::new(pullback))
    }
}

struct RulePullback<R> {
    rule: Arc<R>,
    target: Target,
    x: Vec<Tensor<2>>,
    differentiable: Vec<bool>,
    velocities: Vec<Option<Tensor<3>>>,
}

impl<R: Rule> Pullback for RulePullback<R> {
    fn cotangent(&self, alpha: &Frame) -> Vec<Option<Frame>> {
        let alpha = burn(alpha);
        (0..self.x.len())
            .map(|j| {
                self.differentiable[j]
                    .then(|| frame(self.rule.pull(&self.x, j, alpha.clone()), &self.target))
            })
            .collect()
    }
}

impl<R: Rule> SecondPullback for RulePullback<R> {
    fn second(&self, alpha: &Frame) -> Vec<Option<Frame>> {
        let alpha = burn(alpha);
        (0..self.x.len())
            .map(|j| {
                if !self.differentiable[j] {
                    return None;
                }
                let second = self
                    .rule
                    .second(&self.x, j, &self.velocities, alpha.clone())?;
                Some(frame(second, &self.target))
            })
            .collect()
    }
}

fn burn(frame: &Frame) -> Tensor<3> {
    frame.clone().into_burn()
}

pub(crate) fn value(tensor: Tensor<2>, target: &Target) -> Value {
    match Points::from_burn(tensor, target) {
        Ok(points) => Value::Continuous(points),
        Err(error) => unreachable!("a kernel keeps its target's precision and device: {error}"),
    }
}

pub(crate) fn frame(tensor: Tensor<3>, target: &Target) -> Frame {
    match Frame::from_burn(tensor, target) {
        Ok(frame) => frame,
        Err(error) => unreachable!("a kernel keeps its target's precision and device: {error}"),
    }
}

/// `[N, d]` as `[N, d, K]`, the same in every column.
pub(crate) fn columns(x: &Tensor<2>, k: usize) -> Tensor<3> {
    let [n, d] = x.dims();
    x.clone().unsqueeze_dim::<3>(2).expand([n, d, k])
}

/// A frame of one column, `[N, d, 1]`, repeated to `k` columns.
pub(crate) fn widen(alpha: Tensor<3>, k: usize) -> Tensor<3> {
    let [n, d, _] = alpha.dims();
    alpha.expand([n, d, k])
}
