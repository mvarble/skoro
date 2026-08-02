//! Built-in [`DiscreteOperation`]s backing the discrete `Q` operation library.
//!
//! Each kernel is a small, forward-only function `Vec<Input> ->
//! Tensor<Int>`. Inputs may be discrete (the common case) or continuous (for the
//! discretization bridges).

use skoro_backend::{Backend, DiscretePoint};
use skoro_operation::{Input, discrete::DiscreteOperation};

/// Elementwise integer addition of two discrete inputs.
#[derive(Debug, Clone, Copy)]
pub struct IntAdd;

impl<B: Backend> DiscreteOperation<B> for IntAdd {
    fn forward(&self, inputs: Vec<Input<B>>) -> DiscretePoint<B> {
        let [x, y] = inputs.try_into().unwrap();
        x.into_int() + y.into_int()
    }
}

/// Elementwise integer subtraction (`x - y`).
#[derive(Debug, Clone, Copy)]
pub struct IntSub;

impl<B: Backend> DiscreteOperation<B> for IntSub {
    fn forward(&self, inputs: Vec<Input<B>>) -> DiscretePoint<B> {
        let [x, y] = inputs.try_into().unwrap();
        x.into_int() - y.into_int()
    }
}

/// Elementwise integer multiplication.
#[derive(Debug, Clone, Copy)]
pub struct IntMul;

impl<B: Backend> DiscreteOperation<B> for IntMul {
    fn forward(&self, inputs: Vec<Input<B>>) -> DiscretePoint<B> {
        let [x, y] = inputs.try_into().unwrap();
        x.into_int() * y.into_int()
    }
}

/// The comparison performed by [`IntCompare`].
#[derive(Debug, Clone, Copy)]
pub enum Comparison {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Elementwise integer comparison producing a boolean (`0`/`1`) tensor.
#[derive(Debug, Clone, Copy)]
pub struct IntCompare(pub Comparison);

impl<B: Backend> DiscreteOperation<B> for IntCompare {
    fn forward(&self, inputs: Vec<Input<B>>) -> DiscretePoint<B> {
        let [x, y] = inputs.try_into().unwrap();
        let x = x.into_int();
        let y = y.into_int();
        let mask = match self.0 {
            Comparison::Eq => x.equal(y),
            Comparison::Ne => x.equal(y).bool_not(),
            Comparison::Lt => x.lower(y),
            Comparison::Le => x.lower_equal(y),
            Comparison::Gt => x.greater(y),
            Comparison::Ge => x.greater_equal(y),
        };
        mask.int()
    }
}

/// The logical operation performed by [`BoolLogic`].
#[derive(Debug, Clone, Copy)]
pub enum Logic {
    And,
    Or,
}

/// Elementwise boolean logic on two `0`/`1` integer tensors.
#[derive(Debug, Clone, Copy)]
pub struct BoolLogic(pub Logic);

impl<B: Backend> DiscreteOperation<B> for BoolLogic {
    fn forward(&self, inputs: Vec<Input<B>>) -> DiscretePoint<B> {
        let [x, y] = inputs.try_into().unwrap();
        let x = x.into_int().bool();
        let y = y.into_int().bool();
        let out = match self.0 {
            Logic::And => x.bool_and(y),
            Logic::Or => x.bool_or(y),
        };
        out.int()
    }
}

/// Elementwise boolean negation of a `0`/`1` integer tensor.
#[derive(Debug, Clone, Copy)]
pub struct BoolNot;

impl<B: Backend> DiscreteOperation<B> for BoolNot {
    fn forward(&self, inputs: Vec<Input<B>>) -> DiscretePoint<B> {
        let [x] = inputs.try_into().unwrap();
        x.into_int().bool().bool_not().int()
    }
}
