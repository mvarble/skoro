//! The discrete operations, the embedding of a discrete value as a real, and the discretization
//! of a real by its sign: all of order zero, carrying no derivatives.

use std::sync::Arc;

use skoro_tensor::burn::{Bool, Int, Tensor};
use skoro_tensor::{Integers, Target, Value};

use crate::{Error, Forward, Kernel, Lowered, Lowering, Operation, Param, Space};

/// What an order-zero kernel computes, on its target.
type Compute = fn(&[Value], &Target) -> Value;

struct Discrete {
    compute: Compute,
    target: Target,
}

impl Forward for Discrete {
    fn forward(&self, inputs: &[Value]) -> Value {
        (self.compute)(inputs, &self.target)
    }
}

fn integers(value: &Value) -> Tensor<2, Int> {
    match value {
        Value::Discrete(integers) => integers.clone().into_burn(),
        Value::Continuous(_) => unreachable!("checked when the node was added: a discrete input"),
    }
}

fn points(value: &Value) -> Tensor<2> {
    match value {
        Value::Continuous(points) => points.clone().into_burn(),
        Value::Discrete(_) => unreachable!("checked when the node was added: a continuous input"),
    }
}

fn discrete(tensor: Tensor<2, Int>, target: &Target) -> Value {
    match Integers::from_burn(tensor, target) {
        Ok(integers) => Value::Discrete(integers),
        Err(error) => unreachable!("integer arithmetic keeps its target's integers: {error}"),
    }
}

fn boolean(mask: Tensor<2, Bool>, target: &Target) -> Value {
    match Integers::from_mask(mask, target) {
        Ok(integers) => Value::Discrete(integers),
        Err(error) => unreachable!("a comparison stays on its target's device: {error}"),
    }
}

macro_rules! discrete {
    ($(#[$doc:meta])* $ty:ident, $name:literal, $output:expr, $compute:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub struct $ty;

        impl Operation for $ty {
            fn name(&self) -> &'static str {
                $name
            }

            fn params(&self) -> Vec<Param> {
                Vec::new()
            }

            fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
                let output: fn(&[Space]) -> Option<Space> = $output;
                output(inputs).ok_or_else(|| Error::SpaceMismatch {
                    operation: $name,
                    inputs: inputs.to_vec(),
                })
            }

            fn lower(
                &self,
                inputs: &[Space],
                lowering: &mut Lowering<'_>,
            ) -> Result<Lowered, Error> {
                self.output(inputs)?;
                let kernel = Discrete {
                    compute: $compute,
                    target: lowering.target().clone(),
                };
                Ok(Lowered::Kernel(Kernel::Zero(Arc::new(kernel))))
            }
        }
    };
}

fn two_integers(inputs: &[Space], output: Space) -> Option<Space> {
    matches!(inputs, [Space::Integer, Space::Integer]).then_some(output)
}

discrete!(
    /// `i + j`.
    IntegerAdd,
    "skoro.integer.add",
    |inputs| two_integers(inputs, Space::Integer),
    |x, target| discrete(integers(&x[0]) + integers(&x[1]), target)
);
discrete!(
    /// `i - j`.
    IntegerSub,
    "skoro.integer.sub",
    |inputs| two_integers(inputs, Space::Integer),
    |x, target| discrete(integers(&x[0]) - integers(&x[1]), target)
);
discrete!(
    /// `i j`.
    IntegerMul,
    "skoro.integer.mul",
    |inputs| two_integers(inputs, Space::Integer),
    |x, target| discrete(integers(&x[0]) * integers(&x[1]), target)
);
discrete!(
    /// `i = j`, a boolean.
    IntegerEq,
    "skoro.integer.eq",
    |inputs| two_integers(inputs, Space::BOOLEAN),
    |x, target| boolean(integers(&x[0]).equal(integers(&x[1])), target)
);
discrete!(
    /// `i < j`, a boolean.
    IntegerLt,
    "skoro.integer.lt",
    |inputs| two_integers(inputs, Space::BOOLEAN),
    |x, target| boolean(integers(&x[0]).lower(integers(&x[1])), target)
);
discrete!(
    /// `i ≤ j`, a boolean.
    IntegerLe,
    "skoro.integer.le",
    |inputs| two_integers(inputs, Space::BOOLEAN),
    |x, target| boolean(integers(&x[0]).lower_equal(integers(&x[1])), target)
);
discrete!(
    /// A discrete value read as a real: an embedding, with no derivative.
    ToReal,
    "skoro.to_real",
    |inputs| match inputs {
        [Space::Integer | Space::Categorical(_)] => Some(Space::Real),
        _ => None,
    },
    |x, target| {
        let Value::Discrete(integers) = &x[0] else {
            unreachable!("checked when the node was added: a discrete input")
        };
        Value::Continuous(integers.to_points(target))
    }
);
discrete!(
    /// `1[x ≥ 0]`, a boolean: a discretization, passing no cotangent back.
    IsNonnegative,
    "skoro.is_nonnegative",
    |inputs| matches!(inputs, [Space::Real]).then_some(Space::BOOLEAN),
    |x, target| boolean(points(&x[0]).greater_equal_elem(0.0), target)
);
