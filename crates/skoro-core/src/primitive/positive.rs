//! Selection, and the primitives on positive values, read in the logarithmic charts.

use skoro_tensor::burn::Tensor;

use super::rule::{Rule, RuleKernel, columns, widen};
use crate::{Error, Lowered, Lowering, Operation, Param, Space};

fn space_mismatch(operation: &'static str, inputs: &[Space]) -> Error {
    Error::SpaceMismatch {
        operation,
        inputs: inputs.to_vec(),
    }
}

/// `select(k; x_0, …, x_{n-1}) = x_k`: a `Categorical(n)` index and `n` values of one continuous
/// space, to that space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Select;

impl Operation for Select {
    fn name(&self) -> &'static str {
        "skoro.select"
    }

    fn params(&self) -> Vec<Param> {
        Vec::new()
    }

    fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
        let name = self.name();
        let Some((Space::Categorical(n), values)) = inputs.split_first() else {
            return Err(space_mismatch(name, inputs));
        };
        let Some(first) = values.first() else {
            return Err(space_mismatch(name, inputs));
        };
        if !first.is_continuous() || values.iter().any(|v| v != first) {
            return Err(space_mismatch(name, inputs));
        }
        if *n != values.len() {
            return Err(Error::ShapeMismatch {
                operation: name,
                inputs: inputs.to_vec(),
            });
        }
        Ok(first.clone())
    }

    fn lower(&self, inputs: &[Space], lowering: &mut Lowering<'_>) -> Result<Lowered, Error> {
        self.output(inputs)?;
        Ok(Lowered::Kernel(RuleKernel::kernel(
            Selection,
            lowering.target(),
        )))
    }
}

/// A positive primitive: its name, its input spaces to its output, and its rule.
macro_rules! positive {
    ($(#[$doc:meta])* $ty:ident, $name:literal, $output:expr, $rule:expr) => {
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
                output(inputs).ok_or_else(|| space_mismatch($name, inputs))
            }

            fn lower(
                &self,
                inputs: &[Space],
                lowering: &mut Lowering<'_>,
            ) -> Result<Lowered, Error> {
                self.output(inputs)?;
                Ok(Lowered::Kernel(RuleKernel::kernel($rule, lowering.target())))
            }
        }
    };
}

positive!(
    /// A positive value read as a real one: `Positive(n)` to `Vector(n)`.
    PositiveAsReal,
    "skoro.positive.as_real",
    |inputs| match inputs {
        [Space::Positive(n)] => Some(Space::Vector(*n)),
        _ => None,
    },
    AsReal
);
positive!(
    /// `e^c`, from `Real` or `Vector(n)` to `Positive(n)`: an identity in the charts.
    PositiveExp,
    "skoro.positive.exp",
    |inputs| match inputs {
        [Space::Real] => Some(Space::Positive(1)),
        [Space::Vector(n)] => Some(Space::Positive(*n)),
        _ => None,
    },
    Chart { value: Tensor::exp, sign: 1.0 }
);
positive!(
    /// `log a`, `Positive(n)` to `Vector(n)`: an identity in the charts.
    PositiveLog,
    "skoro.positive.log",
    |inputs| match inputs {
        [Space::Positive(n)] => Some(Space::Vector(*n)),
        _ => None,
    },
    Chart { value: Tensor::log, sign: 1.0 }
);
positive!(
    /// `a ⊙ b`, of one positive space: a sum in the charts.
    PositiveMul,
    "skoro.positive.mul",
    |inputs| match inputs {
        [Space::Positive(n), Space::Positive(m)] if n == m => Some(Space::Positive(*n)),
        _ => None,
    },
    LogSum
);
positive!(
    /// `1 / a`: a negation in the charts.
    PositiveRecip,
    "skoro.positive.recip",
    |inputs| match inputs {
        [Space::Positive(n)] => Some(Space::Positive(*n)),
        _ => None,
    },
    Chart { value: Tensor::recip, sign: -1.0 }
);

/// The index is input 0, as floats; the values follow it.
struct Selection;

impl Selection {
    /// Whether row `r` selects value `i`, `[N, 1]`, expanded to `[N, d]` or `[N, d, K]`.
    fn chosen(x: &[Tensor<2>], i: usize) -> Tensor<2, skoro_tensor::burn::Bool> {
        x[0].clone().equal_elem(i as f64)
    }
}

impl Rule for Selection {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        let [n, s] = x[1].dims();
        (1..x.len()).fold(x[1].zeros_like(), |out, j| {
            out.mask_where(Selection::chosen(x, j - 1).expand([n, s]), x[j].clone())
        })
    }

    fn push(&self, x: &[Tensor<2>], j: usize, t: Tensor<3>) -> Tensor<3> {
        let [n, d, k] = t.dims();
        let chosen = Selection::chosen(x, j - 1)
            .unsqueeze_dim::<3>(2)
            .expand([n, d, k]);
        t.zeros_like().mask_where(chosen, t)
    }

    fn pull(&self, x: &[Tensor<2>], j: usize, alpha: Tensor<3>) -> Tensor<3> {
        self.push(x, j, alpha)
    }
}

/// `a` read as real: in the chart, `t ↦ a ⊙ t`.
struct AsReal;

impl Rule for AsReal {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        x[0].clone()
    }

    fn push(&self, x: &[Tensor<2>], _: usize, t: Tensor<3>) -> Tensor<3> {
        let k = t.dims()[2];
        columns(&x[0], k) * t
    }

    fn pull(&self, x: &[Tensor<2>], _: usize, alpha: Tensor<3>) -> Tensor<3> {
        self.push(x, 0, alpha)
    }

    fn second(
        &self,
        x: &[Tensor<2>],
        _: usize,
        v: &[Option<Tensor<3>>],
        alpha: Tensor<3>,
    ) -> Option<Tensor<3>> {
        let v = v[0].clone()?;
        let k = v.dims()[2];
        Some(columns(&x[0], k) * v * widen(alpha, k))
    }
}

/// A map that is `± identity` in the charts.
struct Chart {
    value: fn(Tensor<2>) -> Tensor<2>,
    sign: f64,
}

impl Rule for Chart {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        (self.value)(x[0].clone())
    }

    fn push(&self, _: &[Tensor<2>], _: usize, t: Tensor<3>) -> Tensor<3> {
        if self.sign < 0.0 { -t } else { t }
    }

    fn pull(&self, _: &[Tensor<2>], _: usize, alpha: Tensor<3>) -> Tensor<3> {
        if self.sign < 0.0 { -alpha } else { alpha }
    }
}

/// `a ⊙ b`: in the charts, `t_a + t_b`.
struct LogSum;

impl Rule for LogSum {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        x[0].clone() * x[1].clone()
    }

    fn push(&self, _: &[Tensor<2>], _: usize, t: Tensor<3>) -> Tensor<3> {
        t
    }

    fn pull(&self, _: &[Tensor<2>], _: usize, alpha: Tensor<3>) -> Tensor<3> {
        alpha
    }
}
