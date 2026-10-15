//! The primitives on Euclidean values, and slicing and concatenation, which also take products.

use std::ops::Range;

use skoro_tensor::burn::Tensor;

use super::rule::{Rule, RuleKernel, columns, widen};
use crate::{Error, Lowered, Lowering, Operation, Param, Space};

fn is_euclidean(space: &Space) -> bool {
    matches!(space, Space::Real | Space::Vector(_) | Space::Matrix(_, _))
}

fn space_mismatch(operation: &'static str, inputs: &[Space]) -> Error {
    Error::SpaceMismatch {
        operation,
        inputs: inputs.to_vec(),
    }
}

fn shape_mismatch(operation: &'static str, inputs: &[Space]) -> Error {
    Error::ShapeMismatch {
        operation,
        inputs: inputs.to_vec(),
    }
}

/// One Euclidean input.
fn one_euclidean(operation: &'static str, inputs: &[Space]) -> Result<Space, Error> {
    match inputs {
        [a] if is_euclidean(a) => Ok(a.clone()),
        _ => Err(space_mismatch(operation, inputs)),
    }
}

/// Two inputs of one Euclidean space: two different spaces are a space mismatch, whatever their
/// sizes.
fn two_euclidean(operation: &'static str, inputs: &[Space]) -> Result<Space, Error> {
    match inputs {
        [a, b] if is_euclidean(a) && a == b => Ok(a.clone()),
        _ => Err(space_mismatch(operation, inputs)),
    }
}

/// Rows and columns of a Euclidean space read as a matrix: a vector is one column.
fn matrix_shape(space: &Space) -> Option<(usize, usize)> {
    match space {
        Space::Matrix(m, n) => Some((*m, *n)),
        Space::Vector(m) => Some((*m, 1)),
        _ => None,
    }
}

/// A primitive whose rule needs nothing from its parameters or its inputs' spaces.
macro_rules! plain {
    ($(#[$doc:meta])* $ty:ident, $name:literal, $output:ident, $rule:expr) => {
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
                $output($name, inputs)
            }

            fn lower(&self, inputs: &[Space], lowering: &mut Lowering<'_>) -> Result<Lowered, Error> {
                self.output(inputs)?;
                Ok(Lowered::Kernel(RuleKernel::kernel($rule, lowering.target())))
            }
        }
    };
}

plain!(
    /// `a + b`, of one Euclidean space.
    Add, "skoro.add", two_euclidean, Linear { sign: 1.0 }
);
plain!(
    /// `a - b`, of one Euclidean space.
    Sub, "skoro.sub", two_euclidean, Linear { sign: -1.0 }
);
plain!(
    /// `-a`, elementwise.
    Neg, "skoro.neg", one_euclidean, Negation
);
plain!(
    /// `a ⊙ b`, elementwise, of one Euclidean space.
    Mul, "skoro.mul", two_euclidean, Product
);
plain!(
    /// `exp a`, elementwise.
    Exp, "skoro.exp", one_euclidean, Exponential
);
plain!(
    /// `log a`, elementwise.
    Log, "skoro.log", one_euclidean, Logarithm
);

plain!(
    /// `sin a`, elementwise.
    Sin, "skoro.sin", one_euclidean, Sine
);
plain!(
    /// `cos a`, elementwise.
    Cos, "skoro.cos", one_euclidean, Cosine
);
plain!(
    /// `atan2(a, b)`, elementwise: the angle in `(-π, π]` of the point `(b, a)`, defined where
    /// `a² + b²` is positive and smooth off the half-line `a = 0, b < 0`.
    Atan2, "skoro.atan2", two_euclidean, Angle
);

/// `s a`, a `Real` times a Euclidean value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Scale;

impl Operation for Scale {
    fn name(&self) -> &'static str {
        "skoro.scale"
    }

    fn params(&self) -> Vec<Param> {
        Vec::new()
    }

    fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
        match inputs {
            [Space::Real, a] if is_euclidean(a) => Ok(a.clone()),
            _ => Err(space_mismatch(self.name(), inputs)),
        }
    }

    fn lower(&self, inputs: &[Space], lowering: &mut Lowering<'_>) -> Result<Lowered, Error> {
        self.output(inputs)?;
        Ok(Lowered::Kernel(RuleKernel::kernel(
            Scaling,
            lowering.target(),
        )))
    }
}

/// `A B`: `Matrix(m, k)` times `Matrix(k, n)`, or times `Vector(k)` to `Vector(m)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MatMul;

impl MatMul {
    fn shapes(&self, inputs: &[Space]) -> Result<(Space, MatrixProduct), Error> {
        let name = self.name();
        let (Space::Matrix(m, k), Some((k2, n))) = (&inputs[0], matrix_shape(&inputs[1])) else {
            return Err(space_mismatch(name, inputs));
        };
        if *k != k2 {
            return Err(shape_mismatch(name, inputs));
        }
        let output = match inputs[1] {
            Space::Vector(_) => Space::Vector(*m),
            _ => Space::Matrix(*m, n),
        };
        Ok((output, MatrixProduct { m: *m, k: *k, n }))
    }
}

impl Operation for MatMul {
    fn name(&self) -> &'static str {
        "skoro.matmul"
    }

    fn params(&self) -> Vec<Param> {
        Vec::new()
    }

    fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
        if inputs.len() != 2 {
            return Err(space_mismatch(self.name(), inputs));
        }
        Ok(self.shapes(inputs)?.0)
    }

    fn lower(&self, inputs: &[Space], lowering: &mut Lowering<'_>) -> Result<Lowered, Error> {
        self.output(inputs)?;
        let (_, rule) = self.shapes(inputs)?;
        Ok(Lowered::Kernel(RuleKernel::kernel(rule, lowering.target())))
    }
}

/// `Σ a_i`, a Euclidean value to `Real`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Sum;

impl Operation for Sum {
    fn name(&self) -> &'static str {
        "skoro.sum"
    }

    fn params(&self) -> Vec<Param> {
        Vec::new()
    }

    fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
        one_euclidean(self.name(), inputs).map(|_| Space::Real)
    }

    fn lower(&self, inputs: &[Space], lowering: &mut Lowering<'_>) -> Result<Lowered, Error> {
        let input = one_euclidean(self.name(), inputs)?;
        let rule = Summation {
            dimension: input.storage(),
        };
        Ok(Lowered::Kernel(RuleKernel::kernel(rule, lowering.target())))
    }
}

/// `Matrix(m, n)` to `Matrix(n, m)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Transpose;

impl Operation for Transpose {
    fn name(&self) -> &'static str {
        "skoro.transpose"
    }

    fn params(&self) -> Vec<Param> {
        Vec::new()
    }

    fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
        match inputs {
            [Space::Matrix(m, n)] => Ok(Space::Matrix(*n, *m)),
            _ => Err(space_mismatch(self.name(), inputs)),
        }
    }

    fn lower(&self, inputs: &[Space], lowering: &mut Lowering<'_>) -> Result<Lowered, Error> {
        let Space::Matrix(m, n) = self.output(inputs)? else {
            unreachable!("a transpose's output is a matrix")
        };
        let rule = Transposition { m: n, n: m };
        Ok(Lowered::Kernel(RuleKernel::kernel(rule, lowering.target())))
    }
}

/// A part of a Euclidean value or a product: the output space, its range of storage and its range
/// of tangent coordinates, which differ when a product has an embedded factor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Slice {
    output: Space,
    storage: Range<usize>,
    tangent: Range<usize>,
}

impl Slice {
    pub fn new(output: Space, storage: Range<usize>, tangent: Range<usize>) -> Slice {
        Slice {
            output,
            storage,
            tangent,
        }
    }
}

impl Operation for Slice {
    fn name(&self) -> &'static str {
        "skoro.slice"
    }

    fn params(&self) -> Vec<Param> {
        let range = |r: &Range<usize>| {
            Param::List(vec![
                Param::Integer(r.start as i64),
                Param::Integer(r.end as i64),
            ])
        };
        vec![
            Param::Space(self.output.clone()),
            range(&self.storage),
            range(&self.tangent),
        ]
    }

    fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
        let name = self.name();
        let [input] = inputs else {
            return Err(space_mismatch(name, inputs));
        };
        let fits = self.storage.start <= self.storage.end
            && self.tangent.start <= self.tangent.end
            && self.storage.end <= input.storage()
            && input.dimension().is_some_and(|d| self.tangent.end <= d)
            && self.output.storage() == self.storage.len()
            && self.output.dimension() == Some(self.tangent.len());
        if is_euclidean(input) {
            if !is_euclidean(&self.output) {
                return Err(space_mismatch(name, inputs));
            }
            if !fits || self.storage != self.tangent {
                return Err(shape_mismatch(name, inputs));
            }
            return Ok(self.output.clone());
        }
        if !matches!(input, Space::Product(_)) {
            return Err(space_mismatch(name, inputs));
        }
        if !fits {
            return Err(shape_mismatch(name, inputs));
        }
        // A product is sliced into a run of its factors: one factor, or the product of several.
        let blocks = input.blocks();
        let run: Vec<_> = blocks
            .iter()
            .filter(|b| self.storage.start <= b.storage.start && b.storage.end <= self.storage.end)
            .collect();
        let covers = run.first().map(|b| (b.storage.start, b.tangent.start))
            == Some((self.storage.start, self.tangent.start))
            && run.last().map(|b| (b.storage.end, b.tangent.end))
                == Some((self.storage.end, self.tangent.end));
        let factors: Vec<Space> = run.iter().map(|b| b.space.clone()).collect();
        let matches = match factors.as_slice() {
            [one] => *one == self.output,
            _ => Space::product(factors.clone()).ok().as_ref() == Some(&self.output),
        };
        if covers && matches {
            Ok(self.output.clone())
        } else {
            Err(space_mismatch(name, inputs))
        }
    }

    fn lower(&self, inputs: &[Space], lowering: &mut Lowering<'_>) -> Result<Lowered, Error> {
        self.output(inputs)?;
        let rule = Slicing {
            storage: self.storage.clone(),
            tangent: self.tangent.clone(),
            dimension: inputs[0].dimension().unwrap_or(0),
        };
        Ok(Lowered::Kernel(RuleKernel::kernel(rule, lowering.target())))
    }
}

/// Values laid end to end, to a Euclidean space or a product. With one input it is a reshape.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Concatenate {
    output: Space,
}

impl Concatenate {
    pub fn new(output: Space) -> Concatenate {
        Concatenate { output }
    }
}

impl Operation for Concatenate {
    fn name(&self) -> &'static str {
        "skoro.concatenate"
    }

    fn params(&self) -> Vec<Param> {
        vec![Param::Space(self.output.clone())]
    }

    fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
        let name = self.name();
        if inputs.is_empty() || !inputs.iter().all(Space::is_continuous) {
            return Err(space_mismatch(name, inputs));
        }
        match &self.output {
            output if is_euclidean(output) => {
                if !inputs.iter().all(is_euclidean) {
                    return Err(space_mismatch(name, inputs));
                }
                if inputs.iter().map(Space::storage).sum::<usize>() != output.storage() {
                    return Err(shape_mismatch(name, inputs));
                }
                Ok(output.clone())
            }
            Space::Product(factors) if **factors == *inputs || [self.output.clone()] == inputs => {
                Ok(self.output.clone())
            }
            _ => Err(space_mismatch(name, inputs)),
        }
    }

    fn lower(&self, inputs: &[Space], lowering: &mut Lowering<'_>) -> Result<Lowered, Error> {
        self.output(inputs)?;
        let mut start = 0;
        let tangent = inputs
            .iter()
            .map(|input| {
                let end = start + input.dimension().unwrap_or(0);
                let range = start..end;
                start = end;
                range
            })
            .collect();
        let rule = Concatenation {
            tangent,
            dimension: start,
        };
        Ok(Lowered::Kernel(RuleKernel::kernel(rule, lowering.target())))
    }
}

// The rules, row by row of the table of Euclidean primitives.

struct Linear {
    sign: f64,
}

impl Rule for Linear {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        if self.sign > 0.0 {
            x[0].clone() + x[1].clone()
        } else {
            x[0].clone() - x[1].clone()
        }
    }

    fn push(&self, _: &[Tensor<2>], j: usize, t: Tensor<3>) -> Tensor<3> {
        if j == 1 && self.sign < 0.0 { -t } else { t }
    }

    fn pull(&self, _: &[Tensor<2>], j: usize, alpha: Tensor<3>) -> Tensor<3> {
        if j == 1 && self.sign < 0.0 {
            -alpha
        } else {
            alpha
        }
    }
}

struct Negation;

impl Rule for Negation {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        -x[0].clone()
    }

    fn push(&self, _: &[Tensor<2>], _: usize, t: Tensor<3>) -> Tensor<3> {
        -t
    }

    fn pull(&self, _: &[Tensor<2>], _: usize, alpha: Tensor<3>) -> Tensor<3> {
        -alpha
    }
}

struct Product;

impl Rule for Product {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        x[0].clone() * x[1].clone()
    }

    fn push(&self, x: &[Tensor<2>], j: usize, t: Tensor<3>) -> Tensor<3> {
        let k = t.dims()[2];
        columns(&x[1 - j], k) * t
    }

    fn pull(&self, x: &[Tensor<2>], j: usize, alpha: Tensor<3>) -> Tensor<3> {
        let k = alpha.dims()[2];
        columns(&x[1 - j], k) * alpha
    }

    fn second(
        &self,
        _: &[Tensor<2>],
        j: usize,
        v: &[Option<Tensor<3>>],
        alpha: Tensor<3>,
    ) -> Option<Tensor<3>> {
        let other = v[1 - j].clone()?;
        let k = other.dims()[2];
        Some(other * widen(alpha, k))
    }
}

/// `s a`: input 0 is `s`, `[N, 1]`, and input 1 is `a`.
struct Scaling;

impl Scaling {
    fn s(x: &[Tensor<2>]) -> Tensor<2> {
        let [n, d] = x[1].dims();
        x[0].clone().expand([n, d])
    }
}

impl Rule for Scaling {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        Scaling::s(x) * x[1].clone()
    }

    fn push(&self, x: &[Tensor<2>], j: usize, t: Tensor<3>) -> Tensor<3> {
        let [n, d] = x[1].dims();
        let k = t.dims()[2];
        match j {
            0 => t.expand([n, d, k]) * columns(&x[1], k),
            _ => columns(&Scaling::s(x), k) * t,
        }
    }

    fn pull(&self, x: &[Tensor<2>], j: usize, alpha: Tensor<3>) -> Tensor<3> {
        let k = alpha.dims()[2];
        match j {
            0 => (columns(&x[1], k) * alpha).sum_dim(1),
            _ => columns(&Scaling::s(x), k) * alpha,
        }
    }

    fn second(
        &self,
        x: &[Tensor<2>],
        j: usize,
        v: &[Option<Tensor<3>>],
        alpha: Tensor<3>,
    ) -> Option<Tensor<3>> {
        let [n, d] = x[1].dims();
        match j {
            0 => {
                let v_a = v[1].clone()?;
                let k = v_a.dims()[2];
                Some((v_a * widen(alpha, k)).sum_dim(1))
            }
            _ => {
                let v_s = v[0].clone()?;
                let k = v_s.dims()[2];
                Some(v_s.expand([n, d, k]) * widen(alpha, k))
            }
        }
    }
}

/// A row of storage `[N, rows * cols]`, column by column, as matrices `[N, rows, cols]`.
fn matrix(x: Tensor<2>, rows: usize, cols: usize) -> Tensor<3> {
    let n = x.dims()[0];
    x.reshape([n, cols, rows]).swap_dims(1, 2)
}

fn unmatrix(x: Tensor<3>) -> Tensor<2> {
    let [n, rows, cols] = x.dims();
    x.swap_dims(1, 2).reshape([n, rows * cols])
}

/// A frame `[N, rows * cols, K]` as matrices `[N, K, rows, cols]`.
fn matrices(t: Tensor<3>, rows: usize, cols: usize) -> Tensor<4> {
    let [n, _, k] = t.dims();
    t.reshape([n, cols, rows, k]).permute([0, 3, 2, 1])
}

fn unmatrices(t: Tensor<4>) -> Tensor<3> {
    let [n, k, rows, cols] = t.dims();
    t.permute([0, 3, 2, 1]).reshape([n, rows * cols, k])
}

/// Matrices `[N, rows, cols]` repeated over `K` columns.
fn repeat(x: Tensor<3>, k: usize) -> Tensor<4> {
    let [n, rows, cols] = x.dims();
    x.unsqueeze_dim::<4>(1).expand([n, k, rows, cols])
}

/// `A B` with `A` of `m × k` and `B` of `k × n`.
struct MatrixProduct {
    m: usize,
    k: usize,
    n: usize,
}

impl MatrixProduct {
    fn a(&self, x: &[Tensor<2>]) -> Tensor<3> {
        matrix(x[0].clone(), self.m, self.k)
    }

    fn b(&self, x: &[Tensor<2>]) -> Tensor<3> {
        matrix(x[1].clone(), self.k, self.n)
    }
}

impl Rule for MatrixProduct {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        unmatrix(self.a(x).matmul(self.b(x)))
    }

    fn push(&self, x: &[Tensor<2>], j: usize, t: Tensor<3>) -> Tensor<3> {
        let k = t.dims()[2];
        unmatrices(match j {
            0 => matrices(t, self.m, self.k).matmul(repeat(self.b(x), k)),
            _ => repeat(self.a(x), k).matmul(matrices(t, self.k, self.n)),
        })
    }

    fn pull(&self, x: &[Tensor<2>], j: usize, alpha: Tensor<3>) -> Tensor<3> {
        let k = alpha.dims()[2];
        let alpha = matrices(alpha, self.m, self.n);
        unmatrices(match j {
            0 => alpha.matmul(repeat(self.b(x).swap_dims(1, 2), k)),
            _ => repeat(self.a(x).swap_dims(1, 2), k).matmul(alpha),
        })
    }

    fn second(
        &self,
        _: &[Tensor<2>],
        j: usize,
        v: &[Option<Tensor<3>>],
        alpha: Tensor<3>,
    ) -> Option<Tensor<3>> {
        let alpha = matrices(alpha, self.m, self.n);
        let [n, _, rows, cols] = alpha.dims();
        Some(unmatrices(match j {
            0 => {
                let v_b = v[1].clone()?;
                let k = v_b.dims()[2];
                let v_b = matrices(v_b, self.k, self.n).swap_dims(2, 3);
                alpha.expand([n, k, rows, cols]).matmul(v_b)
            }
            _ => {
                let v_a = v[0].clone()?;
                let k = v_a.dims()[2];
                let v_a = matrices(v_a, self.m, self.k).swap_dims(2, 3);
                v_a.matmul(alpha.expand([n, k, rows, cols]))
            }
        }))
    }
}

struct Summation {
    dimension: usize,
}

impl Rule for Summation {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        x[0].clone().sum_dim(1)
    }

    fn push(&self, _: &[Tensor<2>], _: usize, t: Tensor<3>) -> Tensor<3> {
        t.sum_dim(1)
    }

    fn pull(&self, _: &[Tensor<2>], _: usize, alpha: Tensor<3>) -> Tensor<3> {
        let [n, _, k] = alpha.dims();
        alpha.expand([n, self.dimension, k])
    }
}

struct Exponential;

impl Rule for Exponential {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        x[0].clone().exp()
    }

    fn push(&self, x: &[Tensor<2>], _: usize, t: Tensor<3>) -> Tensor<3> {
        let k = t.dims()[2];
        columns(&self.value(x), k) * t
    }

    fn pull(&self, x: &[Tensor<2>], _: usize, alpha: Tensor<3>) -> Tensor<3> {
        let k = alpha.dims()[2];
        columns(&self.value(x), k) * alpha
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
        Some(columns(&self.value(x), k) * v * widen(alpha, k))
    }
}

struct Logarithm;

impl Rule for Logarithm {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        x[0].clone().log()
    }

    fn push(&self, x: &[Tensor<2>], _: usize, t: Tensor<3>) -> Tensor<3> {
        let k = t.dims()[2];
        t / columns(&x[0], k)
    }

    fn pull(&self, x: &[Tensor<2>], _: usize, alpha: Tensor<3>) -> Tensor<3> {
        let k = alpha.dims()[2];
        alpha / columns(&x[0], k)
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
        let square = x[0].clone() * x[0].clone();
        Some(-(v * widen(alpha, k)) / columns(&square, k))
    }
}

/// `Matrix(m, n)` to `Matrix(n, m)`, the input being `m × n`.
struct Transposition {
    m: usize,
    n: usize,
}

impl Rule for Transposition {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        unmatrix(matrix(x[0].clone(), self.m, self.n).swap_dims(1, 2))
    }

    fn push(&self, _: &[Tensor<2>], _: usize, t: Tensor<3>) -> Tensor<3> {
        unmatrices(matrices(t, self.m, self.n).swap_dims(2, 3))
    }

    fn pull(&self, _: &[Tensor<2>], _: usize, alpha: Tensor<3>) -> Tensor<3> {
        unmatrices(matrices(alpha, self.n, self.m).swap_dims(2, 3))
    }
}

struct Slicing {
    storage: Range<usize>,
    tangent: Range<usize>,
    /// The input's dimension.
    dimension: usize,
}

impl Rule for Slicing {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        let n = x[0].dims()[0];
        x[0].clone().slice([0..n, self.storage.clone()])
    }

    fn push(&self, _: &[Tensor<2>], _: usize, t: Tensor<3>) -> Tensor<3> {
        let [n, _, k] = t.dims();
        t.slice([0..n, self.tangent.clone(), 0..k])
    }

    fn pull(&self, _: &[Tensor<2>], _: usize, alpha: Tensor<3>) -> Tensor<3> {
        let [n, _, k] = alpha.dims();
        let zeros = Tensor::zeros([n, self.dimension, k], (&alpha.device(), alpha.dtype()));
        zeros.slice_assign([0..n, self.tangent.clone(), 0..k], alpha)
    }
}

struct Concatenation {
    /// Each input's range of the output's tangent coordinates.
    tangent: Vec<Range<usize>>,
    /// The output's dimension.
    dimension: usize,
}

impl Rule for Concatenation {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        Tensor::cat(x.to_vec(), 1)
    }

    fn push(&self, _: &[Tensor<2>], j: usize, t: Tensor<3>) -> Tensor<3> {
        let [n, _, k] = t.dims();
        let zeros = Tensor::zeros([n, self.dimension, k], (&t.device(), t.dtype()));
        zeros.slice_assign([0..n, self.tangent[j].clone(), 0..k], t)
    }

    fn pull(&self, _: &[Tensor<2>], j: usize, alpha: Tensor<3>) -> Tensor<3> {
        let [n, _, k] = alpha.dims();
        alpha.slice([0..n, self.tangent[j].clone(), 0..k])
    }
}

struct Sine;

impl Rule for Sine {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        x[0].clone().sin()
    }

    fn push(&self, x: &[Tensor<2>], _: usize, t: Tensor<3>) -> Tensor<3> {
        let k = t.dims()[2];
        columns(&x[0].clone().cos(), k) * t
    }

    fn pull(&self, x: &[Tensor<2>], _: usize, alpha: Tensor<3>) -> Tensor<3> {
        let k = alpha.dims()[2];
        columns(&x[0].clone().cos(), k) * alpha
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
        Some(-(columns(&self.value(x), k) * v * widen(alpha, k)))
    }
}

struct Cosine;

impl Rule for Cosine {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        x[0].clone().cos()
    }

    fn push(&self, x: &[Tensor<2>], _: usize, t: Tensor<3>) -> Tensor<3> {
        let k = t.dims()[2];
        -(columns(&x[0].clone().sin(), k) * t)
    }

    fn pull(&self, x: &[Tensor<2>], _: usize, alpha: Tensor<3>) -> Tensor<3> {
        let k = alpha.dims()[2];
        -(columns(&x[0].clone().sin(), k) * alpha)
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
        Some(-(columns(&self.value(x), k) * v * widen(alpha, k)))
    }
}

/// `atan2(a, b)`, with `r = a² + b²`: its partial derivatives are `b / r` and `-a / r`.
struct Angle;

impl Angle {
    fn r(x: &[Tensor<2>]) -> Tensor<2> {
        x[0].clone() * x[0].clone() + x[1].clone() * x[1].clone()
    }

    /// `∂_j atan2`: `b / r` for `a`, `-a / r` for `b`.
    fn partial(x: &[Tensor<2>], j: usize) -> Tensor<2> {
        match j {
            0 => x[1].clone() / Angle::r(x),
            _ => -(x[0].clone() / Angle::r(x)),
        }
    }
}

impl Rule for Angle {
    fn value(&self, x: &[Tensor<2>]) -> Tensor<2> {
        x[0].clone().atan2(x[1].clone())
    }

    fn push(&self, x: &[Tensor<2>], j: usize, t: Tensor<3>) -> Tensor<3> {
        let k = t.dims()[2];
        columns(&Angle::partial(x, j), k) * t
    }

    fn pull(&self, x: &[Tensor<2>], j: usize, alpha: Tensor<3>) -> Tensor<3> {
        let k = alpha.dims()[2];
        columns(&Angle::partial(x, j), k) * alpha
    }

    /// `hvp_a = ((a² - b²) v_b - 2ab v_a) / r² ⊙ α` and
    /// `hvp_b = ((a² - b²) v_a + 2ab v_b) / r² ⊙ α`, a missing velocity being zero.
    fn second(
        &self,
        x: &[Tensor<2>],
        j: usize,
        v: &[Option<Tensor<3>>],
        alpha: Tensor<3>,
    ) -> Option<Tensor<3>> {
        let k = v.iter().flatten().next()?.dims()[2];
        let (a, b) = (&x[0], &x[1]);
        let r = Angle::r(x);
        let r2 = r.clone() * r;
        let difference = columns(
            &((a.clone() * a.clone() - b.clone() * b.clone()) / r2.clone()),
            k,
        );
        let twice = columns(&((a.clone() * b.clone()).mul_scalar(2.0) / r2), k);
        // The velocity of the other input enters through the difference, its own through `2ab`.
        let (own, other) = (v[j].clone(), v[1 - j].clone());
        let mut hvp: Option<Tensor<3>> = None;
        let mut add = |term: Tensor<3>| {
            hvp = Some(match hvp.take() {
                Some(sum) => sum + term,
                None => term,
            });
        };
        if let Some(other) = other {
            add(difference * other);
        }
        if let Some(own) = own {
            let signed = if j == 0 { -twice } else { twice };
            add(signed * own);
        }
        Some(hvp? * widen(alpha, k))
    }
}
