//! The scale of a normal, in one of four forms, each read through its whitening `W`.

use skoro_core::scope::{constant, product};
use skoro_core::{Error, Positive, PositiveDyn, Q, QDyn, Space, VectorDyn};

use crate::refuse::refuse;

/// Which of the four forms of the scale a value is, read through `W` in every one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Form {
    /// `σ ∈ ℝ>0`, `Σ = σ²I`: a `Positive(1)`.
    Isotropic,
    /// `σ ∈ ℝ>0^d`, `Σ = diag(σ)²`: a `Positive(d)`.
    Diagonal,
    /// `L` with `Σ = LLᵀ`: a `triangular(d)`.
    CovarianceFactor,
    /// `W` with `Λ = WᵀW`: a `triangular(d)`.
    Whitening,
}

impl Form {
    /// The form's name, as composites and samplers record it.
    pub fn name(&self) -> &'static str {
        match self {
            Form::Isotropic => "isotropic",
            Form::Diagonal => "diagonal",
            Form::CovarianceFactor => "covariance_factor",
            Form::Whitening => "whitening",
        }
    }
}

/// A scale: its form and the node of its value, a constant or any value of the form's space.
#[derive(Clone, Copy, Debug)]
pub struct Scale {
    form: Form,
    value: QDyn,
}

/// The space of a lower-triangular factor of side `d` with a positive diagonal: the diagonal, then
/// the entries strictly below it column by column, `Product([Positive(d), Vector(d(d-1)/2)])`, and
/// `Product([Positive(1)])` when `d = 1`.
pub fn triangular(d: usize) -> Space {
    let diagonal = Space::Positive(d);
    let lower = d * d.saturating_sub(1) / 2;
    let factors = if lower == 0 {
        vec![diagonal]
    } else {
        vec![diagonal, Space::Vector(lower)]
    };
    // Neither factor is discrete, so the product is never refused.
    Space::Product(factors.into())
}

impl Scale {
    /// `σ`, with `Σ = σ²I`.
    pub fn isotropic(sigma: Q<Positive<1>>) -> Scale {
        Scale {
            form: Form::Isotropic,
            value: sigma.untyped(),
        }
    }

    /// `σ`, a `Positive(d)`, with `Σ = diag(σ)²`.
    pub fn diagonal(sigma: impl Into<QDyn>) -> Scale {
        Scale {
            form: Form::Diagonal,
            value: sigma.into(),
        }
    }

    /// `L`, a `triangular(d)`, with `Σ = LLᵀ`.
    pub fn covariance_factor(l: impl Into<QDyn>) -> Scale {
        Scale {
            form: Form::CovarianceFactor,
            value: l.into(),
        }
    }

    /// `W`, a `triangular(d)`, with `Λ = WᵀW`.
    pub fn whitening(w: impl Into<QDyn>) -> Scale {
        Scale {
            form: Form::Whitening,
            value: w.into(),
        }
    }

    /// The constant isotropic scale `σ`.
    #[track_caller]
    pub fn sd(sigma: f64) -> Scale {
        Scale::isotropic(constant(Positive::<1>, [sigma]))
    }

    /// The constant diagonal scale `σ`.
    #[track_caller]
    pub fn sds(sigma: &[f64]) -> Scale {
        Scale::diagonal(constant(PositiveDyn(sigma.len()), sigma))
    }

    /// The covariance `Σ`, `d × d` row by row, as the constant Cholesky factor `L` with
    /// `Σ = LLᵀ`; a matrix that is not symmetric positive definite is refused (`ValueShape`).
    #[track_caller]
    pub fn covariance(sigma: &[f64], d: usize) -> Scale {
        let factor = symmetric(sigma, d).and_then(|sigma| cholesky(&sigma, d));
        Scale::covariance_factor(triangular_constant(factor, d))
    }

    /// The precision `Λ`, `d × d` row by row, as the constant lower-triangular `W` with
    /// `Λ = WᵀW`; a matrix that is not symmetric positive definite is refused (`ValueShape`).
    #[track_caller]
    pub fn precision(lambda: &[f64], d: usize) -> Scale {
        let factor = symmetric(lambda, d).and_then(|lambda| whitening_factor(&lambda, d));
        Scale::whitening(triangular_constant(factor, d))
    }

    pub fn form(&self) -> Form {
        self.form
    }

    pub fn value(&self) -> QDyn {
        self.value
    }
}

/// `matrix` as a `d × d` matrix, row by row, when it has `d²` finite entries and is symmetric to
/// within rounding.
fn symmetric(matrix: &[f64], d: usize) -> Option<Vec<f64>> {
    if d == 0 || matrix.len() != d * d || matrix.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let scale = matrix.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let tolerance = 1e-12 * scale.max(f64::MIN_POSITIVE);
    let asymmetric =
        (0..d).any(|i| (0..i).any(|j| (matrix[i * d + j] - matrix[j * d + i]).abs() > tolerance));
    (!asymmetric).then(|| matrix.to_vec())
}

/// The lower-triangular `L` with `A = LLᵀ`, row by row, reading `A`'s lower triangle; `None` when
/// `A` is not positive definite.
fn cholesky(a: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; d * d];
    for j in 0..d {
        let pivot = a[j * d + j] - (0..j).map(|k| l[j * d + k] * l[j * d + k]).sum::<f64>();
        if !(pivot > 0.0 && pivot.is_finite()) {
            return None;
        }
        let diagonal = pivot.sqrt();
        l[j * d + j] = diagonal;
        for i in j + 1..d {
            let dot: f64 = (0..j).map(|k| l[i * d + k] * l[j * d + k]).sum();
            l[i * d + j] = (a[i * d + j] - dot) / diagonal;
        }
    }
    Some(l)
}

/// The lower-triangular `W` with `Λ = WᵀW`, row by row: `W = J Mᵀ J`, `M` the Cholesky factor of
/// `JΛJ` and `J` the exchange matrix.
fn whitening_factor(lambda: &[f64], d: usize) -> Option<Vec<f64>> {
    let reversed: Vec<f64> = (0..d * d)
        .map(|k| lambda[(d - 1 - k / d) * d + (d - 1 - k % d)])
        .collect();
    let m = cholesky(&reversed, d)?;
    Some(
        (0..d * d)
            .map(|k| m[(d - 1 - k % d) * d + (d - 1 - k / d)])
            .collect(),
    )
}

/// The storage of a lower-triangular factor, row by row, in `triangular(d)`: its diagonal, then
/// its entries strictly below the diagonal, column by column.
pub(crate) fn triangular_storage(factor: &[f64], d: usize) -> (Vec<f64>, Vec<f64>) {
    let diagonal = (0..d).map(|a| factor[a * d + a]).collect();
    let lower = (0..d)
        .flat_map(|b| (b + 1..d).map(move |a| (a, b)))
        .map(|(a, b)| factor[a * d + b])
        .collect();
    (diagonal, lower)
}

/// The constant of `triangular(d)` holding `factor`, or a failed handle (`ValueShape`) for none.
#[track_caller]
fn triangular_constant(factor: Option<Vec<f64>>, d: usize) -> QDyn {
    let Some(factor) = factor else {
        return refuse(
            Error::ValueShape {
                space: triangular(d),
            },
            &[],
        );
    };
    let (diagonal, lower) = triangular_storage(&factor, d);
    let mut parts = vec![constant(PositiveDyn(d), diagonal.as_slice()).untyped()];
    if !lower.is_empty() {
        parts.push(constant(VectorDyn(lower.len()), lower.as_slice()).untyped());
    }
    product(&parts).untyped()
}
