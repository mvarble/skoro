//! The composites of a normal's term, defined once per form of the scale, dimension and part, and
//! cached as the core caches a space's retraction.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use skoro_core::scope::{concatenate, constant};
use skoro_core::{Composite, Error, Param, QDyn, Space, VectorDyn};

use crate::Form;
use crate::scale::triangular;

/// Which of a form's composites.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Part {
    Whiten,
    Normalizer,
    Matrix,
}

type Cache = Mutex<HashMap<(Form, usize, Part), Arc<Composite>>>;

static CACHE: OnceLock<Cache> = OnceLock::new();

/// The space of a scale of `form` for a normal of dimension `d`.
pub(crate) fn scale_space(form: Form, d: usize) -> Space {
    match form {
        Form::Isotropic => Space::Positive(1),
        Form::Diagonal => Space::Positive(d),
        Form::CovarianceFactor | Form::Whitening => triangular(d),
    }
}

/// `skoro.normal.whiten`: `r = W t`, of inputs `Vector(d)` and the form's space, to `Vector(d)`.
pub(crate) fn whiten(form: Form, d: usize) -> Result<Arc<Composite>, Error> {
    cached(form, d, Part::Whiten)
}

/// `skoro.normal.normalizer`: `Σ log W_ii - (d/2) log 2π`, of input the form's space, to `Real`.
pub(crate) fn normalizer(form: Form, d: usize) -> Result<Arc<Composite>, Error> {
    cached(form, d, Part::Normalizer)
}

/// `skoro.normal.matrix`: a factor of `triangular(d)` as the dense lower-triangular
/// `Matrix(d, d)`.
pub(crate) fn matrix(d: usize) -> Result<Arc<Composite>, Error> {
    cached(Form::Whitening, d, Part::Matrix)
}

fn cached(form: Form, d: usize, part: Part) -> Result<Arc<Composite>, Error> {
    let cache = CACHE.get_or_init(Cache::default);
    let key = (form, d, part);
    if let Some(composite) = cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
    {
        return Ok(composite.clone());
    }
    // Defined with the lock released: the whitening's composite asks for the matrix's.
    let composite = define(form, d, part)?;
    let mut cache = cache.lock().unwrap_or_else(PoisonError::into_inner);
    Ok(cache.entry(key).or_insert(composite).clone())
}

fn define(form: Form, d: usize, part: Part) -> Result<Arc<Composite>, Error> {
    let params = vec![Param::Text(form.name().into()), Param::Integer(d as i64)];
    let scale = scale_space(form, d);
    let defined = match part {
        Part::Whiten => {
            let matrix = match form {
                Form::Whitening => Some(matrix(d)?),
                _ => None,
            };
            Composite::define(
                "skoro.normal.whiten",
                params,
                &[Space::Vector(d), scale],
                |q| match (form, &matrix) {
                    (Form::Isotropic, _) => q[1].recip().as_real().sum() * q[0],
                    (Form::Diagonal, _) => q[1].recip().as_real() * q[0],
                    (Form::Whitening, Some(matrix)) => matrix.call(&[q[1]]).matmul(q[0]),
                    _ => forward_substitution(q[1], q[0], d),
                },
            )
        }
        Part::Normalizer => Composite::define("skoro.normal.normalizer", params, &[scale], |q| {
            let log_diagonal = match form {
                Form::Isotropic => q[0].log().sum() * -(d as f64),
                Form::Diagonal => -q[0].log().sum(),
                Form::CovarianceFactor => -q[0].factor(0).log().sum(),
                Form::Whitening => q[0].factor(0).log().sum(),
            };
            log_diagonal + -0.5 * d as f64 * (2.0 * PI).ln()
        }),
        Part::Matrix => Composite::define(
            "skoro.normal.matrix",
            vec![Param::Integer(d as i64)],
            &[triangular(d)],
            |q| {
                let diagonal = q[0].factor(0).as_real();
                let mut columns = Vec::with_capacity(2 * d);
                for b in 0..d {
                    if b > 0 {
                        columns.push(constant(VectorDyn(b), vec![0.0; b].as_slice()).untyped());
                    }
                    columns.push(diagonal.slice(b..b + 1));
                    let below = lower(d, b);
                    if !below.is_empty() {
                        columns.push(q[0].factor(1).slice(below));
                    }
                }
                concatenate(&columns, Space::Matrix(d, d))
            },
        ),
    };
    defined.map_err(|failure| failure.error)
}

/// The range, in the storage of the entries strictly below the diagonal, of column `b`'s.
pub(crate) fn lower(d: usize, b: usize) -> std::ops::Range<usize> {
    let start = b * (d - 1) - b * b.saturating_sub(1) / 2;
    start..start + (d - 1 - b)
}

/// `z` with `L z = t`, row by row: `z_i = (t_i - Σ_{j<i} L_ij z_j) / L_ii`, each division a
/// product with the reciprocal of a positive diagonal entry.
fn forward_substitution(l: QDyn, t: QDyn, d: usize) -> QDyn {
    let reciprocals = l.factor(0).recip().as_real();
    let entry = |q: QDyn, i: usize| q.slice(i..i + 1).sum();
    let mut z: Vec<QDyn> = Vec::with_capacity(d);
    for i in 0..d {
        let mut rest = entry(t, i);
        for (j, zj) in z.iter().enumerate() {
            let lij = entry(l.factor(1), lower(d, j).start + (i - j - 1));
            rest = rest - lij * *zj;
        }
        z.push(rest * entry(reciprocals, i));
    }
    concatenate(&z, Space::Vector(d))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use skoro_core::Values;
    use skoro_core::testing::check_operation;
    use skoro_tensor::{HostData, HostValue, Points, Precision, Value};

    /// A lower-triangular `L` of side 3, row by row, with a positive diagonal.
    const L: [f64; 9] = [1.3, 0.0, 0.0, -0.4, 0.8, 0.0, 0.7, 0.25, 1.6];

    fn host(storage: usize, data: Vec<f64>) -> HostValue {
        HostValue::new(Some(1), storage, HostData::Continuous(data)).unwrap()
    }

    /// `L` in `triangular(3)`'s storage: its diagonal, then its lower entries column by column.
    fn storage() -> Vec<f64> {
        vec![L[0], L[4], L[8], L[3], L[6], L[7]]
    }

    skoro_core::precision_tests! {
        fn forward_substitution_solves(target) {
            let d = 3;
            let t = [0.9, -1.1, 0.35];
            let composite = whiten(Form::CovarianceFactor, d).unwrap();
            let program = composite.compile(&target).unwrap();
            let free = program.free();
            let load = |storage: usize, data: &[f64]| {
                Value::Continuous(Points::from_host(data, 1, storage, &target))
            };
            let at = Values::new()
                .with(free[0].node, load(d, &t))
                .with(free[1].node, load(6, &storage()));
            let out = program.forward(&at).unwrap();
            let Value::Continuous(z) = out.get(program.targets()[0]).unwrap() else {
                panic!("a continuous output")
            };
            // The dense solve on the host.
            let mut expected = [0.0; 3];
            for i in 0..d {
                let rest: f64 = (0..i).map(|j| L[i * d + j] * expected[j]).sum();
                expected[i] = (t[i] - rest) / L[i * d + i];
            }
            let relative = match target.precision() {
                Precision::Double => 1e-12,
                Precision::Single => 1e-5,
            };
            for (got, want) in z.to_host().iter().zip(expected) {
                assert!((got - want).abs() <= relative * want.abs().max(1.0), "{got} against {want}");
            }
            // Its derivatives, in t and in L, against finite differences.
            let inputs = [
                (Space::Vector(d), host(d, t.to_vec())),
                (triangular(d), host(6, storage())),
            ];
            check_operation(composite.operation().as_ref(), &inputs, &target).unwrap();
        }
    }
}
