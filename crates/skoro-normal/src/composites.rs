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
    Root,
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

/// `skoro.normal.root`: the root `C` of the expected information, of input the form's space, to
/// `Matrix(d + p, 2d + k)`, `p = d(d+1)/2` and `k` the dimension of the form's space. Its rows
/// are the `d` residual rows, then `(a, a)`, then `(a, b)` for `a > b` in the triangular
/// storage's order; its columns `x`, `μ`, then the scale's tangent coordinates.
pub(crate) fn root(form: Form, d: usize) -> Result<Arc<Composite>, Error> {
    cached(form, d, Part::Root)
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
        Part::Root => {
            let k = scale.dimension().unwrap_or(0);
            let rows = d + d * (d + 1) / 2;
            Composite::define(
                "skoro.normal.root",
                params,
                std::slice::from_ref(&scale),
                |q| {
                    let columns = root_columns(form, d, q[0]);
                    concatenate(&columns, Space::Matrix(rows, 2 * d + k))
                },
            )
        }
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

/// A dense `d × d` lower-triangular matrix of `Real` handles, `None` above the diagonal.
type Lower = Vec<Vec<Option<QDyn>>>;

/// The `Real` entry `i` of a Euclidean value.
fn entry(q: QDyn, i: usize) -> QDyn {
    q.slice(i..i + 1).sum()
}

fn zero() -> QDyn {
    constant(skoro_core::Real, 0.0).untyped()
}

/// The entries of a factor of `triangular(d)`.
fn factor_entries(factor: QDyn, d: usize) -> Lower {
    let diagonal = factor.factor(0).as_real();
    (0..d)
        .map(|r| {
            (0..d)
                .map(|c| match r.cmp(&c) {
                    std::cmp::Ordering::Less => None,
                    std::cmp::Ordering::Equal => Some(entry(diagonal, r)),
                    std::cmp::Ordering::Greater => {
                        Some(entry(factor.factor(1), lower(d, c).start + (r - c - 1)))
                    }
                })
                .collect()
        })
        .collect()
}

/// The entries of the inverse of a factor of `triangular(d)`, column by column by forward
/// substitution against the columns of the identity.
fn inverse_entries(factor: QDyn, d: usize) -> Lower {
    let columns: Vec<QDyn> = (0..d)
        .map(|c| {
            let unit: Vec<f64> = (0..d).map(|r| if r == c { 1.0 } else { 0.0 }).collect();
            let unit = constant(VectorDyn(d), unit.as_slice()).untyped();
            forward_substitution(factor, unit, d)
        })
        .collect();
    (0..d)
        .map(|r| {
            (0..d)
                .map(|c| (r >= c).then(|| entry(columns[c], r)))
                .collect()
        })
        .collect()
}

/// The columns of the root `C` of eq:archive/normal/math/root, each a `Vector(d + p)`.
fn root_columns(form: Form, d: usize, scale: QDyn) -> Vec<QDyn> {
    let p = d * (d + 1) / 2;
    let rows = d + p;
    // The whitening W, and the factor L = W⁻¹ where the form's T_i need it.
    let (w, l): (Lower, Option<Lower>) = match form {
        Form::Isotropic => {
            let s = scale.recip().as_real().sum();
            let w = (0..d)
                .map(|r| (0..d).map(|c| (r == c).then_some(s)).collect())
                .collect();
            (w, None)
        }
        Form::Diagonal => {
            let s = scale.recip().as_real();
            let w = (0..d)
                .map(|r| (0..d).map(|c| (r == c).then(|| entry(s, r))).collect())
                .collect();
            (w, None)
        }
        Form::CovarianceFactor => (inverse_entries(scale, d), Some(factor_entries(scale, d))),
        Form::Whitening => (factor_entries(scale, d), Some(inverse_entries(scale, d))),
    };
    let column = |entries: Vec<QDyn>| concatenate(&entries, Space::Vector(rows));
    let mut columns = Vec::new();
    // The columns of x and μ: (W; 0) and (-W; 0).
    for sign in [1.0, -1.0] {
        for c in 0..d {
            let entries = (0..rows)
                .map(|r| match w.get(r).and_then(|row| row[c]) {
                    Some(e) => e * sign,
                    None => zero(),
                })
                .collect();
            columns.push(column(entries));
        }
    }
    // The scale's columns, each T_i laid out as (0; T_aa / √2 ...; T_ab ...).
    let lay_out = |t: &dyn Fn(usize, usize) -> Option<QDyn>| {
        let mut entries: Vec<QDyn> = (0..d).map(|_| zero()).collect();
        for a in 0..d {
            entries.push(t(a, a).map_or_else(zero, |e| e * std::f64::consts::FRAC_1_SQRT_2));
        }
        for b in 0..d {
            for a in b + 1..d {
                entries.push(t(a, b).unwrap_or_else(zero));
            }
        }
        column(entries)
    };
    match (form, &l) {
        // T = -2I in the coordinate log σ.
        (Form::Isotropic, _) => columns.push(lay_out(&|a, b| {
            (a == b).then(|| constant(skoro_core::Real, -2.0).untyped())
        })),
        // T = -2 E_aa in the coordinate log σ_a.
        (Form::Diagonal, _) => {
            for i in 0..d {
                columns.push(lay_out(&|a, b| {
                    (a == b && a == i).then(|| constant(skoro_core::Real, -2.0).untyped())
                }));
            }
        }
        (_, Some(l)) => {
            // The coordinates of a triangular scale: log of each diagonal entry, then each entry
            // below the diagonal, column by column.
            let coordinates = (0..d)
                .map(|a| (a, a))
                .chain((0..d).flat_map(|b| (b + 1..d).map(move |a| (a, b))));
            for (a, b) in coordinates {
                // G for the coordinate (a, b), with T = s (G + Gᵀ) and its diagonal entries in
                // their logarithmic charts multiplied by the entry.
                let g = |r: usize, c: usize| -> Option<QDyn> {
                    match form {
                        // G = W E_ab: column b holds W's column a.
                        Form::CovarianceFactor => (c == b).then(|| w[r][a]).flatten(),
                        // G = E_ab L: row a holds L's row b.
                        _ => (r == a).then(|| l[b][c]).flatten(),
                    }
                };
                let (sign, chart) = match form {
                    Form::CovarianceFactor => (-1.0, l[a][a]),
                    _ => (1.0, w[a][a]),
                };
                let t = |r: usize, c: usize| -> Option<QDyn> {
                    let sum = match (g(r, c), g(c, r)) {
                        (Some(x), Some(y)) => Some(x + y),
                        (x, y) => x.or(y),
                    }?;
                    let scaled = sum * sign;
                    Some(match (a == b, chart) {
                        (true, Some(entry)) => scaled * entry,
                        _ => scaled,
                    })
                };
                columns.push(lay_out(&t));
            }
        }
        _ => unreachable!("a triangular form has both its factors"),
    }
    columns
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

        fn composites_pass_check_operation(target) {
            use skoro_core::testing::random;
            // `check_operation` asks a composite's second-order map with its only input's velocity
            // missing for a frame as wide as the velocities, and the harness's kernel for a
            // compiled composite cannot know that width when it is given no velocity at all. A
            // composite of one input is therefore checked with a second, real input that adds
            // zero to its output, so that the check holds every derivative of the composite and
            // that one case is asked of a composite that can answer it.
            let with_real = |composite: Arc<Composite>| {
                let output = composite.output().clone();
                let input = composite.inputs()[0].clone();
                Composite::define("test.normal.with_real", Vec::new(), &[input, Space::Real], |q| {
                    let out = composite.call(&[q[0]]);
                    match &output {
                        Space::Real => out + q[1] * 0.0,
                        Space::Matrix(r, c) => {
                            let zeros = vec![0.0; r * c];
                            let zeros = constant(skoro_core::MatrixDyn(*r, *c), zeros.as_slice());
                            out + q[1] * zeros.untyped()
                        }
                        _ => unreachable!("the composites of one input give a real or a matrix"),
                    }
                })
                .unwrap()
            };
            let real = (Space::Real, random(&Space::Real, 2, 4, -1.0, 1.0));
            let forms = [Form::Isotropic, Form::Diagonal, Form::CovarianceFactor, Form::Whitening];
            for d in 1..=3 {
                let tangent = (Space::Vector(d), random(&Space::Vector(d), 2, 1, -1.0, 1.0));
                for form in forms {
                    let space = scale_space(form, d);
                    let scale = (space.clone(), random(&space, 2, 2, 0.5, 1.5));
                    let checks = [
                        (whiten(form, d).unwrap(), vec![tangent.clone(), scale.clone()]),
                        (with_real(normalizer(form, d).unwrap()), vec![scale.clone(), real.clone()]),
                        (with_real(root(form, d).unwrap()), vec![scale.clone(), real.clone()]),
                    ];
                    for (composite, inputs) in checks {
                        check_operation(composite.operation().as_ref(), &inputs, &target)
                            .unwrap_or_else(|e| panic!("{} {form:?} {d}: {e}", composite.name()));
                    }
                }
                let factor = (triangular(d), random(&triangular(d), 2, 3, 0.5, 1.5));
                let matrix = with_real(matrix(d).unwrap());
                check_operation(matrix.operation().as_ref(), &[factor, real.clone()], &target)
                    .unwrap();
            }
        }
    }
}
