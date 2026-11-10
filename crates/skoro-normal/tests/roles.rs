#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The roles a normal binds, the root of its expected information, and the four forms of its
//! scale.

mod common;

use common::{evaluate, host, inverse};
use skoro_core::scope::{constant, quantity, quantity_in};
use skoro_core::{
    AtMostOne, Frames, Key, NodeId, Observations, Positive, PositiveDyn, QDyn, Query, Role,
    Snapshot, Space, Values, Vector, Writer,
};
use skoro_normal::{Form, Normal, Scale, roles, triangular};
use skoro_tensor::{Frame, HostData, HostValue, Points, Target, Value};

#[test]
fn roles_are_bound() {
    let mut writer = Writer::new();
    let (vector, positive, factor) = writer
        .build(|| {
            let x = quantity("x", Vector::<2>);
            let vector = Normal::on(
                Vector::<2>,
                constant(Vector::<2>, [0.0, 1.0]),
                Scale::sds(&[1.0, 2.0]),
            )
            .score(x);
            let p = quantity("p", Positive::<2>);
            let positive = Normal::on(
                Positive::<2>,
                constant(Positive::<2>, [1.0, 2.0]),
                Scale::sd(0.3),
            )
            .score(p);
            let z = constant(Vector::<2>, [0.5, 0.5]);
            let factor = Normal::on(Vector::<2>, x, Scale::covariance(&[1.0, 0.2, 0.2, 1.0], 2))
                .factor_at(z);
            (vector.id(), positive.id(), factor.id())
        })
        .unwrap();
    let snapshot = writer.snapshot();
    let bound = |term: NodeId| {
        let at_most = |role: &Role<AtMostOne>| snapshot.bound(term, role).unwrap().is_some();
        (
            snapshot.bound(term, &roles::RESIDUAL).is_ok(),
            snapshot.bound(term, &roles::MEAN).is_ok(),
            [
                Form::Isotropic,
                Form::Diagonal,
                Form::CovarianceFactor,
                Form::Whitening,
            ]
            .map(|form| at_most(roles::scale(form))),
            at_most(&roles::LOG_VOLUME),
            at_most(&roles::INFORMATION_ROOT),
            at_most(&roles::TANGENT),
        )
    };
    assert_eq!(
        bound(vector),
        (true, true, [false, true, false, false], false, true, true)
    );
    assert_eq!(
        bound(positive),
        (true, true, [true, false, false, false], true, true, true)
    );
    assert_eq!(
        bound(factor),
        (true, true, [false, false, true, false], false, false, false)
    );
}

/// `rows` copies of one row, as a value on `target`.
fn repeated(row: &[f64], rows: usize, target: &Target) -> Value {
    let data: Vec<f64> = row.iter().copied().cycle().take(rows * row.len()).collect();
    Value::Continuous(Points::from_host(&data, rows, row.len(), target))
}

/// The quantity of a free scale of `form`, of dimension `d`, and the scale on it.
fn free_scale(form: Form, d: usize) -> (QDyn, Scale) {
    match form {
        Form::Isotropic => {
            let s = quantity("scale", Positive::<1>);
            (s.untyped(), Scale::isotropic(s))
        }
        Form::Diagonal => {
            let s = quantity("scale", PositiveDyn(d));
            (s.untyped(), Scale::diagonal(s))
        }
        Form::CovarianceFactor => {
            let s = quantity_in("scale", triangular(d));
            (s, Scale::covariance_factor(s))
        }
        Form::Whitening => {
            let s = quantity_in("scale", triangular(d));
            (s, Scale::whitening(s))
        }
    }
}

/// A value of the scale of `form` for d = 2, in its storage.
fn scale_value(form: Form) -> Vec<f64> {
    match form {
        Form::Isotropic => vec![0.6],
        Form::Diagonal => vec![0.5, 0.8],
        // L = [[0.7, 0], [0.3, 0.9]], and W = [[1.2, 0], [-0.4, 1.5]].
        Form::CovarianceFactor => vec![0.7, 0.9, 0.3],
        Form::Whitening => vec![1.2, 1.5, -0.4],
    }
}

/// The expected negative chart Hessian over `(x, μ, θ)`, its standard errors, and `CᵀC`, on
/// `space` of dimension 2 with a free scale of `form`.
fn information(
    space: Space,
    mean: &[f64],
    form: Form,
    rows: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let target = Target::host();
    let mut writer = Writer::new();
    let (x, mu, s, term) = writer
        .build(|| {
            let mu = quantity_in("mu", space.clone());
            let (s, scale) = free_scale(form, 2);
            let x = quantity_in("x", space.clone());
            let term = Normal::in_space(space.clone(), mu, scale).score(x);
            (x, mu, s, term.id())
        })
        .unwrap();
    let snapshot: Snapshot = writer.snapshot();
    let scale = scale_value(form);
    let k = scale.len();
    let observe = |values: &[f64], q: QDyn| {
        HostValue::new(None, values.len(), HostData::Continuous(values.to_vec()))
            .map(|v| (q, v))
            .unwrap()
    };
    let (m, sv) = (observe(mean, mu), observe(&scale, s));
    let given = Observations::new()
        .observe_dyn(m.0, m.1.clone())
        .observe_dyn(sv.0, sv.1.clone());
    let draws = snapshot
        .generate(&given, &Key::from_seed(3), rows, &target)
        .unwrap();

    // The root, at the scale.
    let root = snapshot
        .bound(term, &roles::INFORMATION_ROOT)
        .unwrap()
        .unwrap();
    let c = &evaluate(&snapshot, &[root], given, &target)[0];
    let columns = 4 + k;
    let height = c.len() / columns;
    let ctc: Vec<f64> = (0..columns * columns)
        .map(|e| {
            let (i, j) = (e / columns, e % columns);
            (0..height)
                .map(|r| c[i * height + r] * c[j * height + r])
                .sum()
        })
        .collect();

    // The chart Hessian of the term at each draw, by one Hessian pass with identity velocities.
    let program = snapshot
        .compile(&Query::log_density_of([term]), &target)
        .unwrap();
    let at = Values::new()
        .with(x, draws.values.get(x).unwrap().clone())
        .with(mu, repeated(mean, rows, &target))
        .with(s, repeated(&scale, rows, &target));
    let blocks = [(x, 0, 2), (mu, 2, 2), (s, 4, k)];
    let mut velocities = Frames::new();
    for &(q, offset, dim) in &blocks {
        let mut data = vec![0.0; rows * dim * columns];
        for row in 0..rows {
            for i in 0..dim {
                data[(row * dim + i) * columns + offset + i] = 1.0;
            }
        }
        velocities.insert(q, Frame::from_host(&data, rows, dim, columns, &target));
    }
    let second = program.hessian_products(&at, &velocities).unwrap();
    let mut sum = vec![0.0; columns * columns];
    let mut squares = vec![0.0; columns * columns];
    for &(q, offset, dim) in &blocks {
        let beta = second.beta.get(q).unwrap().to_host();
        for row in 0..rows {
            for i in 0..dim {
                for j in 0..columns {
                    let h = -beta[(row * dim + i) * columns + j];
                    sum[(offset + i) * columns + j] += h;
                    squares[(offset + i) * columns + j] += h * h;
                }
            }
        }
    }
    let n = rows as f64;
    let mean_h: Vec<f64> = sum.iter().map(|s| s / n).collect();
    let se: Vec<f64> = squares
        .iter()
        .zip(&mean_h)
        .map(|(q, m)| ((q / n - m * m).max(0.0) / n).sqrt())
        .collect();
    (mean_h, se, ctc)
}

#[test]
fn root_squares_to_the_information() {
    let forms = [
        Form::Isotropic,
        Form::Diagonal,
        Form::CovarianceFactor,
        Form::Whitening,
    ];
    let spaces = [
        (Space::Vector(2), vec![0.3, -0.7]),
        (Space::Positive(2), vec![1.5, 0.4]),
    ];
    for (space, mean) in &spaces {
        for form in forms {
            let (expected, se, ctc) = information(space.clone(), mean, form, 100_000);
            for (e, (s, c)) in expected.iter().zip(se.iter().zip(&ctc)) {
                assert!(
                    (e - c).abs() <= 5.0 * s + 1e-9 * c.abs().max(1.0),
                    "{space:?} {form:?}: E[-H] {expected:?} against CᵀC {ctc:?}"
                );
            }
        }
    }
}

/// A scale made from a covariance, row by row, in the open scope.
type MakeScale = fn(&[f64]) -> Scale;

#[test]
fn four_forms_agree() {
    let target = Target::host();
    let (d, mean) = (2, [0.3, -0.7]);
    // One Σ given in each form it can be: isotropic, diagonal, and full.
    let isotropic = [0.36, 0.0, 0.0, 0.36];
    let diagonal = [0.25, 0.0, 0.0, 0.64];
    let full = [0.49, 0.21, 0.21, 0.9];
    let groups: [(&[f64], Vec<MakeScale>); 3] = [
        (
            &isotropic,
            vec![
                |_| Scale::sd(0.6),
                |_| Scale::sds(&[0.6, 0.6]),
                |s| Scale::covariance(s, 2),
                |s| Scale::precision(&inverse(s, 2), 2),
            ],
        ),
        (
            &diagonal,
            vec![
                |_| Scale::sds(&[0.5, 0.8]),
                |s| Scale::covariance(s, 2),
                |s| Scale::precision(&inverse(s, 2), 2),
            ],
        ),
        (
            &full,
            vec![|s| Scale::covariance(s, 2), |s| {
                Scale::precision(&inverse(s, 2), 2)
            }],
        ),
    ];
    let points = [0.5, 0.2, -1.0, 0.4, 0.3, -0.7];
    for (sigma, makers) in groups {
        let mut seen: Vec<[Vec<f64>; 4]> = Vec::new();
        for make in makers {
            let mut writer = Writer::new();
            let (x, parts, term) = writer
                .build(|| {
                    let x = quantity("x", Vector::<2>);
                    let normal = Normal::on(Vector::<2>, constant(Vector::<2>, mean), make(sigma));
                    (x, normal.log_density(x), normal.score(x).id())
                })
                .unwrap();
            let snapshot = writer.snapshot();
            let value = HostValue::new(Some(3), d, HostData::Continuous(points.to_vec())).unwrap();
            let given = Observations::new().observe_dyn(x.untyped(), value);
            let root = snapshot
                .bound(term, &roles::INFORMATION_ROOT)
                .unwrap()
                .unwrap();
            let values = evaluate(
                &snapshot,
                &[parts.log_density.untyped(), parts.residual.untyped(), root],
                given,
                &target,
            );
            // The (x, μ) block of CᵀC, the same Λ whatever the form.
            let c = &values[2];
            let height = d + d * (d + 1) / 2;
            let block: Vec<f64> = (0..16)
                .map(|e| {
                    (0..height)
                        .map(|r| c[(e / 4) * height + r] * c[(e % 4) * height + r])
                        .sum()
                })
                .collect();
            let generated = snapshot
                .generate(&Observations::new(), &Key::from_seed(4), 8, &target)
                .unwrap();
            let draws = host(&generated.values, x.id());
            seen.push([values[0].clone(), values[1].clone(), block, draws]);
        }
        for other in &seen[1..] {
            for (a, b) in seen[0].iter().zip(other) {
                common::close(b, a, 1e-12);
            }
        }
    }
}

skoro_core::precision_tests! {
    fn score_binds_the_tangent(target) {
        // `TANGENT` is `x ⊖ μ`, and the residual is `W` times it: `diag(σ)⁻¹` here.
        let sigma = [0.5, 2.0];
        let cases: [(Space, [f64; 2], [f64; 2]); 2] = [
            (Space::Vector(2), [0.3, -0.7], [1.1, 0.4]),
            (Space::Positive(2), [1.5, 0.4], [2.0, 0.1]),
        ];
        for (space, mean, at) in cases {
            let mut writer = Writer::new();
            let (x, term) = writer
                .build(|| {
                    let x = quantity_in("x", space.clone());
                    let mu = skoro_core::scope::constant_in(
                        space.clone(),
                        HostValue::new(None, 2, HostData::Continuous(mean.to_vec())).unwrap(),
                    );
                    let term = Normal::in_space(space.clone(), mu, Scale::sds(&sigma)).score(x);
                    (x, term)
                })
                .unwrap();
            let snapshot = writer.snapshot();
            let tangent = snapshot.bound(term, &roles::TANGENT).unwrap().unwrap();
            let residual = snapshot.bound(term, &roles::RESIDUAL).unwrap();
            let given = Observations::new()
                .observe_dyn(x, HostValue::new(None, 2, HostData::Continuous(at.to_vec())).unwrap());
            let out = evaluate(&snapshot, &[tangent, residual], given, &target);
            let expected: Vec<f64> = match space {
                Space::Positive(_) => (0..2).map(|i| (at[i] / mean[i]).ln()).collect(),
                _ => (0..2).map(|i| at[i] - mean[i]).collect(),
            };
            common::close(&out[0], &expected, common::tolerance(&target));
            let whitened: Vec<f64> = out[0].iter().zip(sigma).map(|(t, s)| t / s).collect();
            common::close(&out[1], &whitened, common::tolerance(&target));
        }
        // A likelihood factor binds none.
        let mut writer = Writer::new();
        let factor = writer
            .build(|| {
                let x = quantity("x", Vector::<2>);
                Normal::on(Vector::<2>, x, Scale::sd(1.0)).factor_at(constant(Vector::<2>, [0.0, 0.0]))
            })
            .unwrap();
        assert_eq!(writer.snapshot().bound(factor, &roles::TANGENT).unwrap(), None);
    }
}
