//! The quaternion arithmetic on handles of storage: `QDyn`s of `Vector(4)` for a quaternion
//! `(w, x, y, z)`, `Vector(3)` for a vector, and `Matrix(3, 3)` for a matrix stored column by
//! column, built of the core's primitives. Each function adds nodes to the open scope, inside a
//! composite's body.

use std::sync::Arc;

use skoro_core::primitive::Select;
use skoro_core::scope::{apply, concatenate, constant};
use skoro_core::{QDyn, Real, Space};

/// A real constant.
pub(crate) fn real(c: f64) -> QDyn {
    constant(Real, c).untyped()
}

pub(crate) fn zero() -> QDyn {
    real(0.0)
}

pub(crate) fn one() -> QDyn {
    real(1.0)
}

/// Entry `i` of a Euclidean value's storage, as a `Real`.
pub(crate) fn entry(x: QDyn, i: usize) -> QDyn {
    x.slice(i..i + 1).reshape(Space::Real)
}

/// `(w, u)` of a `Vector(4)`: `w` a `Real`, `u` a `Vector(3)`.
pub(crate) fn parts(q: QDyn) -> (QDyn, QDyn) {
    (entry(q, 0), q.slice(1..4))
}

/// The `Vector(4)` of `w` and `u`.
pub(crate) fn quaternion(w: QDyn, u: QDyn) -> QDyn {
    concatenate(&[w, u], Space::Vector(4))
}

/// `(a₁, a₂, a₀)`.
fn yzx(a: QDyn) -> QDyn {
    concatenate(&[a.slice(1..3), a.slice(0..1)], Space::Vector(3))
}

/// `(a₂, a₀, a₁)`.
fn zxy(a: QDyn) -> QDyn {
    concatenate(&[a.slice(2..3), a.slice(0..2)], Space::Vector(3))
}

/// `a × b = (a₁b₂ - a₂b₁, a₂b₀ - a₀b₂, a₀b₁ - a₁b₀)`.
pub(crate) fn cross(a: QDyn, b: QDyn) -> QDyn {
    yzx(a) * zxy(b) - zxy(a) * yzx(b)
}

/// `a · b`.
pub(crate) fn dot(a: QDyn, b: QDyn) -> QDyn {
    (a * b).sum()
}

/// The quaternion product `(w₁ w₂ - u₁·u₂, w₁ u₂ + w₂ u₁ + u₁ × u₂)`.
pub(crate) fn mul(a: QDyn, b: QDyn) -> QDyn {
    let ((wa, ua), (wb, ub)) = (parts(a), parts(b));
    quaternion(wa * wb - dot(ua, ub), wa * ub + wb * ua + cross(ua, ub))
}

/// `q̄ = (w, -u)`.
pub(crate) fn conjugate(q: QDyn) -> QDyn {
    let (w, u) = parts(q);
    quaternion(w, -u)
}

/// `π(q) x = x + 2w (u × x) + 2 u × (u × x)`, for a unit `q`.
pub(crate) fn rotate(q: QDyn, x: QDyn) -> QDyn {
    let (w, u) = parts(q);
    let ux = cross(u, x);
    x + (w * ux + cross(u, ux)) * 2.0
}

/// `π(q)ᵀ x = x - 2w (u × x) + 2 u × (u × x)`, for a unit `q`.
pub(crate) fn unrotate(q: QDyn, x: QDyn) -> QDyn {
    let (w, u) = parts(q);
    let ux = cross(u, x);
    x + (cross(u, ux) - w * ux) * 2.0
}

/// `π(q)` of a unit `q = (w, x, y, z)`, a `Matrix(3, 3)` stored column by column:
///
/// ```text
/// column 0: (1 - 2(y² + z²), 2(xy + wz),     2(xz - wy))
/// column 1: (2(xy - wz),     1 - 2(x² + z²), 2(yz + wx))
/// column 2: (2(xz + wy),     2(yz - wx),     1 - 2(x² + y²))
/// ```
pub(crate) fn matrix(q: QDyn) -> QDyn {
    let (w, x, y, z) = (entry(q, 0), entry(q, 1), entry(q, 2), entry(q, 3));
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, xz, yz) = (x * y, x * z, y * z);
    let (wx, wy, wz) = (w * x, w * y, w * z);
    let diagonal = |a: QDyn, b: QDyn| (a + b) * -2.0 + 1.0;
    let twice = |a: QDyn| a * 2.0;
    concatenate(
        &[
            diagonal(yy, zz),
            twice(xy + wz),
            twice(xz - wy),
            twice(xy - wz),
            diagonal(xx, zz),
            twice(yz + wx),
            twice(xz + wy),
            twice(yz - wx),
            diagonal(xx, yy),
        ],
        Space::Matrix(3, 3),
    )
}

/// `[ω]×`, stored column by column: `(0, ω₂, -ω₁)`, `(-ω₂, 0, ω₀)`, `(ω₁, -ω₀, 0)`.
pub(crate) fn hat(omega: QDyn) -> QDyn {
    let (a, b, c) = (entry(omega, 0), entry(omega, 1), entry(omega, 2));
    concatenate(
        &[zero(), c, -b, -c, zero(), a, b, -a, zero()],
        Space::Matrix(3, 3),
    )
}

/// `1 / a = e^{-log a}`, for `a` positive on the domain.
pub(crate) fn reciprocal(a: QDyn) -> QDyn {
    (a.log() * -1.0).exp()
}

/// `√a = e^{½ log a}`, for `a > 0`.
pub(crate) fn sqrt(a: QDyn) -> QDyn {
    (a.log() * 0.5).exp()
}

/// `|x| = √(Σ xᵢ²)`, for `x ≠ 0`.
pub(crate) fn norm(x: QDyn) -> QDyn {
    sqrt((x * x).sum())
}

/// `x / |x|`, for `x ≠ 0`.
pub(crate) fn normalize(x: QDyn) -> QDyn {
    ((x * x).sum().log() * -0.5).exp() * x
}

/// `if_true` on the rows where the `Boolean` `condition` is true, `if_false` elsewhere.
pub(crate) fn choose(condition: QDyn, if_false: QDyn, if_true: QDyn) -> QDyn {
    apply(Arc::new(Select), &[condition, if_false, if_true])
}

/// Both conditions: `1[a b - ½ ≥ 0]` of their embeddings as reals, since `select` takes
/// continuous values only and cannot choose between Booleans.
pub(crate) fn both(a: QDyn, b: QDyn) -> QDyn {
    (a.to_real() * b.to_real() - 0.5).is_nonnegative()
}

/// `δ(t) = ((1 - s), t/2) / (1 + s)`, `s = |t|²/16`: the chart at the identity.
pub(crate) fn delta(t: QDyn) -> QDyn {
    let s = (t * t).sum() * (1.0 / 16.0);
    reciprocal(s + 1.0) * quaternion(-s + 1.0, t * 0.5)
}

/// The chart's inverse at the identity, `4 vec(d) / (1 + w(d))`.
pub(crate) fn delta_inverse(d: QDyn) -> QDyn {
    let (w, u) = parts(d);
    (reciprocal(w + 1.0) * 4.0) * u
}

#[cfg(test)]
pub(crate) mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use skoro_core::scope::quantity_in;
    use skoro_core::{QDyn, Query, Space, Values, Writer};
    use skoro_tensor::{HostData, HostValue, Target, Value};

    use super::*;

    /// `f` at the vectors `inputs`, one row, on the host.
    pub(crate) fn eval(f: impl FnOnce(&[QDyn]) -> QDyn, inputs: &[&[f64]]) -> Vec<f64> {
        let target = Target::host();
        let mut writer = Writer::new();
        let (roots, out) = writer
            .build(|| {
                let roots: Vec<QDyn> = inputs
                    .iter()
                    .enumerate()
                    .map(|(i, x)| quantity_in(i, Space::Vector(x.len())))
                    .collect();
                let out = f(&roots);
                (roots, out)
            })
            .unwrap();
        let program = writer
            .snapshot()
            .compile(&Query::of([out]), &target)
            .unwrap();
        let mut at = Values::new();
        for (q, x) in roots.iter().zip(inputs) {
            let value = HostValue::new(Some(1), x.len(), HostData::Continuous(x.to_vec()))
                .unwrap()
                .load(1, &target)
                .unwrap()
                .unwrap();
            at.insert(*q, value);
        }
        match program.forward(&at).unwrap().get(out).unwrap() {
            Value::Continuous(points) => points.to_host(),
            Value::Discrete(_) => panic!("a continuous value"),
        }
    }

    pub(crate) fn close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{got:?} against {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-12, "{got:?} against {want:?}");
        }
    }

    fn host_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]
    }

    fn unit(q: [f64; 4]) -> [f64; 4] {
        let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        q.map(|x| x / n)
    }

    const A: [f64; 4] = [0.3, -0.5, 0.7, 0.2];
    const B: [f64; 4] = [-0.4, 0.1, 0.6, -0.8];
    const X: [f64; 3] = [0.9, -1.3, 0.4];

    #[test]
    fn product_is_hamiltons() {
        close(&eval(|q| mul(q[0], q[1]), &[&A, &B]), &host_mul(A, B));
        close(
            &eval(|q| conjugate(q[0]), &[&A]),
            &[A[0], -A[1], -A[2], -A[3]],
        );
    }

    #[test]
    fn cross_and_dot() {
        let a = [1.0, 2.0, 3.0];
        close(
            &eval(|q| cross(q[0], q[1]), &[&a, &X]),
            &[
                2.0 * 0.4 - 3.0 * -1.3,
                3.0 * 0.9 - 1.0 * 0.4,
                1.0 * -1.3 - 2.0 * 0.9,
            ],
        );
        close(&eval(|q| dot(q[0], q[1]), &[&a, &X]), &[0.9 - 2.6 + 1.2]);
    }

    #[test]
    fn rotation_is_the_sandwich_and_the_matrix() {
        let q = unit(A);
        let sandwich = host_mul(
            host_mul(q, [0.0, X[0], X[1], X[2]]),
            [q[0], -q[1], -q[2], -q[3]],
        );
        let rotated = eval(|h| rotate(h[0], h[1]), &[&q, &X]);
        close(&rotated, &sandwich[1..]);
        let back = eval(|h| unrotate(h[0], rotate(h[0], h[1])), &[&q, &X]);
        close(&back, &X);
        // Column j of the stored matrix is π(q) eⱼ.
        let m = eval(|h| matrix(h[0]), &[&q]);
        for j in 0..3 {
            let mut e = [0.0; 3];
            e[j] = 1.0;
            close(
                &m[3 * j..3 * j + 3],
                &eval(|h| rotate(h[0], h[1]), &[&q, &e]),
            );
        }
    }

    #[test]
    fn hat_is_the_cross_product() {
        let omega = [0.3, -1.1, 0.8];
        let m = eval(|h| hat(h[0]), &[&omega]);
        let crossed = eval(|h| cross(h[0], h[1]), &[&omega, &X]);
        // (M x)_i = Σ_j M_ij x_j, M_ij at i + 3j.
        let product: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| m[i + 3 * j] * X[j]).sum())
            .collect();
        close(&product, &crossed);
    }

    #[test]
    fn roots_norms_and_selection() {
        close(&eval(|h| sqrt(entry(h[0], 0)), &[&[2.25]]), &[1.5]);
        close(&eval(|h| reciprocal(entry(h[0], 0)), &[&[4.0]]), &[0.25]);
        close(&eval(|h| norm(h[0]), &[&[3.0, 4.0]]), &[5.0]);
        close(&eval(|h| normalize(h[0]), &[&[3.0, 4.0]]), &[0.6, 0.8]);
        let pick = |a: f64, b: f64| {
            eval(
                |h| {
                    let condition = both(
                        entry(h[0], 0).is_nonnegative(),
                        entry(h[0], 1).is_nonnegative(),
                    );
                    choose(condition, real(-1.0), one())
                },
                &[&[a, b]],
            )[0]
        };
        assert_eq!(
            [
                pick(1.0, 1.0),
                pick(1.0, -1.0),
                pick(-1.0, 1.0),
                pick(-1.0, -1.0)
            ],
            [1.0, -1.0, -1.0, -1.0]
        );
    }

    #[test]
    fn chart_at_the_identity_inverts() {
        let t = [0.7, -2.1, 1.4];
        let d = eval(|h| delta(h[0]), &[&t]);
        let s = t.iter().map(|x| x * x).sum::<f64>() / 16.0;
        close(
            &d,
            &[
                (1.0 - s) / (1.0 + s),
                t[0] / 2.0 / (1.0 + s),
                t[1] / 2.0 / (1.0 + s),
                t[2] / 2.0 / (1.0 + s),
            ],
        );
        close(&eval(|h| delta_inverse(delta(h[0])), &[&t]), &t);
    }
}
