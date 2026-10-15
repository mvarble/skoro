//! The unit sphere `S²` of the acceptance test: storage 3, dimension 2, with the basis `B(x)`
//! smooth off the south pole and the inverse stereographic retraction scaled so that its
//! derivative at 0 is `B(x)`.

use std::sync::{Arc, LazyLock};

use skoro::{Composite, Declares, Embedded, QDyn, Space, Typed};

/// `1 / a`, written `e^{-log a}`, for `a` positive on the domain.
pub fn reciprocal(a: QDyn) -> QDyn {
    (-a.log()).exp()
}

/// `(x₁, x₂)`, `x₃` and `c = 1 / (1 + x₃)`.
fn parts(x: QDyn) -> (QDyn, QDyn, QDyn) {
    let x12 = x.slice(0..2);
    let x3 = x.slice(2..3).reshape(Space::Real);
    (x12, x3, reciprocal(x3 + 1.0))
}

/// `B(x) t`: `(t - c x₁₂ a, -a)` for `a = x₁₂·t`.
pub fn basis_times(x: QDyn, t: QDyn) -> QDyn {
    let (x12, _, c) = parts(x);
    let a = (x12 * t).sum();
    skoro::scope::concatenate(&[t - (c * a) * x12, -a], Space::Vector(3))
}

/// `B(x)ᵀ y`: `y₁₂ - (c e + y₃) x₁₂` for `e = x₁₂·y₁₂`.
pub fn basis_transpose(x: QDyn, y: QDyn) -> QDyn {
    let (x12, _, c) = parts(x);
    let (y12, y3) = (y.slice(0..2), y.slice(2..3).reshape(Space::Real));
    let e = (x12 * y12).sum();
    y12 - (c * e + y3) * x12
}

/// `R(x, t) = ((1 - s) x + B(x) t) / (1 + s)`, `s = |t|² / 4`.
pub fn retraction() -> Arc<Composite> {
    Composite::define(
        "sphere.retract",
        Vec::new(),
        &[Space::Vector(3), Space::Vector(2)],
        |q| {
            let (x, t) = (q[0], q[1]);
            let s = (t * t).sum() * 0.25;
            reciprocal(s + 1.0) * ((-s + 1.0) * x + basis_times(x, t))
        },
    )
    .unwrap()
}

/// `Q(x, y) = 2 B(x)ᵀ y / (1 + xᵀ y)`.
pub fn inverse() -> Arc<Composite> {
    Composite::define(
        "sphere.local",
        Vec::new(),
        &[Space::Vector(3), Space::Vector(3)],
        |q| {
            let (x, y) = (q[0], q[1]);
            let dot = (x * y).sum();
            (reciprocal(dot + 1.0) * 2.0) * basis_transpose(x, y)
        },
    )
    .unwrap()
}

pub static SPHERE: LazyLock<Arc<Embedded>> = LazyLock::new(|| {
    Embedded::declare("sphere", Vec::new(), 3, 2, retraction(), inverse()).unwrap()
});

/// The sphere's marker.
#[derive(Clone, Copy, Debug)]
pub struct Sphere;

impl Typed for Sphere {
    fn admits(space: &Space) -> bool {
        matches!(space, Space::Embedded(e) if **e == **SPHERE)
    }
}

impl Declares for Sphere {
    fn space(&self) -> Space {
        Space::Embedded(SPHERE.clone())
    }
}

/// `B(x)` itself, `3 × 2` by rows, as the mathematics writes it.
pub fn basis(x: [f64; 3]) -> [[f64; 2]; 3] {
    let c = 1.0 / (1.0 + x[2]);
    [
        [1.0 - c * x[0] * x[0], -c * x[0] * x[1]],
        [-c * x[0] * x[1], 1.0 - c * x[1] * x[1]],
        [-x[0], -x[1]],
    ]
}

/// Points of the sphere in `U`, the sphere without its south pole, one per row.
pub fn points(rows: usize) -> Vec<[f64; 3]> {
    (0..rows)
        .map(|i| {
            let (theta, phi) = (0.4 + 1.3 * i as f64, 0.7 + 0.9 * i as f64);
            let z = (theta.cos() * 0.8).clamp(-0.6, 0.95);
            let r = (1.0 - z * z).sqrt();
            [r * phi.cos(), r * phi.sin(), z]
        })
        .collect()
}
