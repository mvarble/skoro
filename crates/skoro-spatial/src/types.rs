//! The three types: the unit quaternions `S³`, the rigid motions `S³ ⋉ ℝ³` and the extended poses
//! `S³ ⋉ (ℝ³ × ℝ³)`, each declared once with the composites of its boundary, and their markers and
//! host values.
//!
//! A type's storage is the quaternion `(w, x, y, z)` followed by its translations, and its tangent
//! coordinates are the rotation's `φ` followed by one of each translation's, in the frame of the
//! point:
//!
//! ```text
//! R(x, t) = (q̂ ⊗ δ(φ), v + π(q̂) η, …)       q̂ = q / |q|
//! Q(x, y) = (δ⁻¹(q̄ ⊗ q'), π(q)ᵀ (v' - v), …)
//! P(x)    = (q̂, v, …)
//! Λ(x, t) = -3 log(1 + |φ|² / 16)
//! ```
//!
//! The retraction normalizes its base point: on the type that changes nothing, and off it, it
//! keeps the rounding of a long run of retractions from being carried into every later step. The
//! inverse does not: its inputs are points of the type. It is undefined at `q' = -q`, the chart's
//! cut: non-finite at a row exactly there, and growing without bound near it, as `4 tan(θ/4)` of
//! the relative angle `θ`, so that within rounding of the cut it is large rather than wrong.

use std::sync::{Arc, LazyLock};

use skoro_core::scope::concatenate;
use skoro_core::{
    Composite, Declares, Embedded, Error, IntoHostValue, Param, QDyn, Registry, Space, Typed,
    Volume,
};
use skoro_tensor::{HostData, HostValue};

use crate::algebra::{conjugate, delta, delta_inverse, mul, normalize, rotate, unrotate};

/// Which of the three types, by its number of translations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Rotation,
    Pose,
    ExtendedPose,
}

impl Kind {
    fn translations(self) -> usize {
        match self {
            Kind::Rotation => 0,
            Kind::Pose => 1,
            Kind::ExtendedPose => 2,
        }
    }

    /// The declaration's name and those of its retraction, inverse, projection and log-volume.
    fn names(self) -> [&'static str; 5] {
        match self {
            Kind::Rotation => [
                "skoro.spatial.rotation",
                "skoro.spatial.rotation.retract",
                "skoro.spatial.rotation.local",
                "skoro.spatial.rotation.project",
                "skoro.spatial.rotation.log_volume",
            ],
            Kind::Pose => [
                "skoro.spatial.pose",
                "skoro.spatial.pose.retract",
                "skoro.spatial.pose.local",
                "skoro.spatial.pose.project",
                "skoro.spatial.pose.log_volume",
            ],
            Kind::ExtendedPose => [
                "skoro.spatial.extended_pose",
                "skoro.spatial.extended_pose.retract",
                "skoro.spatial.extended_pose.local",
                "skoro.spatial.extended_pose.project",
                "skoro.spatial.extended_pose.log_volume",
            ],
        }
    }
}

/// The `i`-th translation's storage of a point, or the `i`-th tangent block after `φ`.
fn block(x: QDyn, offset: usize, i: usize) -> QDyn {
    x.slice(offset + 3 * i..offset + 3 * i + 3)
}

/// The rotation's part and the translations laid end to end, or the rotation's part alone.
fn assemble(rotation: QDyn, translations: Vec<QDyn>, space: Space) -> QDyn {
    if translations.is_empty() {
        return rotation;
    }
    let parts: Vec<QDyn> = std::iter::once(rotation).chain(translations).collect();
    concatenate(&parts, space)
}

fn declare(kind: Kind) -> Result<Arc<Embedded>, Error> {
    let k = kind.translations();
    let (n, d) = (4 + 3 * k, 3 + 3 * k);
    let (storage, tangent) = (Space::Vector(n), Space::Vector(d));
    let [name, retract, local, project, log_volume] = kind.names();
    let define = |name, inputs: &[Space], body: &dyn Fn(&[QDyn]) -> QDyn| {
        Composite::define(name, Vec::new(), inputs, |q| body(q)).map_err(|f| f.error)
    };
    let r = define(retract, &[storage.clone(), tangent.clone()], &|q| {
        let (x, t) = (q[0], q[1]);
        let unit = normalize(x.slice(0..4));
        let moved = (0..k)
            .map(|i| block(x, 4, i) + rotate(unit, block(t, 3, i)))
            .collect();
        assemble(mul(unit, delta(t.slice(0..3))), moved, storage.clone())
    })?;
    let q = define(local, &[storage.clone(), storage.clone()], &|q| {
        let (x, y) = (q[0], q[1]);
        let (a, b) = (x.slice(0..4), y.slice(0..4));
        let moved = (0..k)
            .map(|i| unrotate(a, block(y, 4, i) - block(x, 4, i)))
            .collect();
        assemble(delta_inverse(mul(conjugate(a), b)), moved, tangent.clone())
    })?;
    let p = define(project, std::slice::from_ref(&storage), &|q| {
        let x = q[0];
        let kept = (0..k).map(|i| block(x, 4, i)).collect();
        assemble(normalize(x.slice(0..4)), kept, storage.clone())
    })?;
    let lambda = define(log_volume, &[storage.clone(), tangent.clone()], &|q| {
        let phi = q[1].slice(0..3);
        ((phi * phi).sum() * (1.0 / 16.0) + 1.0).log() * -3.0
    })?;
    Embedded::declare(name, Vec::new(), n, d, r, q, Some(p))?.with_volume(Volume::global(lambda))
}

/// The declaration of `kind`, which `types_are_declared` shows is made without error.
fn declared(kind: Kind) -> Arc<Embedded> {
    match declare(kind) {
        Ok(declaration) => declaration,
        Err(_) => unreachable!("the declarations are checked by `types_are_declared`"),
    }
}

/// `skoro.spatial.rotation`: storage 4, dimension 3.
pub static ROTATION: LazyLock<Arc<Embedded>> = LazyLock::new(|| declared(Kind::Rotation));
/// `skoro.spatial.pose`: storage 7, dimension 6.
pub static POSE: LazyLock<Arc<Embedded>> = LazyLock::new(|| declared(Kind::Pose));
/// `skoro.spatial.extended_pose`: storage 10, dimension 9.
pub static EXTENDED_POSE: LazyLock<Arc<Embedded>> = LazyLock::new(|| declared(Kind::ExtendedPose));

/// A unit quaternion `(w, x, y, z)`: a rotation, and its double cover. Tangent coordinates are the
/// rotation vector in the frame of the point, to first order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Rotation;

/// `(q, p)`: a rigid motion, its tangent coordinates `(φ, ρ)` in the frame of the motion.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Pose;

/// `(q, v, p)`: an extended pose, its tangent coordinates `(φ, η, ρ)` in its own frame.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct ExtendedPose;

macro_rules! marker {
    ($marker:ident, $declaration:ident) => {
        impl Typed for $marker {
            fn admits(space: &Space) -> bool {
                matches!(space, Space::Embedded(e) if **e == **$declaration)
            }
        }

        impl Declares for $marker {
            fn space(&self) -> Space {
                Space::Embedded($declaration.clone())
            }
        }
    };
}

marker!(Rotation, ROTATION);
marker!(Pose, POSE);
marker!(ExtendedPose, EXTENDED_POSE);

/// A quaternion normalized on the host; `None` when it is zero or not finite.
fn unit(q: [f64; 4]) -> Option<[f64; 4]> {
    let norm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    (norm > 0.0 && norm.is_finite()).then(|| q.map(|x| x / norm))
}

/// The storage of a quaternion and translations, normalized; `None` when the quaternion is zero
/// or any entry is not finite.
fn stored(q: [f64; 4], translations: &[[f64; 3]]) -> Option<HostValue> {
    let finite = translations.iter().flatten().all(|x| x.is_finite());
    let q = unit(q).filter(|_| finite)?;
    let data: Vec<f64> = q
        .into_iter()
        .chain(translations.iter().flatten().copied())
        .collect();
    HostValue::new(None, data.len(), HostData::Continuous(data))
}

fn split<const N: usize>(x: [f64; N]) -> ([f64; 4], Vec<[f64; 3]>) {
    let q = [x[0], x[1], x[2], x[3]];
    let translations = x[4..].as_chunks::<3>().0.to_vec();
    (q, translations)
}

/// `(w, x, y, z)`, normalized on the host; `None` for a zero or non-finite quaternion. A sensor that
/// gives `(x, y, z, w)` must be reordered.
impl IntoHostValue<Rotation> for [f64; 4] {
    fn into_host_value(self) -> Option<HostValue> {
        stored(self, &[])
    }
}

/// The rotation `(w, x, y, z)` and the position.
impl IntoHostValue<Pose> for ([f64; 4], [f64; 3]) {
    fn into_host_value(self) -> Option<HostValue> {
        stored(self.0, &[self.1])
    }
}

/// The rotation `(w, x, y, z)` and the position, in storage order.
impl IntoHostValue<Pose> for [f64; 7] {
    fn into_host_value(self) -> Option<HostValue> {
        let (q, translations) = split(self);
        stored(q, &translations)
    }
}

/// The rotation `(w, x, y, z)`, the velocity and the position.
impl IntoHostValue<ExtendedPose> for ([f64; 4], [f64; 3], [f64; 3]) {
    fn into_host_value(self) -> Option<HostValue> {
        stored(self.0, &[self.1, self.2])
    }
}

/// The rotation `(w, x, y, z)`, the velocity and the position, in storage order.
impl IntoHostValue<ExtendedPose> for [f64; 10] {
    fn into_host_value(self) -> Option<HostValue> {
        let (q, translations) = split(self);
        stored(q, &translations)
    }
}

fn plain(
    name: &'static str,
    declaration: &Arc<Embedded>,
    params: &[Param],
) -> Result<Arc<Embedded>, Error> {
    if params.is_empty() {
        Ok(declaration.clone())
    } else {
        Err(Error::InvalidDeclaration {
            name: name.to_string(),
            reason: "it takes no parameters".to_string(),
        })
    }
}

/// Adds the three declarations to `registry`, by name, with no parameters (`DuplicateName` when
/// one is there already).
pub fn register(registry: &mut Registry) -> Result<(), Error> {
    registry.register_embedded("skoro.spatial.rotation", |params| {
        plain("skoro.spatial.rotation", &ROTATION, params)
    })?;
    registry.register_embedded("skoro.spatial.pose", |params| {
        plain("skoro.spatial.pose", &POSE, params)
    })?;
    registry.register_embedded("skoro.spatial.extended_pose", |params| {
        plain("skoro.spatial.extended_pose", &EXTENDED_POSE, params)
    })
}
