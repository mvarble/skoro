---
summary: A crate `skoro-rotations` declaring the unit quaternions, the rigid motions and the extended poses as embedded types with their volumes, and every map of the mathematics as a composite of the core's operations, typed by markers, reached through operators and extension traits, with host helpers for the values a model is given.
depends_on: [archive/rotations/math, archive/core/implementation, archive/measures/implementation, archive/normal/implementation]
katex_macros:
    '\Exp': '\operatorname{Exp}'
    '\Log': '\operatorname{Log}'
---

## Overview

When this is built, a crate `skoro-rotations` gives skoro its rotations and rigid motions as the
mathematics settles them ([The unit quaternions](math.svx#the-unit-quaternions),
[The rigid motions](math.svx#the-rigid-motions)). Three markers, `Rotation`, `Pose` and
`ExtendedPose`, declare the types $S^3$, $S^3 \ltimes \R^3$ and $S^3 \ltimes (\R^3 \times \R^3)$,
each with its chart [](eq:archive/rotations/math/quaternion-chart), its projection and its declared
volume, so that a quantity of any of them can be scored by the normal, observed and simulated like
any other. Every map of [Maps of the rotations](math.svx#maps-of-the-rotations) and
[Maps of the rigid motions](math.svx#maps-of-the-rigid-motions) is a composite of the core's
primitives and selection, reached through operators (`*` composes, and acts on vectors) and through
methods of extension traits on the handles. A model of an attitude integrated from a gyroscope and
corrected by a measured direction reads:

```rust
use skoro::scope::{constant, quantity};
use skoro::{Real, Vector};
use skoro_normal::{Normal, Scale};
use skoro_rotations::{Rotation, RotationMaps, rotation};

writer.build(|| {
    // The attitude at the last tick, and the body rate integrated over the step.
    let q0 = quantity("q0", Rotation);
    let omega = constant(Vector::<3>, [0.01, -0.02, 0.005]);
    let predicted = q0 * rotation::exp(omega * 0.1);          // q₀ ⊗ Exp(ω Δt)

    // The new attitude, with its noise in its own frame.
    let q1 = quantity("q1", Rotation);
    Normal::on(Rotation, predicted, Scale::sd(0.01)).score(q1);

    // A measured direction of gravity in the body frame: the inverse action.
    let g = constant(Vector::<3>, [0.0, 0.0, -1.0]);
    let a = quantity("a", Vector::<3>);
    Normal::on(Vector::<3>, q1.unrotate(g), Scale::sd(0.05)).score(a);
});
```

and a pose graph, with odometry in the frame of the pose and a GPS position in a fixed one:

```rust
use skoro_rotations::{Pose, PoseMaps, pose};

writer.build(|| {
    let t0 = quantity("t0", Pose);
    let t1 = quantity("t1", Pose);
    let odometry = constant(Pose, ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0]));
    Normal::on(Pose, t0 * odometry, Scale::sds(&[0.01, 0.01, 0.01, 0.1, 0.1, 0.1])).score(t1);
    let gps = quantity("gps", Vector::<3>);
    Normal::on(Vector::<3>, t1.position(), Scale::sd(2.0)).score(gps);
});
```

The crate writes no kernel and no derivative. Each map's derivatives of both orders are the core's,
by composition, and by [%full](statement:archive/rotations/math/first-order) they are the Jacobians
of the textbooks. A small `host` module does on the host, in double precision, what the
mathematics leaves to the host: aligning an observed quaternion with its prediction
([The double cover](math.svx#the-double-cover)), and reading a quaternion off a matrix or Euler
angles for a constant.

## Constraints

Read at commit `4239e5c`. Each interface below is pasted from the file named, and followed by what
the design must do because of it.

### Embedded types and their volumes

`crates/skoro-core/src/embedded.rs`:

```rust
impl Embedded {
    /// `retraction`, `Vector(n)` and `Vector(d)` to `Vector(n)`, is the `R` of the mathematics;
    /// `inverse`, two of `Vector(n)` to `Vector(d)`, its `Q`; `projection`, `Vector(n)` to
    /// `Vector(n)`, its `P`, when the type has one. Any other signature is refused
    /// (`InvalidDeclaration`).
    pub fn declare(
        name: &'static str,
        params: Vec<Param>,
        storage: usize,
        dimension: usize,
        retraction: Arc<Composite>,
        inverse: Arc<Composite>,
        projection: Option<Arc<Composite>>,
    ) -> Result<Arc<Embedded>, Error>;

    /// This declaration, with its chart's volume against the measure its retraction normalizes.
    pub fn with_volume(self: Arc<Self>, volume: Volume) -> Result<Arc<Embedded>, Error>;
}

impl Volume {
    /// `Λ`, `Vector(n)` and `Vector(d)` to `Real`, on a global chart.
    pub fn global(log_volume: Arc<Composite>) -> Volume;
}
```

- A type is four composites over its storage: $R$, $Q$, $P$ and $\Lambda$, written on `Vector(n)`
  and `Vector(d)`, never on the type itself. The rotation's arithmetic is therefore written once,
  as functions on storage handles, and used by the three declarations and by the maps alike.
- Two declarations under one name and parameters are one type, so each type is declared once, in a
  `static`, and its marker hands out that declaration.
- The measure is the one the retraction normalizes, $\nu$ of
  [%full](statement:archive/rotations/math/volume) for the rotations and $\nu \times$ Lebesgue for
  the motions, so $\Lambda$ is the math's log-volume as it stands, and every chart is global.

### Markers and host values

`crates/skoro-core/src/q.rs` and `into_host.rs`:

```rust
pub trait Typed: Copy + Send + Sync + 'static { fn admits(space: &Space) -> bool; }
pub trait Declares: Typed { fn space(&self) -> Space; }

/// A Rust value that a marker suggests, as a value on the host shared by every row; `None` when
/// it cannot be one.
pub trait IntoHostValue<T> { fn into_host_value(self) -> Option<HostValue>; }
impl<T, V: IntoHostValue<T>> IntoHostValue<T> for Rows<V> { /* one value per row */ }
```

- `Declares::space` cannot fail, while declaring a type returns a `Result`. The declarations are
  deterministic, so the `static` that holds each one handles the error branch with
  `unreachable!`, as the core does where a branch is provably dead, and a test declares every type
  and defines every composite. The workspace lints deny `unwrap`, `expect` and `panic!`, not
  `unreachable!`.
- `IntoHostValue` returns an `Option`, so a host value that cannot be a point (a zero quaternion,
  a non-finite entry) is refused there, and one that is only near the type is normalized there.
  `Rows` gives every implementation its batched form for free.

### Operators and handles

`crates/skoro-core/src/ops.rs`:

```rust
/// `lhs * rhs`.
pub trait QMul<Rhs: Typed = Self>: Typed {
    type Output: Typed;
    fn mul(lhs: Q<Self>, rhs: Q<Rhs>) -> Q<Self::Output>;
}

impl<T: Typed> Q<T> {
    pub fn retract(self, t: impl Into<QDyn>) -> Q<T>;
    pub fn local(self, from: Q<T>) -> Q<VectorDyn>;
    pub fn log_volume(self, t: impl Into<QDyn>) -> Q<Real>;
    pub fn include(self) -> Q<VectorDyn>;
}

pub fn concatenate(parts: &[QDyn], space: Space) -> QDyn;
pub fn project<T: Declares>(marker: T, w: impl Into<QDyn>) -> Q<T>;
pub fn select<T: Typed>(index: impl Into<QDyn>, values: &[Q<T>]) -> Q<T>;
pub fn product(parts: &[QDyn]) -> Q<ProductDyn>;
```

and the inherent methods of the Euclidean markers: `exp`, `log`, `sin`, `cos`, `atan2`, `sum`,
`slice`, `reshape`, `matmul`, `transpose`, and on `Q<Real>`, `is_nonnegative`.

- `QMul` is the core's trait and the markers are this crate's, so the orphan rule allows `impl QMul
  for Rotation` and `impl QMul<Vector<3>> for Rotation`: `*` composes and acts, as the mathematics
  writes them.
- `Q` is the core's type, so the crate cannot add inherent methods to `Q<Rotation>`. Its methods
  come through extension traits, which must not reuse the names of `Q<T>`'s generic inherent
  methods (`retract`, `local`, `log_volume`, `include`, `narrow`, `untyped`, `id`), since an inherent
  method wins and the trait's would be unreachable by method syntax.
- A value of an embedded type can be neither negated nor scaled, and `concatenate` builds Euclidean
  values and products only. Every map into a type therefore computes storage and ends in
  `project`, as [](eq:archive/core/math/map-into) has it; a choice of sign (the rotation vector,
  alignment, the reading of a matrix) is a selection in storage, before the projection.
- `select` takes a `Categorical` index whose value $k$ picks the $k$-th value; a `Boolean` is
  `Categorical(2)`, false picking the first. A choice among four is a tournament of Boolean
  selections.
- A matrix is stored column by column: $X_{ij}$ is at index $i + 3j$ of a `Matrix(3, 3)`.

### Composites

`crates/skoro-core/src/composite.rs`:

```rust
impl Composite {
    /// Runs `body` with one handle per input; the operations inside build into the composite,
    /// and the handle `body` returns is its output.
    pub fn define(name: &'static str, params: Vec<Param>, inputs: &[Space],
                  body: impl FnOnce(&[QDyn]) -> QDyn) -> Result<Arc<Composite>, Failure>;
    /// Adds one node to the open scope: this composite applied to `inputs`.
    pub fn call(self: &Arc<Self>, inputs: &[QDyn]) -> QDyn;
}
```

- A composite is defined once, over exact input spaces, and compiled for each target when a
  program is. Its body cannot see the target's precision, so each cutoff and each truncation is
  one for both precisions. `Lowering::target` is public, so an `Operation` of the crate's own could
  lower to a composite chosen per precision; see [Series and cutoffs](#series-and-cutoffs) for why
  this design does not need it.
- A call refuses inputs of other spaces with `SpaceMismatch`, at the caller's line, which is how a
  map refuses a wrong input. `VectorDyn(3)` is `Vector(3)`, so dynamic handles of the right size
  are accepted.
- A composite may make constants, of any `Declares` marker, inside its body.

### Testing

`crates/skoro-core/src/testing.rs`:

```rust
pub fn check_operation(operation: &dyn Operation, inputs: &[(Space, HostValue)],
                       target: &Target) -> Result<(), String>;
pub fn check_agreement(a: impl Fn(&[QDyn]) -> QDyn, b: impl Fn(&[QDyn]) -> QDyn,
                       inputs: &[(Space, HostValue)], target: &Target) -> Result<(), String>;
pub fn check_embedded(declaration: &Arc<Embedded>, points: &HostValue,
                      target: &Target) -> Result<(), String>;
pub fn check_volume(declaration: &Arc<Embedded>, points: &HostValue,
                    target: &Target) -> Result<(), String>;
```

with `Tolerance::of(Precision::Double)` a relative $10^{-7}$ and a central difference of step
$10^{-6}$, and single-precision results held to double-precision ones within
`Tolerance::AGREEMENT`, $10^{-4}$, against the larger of the reference's largest magnitude and one.

- `check_embedded` and `check_volume` are the checks the mathematics asks of the types' boundaries
  and volumes ([What the tests check](math.svx#what-the-tests-check)).
- The single-precision agreement is absolute below magnitude one, which is what decides the cutoffs
  of [Series and cutoffs](#series-and-cutoffs).

### The registry

`crates/skoro-core/src/registry.rs`:

```rust
/// How a name finds its code again: constructors by name. A value, not global state; a feature
/// crate exports a function that adds its names to one.
impl Registry {
    pub fn register_embedded(&mut self, name: &'static str, make: MakeEmbedded) -> Result<(), Error>;
}
pub type MakeEmbedded = fn(&[Param]) -> Result<Arc<Embedded>, Error>;
```

- The crate exports `register`, adding its three declarations. Composites are not registered:
  they are stored whole, as the normal's are.

## Design

### The crate

`crates/skoro-rotations`, a member of the workspace, depending on `skoro-core` and `skoro-tensor`
and on nothing else, with `skoro-normal` as a development dependency for the tests that score these
types. The umbrella does not re-export it, as it does not re-export the normal. It has no kernel,
so the rule that a change to a kernel is pushed only after the wgpu tests pass does not apply.

```
crates/skoro-rotations/src/
    lib.rs          pub use of the items below; register()
    algebra.rs      the quaternion arithmetic on storage handles (private)
    series.rs       the coefficients of the removable singularities, guarded (private)
    types.rs        the three declarations, their markers, their host values
    rotation.rs     the maps of the rotations: operators, RotationMaps, free functions
    euler.rs        Sequence, Frame, Euler; the composites of eq:euler and the statement euler
    so3.rs          hat, vee, the Jacobians and their inverses
    pose.rs         the maps of the rigid motions: operators, PoseMaps, free functions
    extended.rs     the maps of the extended poses
    host.rs         alignment and conversions on the host, in double precision
```

One crate holds the three types. The `archive` branch had `skoro-so3` and `skoro-se3`, the second
built on the first's arithmetic through a public module (`skoro_so3::ops::body`). Here every type is
written on the same storage arithmetic, which would otherwise have to be a public interface of one
crate for the other; and a pose is not useful without the rotations, nor the rotations, in practice,
without the maps that carry vectors and positions. Kept from the archive: the body-frame chart of
the motions, $(r, p) \oplus (a, b) = (r \oplus a, p + r b)$, the group operations under their
names (compose, invert, rotate and unrotate, transform and untransform), and methods on handles
through extension traits. Not kept: the matrix storage, the exponential chart, and the fused
kernels of the exponential and the logarithm for each backend, which [Matrices or
quaternions](math.svx#matrices-or-quaternions) replaces with composites; and the archive's
cutoffs per precision, which [Series and cutoffs](#series-and-cutoffs) replaces with one.

### Storage arithmetic

`algebra.rs` holds the arithmetic of [](eq:archive/rotations/math/product) and
[](eq:archive/rotations/math/rotation) on handles of storage, `QDyn`s of `Vector(4)` and
`Vector(3)`, built of the core's primitives. Nothing in it is public; each function adds nodes to the
open scope, inside a composite's body.

```rust
/// `(w, u)` of a `Vector(4)`: `w` a `Real`, `u` a `Vector(3)`.
fn parts(q: QDyn) -> (QDyn, QDyn);
/// The `Vector(4)` of `w` and `u`.
fn quaternion(w: QDyn, u: QDyn) -> QDyn;
/// `a × b`, by slices, products and a concatenation.
fn cross(a: QDyn, b: QDyn) -> QDyn;
fn dot(a: QDyn, b: QDyn) -> QDyn;                       // (a * b).sum()
/// eq:product.
fn mul(a: QDyn, b: QDyn) -> QDyn;
fn conjugate(q: QDyn) -> QDyn;
/// `π(q) x = x + 2w (u × x) + 2 u × (u × x)`, and with `-u`, `π(q)ᵀ x`.
fn rotate(q: QDyn, x: QDyn) -> QDyn;
fn unrotate(q: QDyn, x: QDyn) -> QDyn;
/// eq:rotation, as a `Matrix(3, 3)` concatenated column by column.
fn matrix(q: QDyn) -> QDyn;
/// `[ω]×`, column by column.
fn hat(omega: QDyn) -> QDyn;
/// `1 / a = e^{-log a}`, for `a` positive on the domain.
fn reciprocal(a: QDyn) -> QDyn;
/// `|x| = e^{½ log Σ xᵢ²}` and `x / |x|`, for `x ≠ 0`.
fn norm(x: QDyn) -> QDyn;
fn normalize(x: QDyn) -> QDyn;
/// `select` on two `QDyn`s by a `Boolean`: `if_true` where it is true.
fn choose(condition: QDyn, if_false: QDyn, if_true: QDyn) -> QDyn;
/// `δ(t) = ((1 - s), t/2) / (1 + s)`, `s = |t|²/16`: the chart at the identity.
fn delta(t: QDyn) -> QDyn;
/// The chart's inverse at the identity, `4 vec(d) / (1 + w(d))`.
fn delta_inverse(d: QDyn) -> QDyn;
```

`rotate` and `matrix` assume a unit quaternion, as the mathematics does. The maps that take a value
of a type hand them its inclusion, which is unit up to rounding.

### The three types

`types.rs` declares each type once, in a `LazyLock`, with the composites of its boundary:

| type | name | storage, dimension | $R(x, t)$ | $Q(x, y)$ | $P(x)$ | $\Lambda(x, t)$ |
| --- | --- | --- | --- | --- | --- | --- |
| $S^3$ | `skoro.rotations.rotation` | 4, 3 | $\hat x \otimes \delta(t)$ | $\delta^{-1}(\bar x \otimes y)$ | $\hat x$ | $-3 \log(1 + \tfrac1{16} \lVert t \rVert^2)$ |
| $S^3 \ltimes \R^3$ | `skoro.rotations.pose` | 7, 6 | $(\hat q \otimes \delta(\phi),\ p + \pi(\hat q) \rho)$ | $(\delta^{-1}(\bar q \otimes q'),\ \pi(q)\transpose (p' - p))$ | $(\hat q, p)$ | the same, of $\phi$ |
| $S^3 \ltimes (\R^3)^2$ | `skoro.rotations.extended_pose` | 10, 9 | $(\hat q \otimes \delta(\phi),\ v + \pi(\hat q) \eta,\ p + \pi(\hat q) \rho)$ | likewise | $(\hat q, v, p)$ | the same, of $\phi$ |

with $\hat x = x / \lVert x \rVert$. These are [%full](statement:archive/rotations/math/chart),
[%full](statement:archive/rotations/math/volume) and [%full](statement:archive/rotations/math/motions),
with one addition: the retraction normalizes its base point. On the type that changes nothing, and
its derivatives at the type are the same; off it, it keeps a point that a long run of retractions has
carried off the sphere by rounding from carrying that error into every later step, the way an
estimator's state is updated thousands of times in single precision. The mathematics leaves the
behaviour off the type to this entry, as the core's does. The alternative, leaving the drift to the
maps that project, lost because an estimator that only retracts never projects. The inverse
retraction does not normalize: its inputs are points of the type.

Each composite is written on storage with `algebra.rs`, so the motions' declarations do not depend
on the rotation's statics. The denominators are positive on the domain: $1 + s \ge 1$,
$\lVert x \rVert > 0$, and $1 + w(\bar x \otimes y) > 0$ for $y \ne -x$, where the inverse is
undefined and gives non-finite rows, as the chart's cut.

```rust
pub static ROTATION: LazyLock<Arc<Embedded>> = LazyLock::new(|| {
    match declare_rotation() {
        Ok(declaration) => declaration,
        Err(_) => unreachable!("the rotation's declaration is checked by `types_are_declared`"),
    }
});

fn declare_rotation() -> Result<Arc<Embedded>, Error> {
    let (n, d) = (Space::Vector(4), Space::Vector(3));
    let define = |name, inputs: &[Space], body: fn(&[QDyn]) -> QDyn| {
        Composite::define(name, Vec::new(), inputs, body).map_err(|f| f.error)
    };
    let r = define("skoro.rotations.rotation.retract", &[n.clone(), d.clone()],
                   |q| mul(normalize(q[0]), delta(q[1])))?;
    let q = define("skoro.rotations.rotation.local", &[n.clone(), n.clone()],
                   |q| delta_inverse(mul(conjugate(q[0]), q[1])))?;
    let p = define("skoro.rotations.rotation.project", &[n.clone()], |q| normalize(q[0]))?;
    let lambda = define("skoro.rotations.rotation.log_volume", &[n.clone(), d],
                        |q| ((q[1] * q[1]).sum() * (1.0 / 16.0) + 1.0).log() * -3.0)?;
    Embedded::declare("skoro.rotations.rotation", Vec::new(), 4, 3, r, q, Some(p))?
        .with_volume(Volume::global(lambda))
}
```

The markers are unit structs, each admitting its declaration's space and declaring it:

```rust
/// A unit quaternion `(w, x, y, z)`: a rotation, and its double cover. Tangent coordinates are the
/// rotation vector in the frame of the point, to first order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Rotation;
/// `(q, p)`: a rigid motion, its tangent coordinates `(φ, ρ)` in the frame of the motion.
pub struct Pose;
/// `(q, v, p)`: an extended pose, its tangent coordinates `(φ, η, ρ)` in its own frame.
pub struct ExtendedPose;

impl Typed for Rotation {
    fn admits(space: &Space) -> bool { matches!(space, Space::Embedded(e) if **e == **ROTATION) }
}
impl Declares for Rotation {
    fn space(&self) -> Space { Space::Embedded(ROTATION.clone()) }
}
// likewise for Pose and ExtendedPose

/// The three declarations, by name, with no parameters.
pub fn register(registry: &mut Registry) -> Result<(), Error>;
```

Host values give a type's storage in its order, the scalar part of a quaternion first:

```rust
/// `(w, x, y, z)`, normalized on the host; `None` for a zero or non-finite quaternion. A sensor that
/// gives `(x, y, z, w)` must be reordered.
impl IntoHostValue<Rotation> for [f64; 4];
/// The rotation and the position.
impl IntoHostValue<Pose> for ([f64; 4], [f64; 3]);
impl IntoHostValue<Pose> for [f64; 7];
/// The rotation, the velocity and the position.
impl IntoHostValue<ExtendedPose> for ([f64; 4], [f64; 3], [f64; 3]);
impl IntoHostValue<ExtendedPose> for [f64; 10];
```

### Series and cutoffs

`series.rs` computes the coefficients of
[Removable singularities](math.svx#removable-singularities) and the motions' $D$ and $E$
([](eq:archive/rotations/math/motion-q)), each by the core's guarded pattern: a discretization of
the argument against a cutoff selects the series, the closed form's argument is replaced by a
harmless one on the rows where the series is chosen, and the series' argument by one on the rows
where the closed form is, wherever the series could be undefined.

```rust
/// Which coefficient, of `θ²`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Coefficient { A0, B0, A, B, C, D, E }      // a, b of eq:exp; A, B; C; D, E

/// The coefficient at `θ² = theta2`, a `Real`: the series below `CUTOFF`, the closed form above.
fn coefficient(c: Coefficient, theta2: QDyn) -> QDyn {
    let small = (CUTOFF - theta2).is_nonnegative();
    let safe = choose(small, theta2, one());               // the closed form's argument
    let theta = (safe.log() * 0.5).exp();
    choose(small, closed(c, theta), horner(series(c), theta2))
}

/// `L = 2 atan2(n, w) / n` of eq:log from `n²` and `w`: its series in `x = n²/w²` where
/// `n² < LOG_CUTOFF w²` and `w > 0`, the closed form elsewhere.
fn log_ratio(n2: QDyn, w: QDyn) -> QDyn {
    let small = and(w.is_nonnegative(), (w * w * LOG_CUTOFF - n2).is_nonnegative());
    let n = (choose(small, n2, one()).log() * 0.5).exp();
    let w_safe = choose(small, one(), w);                   // the series' 1/w where it is unused
    let x = n2 * reciprocal(w_safe * w_safe);
    choose(small, n.atan2(w) * reciprocal(n) * 2.0, horner(LOG_SERIES, x) * reciprocal(w_safe) * 2.0)
}
```

`and` of two Booleans is the selection of the second by the first, with false where the first is
false. On a row with $w = 0$ exactly, $n^2 \ge 0 = c\, w^2$, so the series is never chosen there, and
its guard keeps its $1/w$ finite.

One cutoff serves both precisions. A composite cannot see its target's precision, and the
measurements below show that no second one is needed, so the alternative, an operation of the
crate's own lowering to a composite per precision through `Lowering::target`, with registry entries
for each, lost. The two conditions, measured in double precision against the series summed to
convergence, and in single precision against double, at the cutoff:

| coefficient | argument | cutoff | series kept | f64 truncation at the cutoff | f32 closed form above it: value, first, second derivative in $\theta$ |
| --- | --- | --- | --- | --- | --- |
| $a$, $b$ | $\theta^2$ | $0.1$ | through $\theta^{10}$ | below $2^{-53}$ | $b$: $9 \cdot 10^{-9}$, $2 \cdot 10^{-7}$, $4 \cdot 10^{-7}$ |
| $A$ | $\theta^2$ | $0.1$ | through $\theta^{10}$ | below $2^{-53}$ | $3 \cdot 10^{-7}$, $2 \cdot 10^{-6}$, $2 \cdot 10^{-5}$ |
| $B$ | $\theta^2$ | $0.1$ | through $\theta^{10}$ | below $2^{-53}$ | $1 \cdot 10^{-7}$, $3 \cdot 10^{-7}$, $6 \cdot 10^{-7}$ |
| $C$ | $\theta^2$ | $0.1$ | through $\theta^{12}$ | below $2^{-53}$ | no cancellation worse than $A$'s |
| $D$ | $\theta^2$ | $0.1$ | through $\theta^{10}$ | below $2^{-53}$ | $2 \cdot 10^{-7}$, $3 \cdot 10^{-6}$, $4 \cdot 10^{-5}$ |
| $E$ | $\theta^2$ | $0.1$ | through $\theta^{10}$ | below $2^{-53}$ | $2 \cdot 10^{-7}$, $5 \cdot 10^{-6}$, $2 \cdot 10^{-4}$ |
| $L$ | $x = n^2/w^2$ | $0.01$ | through $x^7$ | below $2^{-53}$ | $9 \cdot 10^{-8}$, $3 \cdot 10^{-7}$ |

The single-precision errors are absolute, as `Tolerance::AGREEMENT` holds them, and each
coefficient reaches a map's value multiplied by powers of $\theta$ ($E$ by $\Phi \Psi \Phi^2$, of
order $\theta^3 \lVert \rho \rVert$), so the largest, $E$'s second derivative, contributes about
$10^{-5}$. A cutoff of $\theta^2 = 0.01$ would cost $D$ and $E$ errors of $10^{-1}$ and more in their
second derivatives in single precision; one of $0.1$ for $L$ would need its series through $x^{14}$.
The series are evaluated by Horner's rule, a product and a sum per term. The archive's cutoffs per
precision ($\theta^2 \approx 1.37$ and $0.025$, series through $\theta^8$) are not kept.

### The maps of the rotations

`rotation.rs`. Each map with more than a few nodes is a composite named `skoro.rotations.<map>`,
defined once in a `LazyLock` as the types are; the operators and methods call it.

```rust
/// `q₁ * q₂ = P(q₁ ⊗ q₂)`.
impl QMul for Rotation { type Output = Rotation; }
/// `q * x = π(q) x`, a vector, or each column of a `3 × N` matrix.
impl QMul<Vector<3>> for Rotation { type Output = Vector<3>; }
impl QMul<VectorDyn> for Rotation { type Output = VectorDyn; }
impl<const N: usize> QMul<Matrix<3, N>> for Rotation { type Output = Matrix<3, N>; }

/// The maps of a rotation, as methods of its handle.
pub trait RotationMaps: Sized {
    /// `P(q̄)`.
    fn inverse(self) -> Q<Rotation>;
    /// `P(q̄₁ ⊗ q₂)`: the rotation from `self` to `other`, in the frame of `self`.
    fn between(self, other: Q<Rotation>) -> Q<Rotation>;
    /// `π(q)ᵀ x`: a vector of the outer frame in the frame of `self`.
    fn unrotate(self, x: Q<Vector<3>>) -> Q<Vector<3>>;
    /// eq:rotation.
    fn matrix(self) -> Q<Matrix<3, 3>>;
    /// The adjoint, `π(q)`, carrying a perturbation in the frame of `self` to the outer frame.
    fn adjoint(self) -> Q<Matrix<3, 3>>;
    /// eq:log: the inverse of `exp`, of length in [0, 2π); non-finite at `-1`.
    fn log(self) -> Q<Vector<3>>;
    /// eq:rotation-vector: the rotation vector of the shorter turn, the same at `±q`.
    fn log_shorter(self) -> Q<Vector<3>>;
    /// eq:angle, in [0, π].
    fn angle_to(self, other: Q<Rotation>) -> Q<Real>;
    /// eq:slerp, and with `log_shorter` in place of `log`.
    fn slerp(self, other: Q<Rotation>, t: Q<Real>) -> Q<Rotation>;
    fn slerp_shorter(self, other: Q<Rotation>, t: Q<Real>) -> Q<Rotation>;
    /// The statement euler: the angles of `convention`, `α` and `γ` in (-π, π].
    fn euler(self, convention: Euler) -> Q<Vector<3>>;
    /// The axis and the angle of `log`; the axis `(1, 0, 0)` at the identity.
    fn axis_angle(self) -> (Q<Vector<3>>, Q<Real>);
    /// `sign⟨q, r⟩ q`: the point over this rotation on the side of `reference`.
    fn aligned_with(self, reference: Q<Rotation>) -> Q<Rotation>;
}
impl RotationMaps for Q<Rotation> { /* each a call of its composite, narrowed */ }

pub mod rotation {
    pub fn identity() -> Q<Rotation>;
    /// eq:exp.
    pub fn exp(omega: Q<Vector<3>>) -> Q<Rotation>;
    /// The rotation by `angle` about `axis ≠ 0`.
    pub fn about(axis: Q<Vector<3>>, angle: Q<Real>) -> Q<Rotation>;
    /// `q_e(θ)` about a coordinate axis.
    pub fn elementary(axis: Axis, angle: Q<Real>) -> Q<Rotation>;
    /// eq:two-vectors; non-finite where `b` is opposite to `a`, or either is zero.
    pub fn from_two_vectors(a: Q<Vector<3>>, b: Q<Vector<3>>) -> Q<Rotation>;
    /// eq:euler.
    pub fn from_euler(angles: Q<Vector<3>>, convention: Euler) -> Q<Rotation>;
    /// eq:from-matrix.
    pub fn from_matrix(m: Q<Matrix<3, 3>>) -> Q<Rotation>;
}
```

The exponential is $P(a\,,\ b\, \omega)$ with $a$ and $b$ from `series.rs` of $\theta^2 = \omega \cdot
\omega$; the logarithm is `log_ratio(n², w) u`, and `log_shorter` the same of the storage chosen
between $q$ and $-q$ by $w \ge 0$. The angle is $2 \operatorname{atan2}(\lVert \operatorname{vec}
r \rVert, \lvert w(r) \rvert)$, the absolute value a selection; its norm is undefined at $r = \pm 1$,
where the angle has no derivative and its rows are non-finite in the passes, as a norm's are. The
axis–angle guards the axis with a selection at $n^2 = 0$. Alignment selects between the storage of
$q$ and its negation by $\langle q, r \rangle \ge 0$ and projects.

`from_matrix` computes the four arguments $c_m(X)$ of [](eq:archive/rotations/math/from-matrix) as
linear maps of the stored matrix, chooses one by a tournament on their diagonal entries, and
projects only the one chosen:

```rust
// x = the stored Matrix(3, 3) as a Vector(9); X_ij = x[i + 3j].
let c = [c0(x), c1(x), c2(x), c3(x)];                    // each a Vector(4)
let d = |m: usize| c[m].slice(m..m + 1).reshape(Space::Real);
let first = (d(1) - d(0)).is_nonnegative();               // 1 beats 0
let second = (d(3) - d(2)).is_nonnegative();              // 3 beats 2
let (a, da) = (choose(first, c[0], c[1]), choose(first, d(0), d(1)));
let (b, db) = (choose(second, c[2], c[3]), choose(second, d(2), d(3)));
normalize(choose((db - da).is_nonnegative(), a, b))
```

The chosen argument has norm at least $2$, so the projection never meets a small one.

`so3.rs` holds the Euclidean maps of rotation vectors:

```rust
pub mod so3 {
    /// `[ω]×` and its inverse on skew matrices, the vector of the skew part of any other.
    pub fn hat(omega: Q<Vector<3>>) -> Q<Matrix<3, 3>>;
    pub fn vee(m: Q<Matrix<3, 3>>) -> Q<Vector<3>>;
    /// eq:jacobians and eq:jacobian-inverses; the inverses non-finite at `|ω| = 2π`.
    pub fn jl(omega: Q<Vector<3>>) -> Q<Matrix<3, 3>>;
    pub fn jr(omega: Q<Vector<3>>) -> Q<Matrix<3, 3>>;
    pub fn jl_inv(omega: Q<Vector<3>>) -> Q<Matrix<3, 3>>;
    pub fn jr_inv(omega: Q<Vector<3>>) -> Q<Matrix<3, 3>>;
}
```

each $I + c_1 [\omega]_\times + c_2 [\omega]_\times^2$ with the coefficients of `series.rs`.

### Euler angles

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Axis { X, Y, Z }

/// The twelve sequences: six of Tait–Bryan type and six proper.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Sequence { Xyz, Xzy, Yxz, Yzx, Zxy, Zyx, Xyx, Xzx, Yxy, Yzy, Zxz, Zyz }

/// About the axes as the previous rotations left them, or about the fixed axes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Frame { Intrinsic, Extrinsic }

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Euler { pub sequence: Sequence, pub frame: Frame }

impl Euler {
    /// Yaw, pitch and roll `(ψ, ϑ, φ)`, intrinsic about `(z, y, x)`: the aerospace convention.
    pub const YAW_PITCH_ROLL: Euler = Euler { sequence: Sequence::Zyx, frame: Frame::Intrinsic };
    pub fn axes(self) -> [Axis; 3];
}
```

The enum makes a sequence with two equal neighbouring axes unrepresentable. The angles' vector holds
them in the order of the sequence's axes. Both directions are composites per intrinsic sequence,
`skoro.rotations.from_euler` and `skoro.rotations.euler` with the parameter `Text("zyx")`, cached by
sequence as the normal caches its composites; an extrinsic convention calls the composite of the
reversed sequence with the angles reversed, by slices and a concatenation outside it.

`from_euler` is [](eq:archive/rotations/math/euler), three elementary quaternions multiplied and
projected. `euler` is [%full](statement:archive/rotations/math/euler): for a Tait–Bryan sequence it
first forms the linear rearrangement $(w - u_j,\ u_i - \epsilon u_k,\ u_j + w,\ u_k + \epsilon u_i)$,
then for both kinds

```rust
// (w, ui, uj, ul) read from the quaternion or its rearrangement, eps = ±1 for the sequence.
let (na, nb) = (w * w + ui * ui, uj * uj + ul * ul);
let total = na + nb;
let low = (total * LOCK - nb).is_nonnegative();          // β at 0 (proper) or -π/2 (Tait–Bryan)
let high = (total * LOCK - na).is_nonnegative();         // β at π or π/2
let beta = choose(low, choose(high, two_atan2(norm(nb), norm(na)), pi()), zero());
let sigma = ui.atan2(choose(high, w, one()));             // guarded where (w, ui) vanishes
let delta = (ul * eps).atan2(choose(low, uj, one()));     // guarded where (uj, ul) vanishes
let alpha = choose(low, choose(high, sigma + delta, delta * 2.0), sigma * 2.0);
let gamma = choose(low, choose(high, sigma - delta, zero()), zero());
```

with the norms' arguments guarded as `log_ratio`'s are on the rows they are not used, $\beta$
shifted by $-\tfrac\pi2$ and $\gamma$ multiplied by $-\epsilon$ for a Tait–Bryan sequence, and
$\alpha$ and $\gamma$ wrapped to $(-\pi, \pi]$ as $\operatorname{atan2}(\sin \cdot, \cos \cdot)$.
`LOCK` is $10^{-30}$: the lock is taken only where one norm vanishes to within rounding of nothing,
so that on every other row the angles are the statement's, however ill-conditioned; at an exact
lock, as at the identity of a proper sequence, the convention $\gamma = 0$ gives finite angles and
finite derivatives. A larger threshold would replace the angles by the convention on a band of
rotations that it does not reproduce, by up to the band's $\beta$.

### The maps of the rigid motions

`pose.rs` and `extended.rs`, likewise, with the tangent coordinates rotation first:

```rust
impl QMul for Pose { type Output = Pose; }                     // composition
impl QMul<Vector<3>> for Pose { type Output = Vector<3>; }     // the action on a point, R x + p

pub trait PoseMaps: Sized {
    fn inverse(self) -> Q<Pose>;
    fn between(self, other: Q<Pose>) -> Q<Pose>;              // T₁⁻¹ T₂
    fn rotate(self, x: Q<Vector<3>>) -> Q<Vector<3>>;          // R x: a direction
    fn untransform(self, x: Q<Vector<3>>) -> Q<Vector<3>>;     // Rᵀ (x - p)
    fn rotation(self) -> Q<Rotation>;
    fn position(self) -> Q<Vector<3>>;
    fn matrix(self) -> Q<Matrix<4, 4>>;
    fn adjoint(self) -> Q<Matrix<6, 6>>;                       // eq:motion-adjoint
    fn log(self) -> Q<Vector<6>>;                              // eq:motion-log
    fn log_shorter(self) -> Q<Vector<6>>;
    fn interpolate(self, other: Q<Pose>, t: Q<Real>) -> Q<Pose>;          // eq:motion-slerp, first
    fn interpolate_shorter(self, other: Q<Pose>, t: Q<Real>) -> Q<Pose>;
    fn interpolate_parts(self, other: Q<Pose>, t: Q<Real>) -> Q<Pose>;    // eq:motion-slerp, second
    fn interpolate_parts_shorter(self, other: Q<Pose>, t: Q<Real>) -> Q<Pose>;
    /// The same storage as a value of the product S³ × ℝ³, whose chart moves the position in a
    /// fixed frame.
    fn as_product(self) -> Q<ProductDyn>;
}

pub mod pose {
    pub fn identity() -> Q<Pose>;
    pub fn exp(xi: Q<Vector<6>>) -> Q<Pose>;                   // eq:motion-exp
    pub fn from_parts(q: Q<Rotation>, p: Q<Vector<3>>) -> Q<Pose>;
    pub fn from_product(x: Q<ProductDyn>) -> Q<Pose>;          // SpaceMismatch unless S³ × ℝ³
    /// eq:motion-jacobians, with Q of eq:motion-q.
    pub fn jl(xi: Q<Vector<6>>) -> Q<Matrix<6, 6>>;
    pub fn jr(xi: Q<Vector<6>>) -> Q<Matrix<6, 6>>;
    pub fn jl_inv(xi: Q<Vector<6>>) -> Q<Matrix<6, 6>>;
    pub fn jr_inv(xi: Q<Vector<6>>) -> Q<Matrix<6, 6>>;
}

impl QMul for ExtendedPose { type Output = ExtendedPose; }

pub trait ExtendedPoseMaps: Sized {
    fn inverse(self) -> Q<ExtendedPose>;
    fn between(self, other: Q<ExtendedPose>) -> Q<ExtendedPose>;
    fn rotation(self) -> Q<Rotation>;
    fn velocity(self) -> Q<Vector<3>>;
    fn position(self) -> Q<Vector<3>>;
    fn pose(self) -> Q<Pose>;
    fn matrix(self) -> Q<Matrix<5, 5>>;
    fn adjoint(self) -> Q<Matrix<9, 9>>;                       // eq:extended-adjoint
    fn log(self) -> Q<Vector<9>>;
    fn log_shorter(self) -> Q<Vector<9>>;
    fn interpolate(self, other: Q<ExtendedPose>, t: Q<Real>) -> Q<ExtendedPose>;
    fn interpolate_shorter(self, other: Q<ExtendedPose>, t: Q<Real>) -> Q<ExtendedPose>;
}

pub mod extended_pose {
    pub fn identity() -> Q<ExtendedPose>;
    pub fn exp(xi: Q<Vector<9>>) -> Q<ExtendedPose>;
    pub fn from_parts(q: Q<Rotation>, v: Q<Vector<3>>, p: Q<Vector<3>>) -> Q<ExtendedPose>;
    pub fn jl(xi: Q<Vector<9>>) -> Q<Matrix<9, 9>>;
    pub fn jr(xi: Q<Vector<9>>) -> Q<Matrix<9, 9>>;
    pub fn jl_inv(xi: Q<Vector<9>>) -> Q<Matrix<9, 9>>;
    pub fn jr_inv(xi: Q<Vector<9>>) -> Q<Matrix<9, 9>>;
}
```

A map into a motion computes its storage and projects, so its rotation is normalized and its
translations pass through. `rotation()` projects the slice $0..4$ of the inclusion onto `Rotation`,
and `from_parts` concatenates the inclusions and projects onto the motion; `as_product` and
`from_product` are the identity on storage, a product of `Rotation` and `Vector<3>` built with
`scope::product` and read back by `factor`. The matrices are concatenated column by column, the
adjoints and Jacobians block by block from $\pi(q)$, $[p]_\times$, $J_l$ and $Q$, with the blocks in
the order $(\phi, \rho)$ and $(\phi, \eta, \rho)$. $Q(\phi, \rho)$ of
[](eq:archive/rotations/math/motion-q) is a composite, `skoro.rotations.motion_q`, of two
`Vector(3)`s, with $B$, $D$ and $E$ from `series.rs`; the inverses are the block forms of
[](eq:archive/rotations/math/motion-jacobians), with $J_l^{-1}$ of the rotations.

The extended pose's maps go beyond what the mathematics lists in two places, both by the same
formulas: the inverses of its Jacobians, the block lower-triangular inverse with the blocks
$-J_l^{-1} Q(\phi, \eta) J_l^{-1}$ and $-J_l^{-1} Q(\phi, \rho) J_l^{-1}$, and its interpolation by
the group, $T_1 \Exp(t \Log(T_1^{-1} T_2))$. Its action on points is that of its pose.

### What each map is

Every map of the mathematics' tables, with the item that implements it, its composite, its
equation, its kind, and the test that checks it. *Smooth* maps have derivatives of both orders on
their domain; *removable* ones are smooth, with their singularity at the identity computed by
`series.rs`; *discrete* ones select between smooth formulas, and have no derivative across the jump.

| map | item | composite `skoro.rotations.` | equation | kind | test |
| --- | --- | --- | --- | --- | --- |
| the rotation's chart and volume | `Rotation` | `rotation.*` | quaternion-chart | smooth | `types_pass_the_core_checks` |
| identity | `rotation::identity` | (a constant) | | smooth | `group_laws` |
| composition | `Q<Rotation> * Q<Rotation>` | `compose` | product | smooth | `group_laws` |
| inverse | `inverse` | `inverse` | | smooth | `group_laws` |
| relative rotation | `between` | `between` | | smooth | `group_laws` |
| action, inverse action | `* Q<Vector<3>>`, `unrotate` | `rotate`, `unrotate` | rotation | smooth | `actions_are_the_matrix` |
| matrix, adjoint | `matrix`, `adjoint` | `matrix` | rotation, conjugation | smooth | `actions_are_the_matrix`, `conjugation_is_exact` |
| hat, vee | `so3::hat`, `so3::vee` | (inline) | | smooth | `hat_and_vee` |
| exponential | `rotation::exp` | `exp` | exp | removable | `exp_and_log` |
| logarithm | `log` | `log` | log | removable | `exp_and_log` |
| rotation vector | `log_shorter` | `log_shorter` | rotation-vector | discrete | `exp_and_log` |
| Jacobians and inverses | `so3::jl`, `jr`, `jl_inv`, `jr_inv` | `jl`, `jl_inv` | jacobians, jacobian-inverses | removable | `jacobians_are_the_passes` |
| interpolation | `slerp`, `slerp_shorter` | `slerp`, `slerp_shorter` | slerp | removable, discrete | `interpolations` |
| angle | `angle_to` | `angle` | angle | discrete | `interpolations` |
| elementary, axis–angle | `rotation::elementary`, `about`, `axis_angle` | `about`, `axis_angle` | | smooth, discrete | `constructions` |
| two directions | `rotation::from_two_vectors` | `from_two_vectors` | two-vectors | smooth | `constructions` |
| Euler angles | `rotation::from_euler`, `euler` | `from_euler`, `euler` | euler, statement euler | smooth, discrete | `euler_round_trips` |
| from a matrix | `rotation::from_matrix` | `from_matrix` | from-matrix | discrete | `from_matrix_branches` |
| alignment | `aligned_with`, `host::align` | `align` | | discrete | `alignment` |
| the motions' charts and volumes | `Pose`, `ExtendedPose` | `pose.*`, `extended_pose.*` | statement motions | smooth | `types_pass_the_core_checks` |
| composition, inverse, relative | `*`, `inverse`, `between` | `pose.compose`, … | | smooth | `motion_group_laws` |
| actions | `* Q<Vector<3>>`, `rotate`, `untransform` | `pose.transform`, … | | smooth | `motion_group_laws` |
| parts | `rotation`, `position`, `velocity`, `pose`, `from_parts` | (inline) | | smooth | `parts_and_products` |
| product | `as_product`, `pose::from_product` | (inline) | | smooth | `parts_and_products` |
| matrices | `matrix` | `pose.matrix`, `extended_pose.matrix` | | smooth | `motion_group_laws` |
| adjoints | `adjoint` | `pose.adjoint`, `extended_pose.adjoint` | motion-adjoint, extended-adjoint | smooth | `adjoints_conjugate` |
| exponentials, logarithms | `pose::exp`, `log`, `log_shorter` | `pose.exp`, `pose.log`, … | motion-exp, motion-log | removable, discrete | `motion_exp_and_log` |
| Jacobians | `pose::jl`, …, `extended_pose::jl`, … | `motion_q`, `pose.jl`, … | motion-jacobians, motion-q | removable | `jacobians_are_the_passes` |
| interpolations | `interpolate`, `interpolate_parts`, … | `pose.interpolate`, … | motion-slerp | removable, discrete | `interpolations` |

### Host helpers

`host.rs`, on `[f64; 4]` in the order $(w, x, y, z)$, in double precision, for the values a model
is given:

```rust
pub mod host {
    /// `sign⟨z, r⟩ z`, normalized.
    pub fn align(z: [f64; 4], reference: [f64; 4]) -> [f64; 4];
    pub fn compose(a: [f64; 4], b: [f64; 4]) -> [f64; 4];
    pub fn matrix(q: [f64; 4]) -> [[f64; 3]; 3];
    /// eq:from-matrix, by rows.
    pub fn from_matrix(m: [[f64; 3]; 3]) -> [f64; 4];
    pub fn from_euler(angles: [f64; 3], convention: Euler) -> [f64; 4];
    pub fn euler(q: [f64; 4], convention: Euler) -> [f64; 3];
    pub fn exp(omega: [f64; 3]) -> [f64; 4];
    pub fn log(q: [f64; 4]) -> [f64; 3];
}
```

They are the same formulas in plain `f64`, with branches where the composites select. The
mathematics asks for alignment on the host because an observed value is fixed before it is
observed; `aligned_with` exists as well for a quaternion a model computes.

### Errors and domains

A wrong input is refused when the map is applied, by the composite's call, with `SpaceMismatch`
naming the composite, at the caller's line; nothing else is refused at build time. A row outside a
map's domain gives non-finite values, as the core's own logarithm does outside its: the inverse
retraction at $y = -x$, `log` at $-1$, `jl_inv` and the motions' logarithms at an angle of $2\pi$,
`from_two_vectors` for opposite or zero directions, `angle_to`'s derivatives where the angle is $0$
or $\pi$. Each is said in the item's documentation. The selections of the discrete maps keep their
values and the derivatives of the selected formula finite on both sides of every jump.

## Tests

In `crates/skoro-rotations/tests/`, at both precisions of the Flex device through
`precision_tests!` wherever a pass is compared, with `testing::Tolerance`, at points drawn by
`testing::random` and normalized, together with fixed rows: the identity, a half-turn about each
axis, rotations whose angles sit at $0.9$ and $1.1$ times each cutoff, and for the logarithm a row
with $w < 0$.

- `types_are_declared`: the three declarations and every composite are defined without error, and
  `register` adds three names to a `Registry::core()` and refuses them a second time. Shows the
  `unreachable!` branches are dead.
- `types_pass_the_core_checks`: `testing::check_embedded` and `testing::check_volume` on `Rotation`,
  `Pose` and `ExtendedPose`, at the fixed rows and random ones. Shows
  [%full](statement:archive/rotations/math/chart), [%full](statement:archive/rotations/math/volume)
  and [%full](statement:archive/rotations/math/motions) as built, with the area formula.
- `chart_is_stereographic`: on the host, `Q(x, R(x, t)) = t` at lengths up to $10^3$; the angle of
  $\delta(t)$ is $4 \arctan(\lVert t \rVert / 4)$; $\pi(x \oplus t) = \pi(x)(I + [t]_\times) +
  O(\lVert t \rVert^2)$ at $\lVert t \rVert = 10^{-3}$; and $R$ of a base point scaled by $1.01$ equals
  $R$ of the unit one. Shows the chart is the statement's, and the normalization of the base point.
- `volume_has_the_stated_mass`: $\int_{\R^3} e^{\lambda(t)} \dd t = 16 \pi^2$ by quadrature in polar
  coordinates. Shows the normalization of $\nu$ independently of the area formula.
- `group_laws`: on random rows, associativity, identity and inverse of the composition, `between`
  against `inverse` and `*`, and $\pi$ a homomorphism, the same at $q$ and $-q$, all against host
  references written with matrices.
- `actions_are_the_matrix`: `q * x`, `unrotate`, the action on a `Matrix<3, 4>` of columns and
  `matrix` against the host's $\pi(q)$; `matrix` of a product is the product of the matrices.
- `conjugation_is_exact`: [](eq:archive/rotations/math/conjugation) at lengths $0.1$, $1$ and $10$.
- `hat_and_vee`: $[\omega]_\times x = \omega \times x$; `vee` inverts `hat`, and gives the skew
  part's vector of a matrix that is not skew.
- `exp_and_log`: `log` inverts `exp` on the ball of radius $2\pi$, at lengths up to $6$;
  `log_shorter` is the same at $\pm q$, agrees with `log` where $w > 0$, and has length at most
  $\pi$; `exp` at an angular rate integrated against the matrix exponential of $[\omega]_\times$.
- `series_meet_closed_forms`: every coefficient of `series.rs`, as a map of $\omega$ or of $(n^2,
  w)$, against the series summed to convergence below its cutoff and the closed form above,
  at $0.9$ and $1.1$ times the cutoff, within $10^{-15}$ in double precision; and
  `testing::check_operation` on every composite that uses one, at the identity and at both sides of
  the cutoff. Shows the measured table of [Series and cutoffs](#series-and-cutoffs) holds in the
  code, and that every pass is finite at the identity.
- `composites_pass_check_operation`: `testing::check_operation` on every composite of the crate, at
  the fixed and random rows, at both precisions: tangent, cotangent and second-order maps against
  finite differences in the types' charts. Shows every map has its derivatives of both orders.
- `jacobians_are_the_passes`: the tangent pass of `rotation::exp` at $\omega$, read in the chart at
  `exp(ω)`, equals `so3::jr(ω)`; of `log`, `so3::jr_inv(log q)`; of `pose::exp` and
  `extended_pose::exp`, their `jr`; and each `*_inv` times its Jacobian is the identity. Shows
  [%full](statement:archive/rotations/math/first-order), and with it [](eq:archive/rotations/math/motion-q)
  in the order of the coordinates here.
- `second_order_agrees_on_rotations`: the Hessian pass of a scalar map of a rotation, $x \mapsto
  \langle \pi(x)\, a, b \rangle$, at $x$ equals that of the same map read in exponential coordinates
  by finite differences; for a pose they differ by the term the statement names. Shows the second
  claim of [%full](statement:archive/rotations/math/first-order).
- `interpolations`: `slerp` and its variants at $t = 0$ and $t = 1$ give the endpoints; the angle
  from the start grows linearly in $t$; `slerp_shorter` takes the shorter turn where
  $\langle q_1, q_2 \rangle < 0$ and `slerp` the longer; `angle_to` against the host's
  $\arccos((\operatorname{tr} \pi(\bar q_1 q_2) - 1)/2)$; the motions' interpolations at their
  endpoints, the screw's twist constant in $t$ and the parts' position linear.
- `constructions`: `elementary` and `about` against the host's matrices; `axis_angle` inverts
  `about` and is finite at the identity; `from_two_vectors` carries $a$ to $b$ for random pairs, and is
  non-finite for opposite ones.
- `euler_round_trips`: for all twelve sequences, intrinsic and extrinsic, `euler(from_euler(θ))` is
  $\theta$ modulo $2\pi$ for random angles away from the lock, the same for $q$ and $-q$; the
  extrinsic rotation equals the product of the fixed axes' matrices; at an exact lock the angles are
  finite, $\gamma = 0$, and reproduce the rotation; `Euler::YAW_PITCH_ROLL` against a textbook
  example of yaw, pitch and roll. Shows [%full](statement:archive/rotations/math/euler).
- `from_matrix_branches`: at quaternions whose largest coordinate is each of the four in turn,
  `from_matrix(matrix(q))` is $\pm q$ with that coordinate positive, both through the model and
  through `host::from_matrix`; the matrix's storage order is checked by a rotation about $z$ alone.
  Shows [](eq:archive/rotations/math/from-matrix) and catches a transposed index.
- `alignment`: `aligned_with` and `host::align` return $z$ or $-z$, whichever is on the side of the
  reference, and the same rotation.
- `motion_group_laws`: the composition, inverse, relative motion and actions of `Pose` and
  `ExtendedPose` against their homogeneous matrices on the host, and `matrix` a homomorphism.
- `parts_and_products`: `from_parts` and `rotation`, `position` (and `velocity`, `pose`) invert each
  other; `as_product` and `from_product` are the identity on storage, and the derivative of
  `from_product` in charts is $\operatorname{diag}(I, R\transpose)$.
- `adjoints_conjugate`: $T \Exp(\xi) T^{-1} = \Exp(\operatorname{Ad}(T)\, \xi)$ for both motions, and
  $T \oplus h\xi = (1 \oplus h \operatorname{Ad}(T) \xi)\, T + O(h^2)$.
- `motion_exp_and_log`: `pose::exp` and `extended_pose::exp` against the matrix exponential of the
  twist on the host; their logarithms invert them; `log_shorter` the same at $\pm q$.
- `normal_density_on_rotations`: with `skoro-normal`, the normal's term on `Rotation` integrates to
  one through the chart, $\int_{\R^3} \exp(\logterm(\mu \oplus t) + \lambda_\mu(t))\, \dd t = 1$, by
  quadrature in polar coordinates with $r = 4 \tan(u/4)$, for an isotropic and an anisotropic scale.
- `normal_density_on_poses`: on `Pose`, $\logterm(\mu \oplus t) + \lambda_\mu(t) = \log \mathcal
  N(t; 0, \Sigma)$ at random $t$, which with a global, full chart and a checked volume is the
  density integrating to one, without a quadrature in six dimensions.
- `normal_draws_on_rotations`: over $10^5$ draws, the expectations of three bounded functions of the
  rotation, through $\pi$, agree within five standard errors with their quadratures against
  $\density(q) + \density(-q)$ over the hemisphere $w > 0$. Shows
  [%full](statement:archive/rotations/math/cover).

The core's acceptance test is not touched, as the mathematics settles.

## Open questions

> [!TODO] The names
> The design names the crate `skoro-rotations`, its markers `Rotation`, `Pose` and `ExtendedPose`,
> its extension traits `RotationMaps`, `PoseMaps` and `ExtendedPoseMaps`, and its declarations and
> composites under `skoro.rotations.`. `Rotation` for a unit quaternion hides the double cover in
> its name, where `UnitQuaternion` would show it; `Pose` is the robotics name for a rigid motion.
> Are these the names wanted?
