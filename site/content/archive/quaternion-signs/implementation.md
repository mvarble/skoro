---
summary: The core declares a type stored up to a finite group of linear maps and checks it; skoro-spatial's rotations, motions and extended poses become SO(3), SE(3) and SE₂(3) under the Cayley chart, stored up to the quaternion's sign; their maps that read the sign are replaced by those of the shorter turn; and a pose graph read from g2o text is estimated in whatever signs it is given.
depends_on: [archive/quaternion-signs/math, archive/core/implementation, archive/measures/implementation, archive/rotations/implementation, archive/linearization/implementation]
---

## Overview

When this is built, a model that observes a rotation, a pose or an extended pose writes the same
code it writes today, and the sign of every quaternion it is given stops mattering:

```rust
// One edge of a pose graph read from a g2o file: the measured motion, observed in whatever sign
// the file wrote it, and its information converted to skoro's coordinates.
let z = quantity(format!("edge/{e}"), Pose);
Normal::on(Pose, x[i].between(x[j]), Scale::precision(&edge.precision, 6)).score(z);
// ...
let given = Observations::new().observe(z, edge.measurement); // either sign
```

Four things change.

1. **The core** gains the declaration of an embedded type stored up to a finite group of linear
   maps of its storage, [%full](statement:archive/quaternion-signs/math/quotient):
   `Embedded::stored_up_to`, the group's generators as `Linear` maps, and the checks
   `check_embedded` and `check_volume` extended to the group's conditions. Nothing the core
   computes reads the group; its derivatives are unchanged, as the statement says.
2. **skoro-spatial**'s three types take the Cayley chart of
   [%full](statement:archive/quaternion-signs/math/cayley), with the inverse computed as
   [](eq:archive/quaternion-signs/math/cayley-composite) and the log-volume
   [](eq:archive/quaternion-signs/math/cayley-volume), and declare themselves stored up to the
   sign of their quaternion. The maps that read the representative are replaced by their
   counterparts on the groups, under the plain names
   ([The maps of the rotations](math.svx#the-rotations-the-rigid-motions-and-the-extended-poses)):
   `log`, `axis_angle`, `slerp`, the poses' `log`, `interpolate` and `interpolate_parts`, and the
   extended poses' `log`, are those of the shorter turn, and the `*_shorter` names are removed.
3. **skoro-linearize** refuses a linearization point at which a factor's residual or Jacobian is
   not finite, naming the term and the row (`Error::NotFinite`), which is how a value a half-turn
   from its mean fails, as the mathematics promised, rather than carrying a `NaN` into a step.
4. **An example**, `g2o`, reads a 3D pose graph in g2o's text format, converts each edge's
   information by [](eq:archive/quaternion-signs/math/g2o-precision), and estimates it. Without a
   file it writes and reads the one-turn loop of
   [The problem](math.svx#the-problem), with its vertices in the signs that composing the odometry
   gives, so that the loop closure is on the far side of its prediction at the start, the case the
   stereographic chart got wrong.

## Constraints

Every excerpt below was read at commit `511cc97`.

**The embedded declaration**, `crates/skoro-core/src/embedded.rs`. A declaration is a value,
built by `declare` and refined by `with_volume`, both returning a new `Arc<Embedded>`:

```rust
#[derive(Clone)]
pub struct Embedded {
    name: &'static str,
    params: Vec<Param>,
    storage: usize,
    dimension: usize,
    retraction: Arc<Composite>,
    inverse: Arc<Composite>,
    projection: Option<Arc<Composite>>,
    volume: Option<Volume>,
    flat: bool,
}

impl Embedded {
    pub fn declare(name: &'static str, params: Vec<Param>, storage: usize, dimension: usize,
        retraction: Arc<Composite>, inverse: Arc<Composite>, projection: Option<Arc<Composite>>,
    ) -> Result<Arc<Embedded>, Error>;
    pub fn with_volume(self: Arc<Self>, volume: Volume) -> Result<Arc<Embedded>, Error>;
    pub fn volume(&self) -> Option<&Volume>;
}
```

The group joins it the same way, as a refinement that returns a new declaration and refuses what
it cannot accept with `InvalidDeclaration`. Equality and hashing are by name and parameters
("two declarations under one name and the same parameters are one space, so they must not differ
in any part"), so the group adds nothing to either.

**The checks**, `crates/skoro-core/src/testing.rs`. A crate runs these on its own declaration:

```rust
/// - every inner pass is finite at the declaration's order, `R` at `(x, 0)` and `Q` on the
///   diagonal `(x, x)`, where a textbook formula is often `0/0`;
/// - `R(x, 0) = x`, and `Q(x, R(x, t)) = t` for small random `t`;
/// - `B(x) = ∂₂R(x, 0)` has rank `d`;
/// - when a projection is declared, its inner passes are finite at `x` and `P(x) = x`;
/// - `check_operation` holds on the inclusion, on the retraction at `t = 0` and away from it,
///   and on the projection at the points and off the manifold.
pub fn check_embedded(declaration: &Arc<Embedded>, points: &HostValue, target: &Target)
    -> Result<(), String>;

/// - at tangents `t` of lengths 0.1, 1, 3 and 10 in random directions, ...:
///   `Q(x, R(x, t)) = t`, and `Λ(x, t)` agrees with the area formula, ...
pub fn check_volume(declaration: &Arc<Embedded>, points: &HostValue, target: &Target)
    -> Result<(), String>;
```

Every comparison in them is of `R(x, 0)` with `x`, `P(x)` with `x`, and `Q(x, R(x, t))` with `t`.
The Cayley retraction keeps the side of its base point, $\langle x \oplus t, x \rangle > 0$, so each
holds exactly as written, and no existing check has to compare up to the group. The tangent of
length 10 is a rotation by $2 \arctan 5 \approx 157°$, inside the chart. The area formula is local,
and the covering is a local isometry of the storage, so it holds for the Cayley chart on $S^3$ as
it does on $SO(3)$.

**The declarations**, `crates/skoro-spatial/src/types.rs`. One function declares all three types
from the rotation's chart and the translations' blocks:

```rust
let r = define(retract, &[storage.clone(), tangent.clone()], &|q| {
    let (x, t) = (q[0], q[1]);
    let unit = normalize(x.slice(0..4));
    let moved = (0..k).map(|i| block(x, 4, i) + rotate(unit, block(t, 3, i))).collect();
    assemble(mul(unit, delta(t.slice(0..3))), moved, storage.clone())
})?;
let q = define(local, &[storage.clone(), storage.clone()], &|q| {
    let (x, y) = (q[0], q[1]);
    let (a, b) = (x.slice(0..4), y.slice(0..4));
    let moved = (0..k).map(|i| unrotate(a, block(y, 4, i) - block(x, 4, i))).collect();
    assemble(delta_inverse(mul(conjugate(a), b)), moved, tangent.clone())
})?;
// ...
let lambda = define(log_volume, &[storage.clone(), tangent.clone()], &|q| {
    let phi = q[1].slice(0..3);
    ((phi * phi).sum() * (1.0 / 16.0) + 1.0).log() * -3.0
})?;
Embedded::declare(name, Vec::new(), n, d, r, q, Some(p))?.with_volume(Volume::global(lambda))
```

The change of chart is the change of `delta`, `delta_inverse` and the log-volume's body; the
translations' blocks are untouched. `delta` and `delta_inverse` are used nowhere else.

**The core's primitives**, the table of `archive/core/math`: a reciprocal is $e^{-\log a}$ of a
positive $a$, and there is no signed division. Hence [](eq:archive/quaternion-signs/math/cayley-composite).

**The maps that read the representative**, `crates/skoro-spatial/src/rotation.rs`:

```rust
/// `Log(q) = 2 atan2(n, w) u / n`: the inverse of `exp`, of length in [0, 2π). `Log(-q) ≠
/// Log(q)`. ...
fn log(self) -> Q<Vector<3>>;
/// The rotation vector of the shorter turn, `Log(s(q) q)` with `s(q)` the sign of `w`: the
/// same at `±q`, of length in [0, π], and a discretization at half-turns.
fn log_shorter(self) -> Q<Vector<3>>;
/// The axis and the angle of `log`, the angle in [0, 2π); the axis `(1, 0, 0)` at the
/// identity, where it is a discretization.
fn axis_angle(self) -> (Q<Vector<3>>, Q<Real>);
/// `sign⟨q, r⟩ q`: the point over this rotation on the side of `reference`, ...
fn aligned_with(self, reference: Q<Rotation>) -> Q<Rotation>;
/// `q₁ ⊗ Exp(t Log(q̄₁ ⊗ q₂))`: ... by the arc that avoids `-self`.
fn slerp(self, other: Q<Rotation>, t: Q<Real>) -> Q<Rotation>;
/// The same along the shorter of the two turns ...
fn slerp_shorter(self, other: Q<Rotation>, t: Q<Real>) -> Q<Rotation>;
```

with the composites `skoro.spatial.log`, `log_shorter`, `axis_angle`, `slerp` and
`slerp_shorter`, and, in `crates/skoro-spatial/src/pose.rs` and `extended.rs`, `log` and
`log_shorter` on both motions, `interpolate`, `interpolate_shorter`, `interpolate_parts` and
`interpolate_parts_shorter` on the poses, each built on `log_stored`, `shorter` and
`slerp_stored(a, b, t, shorter_turn)`. The series check of `crates/skoro-spatial/src/series.rs`
names `log_shorter` on all three types. `crates/skoro-spatial/src/host.rs` has `log`, of length in
$[0, 2\pi)$, and `align`.

**Observations and scales.** `Observations::observe(q, value)` takes any `IntoHostValue<Pose>`,
and the spatial types' host values normalize the quaternion and keep its sign. `Scale::precision`
takes a $d \times d$ precision row by row and refuses one that is not symmetric positive definite.
A pose graph's first vertex is held by a constant, and every other vertex is a parameter, scored
by no term ([the probabilistic graph](../core/math.svx#the-probabilistic-graph)). The linearizer
takes such a quantity as free and refuses it only when its rows do not determine it:
`crates/skoro-linearize/tests/refusals.rs` builds two unscored quantities `a` and `b` with one
measurement of `b - a`, and is refused `Error::Undetermined` for the second, the gauge. With the
first vertex constant, a connected graph of relative measurements determines every other.

**What the linearizer does with a non-finite value**, `crates/skoro-linearize/src/assemble.rs`
and `minimize.rs`. `Linearizer::at` reads each factor's residual and Jacobian onto the host once,

```rust
let residuals: Vec<(Vec<f64>, Vec<f64>)> = self
    .factors
    .iter()
    .map(|f| Ok((host(&pass.values, f.residual)?, frame(&pass.frames, f.residual)?)))
    .collect::<Result<_, Error>>()?;
```

and checks neither for finiteness. `LevenbergMarquardt::minimize` accepts a step when
`gain > 0.0`, which a `NaN` gain fails, and ends a row with `Status::DampingExceeded` or
`IterationsExceeded`; its `Error`, `#[non_exhaustive]`, has no variant for a value that is not
finite. A non-finite residual at the start is therefore not refused today: it reaches the
elimination, and the row ends without saying why. The mathematics settled that a value a half-turn
from its mean is left non-finite and that "a model evaluated there fails, saying where"; that
needs a refusal the linearizer does not have.

## Design

### The core: a type stored up to a group

In `crates/skoro-core/src/embedded.rs`:

```rust
/// A signed permutation of a type's storage: entry `i` of `g x` is `signs[i] · x[source[i]]`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Linear {
    source: Vec<usize>,
    signs: Vec<i8>,
}

impl Linear {
    /// The map taking entry `source[i]` to entry `i` with sign `signs[i]`; `None` unless
    /// `source` is a permutation of `0..n` and every sign is `1` or `-1`.
    pub fn new(source: Vec<usize>, signs: Vec<i8>) -> Option<Linear>;
    /// The diagonal map with these signs, each `1` or `-1`; `None` otherwise.
    pub fn signs(signs: &[i8]) -> Option<Linear>;
    pub fn storage(&self) -> usize;
    /// `g x`, for one stored point.
    pub fn apply(&self, x: &[f64]) -> Vec<f64>;
    /// `self` first, then `other`: `x ↦ other(self(x))`.
    pub fn then(&self, other: &Linear) -> Linear;
}

pub struct Embedded {
    // ... as before, and:
    /// The generators of the finite group the type is stored up to; empty when it is stored as
    /// itself.
    identification: Vec<Linear>,
}

impl Embedded {
    /// This declaration, stored up to the finite group `generators` generate: a point of the
    /// type is the orbit of a stored point, `R`, `Q`, `P` and `Λ` must meet the conditions of
    /// the statement, which `check_embedded` and `check_volume` check, and every function the
    /// core computes is the same at every representative. Refuses a generator of another
    /// storage, a group of more than 64 elements, which the comparisons up to the group would
    /// enumerate, a flat declaration, and one already stored up to a group
    /// (`InvalidDeclaration`).
    pub fn stored_up_to(self: Arc<Self>, generators: Vec<Linear>) -> Result<Arc<Embedded>, Error>;
    /// The generators, empty for a type stored as itself.
    pub fn identification(&self) -> &[Linear];
    /// Every element of the group, the identity first, by closing the generators under `then`.
    pub fn group(&self) -> Vec<Linear>;
}
```

Both `Linear` and the new methods are re-exported from `skoro_core` beside `Embedded`.

**Why generators, and why signed permutations.** The statement allows any finite group of linear
maps acting freely, and the checks need only the generators: each condition is preserved under
composition, so it holds on the group when it holds on the generators. `group` exists for the
comparisons up to the group in tests, below. A general matrix would be closed under composition
only within rounding, so that the cube of a rotation by $120°$ would not equal the identity and
`group` could not tell when it was done; the signed permutations compose exactly, are always a
finite group, and are still linear maps, as the statement asks. Every deck transformation skoro
has a use for is one: the sign of a quaternion, of a sphere's point, an exchange of two blocks. A
diagonal sign is all the spatial types need, and `Linear::signs` makes it in one line. The
alternative, a dedicated `Antipodal` flag on the first four storage entries, would put the
rotations' storage layout in the core, which knows nothing of quaternions.

**Why nothing else in the core changes.** The derivative maps of
[%full](statement:archive/core/math/embedded) are those of the statement on $\manifold$, at the
stored representative, and by [%full](statement:archive/quaternion-signs/math/quotient) they are
right for every function invariant under the group; every function the core computes from a
quantity of the type goes through `R`, `Q`, `P` and `Λ`. The domain, the globality and the
fullness of a volume are read on $N$, which changes what a type may declare, not what the core
does with it: a `Volume::global` on a type stored up to a group says the chart is global and full
on the quotient.

**The checks.** In `crates/skoro-core/src/testing.rs`, `check_embedded` gains, when the
declaration is stored up to a group, for each generator $g$ at the points $x$ given:

- **freedom**: $g x \ne x$ on every row, beyond the tolerance;
- **on the manifold**: $g x$ passes the declaration's own checks as $x$ does, by running them at
  the points $g x$;
- **equivariance of R**: $R(g x, t) = g\, R(x, t)$ at $t = 0$, at small $t$, and at the tangents
  of length $0.5$ the retraction's `check_operation` uses;
- **invariance of Q**: $Q(g x, y) = Q(x, y) = Q(x, g y)$ at $y = R(x, t)$ for small $t$;
- **equivariance of P**, when declared: $P(g w) = g\, P(w)$ at the points and off the manifold.

`check_volume` gains $\Lambda(g x, t) = \Lambda(x, t)$ at each of its tangents. Each failure names
the generator by its index and the condition. A new helper compares points as points of $N$:

```rust
/// Whether two stored points, one per row, are within `tolerance` as points of the type: some
/// element of its group carries `b` within it of `a`, row by row.
pub fn close_up_to(declaration: &Embedded, a: &[f64], b: &[f64], tolerance: f64)
    -> Result<(), String>;
```

### skoro-spatial: the Cayley chart

In `crates/skoro-spatial/src/algebra.rs`, `delta` and `delta_inverse` are replaced by:

```rust
/// `γ(φ) = (1, φ/2) / √(1 + |φ|²/4)`, the reciprocal `e^{-½ log(1 + |φ|²/4)}`.
pub(crate) fn cayley(phi: QDyn) -> QDyn {
    let s = ((dot(phi, phi) * 0.25 + 1.0).log() * -0.5).exp();
    quaternion(s, phi * (s * 0.5))
}

/// `2 w vec(r) / w²`, `1/w² = e^{-log w²}`: the coordinate of `r` at the identity, the same at
/// `±r`; non-finite where `w = 0`, the half-turns, and large near them.
pub(crate) fn cayley_inverse(r: QDyn) -> QDyn {
    let (w, u) = parts(r);
    u * (w * ((w * w).log() * -1.0).exp() * 2.0)
}
```

which implement [%full](statement:archive/quaternion-signs/math/cayley) and
[](eq:archive/quaternion-signs/math/cayley-composite). In `types.rs`, the retraction calls `cayley`,
the inverse `cayley_inverse`, and the log-volume becomes
[](eq:archive/quaternion-signs/math/cayley-volume):

```rust
let lambda = define(log_volume, &[storage.clone(), tangent.clone()], &|q| {
    let phi = q[1].slice(0..3);
    ((phi * phi).sum() * 0.25 + 1.0).log() * -2.0
})?;
let sign = Linear::signs(&sign_of_quaternion(n)).ok_or_else(|| invalid("signs"))?;
Embedded::declare(name, Vec::new(), n, d, r, q, Some(p))?
    .stored_up_to(vec![sign])?
    .with_volume(Volume::global(lambda))
```

with `sign_of_quaternion(n)` the vector of $-1$ four times and $1$ for the translations. The
module's documentation and the markers' are rewritten: `Rotation` is "a rotation, stored as a unit
quaternion of either sign", `Pose` and `ExtendedPose` likewise, and the inverse is "undefined at
the half-turns from the base point: non-finite at a row exactly there, and growing as
$2 \tan(\varphi / 2)$ near it", in place of the stereographic chart's cut at `-q`. The names of the
declarations and their composites are kept: a model written against them means what it meant, a
rotation, and is now read correctly in either sign.

**The retraction's normalization of its base point** stays, as it is: it keeps a long run of
retractions on $S^3$, and $P$ is equivariant, so it keeps the side of the stored point.

**Half-turns are left non-finite**, as the mathematics settles: no guard, no selection, and the
documentation of `Normal` on these types and of the inverse says so.

### skoro-spatial: the maps of the groups

| map | before | after |
| --- | --- | --- |
| `RotationMaps::log` | $\operatorname{Log}(q)$, length in $[0, 2\pi)$ | $\operatorname{Log}(s(q)\, q)$, the old `log_shorter` |
| `RotationMaps::log_shorter` | the shorter turn | removed |
| `RotationMaps::axis_angle` | of `Log`, angle in $[0, 2\pi)$ | of the shorter turn, angle in $[0, \pi]$ |
| `RotationMaps::slerp` | the arc avoiding $-q_1$ | the shorter turn, the old `slerp_shorter` |
| `RotationMaps::slerp_shorter` | the shorter turn | removed |
| `RotationMaps::aligned_with` | the representative on the side of `reference` | unchanged; documented as the identity on $SO(3)$ |
| `PoseMaps::log`, `ExtendedPoseMaps::log` | $\operatorname{Log}$ of the stored quaternion | of the shorter turn, the old `log_shorter` |
| `PoseMaps::interpolate`, `interpolate_parts` | the arc avoiding $-q_1$ | the shorter turn |
| every `*_shorter` on the motions | the shorter turn | removed |
| `host::log` | length in $[0, 2\pi)$ | the shorter turn |
| `host::canonical` | | new: `align(q, [1, 0, 0, 0])`, the representative with $w \ge 0$ |

The composites keep their names where the plain method keeps its name (`skoro.spatial.log` now
computes the shorter turn), and the `*_shorter` composites are deleted. Inside the crate,
`log_stored`, `shorter` and `slerp_stored` lose their `shorter_turn` flag: every caller passes the
shorter turn. `axis_angle` becomes the direction and length of the shorter turn's rotation vector,
built as now but on `shorter(q)`: the axis `(1, 0, 0)` at the identity, and at a half-turn the axis
of the selected side, the discretization of `log` at the same rows.

Every other map is unchanged: the group operations and actions, the matrices and adjoints,
`exp` and the Jacobians, `angle_to`, the Euler angles in both directions, the constructions, and
alignment. The mathematics shows each is a map of the group; the tests below check it at both
signs.

**Why remove rather than deprecate.** A term built on a map of the representative would make a
density depend on the sign again, which is what this entry removes; a deprecated method still
compiles into such a term.

### skoro-linearize: a point that is not finite

In `crates/skoro-linearize/src/error.rs`:

```rust
/// A factor whose residual or Jacobian is not finite at the point linearized, at a batch row:
/// a value outside its term's chart, such as a rotation a half-turn from its mean.
NotFinite { term: NodeId, row: usize },
```

displayed as `"term {term:?} is not finite at row {row}"`. `Linearizer::at` and `at_in_chart` scan
each factor's residual and Jacobian rows as they read them onto the host, and return the first
factor and row with an entry that is not finite, in the factors' order. Nothing else changes: a
trial point the minimizer evaluates only through `objective`, and a step that lands on a cut gives
a non-finite objective there, a `NaN` gain, and a rejected step, as now.

**Why refuse at the linearization point.** It is where every estimator meets the model's value
first, with the residuals already on the host, so the check costs a scan of data that is read
anyway. Refusing in the core, at every term's evaluation, would cost a read back from the device
on every pass, and would refuse values that a caller evaluating a density on purpose at a cut may
want to see as non-finite. The alternative within the minimizer, a new `Status`, would let the
other rows of a batch go on, but would not name the term, which is what a user needs to find a
bad measurement.

### The example: a pose graph from g2o

`crates/skoro/examples/common/g2o.rs`, read by a new example `crates/skoro/examples/g2o.rs`:

```rust
/// A 3D pose graph as g2o writes it: the vertices' initial poses and the edges' measurements.
pub struct Graph {
    /// By index, each `(w, x, y, z, px, py, pz)`, the quaternion reordered from g2o's
    /// `qx qy qz qw` and normalized, its sign kept.
    pub vertices: BTreeMap<usize, [f64; 7]>,
    pub edges: Vec<Edge>,
}

pub struct Edge {
    pub from: usize,
    pub to: usize,
    /// The measured motion of `X_from⁻¹ X_to`, as a vertex's pose is stored.
    pub measurement: [f64; 7],
    /// The precision of skoro's residual, rotation first, by
    /// [](eq:archive/quaternion-signs/math/g2o-precision), 6 × 6 row by row.
    pub precision: [f64; 36],
}

/// Reads `VERTEX_SE3:QUAT` and `EDGE_SE3:QUAT` lines; other lines are skipped and counted, and a
/// malformed one is an error naming its line.
pub fn read(text: &str) -> Result<Graph>;

/// The text of a one-turn loop of `n` poses on a circle: every measurement written with `w ≥ 0`,
/// the loop closure near the identity, each perturbed by `noise`; and every vertex at the pose
/// that composing the measured odometry from vertex 0 gives, in the sign the composition gives,
/// so that the last vertex's quaternion is near `-1` and the loop closure is on the far side of
/// its prediction.
pub fn one_turn_loop(n: usize, noise: f64, seed: u64) -> String;
```

The model: vertex `0` is a constant at its initial pose, which fixes the gauge as g2o's datasets
are usually fixed; every other vertex is a quantity, scored by no term; each edge is a quantity
of `Pose`, scored by `Normal::on(Pose, x[from].between(x[to]), Scale::precision(&edge.precision,
6))` and observed at its measurement. The start is each vertex's initial pose, in whatever sign
the file gives. The example runs Levenberg–Marquardt, reports the final cost, and checks it
converged; given a path as its argument it reads that file, and without one it reads
`one_turn_loop(100, 0.01, SEED)`, so that it runs in CI with no dataset. The library gains no file
format: a reader belongs to an example until a second caller needs it. `g2o.rs` uses only the
standard library and `skoro_spatial::host`, and returns its own `Result<T, String>`, so that the
test below can include it by path without the rest of the examples' common modules.

**Why no dataset is vendored.** The usual datasets are megabytes, and their licences are each
their own. The example reads any 3D g2o file a user has, and its own loop exercises the case that
matters.

The example joins the checks of `AGENTS.md` and `.github/workflows/ci.yml`:
`cargo run --release -p skoro --example g2o`. Both files are outside the workflow's gates, and the
change is made with the build.

### What else the change touches

Every use of a removed or renamed map is in skoro-spatial: in its sources (`rotation.rs`,
`pose.rs`, `extended.rs`, including `ExtendedPoseMaps::interpolate_shorter`, `host.rs`, and the
list of `series.rs`) and its tests, `checks.rs`, whose `composites_pass_check_operation` runs
`log_shorter`, `axis_angle`, `slerp`, `slerp_shorter` and the motions' `*_shorter` maps by name,
`interpolations.rs`, `motions.rs`, `rotations.rs` and `types.rs`. No other crate calls any of
them. skoro-linearize's tests use `Rotation` in `linearize.rs` and `replacement.rs` only for the
shape of a linearization, not the chart's values, and are unchanged.

The site's pages quote the changed files by line range: `concepts/rotations-and-motions` quotes
`types.rs`, `rotation.rs`, `pose.rs`, `host.rs` and `series.rs`, and `guide/rotations-and-poses`
quotes `tests/normal.rs`. The pages are closed until the publish stage, so between the build and
the publish they quote whatever lines then stand at those ranges, and `pnpm build` fails if a
range runs past the end of its file. The build keeps every quoted file at least as long as its
ranges reach, so that the site builds throughout, and the publish stage rewrites the quotes.

## Tests

**The core**, in `crates/skoro-core/tests/embedded.rs`, on a test declaration of the lines
through the origin of $\R^2$: the circle $S^1 \subset \R^2$ stored up to $x \mapsto -x$, of storage
$2$ and dimension $1$, with $J$ the quarter-turn,

$$
	R(x, t) = P(x + t J x), \qquad Q(x, y) = \frac{(J x) \cdot y}{x \cdot y}, \qquad
	P(w) = \frac{w}{\lvert w \rvert}, \qquad \Lambda(x, t) = -\log(1 + t^2) ,
$$

the tangent of the angle from $x$ to $y$, which is equivariant and invariant as the statement asks,
global, and full on the quotient, missing only the line orthogonal to $x$. The test writes it with
the core's primitives, the reciprocal of $x \cdot y$ as that of its square, as the rotations do:

- `stored_up_to_refuses_what_it_cannot_accept`: a generator of another storage, a group of more
  than 64 elements (the signed permutations of seven entries generated by a cycle and a sign), a
  flat declaration, and a second call, each `InvalidDeclaration`; and `Linear::new` refuses a
  `source` that is not a permutation and a sign other than $\pm 1$.
- `group_closes_the_generators`: the group of one sign is two elements, the identity first; of two
  commuting signs, four.
- `the_checks_hold_on_the_lines`: `check_embedded` and `check_volume` pass.
- `the_checks_name_a_broken_condition`: the same declaration with the stereographic inverse
  $2 (J x) \cdot y / (1 + x \cdot y)$, which reads the representative, fails `check_embedded`,
  naming the generator and "invariance of Q"; with the identity as its generator, it fails
  "freedom".
- `close_up_to_compares_orbits`: $x$ and $-x$ are close up to the group, and not as stored values.

**The chart**, in `crates/skoro-spatial/tests/types.rs`, `chart_is_stereographic` becomes
`chart_is_cayley`: the angle of $\gamma(t)$ is $2 \arctan(\lvert t \rvert / 2)$; $Q(x, R(x, t)) = t$
at lengths up to $10$; $Q(-x, y) = Q(x, y) = Q(x, -y)$ exactly; $(-x) \oplus t = -(x \oplus t)$;
$\pi(x \oplus t) = \pi(x)(I + [t]_\times) + O(\lvert t \rvert^2)$ against
[](eq:archive/rotations/math/rotation); $Q(x, y)$ non-finite at $x = (1, 0, 0, 0)$ and $y = (0, 1, 0, 0)$, exactly a half-turn at either
precision, and of length $2 \tan(\varphi/2)$ at $\varphi = 3$ rad. `volume_has_the_stated_mass` integrates
$e^{\lambda}$ [](eq:archive/quaternion-signs/math/cayley-volume) over $\R^3$ to $8\pi^2$, the mass of
$\nu_{SO(3)}$ normalized at the chart's centre, half that of $S^3$'s, by the same quadrature.
`types_pass_the_core_checks` now exercises the group's checks on all three types.

**The normal**, in `crates/skoro-spatial/tests/normal.rs`:

- `normal_density_on_rotations` integrates the term through the Cayley chart, $r = 2 \tan(u/2)$
  for $u \in (0, \pi)$, to $1$: the law is on $SO(3)$, with no second sheet.
- `normal_draws_on_rotations` compares the draws' means of three functions of the rotation with
  quadratures of $\density(q)$ alone over the chart at the identity, which reaches all of $SO(3)$;
  the sum $\density(q) + \density(-q)$ of the double cover is gone.
- `the_term_is_the_same_at_either_sign`: the normal's term on each type, at a value and a mean
  given in all four combinations of signs, agrees to rounding, and so do its gradient's chart
  coordinates with respect to both.

**The maps**, across `rotations.rs`, `interpolations.rs` and `motions.rs`:

- `exp_and_log`: `log` inverts `exp` on the ball of radius $\pi$, has length at most $\pi$, and is
  the same at $\pm q$; `axis_angle`'s angle is in $[0, \pi]$, and its axis and angle rebuild the
  rotation through `about`.
- `interpolations`: `slerp` is the shorter turn, at a constant rate, its endpoints right, and the
  same rotation whichever signs its endpoints carry.
- `motion_exp_and_log`: the motions' `log` inverts their `exp` on the ball of radius $\pi$ in the
  rotation, and is the same at $\pm q$.
- `maps_of_the_group_ignore_the_sign`: every map the mathematics lists as a map of the group,
  each composition, action, matrix, adjoint, `angle_to`, Euler convention, `log`, `axis_angle`,
  `slerp`, `interpolate`, `interpolate_parts`, and `aligned_with` up to the group, gives the same
  value, or the same point of the group, when any of its inputs is negated. This is the test that
  the representative maps are gone.
- The tests of `log_shorter`, `slerp_shorter` and the `*_shorter` motions' maps are folded into the
  above, and the series check's list renames `log_shorter` to `log` on all three types.
- `host_maps_agree_with_the_graph` adds `host::log` and `host::canonical`.

**The refusal**, in `crates/skoro-linearize/tests/refusals.rs`:

- `a_non_finite_residual_is_refused`: a normal on a vector whose mean is a constant `NaN` is
  refused at the linearizer's `at` with `Error::NotFinite`, naming its term and row `0`, and in a
  batch of two rows with the `NaN` in the second, naming row `1`.

**The sign of a measurement**, in a new `crates/skoro/tests/signs.rs`, on `one_turn_loop(24, 0,
seed)` and on the same loop with noise, read through `g2o::read` (the test includes the module
by path, as the examples' common modules are shared):

- `a_one_turn_loop_closes`: the fixture's loop closure is on the far side of its prediction at the
  start, which the test asserts first, $\langle z, h(\hat\point) \rangle < 0$; from that start,
  Levenberg–Marquardt converges to a cost at the noise level, and every relative rotation of the
  estimate is within the noise of its measurement: no turn is wound into it. This is the case
  [In a cycle](math.svx#what-a-misaligned-measurement-does) shows the stereographic chart
  cannot satisfy.
- `the_estimate_ignores_every_sign`: negating the quaternion of every measurement, of every
  vertex's start, and of a random half of each, gives estimates equal up to the group, by
  `close_up_to`, and the same final cost to rounding.
- `g2o_information_is_converted`: an edge with a diagonal $\Omega$ and one with a coupling block
  give the precision [](eq:archive/quaternion-signs/math/g2o-precision); and the normal's quadratic
  form $\tangent\transpose \Lambda \tangent$ and g2o's $e\transpose \Omega e$, computed on the host from
  [](eq:archive/quaternion-signs/math/g2o-error), differ by $O(\lvert \tangent \rvert^3)$: at a
  residual of length $0.1$ and again at half of it, the difference falls by a factor between $6$
  and $10$.
- `a_half_turn_is_non_finite`: a rotation observed at $(0, 1, 0, 0)$ with the mean $(1, 0, 0, 0)$,
  exactly a half-turn at either precision, makes the term non-finite, and the linearizer refuses
  it with `Error::NotFinite`, naming that term and row.

**The examples.** `pose_graph` and `sliding_window` run as before; their measurements are drawn
from the model, so their results change only through the chart's third-order term and its
log-volume, within the tolerances their checks use. `g2o` runs on its own loop.

## Open questions

None remain.
