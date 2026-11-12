---
summary: Two examples of the umbrella crate, a pose graph estimated in a batch and a sliding window over an open path, sharing one model builder, with the pieces they need added to the library where a user would need them too; a Levenberg–Marquardt method in skoro-linearize, the inverse retraction of a layout and the observation of generated values in the core, and two maps of the rigid motions on the host.
depends_on: [archive/examples/math, archive/linearization/implementation]
---

## Overview

When this is built, `crates/skoro/examples/` holds two programs, each run by CI and quoted by a
guide page:

- `pose_graph`: [the batch](math.svx#a-levenberg-marquardt-method) on a closed circuit, $N$ datasets
  generated from [the model](math.svx#the-model) and estimated together, with
  [the calibration check](math.svx#calibration) of the Laplace marginals;
- `sliding_window`: [the window](math.svx#a-sliding-window) on an open path, retiring a pose and
  the landmarks it alone sees at every tick, reporting its calibration and its gap
  [](eq:archive/examples/math/gap) to the batch on the same path.

Each exits with an error, naming what failed, when a check of the mathematics fails, so that CI
running it is a test. Both are built from one module of the examples, `common`, which holds the
scenarios, the model and the statistics.

What the examples need and a user would need too is added to the library rather than written in
the examples:

| piece | crate | for |
| --- | --- | --- |
| `LevenbergMarquardt`, `Minimized`, `decrement` | `skoro-linearize` | [the method](math.svx#a-levenberg-marquardt-method): [](eq:archive/examples/math/marquardt), [](eq:archive/examples/math/gain), [](eq:archive/examples/math/schedule), the stopping rule and the check of stationarity |
| `Linearizer::objective` | `skoro-linearize` | $f$ at a trial point, for [](eq:archive/examples/math/gain) |
| `Layout::local` | `skoro-core` | $q^\star \ominus \hat q$ of [](eq:archive/examples/math/nees) and $\tilde \point_K \ominus \hat \point_K$ of [](eq:archive/examples/math/gap) |
| `Observations::observe_value` | `skoro-core` | conditioning on what a generation drew, row by row |
| `host::pose_compose`, `host::pose_transform` | `skoro-spatial` | the start [](eq:archive/examples/math/start) on the host |

The mathematics anticipated that the examples would need the query's gradient laid out by its
quantities, which only the tests have. They do not: stationarity is checked by the undamped
decrement, which the rows give, and `Layout::flatten` already lays out a gradient's frames for a
caller who wants one.

A user meets the method as the batch example does:

```rust
let snapshot = writer.snapshot();
let query = Query::log_density().given(measurements);
let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Residual)?;
let schedule = linearizer.schedule()?;
let minimized = LevenbergMarquardt::default().minimize(&linearizer, &schedule, &start)?;
for (row, outcome) in minimized.rows.iter().enumerate() {
    if outcome.status != Status::Converged {
        return Err(format!("row {row}: {:?} after {} iterations", outcome.status, outcome.iterations).into());
    }
}
let decrements = decrement(&linearizer, &schedule, &minimized.point)?;
```

## Constraints

Everything below was read at commit `06fcb43`.

**The linearizer.** A `Linearizer` is built once per snapshot, query, target and assembly, and
linearizes a batch of points into one `Problem` per row
(`crates/skoro-linearize/src/linearizer.rs`, `assemble.rs`):

```rust
pub struct Linearizer {
    pub(crate) snapshot: Snapshot,
    pub(crate) target: Target,
    pub(crate) assembly: Assembly,
    /// The query's density program: `f`, and the layout.
    pub(crate) density: Arc<Program>,
    pub(crate) factors: Vec<Factor>,
    pub(crate) structure: Arc<Structure>,
    /* the rows program, its seeds and columns, the folded volumes */
}

impl Linearizer {
    pub fn new(snapshot: &Snapshot, query: &Query, target: &Target, assembly: Assembly)
        -> Result<Linearizer, Error>;
    pub fn layout(&self) -> &Layout;
    pub fn structure(&self) -> &Arc<Structure>;
    pub fn schedule(&self) -> Result<Schedule, Error>;
    pub fn schedule_keeping(&self, quantities: &[NodeId]) -> Result<Schedule, Error>;
    pub fn at(&self, point: &Values) -> Result<Linearized, Error>;
    pub fn at_in_chart(&self, point: &Values, quantities: &[NodeId]) -> Result<Linearized, Error>;
}
```

The density program is already compiled and private to the crate; its `forward` gives the query's
log-density per row under `NodeId::DENSITY`. So $f$ at a trial point is one forward pass of a
program the linearizer holds, and the method belongs in the crate that holds it, where it reads the
program without compiling it again.

**Damping is one for the whole batch.** `Linearized::eliminate` appends one `Damping` to every
row (`crates/skoro-linearize/src/solved.rs`):

```rust
pub fn eliminate(&self, schedule: &Schedule, damping: Option<&Damping>) -> Result<Solved, Error> {
    let eliminations = self
        .problems
        .iter()
        .map(|problem| {
            problem
                .eliminate(schedule, damping)
                .map_err(|e| undetermined(e, &self.layout))
        })
        .collect::<Result<_, Error>>()?;
    /* ... */
}
```

while [the method](math.svx#a-levenberg-marquardt-method) gives each row its own $\lambda$ and its
own $D$, read off its own rows. The method therefore eliminates row by row with
`Problem::eliminate`, which is public, and maps a refusal with the crate's `undetermined`, as
`eliminate` does. `Linearized::eliminate` is left as it is.

**The damping and its promise** (`crates/skoro-elimination/src/problem.rs`):

```rust
/// `½ λ τᵀ D τ`: `λ` and the positive diagonal `D`, one entry per coordinate.
pub struct Damping {
    pub lambda: f64,
    pub diagonal: Vec<f64>,
}

impl Problem {
    pub fn factors(&self) -> &[Rows];
    pub fn value(&self, tangent: &[f64]) -> f64;
    /// `½|J τ|² + λ τᵀ D τ`: the decrease of the undamped problem that the damped minimizer
    /// `tangent` promises.
    pub fn promised(&self, tangent: &[f64], damping: &Damping) -> f64;
}
```

`promised` is [](eq:archive/linearization/math/promised) and the $\Delta$ of
[](eq:archive/examples/math/gain); with $\lambda = 0$ it is the decrement
$\tfrac12 \lvert J \tangent_0 \rvert^2$. `value(0)` is $f$ at the point of the linearization, the
constant having been chosen so ([](eq:archive/linearization/math/agreement)). A factor's rows
`Rows { a, b }` have the columns of its blocks in the structure's order, which
`Structure::factors()` and `Structure::range(k)` locate; the diagonal of
[](eq:archive/examples/math/marquardt) is read from them.

**Retracting a batch.** `Layout::retract` stacks the quantities' values into a point of their
product and moves it by the product's retraction, compiled once per target
(`crates/skoro-core/src/layout.rs`):

```rust
pub fn retract(&self, at: &Values, tangent: &Points, target: &Target) -> Result<Values, Error> {
    /* stack `at`'s quantities into one point of the product */
    let program = self.retraction(target)?;           // product.retraction()?.compile(target)
    /* forward the program at (point, tangent), slice the moved point back per quantity */
}
```

The product also has `inverse_retraction()`, the composite of $\point \ominus \mu$ over every
factor (`crates/skoro-core/src/retraction.rs`), which no `Layout` method exposes. `Q::local` gives
it inside a model, but the truth of an example is a value on the host, not a node.

**Observations are host values, row by row.** `Observations::observe_dyn` takes a `HostValue`,
which holds one value for every row or one per row (`crates/skoro-tensor/src/host.rs`):

```rust
/// One value shared by every row of a batch (`rows: None`), or one value per row, pinning the
/// batch to `rows` (`Some(rows)`).
pub fn new(rows: Option<usize>, storage: usize, data: HostData) -> Option<HostValue>;
```

while `Snapshot::generate` gives `Values`, a `Points` of `[rows, storage]` per quantity. Nothing
converts the one into the other; the guide's tests build their values by hand.

**Retirement.** `Writer::retire(addresses, observed)` reads a `Plan` with `E`, `E_O`'s values,
`E_F`, the terms and the blanket; `Replacing::new(&retirement, keep, fixed, target)` linearizes the
retired terms' query free in `E_F` and `keep`, and refuses (`Blanket`) when its free quantities are
not exactly those; `Replacing::at(&point)` gives a `LinearizedReplacement`, whose `build()` is
called inside `Retirement::replace` (`crates/skoro-core/src/retire.rs`,
`crates/skoro-linearize/src/replacing.rs`). The window keeps every blanket quantity, so `keep` is
the plan's blanket and `fixed` is empty. A blanket quantity left unscored needs a proposal only if
it is to be generated, which the window never does.

**The rigid motions.** `Q<Pose>` has `*` for composition and for the action on a point, and
`PoseMaps::{inverse, between, untransform}`; a constant pose is `constant(Pose, [f64; 7])`, the
quaternion `(w, x, y, z)` then the position (`crates/skoro-spatial/src/pose.rs`, `types.rs`).
`skoro_spatial::host` has the rotations' maps on the host (`compose`, `matrix`, `exp`, `log`) and no
pose's.

**The umbrella crate** re-exports `skoro-core` and `skoro-tensor` only, with `skoro-normal` as a
dev-dependency for its tests. Its examples may use the other crates as dev-dependencies, which
creates no cycle, since none of them depends on `skoro`.

**The lints.** The workspace denies `unwrap_used`, `expect_used` and `panic`, examples included:
every example returns `Result<(), Box<dyn std::error::Error>>` and reports a failed check as an
error.

## Design

### The Levenberg–Marquardt method, in `skoro-linearize`

A new module, `crates/skoro-linearize/src/minimize.rs`, re-exported at the crate's root:

```rust
/// The Levenberg–Marquardt method of the examples' mathematics: Marquardt's diagonal, Nielsen's
/// schedule, and a stop on the promised decrease, run on each row of a batch separately.
#[derive(Clone, Debug)]
pub struct LevenbergMarquardt {
    /// `λ₀`: 1e-3.
    pub initial_damping: f64,
    /// `δ`, the floor of `D`: 1e-12.
    pub floor: f64,
    /// `ε`, in nats: 1e-9. Must stay well above `u |f|` at the query's precision.
    pub tolerance: f64,
    /// `λ_max`: 1e10. A row whose damping passes it has not converged.
    pub max_damping: f64,
    /// The bound on iterations, accepted and rejected alike: 100.
    pub max_iterations: usize,
}

impl Default for LevenbergMarquardt { /* the values above */ }

/// How a row's minimization ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    /// A step, accepted or not, promised at most `tolerance`.
    Converged,
    /// `λ` passed `max_damping`: no step decreased `f`.
    DampingExceeded,
    /// `max_iterations` passed first.
    IterationsExceeded,
}

/// One iteration of one row.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Iterate {
    /// `f` at the point the step was taken from.
    pub objective: f64,
    pub damping: f64,
    /// `Δ(τ)`, [](eq:archive/examples/math/gain)'s denominator.
    pub promised: f64,
    /// `ρ_λ`.
    pub gain: f64,
    pub accepted: bool,
}

#[derive(Clone, Debug)]
pub struct RowOutcome {
    pub status: Status,
    pub iterations: usize,
    pub history: Vec<Iterate>,
}

#[derive(Clone, Debug)]
pub struct Minimized {
    /// The final point of every row, at the linearizer's target.
    pub point: Values,
    pub rows: Vec<RowOutcome>,
}

impl LevenbergMarquardt {
    /// Minimizes the linearizer's `f` from `start`, row by row, eliminating along `schedule`
    /// (`Shape` of a schedule of another structure). Refuses the Fisher assembly
    /// (`NotAnIntegral`), whose rows are not the curvature of `f`, and what the linearizer and
    /// the elimination refuse. A row that does not converge is not an error: its outcome says so.
    pub fn minimize(
        &self,
        linearizer: &Linearizer,
        schedule: &Schedule,
        start: &Values,
    ) -> Result<Minimized, Error>;
}

/// `½|J τ₀|²` per row at `at`: the Gauss–Newton decrement, with `τ₀` the undamped step, zero
/// exactly where the query's chart gradient is, the rows being of full rank.
pub fn decrement(linearizer: &Linearizer, schedule: &Schedule, at: &Values) -> Result<Vec<f64>, Error>;
```

The body is the point, so it is sketched:

```rust
pub fn minimize(&self, linearizer: &Linearizer, schedule: &Schedule, start: &Values)
    -> Result<Minimized, Error>
{
    if linearizer.assembly() == Assembly::Fisher {
        return Err(Error::NotAnIntegral);
    }
    let (layout, target) = (linearizer.layout(), linearizer.target());
    let d = layout.dimension();
    let mut at = start.clone();
    let rows = rows_of(&at, layout)?;                       // the batch's number of rows
    let mut state: Vec<Row> = (0..rows)
        .map(|_| Row { lambda: self.initial_damping, beta: 2.0, done: None, history: vec![] })
        .collect();
    while state.iter().any(|r| r.done.is_none()) {
        let linearized = linearizer.at(&at)?;
        let mut steps = vec![0.0; rows * d];
        let mut promised = vec![0.0; rows];
        for (r, problem) in linearized.problems().iter().enumerate() {
            let row = &mut state[r];
            if row.done.is_some() { continue; }
            if row.history.len() >= self.max_iterations { row.done = Some(Status::IterationsExceeded); continue; }
            if row.lambda > self.max_damping { row.done = Some(Status::DampingExceeded); continue; }
            let damping = Damping { lambda: row.lambda, diagonal: marquardt(problem, self.floor) };
            let step = problem
                .eliminate(schedule, Some(&damping))
                .and_then(|e| e.minimizer())
                .map_err(|e| undetermined(e, layout))?;
            promised[r] = problem.promised(&step, &damping);
            steps[r * d..(r + 1) * d].copy_from_slice(&step);
        }
        let trial = layout.retract(&at, &Points::from_host(&steps, rows, d, target), target)?;
        let before: Vec<f64> = linearized.problems().iter().map(|p| p.value(&vec![0.0; d])).collect();
        let after = linearizer.objective(&trial)?;
        let mut accepted = vec![false; rows];
        for r in 0..rows {
            let row = &mut state[r];
            if row.done.is_some() { continue; }
            let gain = (before[r] - after[r]) / promised[r];      // ρ_λ
            let accept = gain > 0.0;
            row.history.push(Iterate { objective: before[r], damping: row.lambda, promised: promised[r], gain, accepted: accept });
            if accept { accepted[r] = true; }                     // taken, converged or not
            if promised[r] <= self.tolerance { row.done = Some(Status::Converged); continue; }
            nielsen(&mut row.lambda, &mut row.beta, gain);        // [](eq:archive/examples/math/schedule)
        }
        at = select_rows(&at, &trial, &accepted, layout, target)?;
    }
    Ok(Minimized { point: at, rows: state.into_iter().map(Row::outcome).collect() })
}
```

with three private helpers in the module:

```rust
/// `D_ii = max(|J_{:,i}|², δ)`, [](eq:archive/examples/math/marquardt): the squared norm of each
/// coordinate's column over every factor's rows.
fn marquardt(problem: &Problem, floor: f64) -> Vec<f64>;

/// [](eq:archive/examples/math/schedule) on one row's `λ` and `β`.
fn nielsen(lambda: &mut f64, beta: &mut f64, gain: f64) {
    if gain > 0.0 {
        *lambda *= (1.0 - (2.0 * gain - 1.0).powi(3)).max(1.0 / 3.0);
        *beta = 2.0;
    } else {
        *lambda *= *beta;
        *beta *= 2.0;
    }
}

/// Each quantity of `layout` from `trial` in the accepted rows and from `at` in the others,
/// through the host.
fn select_rows(at: &Values, trial: &Values, accepted: &[bool], layout: &Layout, target: &Target)
    -> Result<Values, Error>;
```

A few choices in that body:

- **The trial is batched, the decision per row.** One retraction and one forward pass serve every
  row, and a row that is done or rejected keeps its point through `select_rows`. A done row still
  rides along in the passes, which costs its share of them and keeps the batch one batch; the
  alternative, shrinking the batch as rows finish, would change the rows of every `Values` and
  every observation of the query as it went, for a saving the examples do not need.
- **$f$ before the step comes from the problem**, `value(0)`, which is $f$ at the point of
  linearization by construction, so only the trial point costs a pass. A rejected row is
  linearized again at the same point on the next iteration, which repeats work the row has
  already done; caching it would split a batch's problems between two points, for no change in
  any result.
- **$\Delta$ is compared with the tolerance before the acceptance**, as
  [the stopping rule](math.svx#a-levenberg-marquardt-method) says: a row stops on a step that
  promises at most $\epsilon$, taken if its gain is positive and refused if not. Once $\Delta$ is
  rounding the gain has a random sign, or is $0/0$ (which `gain > 0.0` refuses, being false for
  `NaN`), so no gain is formed from it into the schedule: a stopped row does not update $\lambda$.
  Stopping only on an accepted step would let a converged row be rejected until its damping passed
  `max_damping`.
- **A step is refused, not skipped,** when the elimination refuses a block. Damping makes every
  block determined for $\lambda > 0$, so a refusal is a malformed problem, and the error names its
  quantity.

`decrement` linearizes at `at`, eliminates every row along `schedule` with no damping, and returns
`problem.promised(&step, &zero)` with `zero` the damping of $\lambda = 0$.

The alternative to putting the method in the library was to write it in the examples' `common`
module. It lost because both examples run it, a user who reads them wants to call it rather than
copy it, and its correctness then rests on the examples' checks alone, while in the crate it has
tests of its own.

### $f$ at a point, in `skoro-linearize`

```rust
impl Linearizer {
    /// `f = -log p` of the query at `point`, per row: one forward pass of its density program.
    pub fn objective(&self, point: &Values) -> Result<Vec<f64>, Error>;
}
```

It negates what `pass` already reads with `host(&self.density.forward(point)?, NodeId::DENSITY)`.

### The inverse retraction of a layout, in `skoro-core`

```rust
impl Layout {
    /// `of ⊖ at`: the tangent coordinates at `at` that reach `of`, stacked as `retract` stacks a
    /// step, `[rows, dimension]` on `target`, by the product's inverse retraction compiled once per
    /// target. The inverse of `retract` on the chart's image: `local(retract(at, τ), at) = τ`.
    /// Refuses as `retract` does (`MissingValue`, `NotContinuous`).
    pub fn local(&self, of: &Values, at: &Values, target: &Target) -> Result<Points, Error>;
}
```

It stacks `of` and `at` as `retract` stacks `at`, forwards the compiled
`product.inverse_retraction()` at `(of, at)`, in the composite's own order of inputs, and returns
the tangent. The compiled program is cached beside the retraction's, in a second `Mutex<Vec<(Target,
Arc<Program>)>>` field, `locals`, which `Layout::of` initializes empty and `Clone` leaves empty, as
it does `retractions`.

### Observing what was generated, in `skoro-core`

```rust
impl Observations {
    /// This set with `quantity` observed at `value`, one value per row of a `Points`, as
    /// `generate` gives it (`ValueShape` when the value is discrete or of another storage than
    /// the quantity's space, read when the set is used, as `observe` defers a bad value).
    pub fn observe_value(self, quantity: impl Into<QDyn>, value: &Value) -> Observations;
}
```

It reads the points to the host and inserts `HostValue::new(Some(rows), storage, Continuous(data))`.
The window and the batch observe a generated measurement with it in one line, and the examples
contain no conversion of their own.

### Two maps of the rigid motions on the host, in `skoro-spatial`

```rust
// crates/skoro-spatial/src/host.rs

/// `T₁ * T₂ = (P(q₁ ⊗ q₂), p₁ + R₁ p₂)`, in storage order: as `Q<Pose> * Q<Pose>` computes it.
pub fn pose_compose(a: [f64; 7], b: [f64; 7]) -> [f64; 7];

/// `T * x = R x + p`: as `Q<Pose> * Q<Vector<3>>` computes it.
pub fn pose_transform(t: [f64; 7], x: [f64; 3]) -> [f64; 3];
```

Both are written with the module's `compose` and `matrix`, `P` the same sign convention as the
graph's (`align` to the first factor's quaternion is not applied: the graph's composition does not
apply it either, and a test pins the two to each other).

### The examples, in `crates/skoro`

```
crates/skoro/
    Cargo.toml                 dev-dependencies: skoro-normal, skoro-spatial, skoro-linearize,
                               skoro-elimination, faer; an [[example]] for each, test = true
    examples/
        common/mod.rs          pub mod scenario; pub mod model; pub mod laplace; pub mod report;
        common/scenario.rs     the two scenarios and their sightings
        common/model.rs        the model of [](eq:archive/examples/math/model), tick by tick
        common/laplace.rs      normalized errors and the calibration statistic
        common/report.rs       printing, and the checks as errors
        pose_graph.rs
        sliding_window.rs
```

Each example starts with `mod common;`. Cargo does not take `common/mod.rs` for an example, since
it is not a `main.rs`. `common/mod.rs` opens with `#![allow(dead_code)]`, as every `tests/common`
of the workspace does: each example uses part of it, and clippy with `-D warnings` refuses an item
that one of them leaves unused, as a throwaway crate confirmed for an example built normally and
as a test.

**The scenarios.** A scenario is the constants of [the setting](math.svx#setting):

```rust
pub struct Scenario {
    /// `x̄₀ … x̄_K`, storage order.
    pub nominal: Vec<[f64; 7]>,
    /// `m̄₁ … m̄_J`.
    pub landmarks: Vec<[f64; 3]>,
    /// `V`, sorted by tick then landmark.
    pub sightings: Vec<(usize, usize)>,
    pub scales: Scales,
}

pub struct Scales {
    /// `Σ₀`, `Σ_u`, `Σ_y`: standard deviations of `(φ, ρ)`, rotation first.
    pub first: [f64; 6],
    pub motion: [f64; 6],
    pub odometry: [f64; 6],
    /// `Σ_m`, `Σ_z`: isotropic standard deviations in metres.
    pub landmark: f64,
    pub sighting: f64,
}

impl Scenario {
    /// The closed circuit: 73 poses on a circle of radius 10 m traversed 1.2 times, heading along
    /// the path: the height `0.5 sin 2θ` and the pitch the path's climb, `atan(0.1 cos 2θ)`; 24
    /// landmarks on rings of radii 7 and 13 m at alternating heights ±1 m; sighted within `h = 6` m.
    pub fn circuit() -> Scenario;
    /// The open path: 80 poses 1 m apart along `(k, 3 sin(k/10), 0.3 sin(k/7))`, heading along it;
    /// 40 landmarks at `(2j, ±5, 0)`, alternating sides; sighted within `h = 8.5` m. Refuses itself
    /// (`Err`) unless every landmark's sightings are an interval of ticks, as
    /// [the window](math.svx#a-sliding-window) requires.
    pub fn open_path() -> Result<Scenario, Box<dyn std::error::Error>>;
    /// `k(j)`, the first tick each landmark is sighted at; `Err` for a landmark never sighted.
    pub fn first_sightings(&self) -> Result<Vec<usize>, Box<dyn std::error::Error>>;
}
```

The sightings are the pairs with $\lvert \bar p_k - \bar m_j \rvert \le h$, the norm being invariant
under the pose's rotation, so no map of the motions is needed to choose them. The scales are the
same for both scenarios: $\Sigma_0$ of $0.01$ rad and $0.01$ m, $\Sigma_u$ of $0.02$ rad and
$0.05$ m, $\Sigma_y$ of $0.005$ rad and $0.02$ m, $\Sigma_m$ of $2$ m and $\Sigma_z$ of $0.05$ m.
Dead reckoning on the odometry drifts in rotation by about $0.005 \sqrt{72} \approx 0.04$ rad per
axis over the circuit. That is the start's error, not the posterior's: the motion terms and the
sightings that close the loop make every marginal tighter, so the posterior's $\sigma_\phi$ is
expected well within the $0.05$ rad that [the check](math.svx#calibration) assumes. The example
checks the assumption rather than trusting this figure.

**The model, tick by tick.** One builder serves the batch, which calls it for every tick in one
`build`, and the window, which calls it once per tick in a `build` of its own, so that the model is
written once:

```rust
/// The quantities of one tick, as their handles.
pub struct Tick {
    pub pose: Q<Pose>,
    /// The odometry from the previous pose, none at tick 0.
    pub odometry: Option<Q<Pose>>,
    /// The landmarks first sighted at this tick, with their indices.
    pub new_landmarks: Vec<(usize, Q<Vector<3>>)>,
    /// This tick's sightings, with their landmarks' indices.
    pub sightings: Vec<(usize, Q<Vector<3>>)>,
}

/// Adds tick `k` inside the open scope: the pose `pose/k`, scored by the prior at `k = 0` and by
/// the motion term from `previous` after; the odometry `odometry/{k-1}` from `previous`; each
/// landmark `landmark/j` first sighted at `k`, scored by its prior; each sighting `sighting/k/j`.
/// `landmarks` holds every landmark already in the model, by index, and gains the new ones.
pub fn tick(
    scenario: &Scenario,
    k: usize,
    previous: Option<Q<Pose>>,
    landmarks: &mut BTreeMap<usize, Q<Vector<3>>>,
) -> Tick;
```

Inside it, every term of [](eq:archive/examples/math/model) is one `Normal::on(...).score(...)`:
the motion's mean is `previous * constant(Pose, x̄_{k-1}).between(constant(Pose, x̄_k))`, the
command $u_{k-1}$ built from the nominal poses as a mapped value of constants; the odometry's
mean is `previous.between(pose)`; a sighting's is `pose.untransform(landmark)`; the scales are
`Scale::sds` of the rotation-first standard deviations and `Scale::sd` of the isotropic ones. The
odometry from $k - 1$ to $k$ is addressed `odometry/{k-1}`, under the tick it retires with, which
[the window's retirement](math.svx#a-sliding-window) needs: it retires with $\randomPoint_{k-1}$,
not with $\randomPoint_k$.

**Generation and the query.** The batch builds every tick in one `build`, generates $N = 100$ rows
from `Key::from_seed(2026_11_09)` with nothing observed, and keeps the generated poses and
landmarks as the truth. Its query observes every odometry and sighting with `observe_value` at the
generated rows. The start [](eq:archive/examples/math/start) is computed on the host per row with
`pose_compose` and `pose_transform`, from the observed values, and loaded as `Values`.

**The Laplace marginals and the check** (`common/laplace.rs`):

```rust
/// Each row's normalized error of `quantity`, [](eq:archive/examples/math/nees): `chart`, the
/// chart linearization at `mode` computed once for every quantity, eliminated along the schedule
/// keeping `quantity`; its marginal's whitening `W` and mean `μ` per row; and `|W (q* ⊖ q̂ - μ)|²`
/// with `q* ⊖ q̂` from `Layout::local`. For a pose, also the largest eigenvalue of the marginal
/// covariance's rotation block, per row.
pub fn normalized_errors(
    linearizer: &Linearizer,
    chart: &Linearized,
    mode: &Values,
    truth: &Values,
    quantity: NodeId,
) -> Result<(Vec<f64>, Option<Vec<f64>>), Box<dyn std::error::Error>>;

/// The calibration statistic of a set of quantities: per row the fraction `c_r` of them whose
/// error is at most `χ²_d(0.9)`, the mean `c̄`, `s* = max(s, 1/n)` and the sampling part
/// `4 s* / √N` of [](eq:archive/examples/math/check).
pub struct Calibration { pub mean: f64, pub sampling: f64, pub fractions: Vec<f64> }

pub fn calibration(errors: &[Vec<f64>], quantile: f64) -> Calibration;

/// `χ²₆(0.9)` and `χ²₃(0.9)`, and the slopes `χ²_d(α) f_d(χ²_d(α))` of
/// [](eq:archive/examples/math/coverage-error) at them, computed once from the closed forms of the
/// two distribution functions.
pub const CHI2_6_90: f64 = 10.644_640_675_668_418;
pub const CHI2_3_90: f64 = 6.251_388_631_170_324;
pub const SLOPE_6: f64 = 0.367_975;
pub const SLOPE_3: f64 = 0.273_781;
```

The chart linearization is computed once at the mode, and the marginal of each quantity is its
own elimination of it, `schedule_keeping(&[q])`, which for 73 poses and 24 landmarks over 100 rows
is 9 700 eliminations of the whole problem, of dimension $6 \cdot 73 + 3 \cdot 24 = 510$: tens of
seconds in release by an estimate the build measures and reports. The alternative, the joint marginal and its covariance, inverts one dense matrix of the
whole problem's dimension per row and is slower at these sizes. The truth's tangent is taken at
the mode, and the marginal's mean, zero to the tolerance at a converged mode, is subtracted rather
than assumed.

$\sigma_\phi^2$ is the largest eigenvalue of a pose's rotation block of its marginal covariance,
the largest over every pose and row, as the mathematics defines it. The eigenvalues are faer's
self-adjoint eigensolver's, `faer` being a dev-dependency of the umbrella crate for the purpose;
the trace, an upper bound, would be up to three times too strict. If the bound exceeds $0.05^2$, the example fails, saying that the scales left the
regime the check was designed for, rather than check at a tolerance whose premise does not hold.

**The batch example**, `pose_graph.rs`, in order:

1. Build the circuit's model; generate; build the query and the start.
2. `LevenbergMarquardt::default().minimize`; fail unless every row converged.
3. `decrement`; fail unless every row's is at most $10\, \epsilon$.
4. The normalized errors of every pose and landmark at the start and at the mode; the bound on
   $\sigma_\phi^2$; fail if it exceeds $0.05^2$.
5. `calibration` of the poses and the landmarks; fail unless each satisfies
   [](eq:archive/examples/math/check) with its $\delta$.
6. Print the report: per row the iterations and the final $f$, the history of row 0, the mean
   normalized errors at the start and at the mode, and the two calibrations with their tolerances.

**The window example**, `sliding_window.rs`. The truth and the measurements come from the open
path's full model, built and generated as the batch builds and generates the circuit's, and the
batch on that path is run as in steps 1 and 2 for the gap. The window is a second `Writer`:

```rust
let mut window = Writer::new();
let mut landmarks = BTreeMap::new();
let mut live: Live = Live::default();                         // the window's handles and values
let mut point = Values::new();                                // the window's current mode
for k in 0..=scenario.last() {
    let tick = window.build(|| tick(&scenario, k, live.newest(), &mut landmarks))?;
    live.append(&tick, &generated, k);                        // its measurements' observed rows
    point = live.start(point, &tick, &generated)?;            // [](eq:archive/examples/math/start)
    let snapshot = window.snapshot();
    let query = Query::log_density().given(live.observations());
    let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Residual)?;
    let schedule = linearizer.schedule()?;
    let minimized = method.minimize(&linearizer, &schedule, &point)?;
    /* fail unless converged and the decrement is at most 10 ε, at every tick */
    point = minimized.point;
    report.filtering(k, normalized_errors(&linearizer, &point, &truth_of(k), tick.pose.id())?);
    if live.poses() > W {
        let a = k - W;
        let addresses = live.retiring(a);                     // pose/a, odometry/a, sighting/a/*,
                                                              // and landmark/j for each landmark
                                                              // no other window pose sees
        let retirement = window.retire(addresses, live.observed_of(a))?;
        let keep: Vec<QDyn> = retirement.plan().blanket.iter().map(|(q, ..)| *q).collect();
        let replacing = Replacing::new(&retirement, &keep, &Observations::new(), &target)?;
        let replacement = replacing.at(&point)?;              // at the window's mode
        retirement.replace(|_| replacement.build())?;
        live.retire(a);                                       // drop their handles and values
    }
}
```

with $W = 5$. `Live` is a struct of the example's `common` module that holds the window's live
handles, their observed rows, and the landmarks' last sightings among the live poses, from which
`retiring(a)` names the landmarks that $\randomPoint_a$ alone still sees. Retiring a landmark at
`landmark/j` with `Addresses::at`, beside the pose's ticks taken with `Addresses::under`, follows the
core's rule that a collection need not be a namespace. The point given to `Replacing::at` holds every
quantity of $E_F$ and the blanket, since it is the window's whole mode.

At the end, the gap [](eq:archive/examples/math/gap) of every row: the batch's marginal whitening of
$\randomPoint_K$ at its mode, and `Layout::local` of a layout of the batch's $\randomPoint_K$
alone, with the window's value of its own $\randomPoint_K$ placed under the batch's identifier. The
report prints the filtering calibration beside $\alpha$, its sampling part and $\delta_6$, and the
gap's mean and largest value. Neither is checked, as
[the decision](math.svx#a-sliding-window) has it.

**Running them.** CI's Rust job gains two steps after `cargo test`:

```sh
cargo run --release -p skoro --example pose_graph
cargo run --release -p skoro --example sliding_window
```

and the `## Checks` list of `AGENTS.md` gains the same two lines. `cargo clippy --all-targets`
already lints the examples, and `cargo test` builds them.

### What is not kept from the archive

The archive had no example that ran: its example pages were descriptions with listings marked to
come. Its sliding window marginalized by hand, a Schur complement written in the test, a
`demote_to_parameter` and a joint factor; here the core's retirement and the linearization's
replacement do all of it, and the example writes none of their mathematics. Its estimation page
proposed an `Estimator` with a line search; the method here is a damping method alone, as the
mathematics settled, and has no line search.

## Tests

**`skoro-linearize`, `tests/minimize.rs`.** These test the method apart from the examples, on the
crate's own models in `tests/common`.

- `the_marquardt_diagonal_is_the_column_norms`: on a problem written by hand over two blocks,
  of dimensions 2 and 1, with one factor on both and one on the second alone, `marquardt` against
  the column norms summed over the factors, and the floor taking a coordinate whose column is zero
  in every factor. Pins [](eq:archive/examples/math/marquardt), and the mapping of a factor's
  columns to the stacked coordinates through `Structure::range`. (A unit test in `minimize.rs`,
  since `marquardt` is private.)
- `the_schedule_is_nielsens`: `nielsen` at $\varrho_\lambda = 1$ divides $\lambda$ by $3$, at
  $\tfrac12$ keeps it, at $0.9$ multiplies it by $1 - 0.8^3$, and at $0$ and $-1$ multiplies it by
  $\beta$ and doubles $\beta$, which an acceptance resets to $2$. Pins
  [](eq:archive/examples/math/schedule). (A unit test.)
- `levenberg_marquardt_finds_the_mode_on_rotations`: from the chain of rotations' two starting
  points, every row converges and its decrement $\delta$ is at most $10\, \epsilon$. Then two
  checks in the terms the rule bounds, both in nats or derived from them:
  - the query's gradient $g$ by the cotangent pass satisfies
    $\lvert g \rvert^2 \le 2 \cdot 10\, \epsilon \cdot \operatorname{tr}(J\transpose J)$, with
    $J\transpose J$ from the linearization at the mode. The decrement is
    $\tfrac12 g\transpose (J\transpose J)^{-1} g \ge \lvert g \rvert^2 / (2 h_{\max})$, and the
    largest eigenvalue $h_{\max}$ is at most the trace, so this is what a decrement of $10\, \epsilon$
    allows. A fixed bound on $\lvert g \rvert$ would depend on the curvature and on the units of
    the coordinates, as [the stopping rule](math.svx#a-levenberg-marquardt-method) says of a
    gradient's norm.
  - the undamped loop of `steps_converge_on_rotations`, run from the same points, reaches a mode
    whose decrement is also at most $10\, \epsilon$. With $\tangent$ = `Layout::local` of the loop's
    mode at the method's, $\tfrac12 \lvert J \tangent \rvert^2$, which is `promised(τ, 0)` of the
    method's linearization, is at most $40\, \epsilon$: each point is within its decrement of the
    mode in the metric of $J$, and the triangle inequality gives
    $\tfrac12 \lvert J \tangent \rvert^2 \le 2 (\delta_1 + \delta_2)$.

  Shows the method stops at a stationary point of $f$, log-volumes included, to its tolerance, and
  at the point Gauss–Newton reaches.
- `levenberg_marquardt_finds_the_linear_gaussian_mean`: on a linear-Gaussian chain of three
  scalars, the row converges, and with $\mu$ and $\Lambda = J\transpose J$ the closed-form
  posterior mean and precision, $\tfrac12 (x - \mu)\transpose \Lambda (x - \mu) \le 10\,
  \epsilon$ at the mode $x$. In a linear model that quadratic is exactly $f(x) - f(\mu)$, so the
  check is the tolerance itself, in nats; computed from $\Lambda$ on the host, it does not rest on
  the difference of two values of $f$, which rounding would swamp. Shows the method finds the mode
  where the linearization is exact, to its tolerance.
- `rows_are_minimized_independently`: a batch of two rows of different starts gives each row the
  mode and the history that the row alone gives, step for step. Shows that rows share passes and
  nothing else: no row's damping, acceptance or stop leaks into another's.
- `a_rejected_step_keeps_the_point`: on a scalar model with a strongly nonlinear mean, `z ~ N(x³,
  0.1²)` observed and `x ~ N(0, 1)`, from a start far from the mode with $\lambda_0 = 10^{-8}$, the
  history has a rejection; after each, the next iterate's `objective` is the same and its `damping`
  is $\beta$ times larger. Shows rejection leaves the point and grows the damping.
- `non_convergence_is_an_outcome`: `max_iterations: 1` gives `Status::IterationsExceeded` and no
  error; `max_damping` below $\lambda_0$ gives `Status::DampingExceeded`.
- `rows_at_rounding_converge`: a model with many terms, so that $\lvert f \rvert$ is in the
  thousands, and nearly linear, $x \sim \mathcal N(0, 1)$ on $\R^3$ with 2 000 observed
  $z_i \sim \mathcal N(x + 0.01\, x \odot x, I)$, minimized from 1 000 rows of starts near the
  mode: every row ends `Converged`. Rows whose last accepted step promised just over $\epsilon$ take
  a next step whose gain is rounding; a rule that stopped only on an accepted step would send some
  to `max_damping`, and this test is what shows that the rule of the mathematics, which stops on
  the promise, does not. Also checks that a row's final `Iterate` has `promised` at most $\epsilon$.
- `the_fisher_assembly_is_refused`: `minimize` with a Fisher linearizer is `NotAnIntegral`.
- `the_objective_is_the_negative_log_density`: `Linearizer::objective` against the negated
  `DENSITY` of the query's own compiled program, per row, and against `value(0)` of the
  linearization at the same point.

**`skoro-core`.**

- `local_inverts_retract` (`tests/layout.rs`): on a layout of a real, a vector, a positive vector and
  a circle, an embedded type declared in the test file as `tests/embedded.rs` declares its own
  (the core has no sphere of its own), `local(retract(at, τ), at)` equals `τ` to rounding, row by row, for a batch of
  three; and `local(at, at)` is zero. Shows the stacking and the inputs' order are right on every
  kind of factor, flat, positive and curved.
- `local_refuses_as_retract_does`: a missing quantity is `MissingValue`, a discrete one
  `NotContinuous`.
- `generated_values_are_observed_row_by_row` (`crates/skoro/tests/generation.rs`, beside the
  generation tests and their normal helpers, which the core's own tests do not have): generate
  four rows of that file's chain of three scalars, observe the two drawn quantities with
  `observe_value`, and the query's log-density per row equals the closed form at each row's own
  values. Shows the rows are kept apart, not broadcast.
- `a_discrete_generated_value_is_refused`: `observe_value` of a discrete value, and of a
  continuous one of another storage than the quantity's, is refused with `ValueShape` when the set
  is used.

**`skoro-spatial`, `tests/host.rs`.**

- `host_pose_maps_agree_with_the_graph`: `pose_compose` and `pose_transform` against `Q<Pose> *
  Q<Pose>` and `Q<Pose> * Q<Vector<3>>` evaluated by a program, on twenty random poses and points,
  to $10^{-14}$, including quaternions of either sign. Shows the start computed on the host is the
  mean the model computes.

**The examples**, run by CI in release. Their checks are the tests of the whole:

- `the_scenarios_are_sighted`: in both scenarios every landmark is sighted, `first_sightings`
  being `Ok`, and `open_path` does not refuse itself, so every landmark's sightings on the open
  path are an interval of ticks. Shows the constants give the structure the window needs before
  anything is drawn. (A unit test in the examples' `common`, beside `the_chi_squared_constants`.)

- `the_chi_squared_constants`: the distribution function at `CHI2_6_90` by its closed form
  $1 - e^{-x/2}(1 + x/2 + x^2/8)$, and at `CHI2_3_90` by Simpson's rule on the density
  $\sqrt{x}\, e^{-x/2} / \sqrt{2\pi}$ (the standard library has no $\operatorname{erf}$), each
  $0.9$ to $10^{-9}$; and the slopes against $x f_d(x)$ at them, to $10^{-6}$. (A unit test in the examples' `common`;
  each example is declared in `Cargo.toml` with `test = true`, so `cargo test` builds it as a
  test harness and runs its unit tests, not its `main`.)
- `pose_graph`: every row converged; every decrement at most $10\, \epsilon$; the bound on
  $\sigma_\phi^2$ at most $0.05^2$; the poses' and the landmarks' calibrations within
  [](eq:archive/examples/math/check). Fails with the figures when any does not hold.
- `sliding_window`: every tick of every row converged with its decrement at most $10\, \epsilon$;
  the open path's sightings are intervals; the batch on the open path converged. Its calibration
  and its gap are printed and not checked.

## Open questions

None. The rest is decided. The method is the library's, in `skoro-linearize`, with per-row damping by
row-wise elimination; the examples are the umbrella crate's, sharing one model builder; the inverse
retraction, the observation of generated values and the host maps of the motions are added where a
user would look for them.

## Revisions

**2026-11-10: the stopping rule.** The mathematics was reopened because a row stopped only on an
accepted step that promised $\Delta \le \epsilon$, which fails rows that have converged when
$\lvert f \rvert$ is in the thousands. This document had proposed the amendment as an open
question, and the mathematics now states it. What changed here: `Status::Converged` is a step's
promise, accepted or not; `minimize` compares $\Delta$ with the tolerance before the acceptance,
takes the step if its gain is positive, and does not update $\lambda$ for a row that stops; the
test `rows_at_rounding_converge` pins it; and the open question is closed.

**2026-11-10: the method's tests and its tolerance.** The build showed that two tests of the
method asked more than its stopping rule bounds. `levenberg_marquardt_finds_the_mode_on_rotations`
asked for the query's gradient below $10^{-6}$ and for the Gauss–Newton loop's mode to $10^{-8}$,
and `levenberg_marquardt_finds_the_linear_gaussian_mean` for the posterior mean to $10^{-10}$. The
rule stops on a promise of at most $\epsilon = 10^{-9}$ nats. A decrement $\delta$ allows a gradient
up to about $\sqrt{2 \delta h}$ for a curvature $h$, which in these models is of the order of
$10^2$, and the last step, damped by $\lambda \approx 10^{-4}$, leaves the point some $10^{-10}$
from the mode. Run, every row converged without a rejection, with a final decrement near
$10^{-13}$, a gradient of $4 \times 10^{-6}$ to $8 \times 10^{-6}$ and the linear mean within
$5 \times 10^{-10}$. The checks are restated in terms the rule bounds: on rotations, the gradient
against $2 \cdot 10\, \epsilon \cdot \operatorname{tr}(J\transpose J)$, which a decrement of
$10\, \epsilon$ allows, and the Gauss–Newton loop's mode within $40\, \epsilon$ in the metric of
$J$; in the linear model, the mode's excess of $f$ over the mean's, computed in closed form, at
most $10\, \epsilon$.

Three tests were described in ways the build could not or did not follow, and are described here
as built: `the_marquardt_diagonal_is_the_column_norms` uses a problem written by hand, since a unit
test does not reach the integration tests' models; `local_inverts_retract` uses a circle declared
in its file, since the core has no test sphere; and `generated_values_are_observed_row_by_row`
lives in the umbrella crate's generation tests and uses their chain of scalars, since the guide's
model needs the normal, which the core's tests do not have.

**2026-11-10: the open path's range.** The build showed that the open path, with its landmarks at
$(2j, \pm 5, 0)$ and a range of $h = 7$ m, sights ten of its forty landmarks never: the path swings
$3$ m to either side, and a landmark on the side it has swung away from stays $7.5$ to $8$ m from
every pose. [The window](math.svx#a-sliding-window) needs every landmark sighted, its sightings an
interval of ticks, and `Scenario::open_path` refuses itself otherwise, as designed. The scenario's
constants are revised so that every landmark is sighted over one interval: the landmarks stay at
$(2j, \pm 5, 0)$, and the range becomes $h = 8.5$ m. Every landmark is then sighted, over one
interval of $5$ to $16$ ticks. The alternatives were landmarks at $\pm 4$ m with $h = 7.5$ m, or
at $\pm 3.5$ m with $h = 7$ m, which sight every landmark too; the range was changed rather than
the layout. A test, `the_scenarios_are_sighted`, now checks the scenarios' structure before any
example runs. And the circuit's pitch, which this document said followed $0.5 \sin 2\theta$ with
its height, is said as the build has it: the pitch is the path's climb, so that each pose heads
along the path.
