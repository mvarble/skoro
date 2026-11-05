---
summary: The wgpu tests admit their adapter against the stated error of its sine and cosine before they run; the rotations' coefficients move their cutoff to θ² = 1 with series long enough for it; the test sphere's exponential and log-volume move theirs to r² = 0.1.
depends_on: [archive/wgpu-accuracy/math]
---

## Overview

When this is built, three things are different, and none of them is visible to a user of the library
outside its tests.

1. **Every wgpu test admits its adapter first.** `skoro_core::testing::wgpu()`, which every
   `name::wgpu` instance of `precision_tests!` calls, measures the adapter's sine and cosine once per
   test binary, against double precision on the host, and refuses the adapter when its error exceeds
   the bounds of [](eq:archive/wgpu-accuracy/math/design-error). This is the admission of
   [](eq:archive/wgpu-accuracy/math/admission). A refused adapter fails every wgpu test with a message
   naming the error measured, the argument at which it was reached, and the bound exceeded:

   ```text
   the wgpu adapter is not admitted: its sine and cosine err by 2.1e-6 at x = 0.73 on [-π, π],
   above the 9.5e-7 the cutoffs are designed for (archive/wgpu-accuracy/math)
   ```

2. **The rotations' coefficients switch at $\theta^2 = 1$.** `skoro-spatial`'s `series::CUTOFF`
   moves from $0.1$ to $1$, and each series of `a`, `b`, `A`, `B`, `C`, `D` and `E` keeps as many
   terms as [](eq:archive/wgpu-accuracy/math/truncation-condition) asks there. The closed form above
   the cutoff then meets [](eq:archive/wgpu-accuracy/math/closed-condition) at $\varepsilon_\star$
   with the margin found under "Conditions on a cutoff" of the mathematics. `LOG_CUTOFF`, the cutoff
   of the logarithm's `L`, does not move: its closed form divides an $\operatorname{atan2}$, which the
   mathematics found loses nothing.

3. **The test sphere's exponential switches at $r^2 = 0.1$.** The fixture in
   `crates/skoro/tests/common/sphere.rs` splits its one cutoff `CUT` in two. Its retraction and its
   log-volume $\Lambda$ move to $r^2 = 0.1$, with series of $\cos r$, $\sin r / r$ and $\Lambda$ long
   enough there. Its logarithm keeps $10^{-8}$.

The three tests that fail on the adapter today pass, with no tolerance changed:
`series_meet_closed_forms::wgpu` (in `series.rs`'s tests) and `composites_pass_check_operation::wgpu`
(in `tests/checks.rs`) in `skoro-spatial`, and `exponential_sphere_passes_check_volume::wgpu` in
`skoro`.

What a person running the tests meets is the admission. With an adapter, `cargo test --workspace
--features wgpu` is unchanged when the adapter is admitted. When it is not, every wgpu test fails at
the same first line, rather than three of them failing in a coefficient's second derivative.

## Constraints

Read at commit `2e4ad64`.

**The wgpu target of the tests.** `crates/skoro-core/src/testing.rs`:

```rust
/// The test adapter at single precision, or an error naming the missing adapter.
#[cfg(feature = "wgpu")]
pub fn wgpu() -> Result<Target, String> {
    let device = skoro_tensor::test_adapter(skoro_tensor::wgpu_devices())?;
    Target::new(device, Precision::Single).map_err(|e| e.to_string())
}
```

and the macro that calls it, in the same file:

```rust
            $crate::__if_wgpu! {
                #[test]
                fn wgpu() {
                    let $target = $crate::testing::wgpu().unwrap();
                    $body
                }
            }
```

Every wgpu test in the workspace reaches its target through `testing::wgpu()`, and a missing adapter
already fails there, naming itself (`skoro_tensor::test_adapter`). Admission belongs at the same
point, for the same reason. It is a precondition of every check that compares single precision with
double, and a failure there is reported before any check runs. Because the function returns
`Result<Target, String>` and the macro unwraps it, an admission failure needs no new path to the
test's output.

**The agreement and the checks' tolerances.** `crates/skoro-core/src/testing.rs`:

```rust
impl Tolerance {
    /// How closely a single-precision result is held to the double-precision one.
    pub const AGREEMENT: f64 = 1e-4;

    pub fn of(precision: Precision) -> Tolerance {
        match precision {
            Precision::Double => Tolerance { step: 1e-6, relative: 1e-7 },
            Precision::Single => Tolerance { step: 1e-2, relative: 5e-3 },
        }
    }
}
```

None of these changes. The bounds of [](eq:archive/wgpu-accuracy/math/design-error) are not a
tolerance of any check: the mathematics says so under "In skoro's terms". They do not go on
`Tolerance`.

**The coefficients.** `crates/skoro-spatial/src/series.rs`:

```rust
/// The cutoff in `θ²` below which the series of `a`, `b`, `A`, `B`, `C`, `D` and `E` are used.
pub(crate) const CUTOFF: f64 = 0.1;

/// The cutoff in `x = n²/w²` below which the series of `L` is used.
pub(crate) const LOG_CUTOFF: f64 = 0.01;

/// The coefficient `c` at `θ² = theta2`, a `Real`: the series below `CUTOFF`, the closed form
/// above, its argument guarded where the series is chosen.
pub(crate) fn coefficient(c: Coefficient, theta2: QDyn) -> QDyn {
    let small = (theta2 * -1.0 + CUTOFF).is_nonnegative();
    let safe = choose(small, theta2, one());
    let theta = sqrt(safe);
    choose(small, closed(c, theta, safe), horner(series(c), theta2))
}
```

The series are tables built at compile time by `table!` from closed expressions in `factorial`,
`sign` and `four`, six terms each, and `C`'s seven are written out with the Bernoulli numbers
$B_2 \ldots B_{14}$. One `CUTOFF` serves every coefficient, and the guard is `choose(small, theta2,
one())`. The guard puts $\theta^2 = 1$ into the closed form on the rows where the series is chosen.
At the new cutoff, that is the cutoff itself, which is harmless, since the closed form is defined
there. The integration tests carry their own copy of the constant
(`crates/skoro-spatial/tests/common/mod.rs`, `pub const CUTOFF: f64 = 0.1`), from which
`fixed_quaternions` and `checks.rs` place their rows at $0.9$ and $1.1$ times it.

**The test sphere.** `crates/skoro/tests/common/sphere.rs`:

```rust
/// The squared radius below which the series is used.
pub const CUT: f64 = 1e-8;
```

It is used in three places. The retraction `exponential(near)` uses it to select
`-ss * 0.5 + 1.0 + ss * ss * (1.0 / 24.0)` for $\cos r$ and the matching two terms for
$\sin r / r$. `logarithm()` uses it for the ratio $\theta / r$. `exponential_log_volume(true)` uses it
for $-r^2/6 - r^4/180$. The second-order map of $\Lambda$ that `check_volume` takes at $t = 0$ is
compared with central differences at the single-precision step of $10^{-2}$, where $r^2 = 10^{-4}$.
`check_volume` also compares $\Lambda$ with the area formula. The area formula takes the retraction's
Jacobian at the same tangents, so the retraction and $\Lambda$ must switch at the same radius: with
$\Lambda$ alone moved, that comparison would set the series of one against the closed form of the
other on the GPU.

**The archive.** The archive's exponential (`crates/skoro-so3/src/ops/exp/flex.rs` on `archive`)
chose its cutoff by precision. It used $\theta^2 = 1.3661$ when the element type was single and
$0.024513$ when it was double, from balancing a truncation of $O(\theta^{10} / 11!)$ against each
width's rounding. Two things are kept: a large cutoff in single precision, and series long enough to
reach it. The per-precision cutoff is not kept. A composite sees neither its precision nor its device,
so one cutoff serves every target, as the mathematics has it. That cutoff is met in double precision
by lengthening the series rather than by lowering the cutoff.

## Design

### Admission

Admission lives in `skoro_core::testing`, beside `wgpu()`, and is public, since `skoro-spatial` and
`skoro` reach the wgpu target through the macro. It is test support alone. `Target::new` and
`skoro-tensor` do not change: an unadmitted adapter computes correctly, only less accurately than the
design assumes.

```rust
/// The absolute error of a target's sine and cosine, measured against double precision on the
/// host, and the bounds the removable singularities' cutoffs are designed for
/// (archive/wgpu-accuracy/math, "The error designed for").
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Admission {
    /// The largest error measured on `[-π, π]`, and the argument it was reached at.
    pub near: (f64, f64),
    /// The largest error measured on `π < |x| ≤ FAR_RANGE`, and the argument.
    pub far: (f64, f64),
}

impl Admission {
    /// `ε⋆`: the error designed for on `[-π, π]`.
    pub const NEAR: f64 = 9.5367431640625e-7; // 2⁻²⁰
    /// `ε⋆ far`: the error designed for beyond `π`.
    pub const FAR: f64 = 1.52587890625e-5; // 2⁻¹⁶
    /// The largest argument measured beyond `π`.
    pub const FAR_RANGE: f64 = 100.0;

    /// The error of `target`'s sine and cosine over the inputs `inputs()`.
    pub fn measure(target: &Target) -> Result<Admission, String>;

    /// `Ok` when both errors are within their bounds; otherwise a message naming the error, its
    /// argument, the bound exceeded, and the document the bound is stated in.
    pub fn admit(&self) -> Result<(), String>;
}
```

`wgpu()` becomes

```rust
#[cfg(feature = "wgpu")]
pub fn wgpu() -> Result<Target, String> {
    static ADMITTED: OnceLock<Result<(), String>> = OnceLock::new();
    let device = skoro_tensor::test_adapter(skoro_tensor::wgpu_devices())?;
    let target = Target::new(device, Precision::Single).map_err(|e| e.to_string())?;
    ADMITTED
        .get_or_init(|| Admission::measure(&target)?.admit())
        .clone()
        .map_err(|e| format!("the wgpu adapter is not admitted: {e}"))?;
    Ok(target)
}
```

The measurement runs once per test binary, because `test_adapter` returns the same first adapter
every time. Once it has run, a test's admission costs a lock. The `OnceLock` caches the outcome, not
the `Admission`, so that a failed measurement, such as a program that does not compile, also fails
every test with the same message.

**The inputs.** `inputs()` is private and deterministic. It returns:

- on $[-\pi, \pi]$: 4096 arguments spaced logarithmically from $10^{-7}$ to $\pi$, and 4096 spaced
  evenly on $[0, \pi]$, each with its negative, and $0$;
- beyond $\pi$: 4096 arguments spaced evenly on $(\pi, 100]$, each with its negative.

This is dense near $0$, where the closed forms are used just above their cutoffs, as dense near
$\pi$, and as wide as the measurements of the mathematics. Each argument is rounded to single
precision before anything else (`x as f32 as f64`), so that the reference $\sin x$ is taken at the
input the device actually receives. Loading the value onto a single-precision target would round it
in the same way. The far range of $100$ is the range measured. The tests' arguments stay well inside
it: `testing::random` draws their inputs from ranges of a few units, and the largest rotation angle the
spatial tests place by hand is $3.0$, among the `twists` of `tests/checks.rs`.

**The measurement.** `measure` builds one program of the arguments as a batch of `Space::Real` rows,
with targets `x.sin()` and `x.cos()`, through the same `Writer`, `Query::of` and `compile` that every
test uses. The adapter is therefore measured through the kernels the checks run, not through burn
directly. It runs `forward`, reads both targets back with `to_host`, and takes the largest
$\max(\lvert \widetilde\sin x - \sin x \rvert, \lvert \widetilde\cos x - \cos x \rvert)$ over each
range, with $\sin x$ and $\cos x$ computed in `f64` on the host. Its error, $2^{-53}$, is negligible.

The alternative was a separate test, `the_adapter_is_admitted`, alongside the others. It lost because
it leaves the other tests running on an unadmitted adapter and failing in their coefficients. The
mathematics asks that a device not admitted fail at admission.

### The rotations' coefficients

In `crates/skoro-spatial/src/series.rs`:

```rust
/// The cutoff in `θ²` below which the series of `a`, `b`, `A`, `B`, `C`, `D` and `E` are used,
/// chosen so that the closed form above it meets the GPU's error designed for
/// (archive/wgpu-accuracy/math, "Conditions on a cutoff").
pub(crate) const CUTOFF: f64 = 1.0;
```

The series' lengths are those [](eq:archive/wgpu-accuracy/math/truncation-condition) asks at
$\theta_c^2 = 1$. The tail $\sum_{k \ge N} \lvert c_k \rvert\, k^{\underline m}\, x^{k - m}$ is bounded
for $m \le 2$ against $2^{-53} \lvert f(1) \rvert$, with $k^{\underline m}$ the falling factorial:

| coefficient | terms | through |
| --- | --- | --- |
| `a`, `b` | 8 | $\theta^{14}$ |
| `A`, `B`, `D`, `E` | 9 | $\theta^{16}$ |
| `C` | 12 | $\theta^{22}$ |

The `table!` lines change only in their lengths. `C_SERIES` becomes a `table!` too, with the term
`sign(k) * BERNOULLI[k] / factorial(2 * k + 2)`. Its Bernoulli numbers $B_2 \ldots B_{24}$ move from
the tests into the module, as

```rust
/// The Bernoulli numbers `B₂, B₄, …, B₂₄`.
const BERNOULLI: [f64; 12] = [1.0 / 6.0, -1.0 / 30.0, /* … */ -236364091.0 / 2730.0];
```

and the tests' references are computed independently of the table (see [Tests](#tests)).
`factorial(24)` is not exact in `f64`, but its rounding is a relative $2^{-53}$, which is the
rounding the condition allows. `horner`, `closed` and `coefficient` do not change. `LOG_CUTOFF` and
`LOG_SERIES` do not change. The module's doc comment already says what the cutoff must satisfy, and
gains the reference to the GPU's error.

In `crates/skoro-spatial/tests/common/mod.rs`, `CUTOFF` becomes `1.0` with the library's.
`fixed_quaternions` and `checks.rs` follow it unchanged, at $\theta^2 = 0.9$ and $1.1$.

The alternative was a cutoff of $\theta^2 = 0.5$, with one term fewer in most series. It lost
because the mathematics measured the GPU's errors at $\theta^2 = 1$ and not at $0.5$. Scaling from
the measurement at $0.1$ suggests $0.5$ would serve, but only by about one order of margin over the
CPU, against two to three at $1$. A Horner step costs a product and a sum. The archive's $1.37$ is
evidence that a cutoff of this size works in single precision.

### The test sphere

In `crates/skoro/tests/common/sphere.rs`, `CUT` is split:

```rust
/// The squared radius below which the exponential retraction and its log-volume use their series,
/// chosen so that their closed forms above it meet the GPU's error designed for
/// (archive/wgpu-accuracy/math, "Conditions on a cutoff").
pub const EXP_CUT: f64 = 0.1;

/// The squared radius below which the logarithm's ratio `θ / r` uses its series. Its closed form
/// divides an `atan2`, which loses nothing to the GPU, and its series `1 + r²/6` holds only near
/// `r = 0`.
pub const LOG_CUT: f64 = 1e-8;
```

Raising the logarithm's cutoff would need the series of $\theta / \sin \theta$ in $r = \sin\theta$.
That series converges only for $r < 1$, and only on the half of the sphere where $x^\top y > 0$, and
nothing asks for it.

The series become tables beside the cutoffs, summed by a local `horner` (the spatial crate's is
`pub(crate)`):

```rust
/// `cos r = Σ (-1)^k r^{2k} / (2k)!`, eight terms.
const COS_SERIES: [f64; 8] = /* … */;
/// `sin r / r = Σ (-1)^k r^{2k} / (2k+1)!`, eight terms.
const SINC_SERIES: [f64; 8] = /* … */;
/// `Λ = log(sin r / r) = Σ_{n ≥ 1} (-1)^n 2^{2n-1} B_{2n} r^{2n} / (n (2n)!)`, ten terms from
/// `r²`, with the Bernoulli numbers `B₂ … B₂₀`: `-r²/6 - r⁴/180 - r⁶/2835 - …`.
const LOG_VOLUME_SERIES: [f64; 10] = /* … */;

fn horner(coefficients: &[f64], x: QDyn) -> QDyn;
```

`exponential(near)` selects `horner(&COS_SERIES, ss)` and `horner(&SINC_SERIES, ss)` below
`EXP_CUT`, for `Near::Selected` and `Near::Guarded` alike. `Near::Closed` stays the bare closed form:
it exists to show that a selection without a guard fails at $t = 0$, and no cutoff enters it.
`exponential_log_volume(true)` selects `ss * horner(&LOG_VOLUME_SERIES, ss)` below `EXP_CUT`.
`logarithm()` uses `LOG_CUT`. The lengths are those [](eq:archive/wgpu-accuracy/math/truncation-condition)
asks at $r_c^2 = 0.1$, by the same tail bound as the rotations' coefficients.

The second-order map at $t = 0$ and the central differences at $r^2 = 10^{-4}$ are then both taken
from the series. Above $r^2 = 0.1$, the closed form's errors at $\varepsilon_\star$ are within the
check's tolerance by the scaling of the mathematics.

### What the build confirms, and what it cannot

The margin at $\varepsilon_\star$ is the mathematics' scaling of the errors measured on one adapter.
The build confirms it on that adapter alone: the wgpu tests pass there, with the adapter admitted.
On an adapter whose error lies between the one measured and $\varepsilon_\star$, the design rests on
the scaling and on the margin of two to three orders at $\theta^2 = 1$. Nothing in the test suite
can confirm it there without such an adapter. The commit that changes the cutoffs states, as
`AGENTS.md` asks, that `cargo test --workspace --features wgpu` passed, on which adapter, and the
`Admission` measured.

## Tests

The acceptance criterion is that these three, failing today, pass under `cargo test --workspace
--features wgpu` on the adapter, with no tolerance changed:

- `skoro-spatial`'s `series::tests::series_meet_closed_forms::wgpu`;
- `skoro-spatial`'s `composites_pass_check_operation::wgpu`, in `tests/checks.rs`;
- `skoro`'s `exponential_sphere_passes_check_volume::wgpu` in `tests/sphere.rs`.

They need no change of their own. They place their points at $0.9$ and $1.1$ times the cutoff, and
at the central-difference step, and follow the new constants. Every other test, wgpu or not, still
passes. New and changed tests:

**Admission**, in `crates/skoro-core/src/testing.rs`'s tests:

- `admission_names_the_error_and_the_bound`: `Admission { near: (2e-6, 0.73), far: (1e-6, 50.0) }
  .admit()` fails with a message containing `2e-6` (or its formatting), `0.73`, the bound $2^{-20}$
  and `archive/wgpu-accuracy/math`. An `Admission` within both bounds is admitted, and one within
  `NEAR` but over `FAR` fails naming `FAR`. This checks the comparison and the message without a
  device that fails.
- `the_host_is_admitted_in_single_precision`: `Admission::measure(&flex(Single))` is admitted, with
  `near.0` below $10^{-7}$. The CPU's sine is accurate relative to its value, so its absolute error on
  $[-\pi, \pi]$ is a few single-precision roundings of numbers at most $1$. This shows the
  measurement measures, and gives a known answer for the program it compiles. It is not gated on
  `wgpu`.
- `the_adapter_is_admitted` (with `wgpu`): `Admission::measure(&wgpu-target)` is admitted, and the
  test prints the `Admission`, so that the build's commit can quote it. On the adapter of the
  mathematics, `near.0` is about $3.8 \cdot 10^{-7}$ and `far.0` about $7.8 \cdot 10^{-6}$, and a
  value far from these says the measurement differs from the one the mathematics made.

**The rotations' coefficients**, in `crates/skoro-spatial/src/series.rs`'s tests:

- `series_are_truncated_below_rounding` (new): for each coefficient and $m \le 2$, the tail beyond
  the table's last term, at `CUTOFF`, is below $2^{-53}\, \lvert f \rvert$. This is
  [](eq:archive/wgpu-accuracy/math/truncation-condition) itself. For `a`, `b`, `A`, `B`, `D` and `E`,
  the tail is summed from the closed expressions of the terms for forty terms past the table, which
  bounds it since the terms fall factorially. For `C`, the tail is bounded with
  $\lvert B_{2n} \rvert / (2n)! = 2 \zeta(2n) / (2\pi)^{2n}$ and $\zeta(2n) \le 1 + 2^{-2n} + 2^{1-2n}
  / (2n - 1)$, a geometric series in $x / 4\pi^2$. This is the test that fails if the cutoff is
  raised without the terms.
- `series_meet_closed_forms` (changed reference): `summed`, its reference sum to convergence, takes
  twenty terms rather than twelve. `C`'s terms become $2 \zeta(2k + 2) / (2\pi)^{2k + 2}$, with
  $\zeta$ summed directly, rather than the `BERNOULLI` table. Without that change, a reference of
  twelve terms from the table the series is built from would compare `C`'s series with itself below
  the cutoff. The comment "twelve terms being far below rounding at θ² ≤ 0.11" becomes one for
  $\theta^2 \le 1.1$. The test's points follow `CUTOFF`. Its far length `1.3` now sits just above
  $1.1$, so `2.5` is added to `lengths`.
- `coefficients_are_their_closed_forms_away_from_the_identity` (changed points): its $\theta^2 = 0.5$
  is now below the cutoff and would compare the series with the closed form. The points become
  $1.5$, $2$ and $9$, all above it.
- `the_bernoulli_numbers_are_the_zeta_values` (new): each entry of `BERNOULLI` against $(-1)^{n+1}
  2 \zeta(2n) (2n)! / (2\pi)^{2n}$, to $10^{-14}$ relative. The table now builds a series, so it is
  checked against an independent expression of the same numbers.

**The test sphere**, in `crates/skoro/tests/sphere.rs`:

- `exponential_series_are_truncated_below_rounding` (new): as for the coefficients, the tails of
  `COS_SERIES`, `SINC_SERIES` and `LOG_VOLUME_SERIES` at `EXP_CUT`, for $m \le 2$, are below
  $2^{-53}$ times each function's value there. $\Lambda$'s tail is bounded with $\lvert \lambda_n \rvert
  = \zeta(2n) / (n \pi^{2n})$, a geometric series in $r^2 / \pi^2$.
- `exponential_series_meet_closed_forms` (new): each series at $0.9$ `EXP_CUT`, in double precision
  on the host, against its closed form, $\cos r$, $\sin r / r$ and $\log(\sin r / r)$, to $10^{-15}$.
  The closed forms are accurate to that there, since $r^2 = 0.09$ is far from where they cancel in
  double precision. The test also compares `LOG_VOLUME_SERIES` with $-\zeta(2n) / (n \pi^{2n})$, the
  coefficients from an expression independent of its Bernoulli numbers.
- The existing checks of the sphere, `check_embedded`'s and `check_volume`'s among them, run
  unchanged at both precisions and on wgpu, and now exercise the series out to $r^2 = 0.1$. The two
  that expect `Near::Closed` and `Near::Selected` to fail at $t = 0$ still do, since both still
  evaluate the closed form at $r = 0$.
