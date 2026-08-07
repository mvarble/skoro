//! Frozen `f64` references for `exp_hvp` and `log_hvp`, and the macro that checks a backend
//! against them.
//!
//! Each map's Jacobian coefficients exist twice — once in `ops::*::ndarray` and once transcribed by
//! hand into `ops::*::cube` — and the two can drift apart silently. These tables are what stops
//! that: they are the only thing in the workspace comparing the GPU kernels against a finer
//! reference. The finite-difference tests in [`super::numerical`] cannot do it, because their
//! tolerance floor is set by the differencing rather than by the ops.
//!
//! What they cover, established by mutation:
//!
//! - reverting `log`'s coefficients to the pre-fix form (an `f64`-calibrated cutoff with a
//!   zeroth-order Taylor branch) moves `log_hvp` by 2.1e-5 against a 1e-6 tolerance
//! - swapping two coefficient fields at a `cube` call site moves `exp_hvp` well past tolerance
//!
//! What they do **not** cover: swapping the current `f32` and `f64` cutoffs for `log`. The primed
//! coefficient contributes only ~1% of `log_hvp` over the band where the two cutoffs disagree, so
//! the resulting error lands below `f32` round-off in the output. The cutoff choice is tested
//! directly instead, by the unit tests beside each `*_jacobian_coeffs` function.
//!
//! The reference is a table rather than a live call into `ops::*::ndarray`, because a backend test
//! executable must contain exactly one backend at one precision — an `NdArray` tensor at `f32` and
//! one at `f64` in the same binary collapse to a single `DType`. Checking the GPU against a CPU
//! reference in one process is precisely the arrangement that breaks.
//!
//! To regenerate after an intentional change to the math, run the tables out of an `f64`-only
//! executable (assert `out.dtype()` is `DType::F64` while you do — a silently demoted reference
//! would still pass every check here).

/// A generic unit axis, so no component of `ω` is incidentally zero.
pub const AXIS: [f64; 3] = [0.26726124, 0.53452248, 0.80178373];

/// The velocity `v` that the HVPs differentiate `J(ω)` along. Chosen with a large component along
/// [`AXIS`]: the primed coefficients enter only through a `(v·ω)` factor, so a `v` near-orthogonal
/// to `ω` hides them — with the previous `[0.3, -0.7, 0.2]` the dot product was -0.13 and reverting
/// `log`'s coefficients moved the result by only 10x the tolerance instead of 10000x.
pub const VELOCITY: [f64; 3] = [0.5, 0.6, 0.9];

/// A single-column cotangent frame.
pub const ALPHA: [f64; 3] = [1.0, -2.0, 0.5];

/// Decades near the origin, then a uniform sweep out to `θ = 3`, spanning every branch cutoff in
/// play (`exp`: `θ² = 1.366` / `0.0245`; `log`: `θ² = 5.0` / `0.3`). Written as literals so the
/// tables cannot silently desynchronise from the points they were generated at.
#[rustfmt::skip]
pub const THETAS: [f64; 14] = [
    0.0001, 0.00030000000000000003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.30000000000000004,
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
];

/// `exp_hvp(θ·AXIS, VELOCITY, ALPHA)` evaluated on `NdArrayBackend<f64, i32>`, one row per
/// [`THETAS`] entry.
#[rustfmt::skip]
pub const EXP_HVP: [[f64; 3]; 14] = [
    [1.049956344622252, 0.32507215960978375, -0.8000289515868005],
    [1.0498690176178003, 0.32521647327407377, -0.8000868444739532],
    [1.049563202515623, 0.32572151272390215, -0.8002893615520291],
    [1.0486879835926537, 0.32716398110908107, -0.8008670553515576],
    [1.04560769339738, 0.33220672656672706, -0.8028781567707979],
    [1.0366616365720467, 0.34656294226465273, -0.8085308865543369],
    [1.00368402058441, 0.3961644133465692, -0.8272093709688204],
    [0.8959607587113334, 0.5312808224424864, -0.8706828833851075],
    [0.7703609608411376, 0.6541836834813041, -0.898693801203838],
    [0.39940563387024297, 0.8905832964873682, -0.8976020488537763],
    [-0.006126275045297058, 1.0028613902679615, -0.7975895858813009],
    [-0.38508512717992, 0.9806972201584007, -0.6153487518639466],
    [-0.682697833384446, 0.8360556950969344, -0.38026673634549746],
    [-0.8599696223443749, 0.6003496319895723, -0.12903504423150303],
];

/// `log_hvp(θ·AXIS, VELOCITY, ALPHA)` evaluated on `NdArrayBackend<f64, i32>`, one row per
/// [`THETAS`] entry. `log`'s coefficients are a different pair from `exp`'s and were broken in a
/// different way, so they need their own probe.
#[rustfmt::skip]
pub const LOG_HVP: [[f64; 3]; 14] = [
    [-1.0500218263347365, -0.3249639197322632, 0.7999855233494366],
    [-1.0500654790043835, -0.32489175919650515, 0.7999565700481874],
    [-1.0502182633545458, -0.32463919731090357, 0.7998552334893207],
    [-1.0506547902377394, -0.32391759164839545, 0.7995657003456582],
    [-1.0521826407271762, -0.32139196138099724, 0.7985523298481552],
    [-1.0565480962893499, -0.31417559981783305, 0.7956568672361851],
    [-1.0718335915520107, -0.28890788160352665, 0.7855182515957404],
    [-1.115675499339186, -0.2164383164298143, 0.7564320050622076],
    [-1.1600466091169674, -0.14310458916173782, 0.7269739109237199],
    [-1.2757853122663503, 0.04808452992237962, 0.6499458505554888],
    [-1.4040020944427187, 0.25963967075381483, 0.5641301964246723],
    [-1.554105339631018, 0.5068955217735751, 0.46285973332728636],
    [-1.7408246237051483, 0.8138310628605923, 0.3356401829861475],
    [-1.9897541636676843, 1.2220631846468826, 0.16414031198594697],
];

/// Check one backend's `exp_hvp` or `log_hvp` against its table at a scaled tolerance.
///
/// The HVP is the probe because it is the only entry point that uses the primed coefficients, the
/// ones that cancel worst. Tolerances are measured floors backed off for margin: `f32` runs at
/// 1e-6 against a floor of 3.6e-7, `f64` at 1e-12 against a floor below 1e-15.
///
/// A typo in a leading-order constant shows up. A typo in the last term or two of a series does
/// not — those sit below `f32` round-off, which is the reason the series stops where it does.
#[macro_export]
macro_rules! test_hvp_reference {
    ($name:ident, $hvp:path, $table:ident, $tolerance:expr$(,)?) => {
        #[test]
        pub fn $name() {
            use super::B;
            use super::common::coefficients::{ALPHA, AXIS, THETAS, VELOCITY, $table};
            use skoro_model::test_utils::tensor;

            for (theta, want) in THETAS.iter().zip($table.iter()) {
                // Scale by the magnitude of the whole result, not of the component: these outputs
                // pass through zero as θ sweeps, and a per-component relative error blows up there
                // while saying nothing about accuracy.
                let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                let device = Default::default();
                let got = $hvp(
                    tensor::<B, 2>([AXIS.map(|x| x * theta)], &device),
                    tensor::<B, 2>([VELOCITY], &device),
                    tensor::<B, 3>([[[ALPHA[0]], [ALPHA[1]], [ALPHA[2]]]], &device),
                )
                .into_data()
                .convert::<f64>()
                .to_vec::<f64>()
                .unwrap();

                for (i, (&want, &got)) in want.iter().zip(got.iter()).enumerate() {
                    let error = (got - want).abs() / scale;
                    assert!(
                        error < $tolerance,
                        "θ={theta}: component {i} is {got} vs {want} (f64 reference), \
                         scaled error {error:e} exceeds {:e}",
                        $tolerance as f64,
                    );
                }
            }
        }
    };
}
