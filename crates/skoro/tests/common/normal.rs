//! A normal distribution of a real quantity, private to the tests: its log-density a composite,
//! and a sampler by inversion of its distribution function.
#![allow(clippy::excessive_precision)] // AS 241's coefficients, as published

use std::f64::consts::PI;
use std::sync::Arc;

use skoro::tensor::{Points, Target, Value};
use skoro::{Composite, Key, Param, Proposal, Q, QDyn, Real, Sample, Sampler, Space};

/// `log N(x; mean, sd²)`.
pub fn log_normal(x: f64, mean: f64, sd: f64) -> f64 {
    let z = (x - mean) / sd;
    -0.5 * z * z - sd.ln() - 0.5 * (2.0 * PI).ln()
}

/// `log N(x; mean, sd²)`, as the composite `test.normal` of `x` and `mean`.
pub fn log_density(x: impl Into<QDyn>, mean: impl Into<QDyn>, sd: f64) -> Q<Real> {
    let composite = Composite::define(
        "test.normal",
        vec![Param::Float(sd)],
        &[Space::Real, Space::Real],
        move |q| {
            let z = (q[0] - q[1]) * (1.0 / sd);
            z * z * -0.5 - (sd.ln() + 0.5 * (2.0 * PI).ln())
        },
    )
    .unwrap();
    composite.call(&[x.into(), mean.into()]).narrow::<Real>()
}

/// Draws from `N(mean, sd²)`, `mean` its one input, by inversion of a variate of each row's key.
#[derive(Debug)]
pub struct Normal {
    pub sd: f64,
}

impl Normal {
    /// The draw from the key of one row, on the host.
    pub fn draw_one(&self, key: &Key, mean: f64) -> f64 {
        mean + self.sd * quantile(key.uniforms(1)[0])
    }
}

impl Sample for Normal {
    fn name(&self) -> &'static str {
        "test.normal"
    }

    fn params(&self) -> Vec<Param> {
        vec![Param::Float(self.sd)]
    }

    fn draw(&self, inputs: &[Value], keys: &[Key], target: &Target) -> Value {
        let Value::Continuous(mean) = &inputs[0] else {
            panic!("a real mean")
        };
        let mean = mean.to_host();
        let draws: Vec<f64> = keys
            .iter()
            .enumerate()
            .map(|(row, key)| self.draw_one(key, mean[row]))
            .collect();
        Value::Continuous(Points::from_host(&draws, keys.len(), 1, target))
    }
}

/// An exact sampler of `N(mean, sd²)`.
pub fn exact(mean: impl Into<QDyn>, sd: f64) -> Sampler {
    Sampler {
        sample: Arc::new(Normal { sd }),
        inputs: vec![mean.into()],
        log_density: None,
    }
}

/// A sampler of `x` from `N(mean, sd²)` for a term that is not that density.
pub fn inexact(x: impl Into<QDyn>, mean: impl Into<QDyn> + Copy, sd: f64) -> Sampler {
    Sampler {
        sample: Arc::new(Normal { sd }),
        inputs: vec![mean.into()],
        log_density: Some(log_density(x, mean, sd)),
    }
}

/// A proposal of the parameter `x` from `N(mean, sd²)`.
pub fn proposal(x: impl Into<QDyn>, mean: impl Into<QDyn> + Copy, sd: f64) -> Proposal {
    Proposal {
        sample: Arc::new(Normal { sd }),
        inputs: vec![mean.into()],
        log_density: log_density(x, mean, sd),
    }
}

/// The standard normal quantile, by Wichura's algorithm AS 241 (PPND16), to about 1e-16.
pub fn quantile(p: f64) -> f64 {
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q * polynomial(r, &A) / polynomial(r, &B);
    }
    let r = if q < 0.0 { p } else { 1.0 - p };
    let r = (-r.ln()).sqrt();
    let x = if r <= 5.0 {
        let r = r - 1.6;
        polynomial(r, &C) / polynomial(r, &D)
    } else {
        let r = r - 5.0;
        polynomial(r, &E) / polynomial(r, &F)
    };
    if q < 0.0 { -x } else { x }
}

fn polynomial(x: f64, coefficients: &[f64; 8]) -> f64 {
    coefficients.iter().rev().fold(0.0, |sum, c| sum * x + c)
}

const A: [f64; 8] = [
    3.387_132_872_796_366_5,
    133.141_667_891_784_38,
    1_971.590_950_306_551_3,
    13_731.693_765_509_461,
    45_921.953_931_549_87,
    67_265.770_927_008_7,
    33_430.575_583_588_13,
    2_509.080_928_730_122_7,
];
const B: [f64; 8] = [
    1.0,
    42.313_330_701_600_91,
    687.187_007_492_057_9,
    5_394.196_021_424_751,
    21_213.794_301_586_597,
    39_307.895_800_092_71,
    28_729.085_735_721_943,
    5_226.495_278_852_545,
];
const C: [f64; 8] = [
    1.423_437_110_749_683_5,
    4.630_337_846_156_546,
    5.769_497_221_460_691,
    3.647_848_324_763_204_5,
    1.270_458_252_452_368_4,
    0.241_780_725_177_450_6,
    0.022_723_844_989_269_184,
    7.745_450_142_783_414e-4,
];
const D: [f64; 8] = [
    1.0,
    2.053_191_626_637_759,
    1.676_384_830_183_803_8,
    0.689_767_334_985_1,
    0.148_103_976_427_480_08,
    0.015_198_666_563_616_457,
    5.475_938_084_995_345e-4,
    1.050_750_071_644_416_9e-9,
];
const E: [f64; 8] = [
    6.657_904_643_501_103,
    5.463_784_911_164_114,
    1.784_826_539_917_291_3,
    0.296_560_571_828_504_9,
    0.026_532_189_526_576_124,
    0.001_242_660_947_388_078_4,
    2.711_555_568_743_487_6e-5,
    2.010_334_399_292_288_1e-7,
];
const F: [f64; 8] = [
    1.0,
    0.599_832_206_555_888,
    0.136_929_880_922_735_8,
    0.014_875_361_290_850_615,
    7.868_691_311_456_133e-4,
    1.846_318_317_510_054_8e-5,
    1.421_511_758_316_446e-7,
    2.044_263_103_389_939_8e-15,
];
