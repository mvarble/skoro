//! The Laplace marginals' normalized errors, and the calibration statistic of a set of them.

use faer::{Mat, Side};
use skoro::{Layout, NodeId, Values};
use skoro_linearize::{Linearized, Linearizer};

/// `χ²₆(0.9)` and `χ²₃(0.9)`, and the slopes `χ²_d(α) f_d(χ²_d(α))` of the coverage's error at
/// them, computed once from the closed forms of the two distribution functions.
pub const CHI2_6_90: f64 = 10.644_640_675_668_418;
pub const CHI2_3_90: f64 = 6.251_388_631_170_324;
pub const SLOPE_6: f64 = 0.367_975;
pub const SLOPE_3: f64 = 0.273_781;

/// A quantity's Laplace marginals, row by row: what the check reads, and what a figure draws.
pub struct Marginals {
    /// Each row's normalized error.
    pub errors: Vec<f64>,
    /// For a pose, each row's largest rotational variance.
    pub spread: Option<Vec<f64>>,
    /// Each row's covariance of the position in the world's axes, by rows: a landmark's own, and
    /// a pose's `R Σ_ρ Rᵀ`, its tangent `ρ` being along the pose's own axes.
    pub position: Vec<[f64; 9]>,
}

/// Each row's normalized error of `quantity`, `|W (q* ⊖ q̂ - μ)|²`: `chart`, the chart
/// linearization at `mode` computed once for every quantity, eliminated along the schedule
/// keeping `quantity`; its marginal's whitening `W` and mean `μ` per row; and `q* ⊖ q̂` from
/// `Layout::local`. For a pose, also the largest eigenvalue of the marginal covariance's rotation
/// block, per row; and each row's covariance of the position.
pub fn normalized_errors(
    linearizer: &Linearizer,
    chart: &Linearized,
    mode: &Values,
    truth: &Values,
    quantity: NodeId,
) -> Result<Marginals, Box<dyn std::error::Error>> {
    let schedule = linearizer.schedule_keeping(&[quantity])?;
    let marginals = chart.eliminate(&schedule, None)?.marginal()?;
    let layout = Layout::new(linearizer.snapshot(), &[quantity])?;
    let d = layout.dimension();
    let tangent = layout.local(truth, mode, linearizer.target())?.to_host();
    // A pose stores a quaternion and a position; its tangent is (φ, ρ), rotation first.
    let pose = layout.product().storage() == 7;
    let mut errors = Vec::with_capacity(marginals.len());
    let mut spreads = Vec::new();
    let mut position = Vec::with_capacity(marginals.len());
    let at = match mode.get(quantity) {
        Some(skoro::tensor::Value::Continuous(points)) => points.to_host(),
        _ => return Err(format!("no mode for {quantity:?}").into()),
    };
    let storage = layout.product().storage();
    for (r, marginal) in marginals.iter().enumerate() {
        let marginal = &marginal.marginal;
        let tau = if tangent.len() == d {
            &tangent[..]
        } else {
            &tangent[r * d..(r + 1) * d]
        };
        let centred: Vec<f64> = tau.iter().zip(&marginal.mean).map(|(t, m)| t - m).collect();
        let whitened = marginal.whitening.apply(&centred);
        errors.push(whitened.iter().map(|v| v * v).sum());
        let covariance = marginal.covariance();
        let offset = if pose { 3 } else { 0 };
        let sigma = |i: usize, j: usize| covariance.get(offset + i, offset + j);
        position.push(if pose {
            let row = if at.len() == storage { 0 } else { r };
            let q = &at[row * storage..row * storage + 4];
            let rot = skoro_spatial::host::matrix([q[0], q[1], q[2], q[3]]);
            std::array::from_fn(|n| {
                let (i, j) = (n / 3, n % 3);
                (0..3)
                    .flat_map(|a| (0..3).map(move |b| (a, b)))
                    .map(|(a, b)| rot[i][a] * sigma(a, b) * rot[j][b])
                    .sum()
            })
        } else {
            std::array::from_fn(|n| sigma(n / 3, n % 3))
        });
        if pose {
            let block = Mat::from_fn(3, 3, |i, j| covariance.get(i, j));
            let eigenvalues = block
                .self_adjoint_eigenvalues(Side::Lower)
                .map_err(|e| format!("the rotation block's eigenvalues: {e:?}"))?;
            spreads.push(eigenvalues.into_iter().fold(0.0, f64::max));
        }
    }
    Ok(Marginals {
        errors,
        spread: pose.then_some(spreads),
        position,
    })
}

/// The calibration statistic of a set of quantities: per row the fraction `c_r` of them whose
/// error is at most `χ²_d(0.9)`, the mean `c̄`, `s* = max(s, 1/n)` and the sampling part
/// `4 s* / √N` of the check.
#[derive(Clone, Debug)]
pub struct Calibration {
    pub mean: f64,
    pub sampling: f64,
    pub fractions: Vec<f64>,
}

/// `errors[i][r]`: quantity `i`'s normalized error in row `r`.
pub fn calibration(errors: &[Vec<f64>], quantile: f64) -> Calibration {
    let n = errors.len();
    let rows = errors.first().map_or(0, Vec::len);
    let fractions: Vec<f64> = (0..rows)
        .map(|r| errors.iter().filter(|e| e[r] <= quantile).count() as f64 / n as f64)
        .collect();
    let mean = fractions.iter().sum::<f64>() / rows as f64;
    let variance =
        fractions.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / (rows as f64 - 1.0).max(1.0);
    let floored = variance.sqrt().max(1.0 / n as f64);
    Calibration {
        mean,
        sampling: 4.0 * floored / (rows as f64).sqrt(),
        fractions,
    }
}

#[cfg(test)]
mod tests {
    use super::{CHI2_3_90, CHI2_6_90, SLOPE_3, SLOPE_6};

    /// The chi-squared density with 3 degrees of freedom, `√x e^{-x/2} / √(2π)`.
    fn density_3(x: f64) -> f64 {
        x.sqrt() * (-x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }

    #[test]
    fn the_chi_squared_constants() {
        // d = 6 in closed form.
        let x = CHI2_6_90;
        let cdf = 1.0 - (-x / 2.0).exp() * (1.0 + x / 2.0 + x * x / 8.0);
        assert!((cdf - 0.9).abs() < 1e-9, "{cdf}");
        let slope = x * x * x * (-x / 2.0).exp() / 16.0;
        assert!((slope - SLOPE_6).abs() < 1e-6, "{slope}");
        // d = 3 by Simpson's rule on the density, written in t = √x so that the integrand is
        // smooth at zero: ∫₀^x f₃ = ∫₀^√x 2t f₃(t²) dt.
        let x = CHI2_3_90;
        let n = 2000;
        let h = x.sqrt() / n as f64;
        let g = |t: f64| 2.0 * t * density_3(t * t);
        let interior: f64 = (1..n)
            .map(|i| if i % 2 == 1 { 4.0 } else { 2.0 } * g(i as f64 * h))
            .sum();
        let cdf = h / 3.0 * (g(0.0) + interior + g(x.sqrt()));
        assert!((cdf - 0.9).abs() < 1e-9, "{cdf}");
        let slope = x * density_3(x);
        assert!((slope - SLOPE_3).abs() < 1e-6, "{slope}");
    }
}
