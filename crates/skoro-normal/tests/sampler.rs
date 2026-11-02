#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The normal's sampler: the quantile it draws through.

use skoro_normal::quantile;

#[test]
fn quantile_matches_reference() {
    // Φ⁻¹ by root-finding at 50 digits, independently of AS 241.
    let reference = [
        (0.5, 0.0),
        (0.3, -0.524_400_512_708_040_8),
        (0.975, 1.959_963_984_540_054_2),
        (0.025, -1.959_963_984_540_054_2),
        (0.999, 3.090_232_306_167_813_5),
        (1e-10, -6.361_340_902_404_056),
        (1e-300, -37.047_096_299_361_2),
    ];
    for (p, x) in reference {
        let error = (quantile(p) - x).abs() / x.abs().max(1.0);
        assert!(error < 1e-15, "Φ⁻¹({p}) = {} against {x}", quantile(p));
    }
}
