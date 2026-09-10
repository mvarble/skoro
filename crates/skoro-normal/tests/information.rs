//! The Gaussian Fisher root, against host references.
//!
//! One backend for the binary (`NdArrayBackend<f64, i32>`), with tensors built through
//! `TensorData` so the tight tolerances below are not undone by an `f32` round trip.

use skoro_normal::information::gaussian_root_matrix;

/// `vec(C) = K · vec(W)` for `d = 2` and `W = [[1, 0], [0.5, 1]]`, against `C` worked by hand.
///
/// With `G_q = W E_q`, the `L` columns are `(G_q + G_qᵀ)/√2` for `q = (0,0), (1,0), (1,1)`:
/// `[[2, 0.5], [0.5, 0]]`, `[[0, 1], [1, 0]]` and `[[0, 0], [0, 2]]`, each over `√2`.
#[test]
fn gaussian_root_matrix_matches_the_hand_derivation_at_d_2() {
    let (rows, cols, k) = gaussian_root_matrix(2);
    // C is (d + d²) × (2d + d_L) = 6 × 7.
    assert_eq!((rows, cols), (42, 3));

    let w = [1.0, 0.5, 1.0]; // packed [w00, w10, w11]
    let c: Vec<f64> = (0..rows)
        .map(|r| (0..cols).map(|q| k[r * cols + q] * w[q]).sum())
        .collect();

    let s = std::f64::consts::FRAC_1_SQRT_2;
    #[rustfmt::skip]
    let expected: [[f64; 7]; 6] = [
        // x         μ           L(0,0)   L(1,0)  L(1,1)
        [1.0, 0.0,  -1.0,  0.0,  0.0,     0.0,    0.0    ],
        [0.5, 1.0,  -0.5, -1.0,  0.0,     0.0,    0.0    ],
        [0.0, 0.0,   0.0,  0.0,  2.0 * s, 0.0,    0.0    ], // (r, c) = (0, 0)
        [0.0, 0.0,   0.0,  0.0,  0.5 * s, s,      0.0    ], // (0, 1)
        [0.0, 0.0,   0.0,  0.0,  0.5 * s, s,      0.0    ], // (1, 0)
        [0.0, 0.0,   0.0,  0.0,  0.0,     0.0,    2.0 * s], // (1, 1)
    ];
    for (row, entries) in expected.iter().enumerate() {
        for (col, want) in entries.iter().enumerate() {
            let got = c[row * 7 + col];
            assert!(
                (got - want).abs() < 1e-12,
                "C[{row}][{col}] = {got}, expected {want}"
            );
        }
    }
}
