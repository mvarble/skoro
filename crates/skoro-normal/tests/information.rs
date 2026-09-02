use skoro_normal::information::gaussian_root_matrix;

/// `d = 2`: `c_rows = d + d² = 6`, `c_cols = 2d + d(d+1)/2 = 7`, so `rows = 42`, `cols = 3` —
/// this shape alone catches an off-by-one in `c_rows`/`c_cols` before any graph is involved.
///
/// For `W = [[1, 0], [0.5, 1]]` (packed `[w00, w10, w11] = [1, 0.5, 1]`), hand-derive a few
/// representative entries of `C` (via `vec(C) = K · vec(W)`) rather than the whole 42-entry
/// matrix — enough to catch a sign or transpose error in each of the two blocks §4.1 describes.
#[test]
fn gaussian_root_matrix_matches_the_hand_derivation_at_d_2() {
    let (rows, cols, k) = gaussian_root_matrix(2);
    assert_eq!((rows, cols), (42, 3));

    let w = [1.0, 0.5, 1.0]; // [w00, w10, w11], packed row-major lower-triangular
    let vec_c: Vec<f64> = (0..rows)
        .map(|r| (0..cols).map(|c| k[r * cols + c] * w[c]).sum())
        .collect();
    let c_cols = 7;
    let at = |row: usize, col: usize| vec_c[row * c_cols + col];

    // (x, mu) block: C = [W, -W, 0], rows 0..2.
    approx::assert_abs_diff_eq!(at(0, 0), 1.0, epsilon = 1e-12); // W[0][0]
    approx::assert_abs_diff_eq!(at(0, 2), -1.0, epsilon = 1e-12); // -W[0][0]
    approx::assert_abs_diff_eq!(at(1, 0), 0.5, epsilon = 1e-12); // W[1][0]
    approx::assert_abs_diff_eq!(at(1, 1), 1.0, epsilon = 1e-12); // W[1][1]
    approx::assert_abs_diff_eq!(at(1, 3), -1.0, epsilon = 1e-12); // -W[1][1]
    approx::assert_abs_diff_eq!(at(0, 4), 0.0, epsilon = 1e-12); // zero block over L

    // L block: rows 2..6 index (r, c) row-major; cols 4..7 index q = packed(a, b).
    let s = std::f64::consts::FRAC_1_SQRT_2;
    // row (r=0, c=0) -> C row 2; S_0[0][0] = sqrt(2) * w00.
    approx::assert_abs_diff_eq!(at(2, 4), 2.0f64.sqrt() * 1.0, epsilon = 1e-12);
    // row (r=1, c=1) -> C row 5; S_2[1][1] = sqrt(2) * w11.
    approx::assert_abs_diff_eq!(at(5, 6), 2.0f64.sqrt() * 1.0, epsilon = 1e-12);
    // row (r=0, c=1) -> C row 3; S_0[0][1] = w10 / sqrt(2), S_1[0][1] = w11 / sqrt(2).
    approx::assert_abs_diff_eq!(at(3, 4), s * 0.5, epsilon = 1e-12);
    approx::assert_abs_diff_eq!(at(3, 5), s * 1.0, epsilon = 1e-12);
    // Symmetry the construction claims: row (r=1, c=0) -> C row 4 should equal row (0,1)'s C_L
    // entries, since S_q is symmetric by construction.
    assert_eq!(at(4, 4), at(3, 4));
    assert_eq!(at(4, 5), at(3, 5));
}
