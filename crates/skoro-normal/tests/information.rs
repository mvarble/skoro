use skoro_backend::{NdArrayBackend, Tensor, TensorData};
use skoro_core::error::SkoroError;
use skoro_dense::{dense::Dense, factor::spd_solve};
use skoro_linalg::{matrix::MatmulOps, tril::Tril};
use skoro_model::{
    ModelBuilderOps, ModelClientOps,
    hash_map_builders::points,
    information::{information, information_root},
    jacobian::jacobian_vjp,
    model::ModelClient,
    test_utils::into_vec,
};
use skoro_normal::distribution::Normal;
use skoro_normal::information::gaussian_root_matrix;
use skoro_pointwise::PointwiseOps;
use skoro_riemann::{
    euclidean::{EuclideanD, VectorD},
    matrix::{EuclideanMatrixD, MatrixD},
};
use skoro_se3::quantity::SE3;

type B = NdArrayBackend<f64, i32>;

fn row(values: &[f64]) -> Tensor<B, 2> {
    let n = values.len();
    Tensor::<B, 1>::from_data(TensorData::new(values.to_vec(), [n]), &Default::default())
        .reshape([1, n])
}

/// Unpack a row-major-packed `Tril(d)` vector into a dense `d x d` `Dense`.
fn unpack_tril(packed: &[f64], d: usize) -> Dense {
    let mut m = vec![0.0; d * d];
    let mut k = 0;
    for i in 0..d {
        for j in 0..=i {
            m[i * d + j] = packed[k];
            k += 1;
        }
    }
    Dense::from_row_major(d, d, m).unwrap()
}

/// `Σ⁻¹` for the covariance packed as `l_packed`, via the host `spd_solve` route §4.2 names.
fn precision_from_tril(l_packed: &[f64], d: usize) -> Dense {
    let l = unpack_tril(l_packed, d);
    let sigma = l.matmul(&l.transpose()).unwrap();
    spd_solve(&sigma, &Dense::identity(d)).unwrap()
}

fn assert_dense_close(got: &Dense, want: &Dense, epsilon: f64) {
    assert_eq!((got.rows(), got.cols()), (want.rows(), want.cols()));
    for (g, w) in got.data().iter().zip(want.data()) {
        approx::assert_abs_diff_eq!(g, w, epsilon = epsilon);
    }
}

/// Elementwise sum; `Dense` has no addition of its own.
fn add_dense(a: &Dense, b: &Dense) -> Dense {
    let data: Vec<f64> = a.data().iter().zip(b.data()).map(|(x, y)| x + y).collect();
    Dense::from_row_major(a.rows(), a.cols(), data).unwrap()
}

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

#[test]
fn observed_gaussian_information_is_h_transpose_precision_h() {
    let model = ModelClient::<B>::default();
    let theta = model.add_parameter(EuclideanD(2));
    let a = model.add_constant(
        EuclideanMatrixD(3, 2),
        &MatrixD::new(3, 2, &[0.5, 0.2, 0.1, 0.6, 0.3, 0.4]),
    );
    let c = model.add_constant(EuclideanD(3), &VectorD::new(&[1.0, 1.0, 1.0]));
    // A*theta_val + c > 0 for the theta this test evaluates at.
    let linear = a.matmul(&theta).unwrap();
    let mu = (linear + c).unwrap().log().unwrap();

    let l_packed = [1.2, 0.1, 0.9, -0.2, 0.3, 1.1];
    let l = model.add_constant(Tril(3), &VectorD::new(&l_packed));
    let y = model.add_quantity(Normal::new(mu.clone(), l)).unwrap();
    y.observe(&VectorD::new(&[0.9, 1.0, 1.1])).unwrap();

    let theta_val = points().with_quantity(&theta, row(&[0.3, -0.2])).build();
    let free = model.tangent_layout(&[theta.id()]).unwrap();

    let info = information(&model, &theta_val, &free).unwrap();
    let outputs = model.tangent_layout(&[mu.id()]).unwrap();
    let h = Dense::from_tensor(jacobian_vjp(&model, &theta_val, &free, &outputs).unwrap()).unwrap();
    let sigma_inv = precision_from_tril(&l_packed, 3);
    let expected = h
        .transpose()
        .matmul(&sigma_inv)
        .unwrap()
        .matmul(&h)
        .unwrap();

    assert_dense_close(&Dense::from_tensor(info).unwrap(), &expected, 1e-9);

    let root = information_root(&model, &theta_val, &free).unwrap();
    assert_eq!(root.dims(), [1, 3 + 9, 2]);
}

#[test]
fn gaussian_prior_information_is_the_precision() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.5, -0.5]));
    let l_packed = [1.4, 0.3, 0.8];
    let l = model.add_constant(Tril(2), &VectorD::new(&l_packed));
    let x = model.add_quantity(Normal::new(mu, l)).unwrap();

    let values = points().with_quantity(&x, row(&[0.2, 0.1])).build();
    let free = model.tangent_layout(&[x.id()]).unwrap();
    let info = information(&model, &values, &free).unwrap();

    let expected = precision_from_tril(&l_packed, 2);
    assert_dense_close(&Dense::from_tensor(info).unwrap(), &expected, 1e-9);
}

#[test]
fn covariance_information_matches_the_trace_formula() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
    let l = model.add_parameter(Tril(2));
    let l_val = [1.5, -0.4, 0.8];
    let x = model.add_quantity(Normal::new(mu, l.clone())).unwrap();
    x.observe(&VectorD::new(&[0.3, -0.6])).unwrap();

    let values = points().with_quantity(&l, row(&l_val)).build();
    let free = model.tangent_layout(&[l.id()]).unwrap();
    let info = information(&model, &values, &free).unwrap();

    // Host F_LL per spec section 4.2: dSigma_q = E_q L^T + L E_q^T,
    // F_LL[q][q'] = 1/2 trace(Sigma^-1 dSigma_q Sigma^-1 dSigma_q').
    let d = 2;
    let l_dense = unpack_tril(&l_val, d);
    let sigma = l_dense.matmul(&l_dense.transpose()).unwrap();
    let sigma_inv = spd_solve(&sigma, &Dense::identity(d)).unwrap();
    let d_l = d * (d + 1) / 2;
    let mut expected = vec![0.0; d_l * d_l];
    let packed = |i: usize, j: usize| i * (i + 1) / 2 + j;
    let e = |a: usize, b: usize| {
        let mut m = vec![0.0; d * d];
        m[a * d + b] = 1.0;
        Dense::from_row_major(d, d, m).unwrap()
    };
    for a in 0..d {
        for b in 0..=a {
            for a2 in 0..d {
                for b2 in 0..=a2 {
                    let dsq = e(a, b).matmul(&l_dense.transpose()).unwrap();
                    let dsq_t = l_dense.matmul(&e(a, b).transpose()).unwrap();
                    let dsq = add_dense(&dsq, &dsq_t);
                    let dsq2 = {
                        let x = e(a2, b2).matmul(&l_dense.transpose()).unwrap();
                        let y = l_dense.matmul(&e(a2, b2).transpose()).unwrap();
                        add_dense(&x, &y)
                    };
                    let trace = sigma_inv
                        .matmul(&dsq)
                        .unwrap()
                        .matmul(&sigma_inv)
                        .unwrap()
                        .matmul(&dsq2)
                        .unwrap();
                    let t: f64 = (0..d).map(|i| trace.get(i, i)).sum();
                    expected[packed(a, b) * d_l + packed(a2, b2)] = 0.5 * t;
                }
            }
        }
    }
    let expected = Dense::from_row_major(d_l, d_l, expected).unwrap();
    assert_dense_close(&Dense::from_tensor(info).unwrap(), &expected, 1e-9);
}

#[test]
fn covariance_information_matches_the_score_covariance() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
    let l = model.add_parameter(Tril(2));
    let l_val = [1.5, -0.4, 0.8];
    let x = model.add_quantity(Normal::new(mu, l.clone())).unwrap();

    // Draw x ~ N(mu, L Lᵀ) n times at the fixed L, one row per draw. A Parameter must be
    // supplied at the running batch, so L is repeated across the rows.
    let n = 40_000;
    let l_batched = Tensor::<B, 2>::from_data(
        TensorData::new(l_val.repeat(n), [n, 3]),
        &Default::default(),
    );
    let generated = model
        .generate_from(&points().with_quantity(&l, l_batched.clone()).build(), n)
        .unwrap();
    let (x_sampled, _) = generated.variables.get(&x.id()).expect("x is sampled");
    let x_sampled = x_sampled.clone().unwrap_float();

    // The score with respect to L at every draw: [n, 3].
    let values = points()
        .with_quantity(&x, x_sampled)
        .with_quantity(&l, l_batched)
        .build();
    let ord1 = model.log_density_ord1(&values).unwrap();
    let scores = into_vec(ord1.cotangents[&l.id()].clone());

    // mean_i(g_i g_iᵀ) over the batch, as a 3x3 host matrix.
    let mut score_cov = vec![0.0; 9];
    for i in 0..n {
        for p in 0..3 {
            for q in 0..3 {
                score_cov[p * 3 + q] += scores[i * 3 + p] * scores[i * 3 + q] / n as f64;
            }
        }
    }

    // The expected information at batch 1. `x` is an unobserved leaf, so it needs a value
    // even though the root does not depend on it.
    let free = model.tangent_layout(&[l.id()]).unwrap();
    let at = points()
        .with_quantity(&x, row(&[0.0, 0.0]))
        .with_quantity(&l, row(&l_val))
        .build();
    let info = Dense::from_tensor(information(&model, &at, &free).unwrap()).unwrap();

    for (g, w) in info.data().iter().zip(&score_cov) {
        approx::assert_abs_diff_eq!(g, w, epsilon = 0.05 * g.abs().max(1.0));
    }
}

#[test]
fn information_is_the_gram_of_the_root() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.5, -0.5]));
    let l = model.add_constant(Tril(2), &VectorD::new(&[1.4, 0.3, 0.8]));
    let x = model.add_quantity(Normal::new(mu, l)).unwrap();

    let values = points().with_quantity(&x, row(&[0.2, 0.1])).build();
    let free = model.tangent_layout(&[x.id()]).unwrap();

    let info = information(&model, &values, &free).unwrap();
    let root = information_root(&model, &values, &free).unwrap();
    let gram = root.clone().swap_dims(1, 2).matmul(root);

    for (g, w) in into_vec(info).iter().zip(into_vec(gram)) {
        approx::assert_abs_diff_eq!(g, &w, epsilon = 1e-12);
    }
}

#[test]
fn se3_root_is_sized_by_dimension() {
    let model = ModelClient::<B>::default();
    let mu = model.add_parameter(SE3);
    let l = model.add_parameter(Tril(6));
    let x = model
        .add_quantity(Normal::new(mu.clone(), l.clone()))
        .unwrap();

    let density = model.log_density_of(&x).unwrap();
    let root_id = model.bound(density, skoro_model::roles::FISHER_ROOT)[0];
    let shape = model.server_call(|s| {
        let qty = s.graph().node(root_id).unwrap().quantity_type();
        let m = qty
            .as_any()
            .downcast_ref::<EuclideanMatrixD>()
            .expect("the root is a matrix quantity");
        (m.0, m.1)
    });
    assert_eq!(
        shape,
        (42, 33),
        "6 + 36 rows over 6 + 6 + 21 argument columns"
    );

    let args = model.bound(density, skoro_model::roles::ARGUMENT);
    let layout = model.tangent_layout(&args).unwrap();
    assert_eq!(layout.total(), 6 + 6 + 21);
}

#[test]
fn a_constant_covariance_folds_the_root() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
    let l_const = model.add_constant(Tril(2), &VectorD::new(&[1.0, 0.0, 1.0]));
    let x = model
        .add_quantity(Normal::new(mu.clone(), l_const))
        .unwrap();
    let density = ModelClientOps::log_density_of(&model, &x).unwrap();
    let root_id = ModelClientOps::bound(&model, density, skoro_model::roles::FISHER_ROOT)[0];
    assert!(model.server_call(|s| matches!(
        s.graph().node(root_id),
        Some(skoro_graph::node::Node::Mapped(
            skoro_graph::node::MappedNode { value: Some(_), .. }
        ))
    )));

    let mu2 = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
    let l_param = model.add_parameter(Tril(2));
    let x2 = model.add_quantity(Normal::new(mu2, l_param)).unwrap();
    let density2 = ModelClientOps::log_density_of(&model, &x2).unwrap();
    let root_id2 = ModelClientOps::bound(&model, density2, skoro_model::roles::FISHER_ROOT)[0];
    assert!(model.server_call(|s| matches!(
        s.graph().node(root_id2),
        Some(skoro_graph::node::Node::Mapped(
            skoro_graph::node::MappedNode { value: None, .. }
        ))
    )));
}

#[test]
fn a_mismatched_covariance_is_refused() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(3), &VectorD::new(&[0.0, 0.0, 0.0]));
    let l = model.add_constant(Tril(2), &VectorD::new(&[1.0, 0.0, 1.0]));
    assert!(matches!(
        model.add_quantity(Normal::new(mu, l)),
        Err(SkoroError::ShapeMismatch {
            expected: 3,
            found: 2
        })
    ));
}

/// A proposal-only term is not a summand of the model's log-density, so it contributes no
/// expected information — even though a Gaussian proposal binds `FISHER_ROOT` and `RESIDUAL`
/// exactly as a scored Gaussian does. Without a density filter in `information_blocks` the
/// proposal's root would be stacked in and every information matrix in the model would be wrong.
#[test]
fn a_proposal_contributes_no_information() {
    let build = |propose: bool| {
        let model = ModelClient::<B>::default();
        let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.5, -0.5]));
        let l_packed = [1.4, 0.3, 0.8];
        let l = model.add_constant(Tril(2), &VectorD::new(&l_packed));
        let x = model.add_quantity(Normal::new(mu, l)).unwrap();

        let mut values = points().with_quantity(&x, row(&[0.2, 0.1]));
        // An unrelated quantity the posterior does not score, generated from its own Gaussian.
        let proposed = if propose {
            let nu = model.add_constant(EuclideanD(2), &VectorD::new(&[1.0, 2.0]));
            let m = model.add_constant(Tril(2), &VectorD::new(&[0.9, -0.2, 1.1]));
            let z = model.add_proposal(Normal::new(nu, m)).unwrap();
            values = values.with_quantity(&z, row(&[0.7, -0.3]));
            Some(z)
        } else {
            None
        };

        let free = model.tangent_layout(&[x.id()]).unwrap();
        let values = values.build();
        let info = Dense::from_tensor(information(&model, &values, &free).unwrap()).unwrap();
        let root = Dense::from_tensor(information_root(&model, &values, &free).unwrap()).unwrap();
        drop(proposed);
        (info, root)
    };

    let (info, root) = build(false);
    let (info_with_proposal, root_with_proposal) = build(true);

    assert_dense_close(&info_with_proposal, &info, 1e-12);
    assert_eq!(
        (root_with_proposal.rows(), root_with_proposal.cols()),
        (root.rows(), root.cols()),
        "the proposal must contribute no rows to the information root"
    );
    assert_dense_close(&root_with_proposal, &root, 1e-12);
}
