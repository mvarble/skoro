//! The Gaussian Fisher root, against host references.
//!
//! One backend for the binary (`Flex` under the `f64` float policy), with tensors built through
//! `TensorData` so the tight tolerances below are not undone by an `f32` round trip. The host
//! references are small dense row-major matrices, at most 3 × 3, handled by the helpers below.

use skoro_backend::{Flex, Tensor, TensorData};
use skoro_core::{error::SkoroError, kind::Continuous};
use skoro_graph::node::{MappedNode, Node};
use skoro_linalg::{
    matrix::{MatmulOps, matmul},
    tril::Tril,
};
use skoro_model::{
    ModelBuilderOps, ModelReaderOps,
    coverage::{require_role, terms_missing},
    hash_map_builders::points,
    information::{information, information_root},
    jacobian::jacobian_vjp,
    model::ModelClient,
    roles::{ARGUMENT, FISHER_ROOT},
};
use skoro_normal::{
    distribution::Normal,
    facade::constant_covariance_identity,
    information::{gaussian_root_map, gaussian_root_matrix},
};
use skoro_pointwise::PointwiseOps;
use skoro_riemann::{
    euclidean::{EuclideanD, VectorD},
    matrix::{EuclideanMatrixD, MatrixD},
};
use skoro_se3::quantity::SE3;

type B = Flex;
skoro_backend::f64_test_binary!(Flex);

/// `L0 = [[1.5, 0], [-0.4, 0.8]]`, packed.
const L0: [f64; 3] = [1.5, -0.4, 0.8];

fn row(values: &[f64]) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(values.to_vec(), [1, values.len()]),
        &Default::default(),
    )
}

fn into_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f64> {
    t.into_data().to_vec::<f64>().unwrap()
}

fn assert_close(got: &[f64], want: &[f64], epsilon: f64) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < epsilon, "entry {i}: {g} vs {w}");
    }
}

// ---- host references ----

fn packed(i: usize, j: usize) -> usize {
    i * (i + 1) / 2 + j
}

/// The dense `d × d` lower triangle a packed `Tril(d)` value stands for.
fn unpack_tril(values: &[f64], d: usize) -> Vec<f64> {
    let mut dense = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            dense[i * d + j] = values[packed(i, j)];
        }
    }
    dense
}

/// `a · b` for `a: rows × inner` and `b: inner × cols`.
fn mat_mul(a: &[f64], b: &[f64], rows: usize, inner: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = (0..inner).map(|k| a[r * inner + k] * b[k * cols + c]).sum();
        }
    }
    out
}

fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Gauss–Jordan inverse with partial pivoting.
fn inverse(a: &[f64], d: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; d * d];
    for i in 0..d {
        inv[i * d + i] = 1.0;
    }
    for col in 0..d {
        let pivot = (col..d)
            .max_by(|&x, &y| m[x * d + col].abs().total_cmp(&m[y * d + col].abs()))
            .unwrap();
        for k in 0..d {
            m.swap(col * d + k, pivot * d + k);
            inv.swap(col * d + k, pivot * d + k);
        }
        let scale = m[col * d + col];
        for k in 0..d {
            m[col * d + k] /= scale;
            inv[col * d + k] /= scale;
        }
        for r in (0..d).filter(|&r| r != col) {
            let factor = m[r * d + col];
            for k in 0..d {
                m[r * d + k] -= factor * m[col * d + k];
                inv[r * d + k] -= factor * inv[col * d + k];
            }
        }
    }
    inv
}

/// `Σ⁻¹` for `Σ = L Lᵀ`.
fn precision(l_packed: &[f64], d: usize) -> Vec<f64> {
    let l = unpack_tril(l_packed, d);
    inverse(&mat_mul(&l, &transpose(&l, d, d), d, d, d), d)
}

/// `F_LL[q][q'] = ½ tr(Σ⁻¹ dΣ_q Σ⁻¹ dΣ_q')` with `dΣ_q = E_q Lᵀ + L E_qᵀ`, over the packed
/// coordinates of `L`.
fn covariance_information(l_packed: &[f64], d: usize) -> Vec<f64> {
    let l = unpack_tril(l_packed, d);
    let lt = transpose(&l, d, d);
    let sigma_inv = precision(l_packed, d);
    let d_l = d * (d + 1) / 2;
    let mut d_sigma = vec![Vec::new(); d_l];
    for a in 0..d {
        for b in 0..=a {
            let mut e = vec![0.0; d * d];
            e[a * d + b] = 1.0;
            d_sigma[packed(a, b)] = add(
                &mat_mul(&e, &lt, d, d, d),
                &mat_mul(&l, &transpose(&e, d, d), d, d, d),
            );
        }
    }
    let mut f = vec![0.0; d_l * d_l];
    for q in 0..d_l {
        for q2 in 0..d_l {
            let left = mat_mul(&sigma_inv, &d_sigma[q], d, d, d);
            let right = mat_mul(&sigma_inv, &d_sigma[q2], d, d, d);
            let product = mat_mul(&left, &right, d, d, d);
            f[q * d_l + q2] = 0.5 * (0..d).map(|i| product[i * d + i]).sum::<f64>();
        }
    }
    f
}

// ---- tests ----

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
    let expected: [f64; 42] = [
        // x         μ           L(0,0)   L(1,0)  L(1,1)
        1.0, 0.0,  -1.0,  0.0,  0.0,     0.0,    0.0,
        0.5, 1.0,  -0.5, -1.0,  0.0,     0.0,    0.0,
        0.0, 0.0,   0.0,  0.0,  2.0 * s, 0.0,    0.0,     // (r, c) = (0, 0)
        0.0, 0.0,   0.0,  0.0,  0.5 * s, s,      0.0,     // (0, 1)
        0.0, 0.0,   0.0,  0.0,  0.5 * s, s,      0.0,     // (1, 0)
        0.0, 0.0,   0.0,  0.0,  0.0,     0.0,    2.0 * s, // (1, 1)
    ];
    assert_close(&c, &expected, 1e-12);
}

/// An observed Gaussian with a nonlinear mean `μ = log(Aθ + c)`: its information over `θ` is
/// `Hᵀ Σ⁻¹ H`, with `H = ∂μ/∂θ`.
#[test]
fn observed_gaussian_information_is_h_transpose_precision_h() {
    let model = ModelClient::<B>::default();
    let theta = model.add_parameter(EuclideanD(2));
    let a = model.add_constant(
        EuclideanMatrixD(3, 2),
        &MatrixD::new(3, 2, &[0.5, 0.2, 0.1, 0.6, 0.3, 0.4]),
    );
    let c = model.add_constant(EuclideanD(3), &VectorD::new(&[1.0, 1.0, 1.0]));
    // Aθ + c > 0 at the θ below, so the logarithm is defined.
    let mu = (a.matmul(&theta).unwrap() + c).unwrap().log().unwrap();

    let l_packed = [1.2, 0.1, 0.9, -0.2, 0.3, 1.1];
    let l = model.add_constant(Tril(3), &VectorD::new(&l_packed));
    let y = model.add_quantity(Normal::new(mu.clone(), l)).unwrap();
    model.observe(&y, &VectorD::new(&[0.9, 1.0, 1.1])).unwrap();

    let values = points().with_quantity(&theta, row(&[0.3, -0.2])).build();
    let free = model.tangent_layout(&[theta.id()]).unwrap();

    let root = information_root(&model, &values, &free).unwrap();
    assert_eq!(root.dims(), [1, 3 + 9, 2]);

    let mean_rows = model.tangent_layout(&[mu.id()]).unwrap();
    let h = into_vec(jacobian_vjp(&model, &values, &free, &mean_rows).unwrap());
    let expected = mat_mul(
        &mat_mul(&transpose(&h, 3, 2), &precision(&l_packed, 3), 2, 3, 3),
        &h,
        2,
        3,
        2,
    );
    let info = information(&model, &values, &free).unwrap();
    assert_eq!(info.dims(), [1, 2, 2]);
    assert_close(&into_vec(info), &expected, 1e-9);
}

/// Over its own quantity, an unobserved Gaussian's information is its precision.
#[test]
fn gaussian_prior_information_is_the_precision() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.5, -0.5]));
    let l_packed = [1.4, 0.3, 0.8];
    let l = model.add_constant(Tril(2), &VectorD::new(&l_packed));
    let x = model.add_quantity(Normal::new(mu, l)).unwrap();

    let values = points().with_quantity(&x, row(&[0.2, 0.1])).build();
    let free = model.tangent_layout(&[x.id()]).unwrap();

    let info = into_vec(information(&model, &values, &free).unwrap());
    assert_close(&info, &precision(&l_packed, 2), 1e-9);
}

/// Over the covariance factor, the information is the trace formula of §4.2.
#[test]
fn covariance_information_matches_the_trace_formula() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
    let l = model.add_parameter(Tril(2));
    let x = model.add_quantity(Normal::new(mu, l.clone())).unwrap();
    model.observe(&x, &VectorD::new(&[0.3, -0.6])).unwrap();

    let values = points().with_quantity(&l, row(&L0)).build();
    let free = model.tangent_layout(&[l.id()]).unwrap();

    let info = information(&model, &values, &free).unwrap();
    assert_eq!(info.dims(), [1, 3, 3]);
    assert_close(&into_vec(info), &covariance_information(&L0, 2), 1e-9);
}

/// The second Bartlett identity: the covariance of the score over draws of `x` is the
/// information. This checks the root against the model's own gradients rather than a formula.
#[test]
fn covariance_information_matches_the_score_covariance() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
    let l = model.add_parameter(Tril(2));
    let x = model.add_quantity(Normal::new(mu, l.clone())).unwrap();

    <B as burn::tensor::backend::Backend>::seed(&Default::default(), 0x5eed);
    let n = 40_000;
    let l_batch =
        Tensor::<B, 2>::from_data(TensorData::new(L0.repeat(n), [n, 3]), &Default::default());
    let generated = model
        .generate_from(&points().with_quantity(&l, l_batch.clone()).build(), n)
        .unwrap();
    let (sampled, _) = generated.variables.get(&x.id()).expect("x is sampled");
    let sampled = sampled.unwrap_float_ref().clone();
    assert_eq!(sampled.dims(), [n, 2]);

    let values = points()
        .with_quantity(&x, sampled)
        .with_quantity(&l, l_batch)
        .build();
    let scores = into_vec(model.log_density_ord1(&values).unwrap().cotangents[&l.id()].clone());
    assert_eq!(scores.len(), n * 3);

    let mut score_covariance = [0.0; 9];
    for draw in scores.chunks_exact(3) {
        for p in 0..3 {
            for q in 0..3 {
                score_covariance[p * 3 + q] += draw[p] * draw[q] / n as f64;
            }
        }
    }

    let f = covariance_information(&L0, 2);
    for (i, (got, want)) in score_covariance.iter().zip(&f).enumerate() {
        let tolerance = 0.05 * want.abs().max(1.0);
        assert!(
            (got - want).abs() < tolerance,
            "entry {i}: score covariance {got} vs information {want}"
        );
    }
}

/// `information` is the Gram matrix of `information_root`, by construction.
#[test]
fn information_is_the_gram_of_the_root() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.5, -0.5]));
    let l = model.add_parameter(Tril(2));
    let x = model.add_quantity(Normal::new(mu, l.clone())).unwrap();

    let values = points()
        .with_quantity(&x, row(&[0.2, 0.1]))
        .with_quantity(&l, row(&L0))
        .build();
    let free = model.tangent_layout(&[x.id(), l.id()]).unwrap();

    let info = information(&model, &values, &free).unwrap();
    let root = information_root(&model, &values, &free).unwrap();
    let gram = root.clone().swap_dims(1, 2).matmul(root);
    assert_eq!(info.dims(), [1, 5, 5]);
    assert_close(&into_vec(info), &into_vec(gram), 1e-12);
}

/// On `SE3` the root is sized by manifold dimension: 6 rows for `x`, 36 for `L`, over the 6 + 6 +
/// 21 columns of `[x, μ, L]`.
#[test]
fn se3_root_is_sized_by_dimension() {
    let model = ModelClient::<B>::default();
    let mu = model.add_parameter(SE3);
    let l = model.add_parameter(Tril(6));
    let x = model.add_quantity(Normal::new(mu, l)).unwrap();

    let density = model.log_density_of(&x).unwrap();
    let roots = model.bound(density, FISHER_ROOT);
    assert_eq!(roots.len(), 1);
    let root_type = model
        .server_call(|server| {
            server
                .graph()
                .get_quantity_type::<Continuous, EuclideanMatrixD>(roots[0])
        })
        .unwrap();
    assert_eq!(root_type, EuclideanMatrixD(42, 33));

    let arguments = model.bound(density, ARGUMENT);
    assert_eq!(
        model.tangent_layout(&arguments).unwrap().total(),
        6 + 6 + 21
    );
}

/// A constant covariance makes the root a constant, folded when it is built; a free one does not.
#[test]
fn a_constant_covariance_folds_the_root() {
    let model = ModelClient::<B>::default();
    let root_of = |l: skoro_model::q::Q<B, Tril>| {
        let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
        let x = model.add_quantity(Normal::new(mu, l)).unwrap();
        let root = model.bound(model.log_density_of(&x).unwrap(), FISHER_ROOT)[0];
        model.server_call(|server| match server.graph().node(root) {
            Some(Node::Mapped(MappedNode { value, .. })) => value.is_some(),
            _ => panic!("the root should be a mapped node"),
        })
    };

    let constant = model.add_constant(Tril(2), &VectorD::new(&[1.0, 0.0, 1.0]));
    assert!(root_of(constant), "a constant covariance folds the root");
    let free = model.add_parameter(Tril(2));
    assert!(
        !root_of(free),
        "a free covariance leaves it to be evaluated"
    );
}

/// A covariance factor sized for a different dimension is refused when the quantity is added.
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

/// The graph's root is the sparse map applied to `W`: for d = 1…6 it equals the dense reference
/// `K · vec(W)` on every batch row.
#[test]
fn information_root_matches_the_dense_reference() {
    for d in 1..=6 {
        let d_l = d * (d + 1) / 2;
        let batch = 2;
        let model = ModelClient::<B>::default();
        let w = model.add_parameter(Tril(d));
        let root = skoro_normal::information::information_root(&w, d).unwrap();

        let packed_w: Vec<f64> = (0..batch * d_l)
            .map(|i| 0.5 + (i as f64 * 0.37).sin())
            .collect();
        let values = points()
            .with_quantity(
                &w,
                Tensor::<B, 2>::from_data(
                    TensorData::new(packed_w.clone(), [batch, d_l]),
                    &Default::default(),
                ),
            )
            .build();
        let got = into_vec(root.evaluate(&values).unwrap());

        let (rows, cols, k) = gaussian_root_matrix(d);
        assert_eq!(got.len(), batch * rows);
        for b in 0..batch {
            let wb = &packed_w[b * d_l..(b + 1) * d_l];
            let want: Vec<f64> = (0..rows)
                .map(|r| (0..cols).map(|c| k[r * cols + c] * wb[c]).sum())
                .collect();
            assert_close(&got[b * rows..(b + 1) * rows], &want, 1e-12);
        }
    }
}

/// The map behind the root stores only its nonzeros: at d = 15 it is a 36,000 × 120 matrix with
/// 1,585 of them, rather than 4.32 million dense entries.
#[test]
fn the_gaussian_root_map_stores_only_its_nonzeros() {
    let map = gaussian_root_map(15);
    assert_eq!(map.dims(), [36_000, 120]);
    assert_eq!(map.nnz(), 1_585);
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
        let info = information(&model, &values, &free).unwrap();
        let root = information_root(&model, &values, &free).unwrap();
        drop(proposed);
        (info.dims(), into_vec(info), root.dims(), into_vec(root))
    };

    let (info_dims, info, root_dims, root) = build(false);
    let (info_dims_with, info_with, root_dims_with, root_with) = build(true);

    assert_eq!(info_dims_with, info_dims);
    assert_close(&info_with, &info, 1e-12);
    assert_eq!(
        root_dims_with, root_dims,
        "the proposal must contribute no rows to the information root"
    );
    assert_close(&root_with, &root, 1e-12);
}

/// A hierarchical Gaussian built from repeated `add_quantity` needs no Fisher bookkeeping: every
/// `Normal` binds a root whatever kind of node its mean is, so the chain is fully covered.
///
/// The chain is the shape `window.rs` grows a tick at a time — a state, a mapped mean over it, a
/// second state, a measurement — which is the case an estimator actually meets. `require_role`
/// passing is the machine-checkable form of "the information covers the whole log-density", and
/// it is the precondition a Fisher step pairs with a gradient over the same terms.
#[test]
fn a_hierarchical_gaussian_has_no_gaps() {
    let model = ModelClient::<B>::default();
    let prior_mean = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
    let cov = constant_covariance_identity(&model, 2);
    let f_mat = model.add_constant(
        EuclideanMatrixD(2, 2),
        &MatrixD::new(2, 2, &[2.0, 0.0, 0.0, 2.0]),
    );

    let x0 = model
        .add_quantity(Normal::new(prior_mean, cov.clone()))
        .unwrap();
    let x1_mean = matmul(&f_mat, &x0).unwrap();
    let x1 = model
        .add_quantity(Normal::new(x1_mean, cov.clone()))
        .unwrap();
    let z_mean = matmul(&f_mat, &x1).unwrap();
    let z = model.add_quantity(Normal::new(z_mean, cov)).unwrap();
    model.observe(&z, &VectorD::new(&[0.1, -0.1])).unwrap();

    assert_eq!(model.factors().len(), 3);
    assert_eq!(require_role(&model, FISHER_ROOT), Ok(()));
    assert!(terms_missing(&model, FISHER_ROOT).is_empty());

    // And the assembled root really does carry a block per term: 3 + 9 rows each, over the
    // two free states' 4 tangent coordinates.
    let values = points()
        .with_quantity(&x0, row(&[0.0, 0.0]))
        .with_quantity(&x1, row(&[0.0, 0.0]))
        .build();
    let free = model.tangent_layout(&[x0.id(), x1.id()]).unwrap();
    assert_eq!(
        information_root(&model, &values, &free).unwrap().dims(),
        [1, 3 * (2 + 2 * 2), 4]
    );
}
