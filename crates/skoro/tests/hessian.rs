//! The dense Hessian of the log-density, against an exact answer and against differences of the
//! gradient.
//!
//! The Hessian is defined as `d/dt g(x ⊕ t eᵢ)`, `g` the gradient in the chart at each point, so
//! the reference is the same thing taken numerically: central differences of `log_density_ord1`
//! at points retracted through the layout. That holds on curved manifolds as on flat ones, which
//! is why `SE3` is here.

use skoro_backend::{Flex, Tensor, TensorData};
use skoro_core::id::Id;
use skoro_dense::{dense::Dense, factor::spd_solve};
use skoro_graph::point_map::PointMap;
use skoro_linalg::tril::Tril;
use skoro_model::{
    ModelBuilderOps, ModelReaderOps, TangentLayout, hash_map_builders::point_map,
    hessian::hessian_hvp, model::ModelClient,
};
use skoro_normal::{distribution::Normal, facade::constant_covariance_diagonal};
use skoro_riemann::euclidean::{EuclideanD, VectorD};
use skoro_se3::quantity::{SE3, SE3Pose};
use skoro_so3::quantity::SO3Matrix;

type B = Flex;
skoro_backend::f64_test_binary!(Flex);

fn row(values: &[f64]) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(values.to_vec(), [1, values.len()]),
        &Default::default(),
    )
}

fn host(t: Tensor<B, 3>) -> Dense {
    Dense::from_tensor(t).unwrap()
}

/// `∇ log p` over `free`, flattened: `[1, D]`.
fn gradient(model: &ModelClient<B>, at: &PointMap<B>, free: &TangentLayout) -> Vec<f64> {
    let g = model.log_density_ord1(at.as_ref()).unwrap().cotangents;
    free.flatten_vectors(&g, 1, &Default::default())
        .unwrap()
        .into_data()
        .to_vec::<f64>()
        .unwrap()
}

/// Column `i` is `(g(x ⊕ h eᵢ) − g(x ⊕ −h eᵢ)) / 2h`.
fn differenced(model: &ModelClient<B>, at: &PointMap<B>, free: &TangentLayout, h: f64) -> Dense {
    let d = free.total();
    let mut out = Dense::zeros(d, d);
    for i in 0..d {
        let step = |s: f64| {
            let delta: Vec<f64> = (0..d).map(|j| if i == j { s } else { 0.0 }).collect();
            gradient(model, &at.retract_flat(free, row(&delta)).unwrap(), free)
        };
        let (plus, minus) = (step(h), step(-h));
        for r in 0..d {
            out.set(r, i, (plus[r] - minus[r]) / (2.0 * h));
        }
    }
    out
}

#[test]
fn a_gaussian_in_its_value_and_mean_has_the_precision_as_its_hessian() {
    let model = ModelClient::<B>::default();
    let mu = model.add_parameter(EuclideanD(2));
    let l = model.add_constant(Tril(2), &VectorD::new(&[1.5, -0.4, 0.8]));
    let x = model.add_quantity(Normal::new(mu.clone(), l)).unwrap();
    let at = point_map()
        .with_quantity(&x, row(&[0.3, -1.2]))
        .with_quantity(&mu, row(&[1.0, 0.5]))
        .build();
    let free = model.tangent_layout(&[x.id(), mu.id()]).unwrap();

    let h = host(hessian_hvp(&model, at.as_ref(), &free).unwrap());

    // log p = −½ (x − μ)ᵀ Λ (x − μ) + const, Λ = (L Lᵀ)⁻¹, so H = −[[Λ, −Λ], [−Λ, Λ]].
    let l = Dense::from_row_major(2, 2, vec![1.5, 0.0, -0.4, 0.8]).unwrap();
    let lambda = spd_solve(&l.matmul(&l.transpose()).unwrap(), &Dense::identity(2)).unwrap();
    let mut want = Dense::zeros(4, 4);
    want.set_block(0, 0, &lambda.neg());
    want.set_block(0, 2, &lambda);
    want.set_block(2, 0, &lambda);
    want.set_block(2, 2, &lambda.neg());
    assert!(h.max_abs_diff(&want).unwrap() < 1e-10, "{h:?}");
}

/// A covariance factor that is itself free: the curvature of `log|det W|` and of the whitened
/// residual together, through `invert`, `diag`, `abs` and `log`.
#[test]
fn a_free_covariance_factor_matches_differences_of_the_gradient() {
    let model = ModelClient::<B>::default();
    let mu = model.add_constant(EuclideanD(2), &VectorD::new(&[0.2, -0.1]));
    let l = model.add_parameter(Tril(2));
    for y in [[0.3, -0.6], [1.1, 0.2], [-0.4, 0.9]] {
        let x = model
            .add_quantity(Normal::new(mu.clone(), l.clone()))
            .unwrap();
        model.observe(&x, &VectorD::new(&y)).unwrap();
    }
    let at = point_map()
        .with_quantity(&l, row(&[1.5, -0.4, 0.8]))
        .build();
    let free = model.tangent_layout(&[l.id()]).unwrap();

    let h = host(hessian_hvp(&model, at.as_ref(), &free).unwrap());
    let want = differenced(&model, &at, &free, 1e-5);
    assert!(h.max_abs_diff(&want).unwrap() < 1e-6, "{h:?}\n{want:?}");
}

fn pose(theta: f64, t: [f64; 3]) -> SE3Pose<f64> {
    let (c, s) = (theta.cos(), theta.sin());
    SE3Pose::new(SO3Matrix([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]), t)
}

/// On `SE3` the matrix is the derivative of the chart gradient, taken away from any stationary
/// point, which is exactly what the differences compute.
#[test]
fn a_pose_matches_differences_of_the_gradient_along_retractions() {
    let model = ModelClient::<B>::default();
    let prior = model.add_constant(SE3, &pose(0.3, [1.0, -2.0, 0.5]));
    let p_cov = constant_covariance_diagonal(&model, &[0.5, 0.4, 0.3, 1.0, 1.5, 0.8]);
    let r_cov = constant_covariance_diagonal(&model, &[0.3, 0.2, 0.4, 0.5, 0.9, 0.6]);
    let p = model.add_quantity(Normal::new(prior, p_cov)).unwrap();
    let z = model.add_quantity(Normal::new(p.clone(), r_cov)).unwrap();
    model.observe(&z, &pose(-0.8, [3.0, 1.0, -1.0])).unwrap();

    let (c, s) = (1.1f64.cos(), 1.1f64.sin());
    let at = point_map()
        .with_quantity(
            &p,
            row(&[c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0, 2.0, 0.5, 1.5]),
        )
        .build();
    let ids: Vec<Id> = vec![p.id()];
    let free = model.tangent_layout(&ids).unwrap();

    let h = host(hessian_hvp(&model, at.as_ref(), &free).unwrap());
    assert_eq!(
        (h.rows(), h.cols()),
        (6, 6),
        "sized by dimension, not storage"
    );
    let want = differenced(&model, &at, &free, 1e-5);
    assert!(h.max_abs_diff(&want).unwrap() < 1e-5, "{h:?}\n{want:?}");
}
