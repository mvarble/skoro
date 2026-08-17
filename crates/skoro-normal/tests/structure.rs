//! The Gaussian structure claims, checked against what the density actually computes.
//!
//! Two things are at stake. First, that `whitened_residual` is really the `r` in
//! `−log p = ½‖r‖² + n(θ)` — if it is not, a least-squares optimizer minimizes something
//! unrelated to the model, silently. Second, that the builders record honestly: the
//! certificate names the density and residual nodes that the graph actually holds.
//!
//! `SE3` is exercised deliberately. Its manifold dimension is 6 and its storage is 12, so a
//! residual sized by storage rather than by dimension is visible here and invisible on
//! `Real` or `EuclideanD`.

use skoro_backend::{NdArrayBackend, Point, TensorData};
use skoro_core::id::Id;
use skoro_linalg::tril::{Tril, TrilOps};
use skoro_linalg::vector::LinalgVectorOps;
use skoro_model::{
    ModelBuilderOps, ModelClientOps, NlsModelClient, NlsStructure, WhitenedResidual,
    hash_map_builders::points, model::ModelClient, q::Q,
};
use skoro_normal::{
    distribution::Normal,
    facade::{GaussianModelClient, GaussianModelClientOps, constant_covariance_diagonal},
    structure::{GaussianFactor, GaussianStructure},
};
use skoro_operation::{Input, ValueMap};
use skoro_pointwise::PointwiseOps;
use skoro_riemann::{
    euclidean::{EuclideanD, VectorD},
    real::Real,
};
use skoro_se3::quantity::{SE3, SE3Pose};
use skoro_so3::quantity::SO3Matrix;

type B = NdArrayBackend<f64, i32>;

fn point(values: &[f64]) -> Input<B> {
    Input::float(Point::<B>::from_data(
        TensorData::new(values.to_vec(), [1, values.len()]),
        &Default::default(),
    ))
}

fn scalar(t: skoro_backend::Tensor<B, 2>) -> f64 {
    t.into_data().to_vec::<f64>().unwrap()[0]
}

fn vector(t: skoro_backend::Tensor<B, 2>) -> Vec<f64> {
    t.into_data().to_vec::<f64>().unwrap()
}

/// `log p(x) = constant − ½‖r(x)‖² + log|L⁻¹|`, with `‖r‖²` summed **on the host** from the
/// evaluated residual rather than through the graph's `norm_squared`.
///
/// Taking the sum of squares outside the graph is what stops this from being a restatement
/// of `log_density`'s own body: the only thing shared between the two sides is the residual
/// node itself, which is exactly the thing under test.
fn assert_density_is_the_sum_of_squares<M>(
    client: &ModelClient<B>,
    distr: &Normal<B, M>,
    x: &Q<B, M>,
    points: ValueMap<B>,
) where
    M: skoro_graph::quantity::ContinuousQuantity<B>,
{
    let residual = WhitenedResidual::whitened_residual(distr, x).unwrap();
    let log_det = distr
        .whitening()
        .invert()
        .unwrap()
        .diag()
        .unwrap()
        .abs()
        .unwrap()
        .log()
        .unwrap()
        .sum()
        .unwrap();

    let r = vector(client.evaluate(&points, &residual).unwrap());
    let sum_of_squares: f64 = r.iter().map(|v| v * v).sum();
    let log_det = scalar(client.evaluate(&points, &log_det).unwrap());
    let density = scalar(client.log_density(&points).unwrap());

    approx::assert_abs_diff_eq!(
        density,
        distr.constant() - 0.5 * sum_of_squares + log_det,
        epsilon = 1e-9
    );
}

#[test]
fn a_real_density_is_its_squared_residual() {
    let client = ModelClient::<B>::default();
    let mean = client.add_constant(Real, &-1.75f64);
    let std_dev = constant_covariance_diagonal(&client, &[2.5]);
    let distr = Normal::new(mean, std_dev);
    let x = client.add_quantity(distr.clone()).unwrap();

    let values = points().with_quantity(&x, point(&[3.25])).build();
    assert_density_is_the_sum_of_squares(&client, &distr, &x, values);
}

/// A non-diagonal covariance, so the whitening actually mixes components: a `Tril` treated
/// as diagonal would pass a diagonal fixture.
#[test]
fn a_euclidean_density_is_its_squared_residual() {
    let client = ModelClient::<B>::default();
    let mean = client.add_constant(EuclideanD(3), &VectorD::new(&[10.0, -20.0, 30.0]));
    let std_dev = client.add_constant(
        Tril(3),
        // Packed row-major lower triangle, far from diagonal.
        &VectorD::new(&[2.0, -1.5, 3.0, 0.75, 2.5, 1.25]),
    );
    let distr = Normal::new(mean, std_dev);
    let x = client.add_quantity(distr.clone()).unwrap();

    let values = points()
        .with_quantity(&x, point(&[11.5, -18.25, 33.0]))
        .build();
    assert_density_is_the_sum_of_squares(&client, &distr, &x, values.clone());

    // And the residual really is `L⁻¹ (x − μ)`: check `L r = x − μ` by hand, which needs
    // no inverse and no library.
    let residual = WhitenedResidual::whitened_residual(&distr, &x).unwrap();
    let r = vector(client.evaluate(&values, &residual).unwrap());
    assert_eq!(r.len(), 3, "the residual is sized by manifold dimension");

    let packed = [2.0, -1.5, 3.0, 0.75, 2.5, 1.25];
    let expected = [11.5 - 10.0, -18.25 + 20.0, 33.0 - 30.0];
    for i in 0..3 {
        let lr: f64 = (0..=i).map(|j| packed[i * (i + 1) / 2 + j] * r[j]).sum();
        approx::assert_abs_diff_eq!(lr, expected[i], epsilon = 1e-9);
    }
}

/// `SE3` has dimension 6 and storage 12. The residual must be 6 wide.
#[test]
fn an_se3_density_is_its_squared_residual() {
    let client = ModelClient::<B>::default();
    // A rotation about z by 0.4 rad, translated. Far enough from the identity that a
    // missing Jacobian in the inverse retraction would show.
    let (c, s) = (0.4f64.cos(), 0.4f64.sin());
    let mean = client.add_constant(
        SE3,
        &SE3Pose::new(
            SO3Matrix([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]),
            [1.5, -2.25, 0.75],
        ),
    );
    let std_dev = constant_covariance_diagonal(&client, &[1.0, 2.0, 0.5, 3.0, 1.5, 0.25]);
    let distr = Normal::new(mean, std_dev);
    let x = client.add_quantity(distr.clone()).unwrap();

    // A different pose: rotation about z by 0.9 rad.
    let (c2, s2) = (0.9f64.cos(), 0.9f64.sin());
    let values = points()
        .with_quantity(
            &x,
            point(&[c2, -s2, 0.0, s2, c2, 0.0, 0.0, 0.0, 1.0, 2.0, -1.0, 1.25]),
        )
        .build();

    let residual = WhitenedResidual::whitened_residual(&distr, &x).unwrap();
    assert_eq!(
        vector(client.evaluate(&values, &residual).unwrap()).len(),
        6,
        "an SE3 residual is 6 wide — a dimension, not the 12-wide storage"
    );
    assert_density_is_the_sum_of_squares(&client, &distr, &x, values);
}

/// `Normal`'s normalizer depends on the covariance factor and on nothing else, which is
/// what makes the least-squares precondition checkable without inspecting an operation.
#[test]
fn only_the_covariance_reaches_the_normalizer() {
    let client = ModelClient::<B>::default();
    let mean = client.add_parameter(EuclideanD(2));
    let std_dev = constant_covariance_diagonal(&client, &[1.0, 2.0]);
    let distr = Normal::new(mean.clone(), std_dev.clone());

    assert_eq!(
        WhitenedResidual::normalizer_parents(&distr),
        &[std_dev.id()],
        "the mean must not be reported: it does not appear in the normalizer"
    );
    assert_eq!(GaussianFactor::mean(&distr).id(), mean.id());
    assert_eq!(GaussianFactor::whitening(&distr).id(), std_dev.id());
}

/// The Gaussian facade records every factor it builds, and the ids it records are the ones
/// the graph holds.
#[test]
fn the_gaussian_facade_records_what_it_built() {
    let mut model = GaussianModelClient::<B>::default();
    let mean_a = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
    let cov_a = constant_covariance_diagonal(&model, &[1.0, 2.0]);
    let a = model.add_gaussian_quantity(&mean_a, &cov_a).unwrap();

    let mean_b = model.add_constant(Real, &1.0f64);
    let cov_b = constant_covariance_diagonal(&model, &[0.5]);
    let b = model.add_gaussian_quantity(&mean_b, &cov_b).unwrap();

    let blocks: Vec<_> = GaussianStructure::blocks(&model).to_vec();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].mean, mean_a.id());
    assert_eq!(blocks[0].whitening, cov_a.id());
    assert_eq!(blocks[1].mean, mean_b.id());
    assert_eq!(blocks[1].whitening, cov_b.id());

    // The certificate agrees with the blocks, factor for factor and in order.
    let factors = NlsStructure::certificate(&model).factors();
    assert_eq!(factors.len(), 2);
    for (factor, block) in factors.iter().zip(&blocks) {
        assert_eq!(factor.density, block.density);
        assert_eq!(factor.residual, block.residual);
        assert_eq!(factor.normalizer_parents, vec![block.whitening]);
    }

    // Each recorded density really names a log-density node: evaluating the two by id and
    // summing must reproduce the model's total log-density. An id that named something else
    // — the outermost `+` of the sugar, say — would not add up.
    let values = points()
        .with_quantity(&a, point(&[0.5, -1.25]))
        .with_quantity(&b, point(&[2.25]))
        .build();
    let density_ids: Vec<Id> = factors.iter().map(|f| f.density).collect();
    let evaluated = model.evaluate_ids(&values, &density_ids).unwrap();
    let summed: f64 = density_ids
        .iter()
        .map(|id| {
            let (value, _) = evaluated.evaluations.get(id).expect("a recorded density");
            scalar(value.as_float().clone())
        })
        .sum();
    approx::assert_abs_diff_eq!(
        summed,
        scalar(model.log_density(&values).unwrap()),
        epsilon = 1e-9
    );

    // The row layout of `J` is the residuals, sized by dimension: 2 then 1.
    let layout = NlsStructure::residual_layout(&model).unwrap();
    assert_eq!(layout.total(), 3);
    assert_eq!(
        layout
            .blocks()
            .iter()
            .map(|b| b.dimension)
            .collect::<Vec<_>>(),
        vec![2, 1]
    );
}

/// The distribution-generic builder records the same way. It accepts any
/// `WhitenedResidual`, not only `Normal` — the bound is what it enforces, not the family.
#[test]
fn the_least_squares_builder_records_what_it_built() {
    let mut model = NlsModelClient::<B>::default();
    let mean = model.add_constant(EuclideanD(2), &VectorD::new(&[3.0, -1.0]));
    let cov = constant_covariance_diagonal(&model, &[1.0, 2.0]);
    let x = model
        .add_residual_quantity(Normal::new(mean, cov.clone()))
        .unwrap();

    let factors = NlsStructure::certificate(&model).factors();
    assert_eq!(factors.len(), 1);
    assert_eq!(factors[0].normalizer_parents, vec![cov.id()]);

    // The recorded density is the model's whole log-density, this being its only factor.
    let values = points().with_quantity(&x, point(&[3.5, -0.25])).build();
    let evaluated = model.evaluate_ids(&values, &[factors[0].density]).unwrap();
    let (density, _) = evaluated
        .evaluations
        .get(&factors[0].density)
        .expect("the recorded density");
    approx::assert_abs_diff_eq!(
        scalar(density.as_float().clone()),
        scalar(model.log_density(&values).unwrap()),
        epsilon = 1e-9
    );

    let layout = NlsStructure::residual_layout(&model).unwrap();
    assert_eq!(layout.total(), 2);
}

/// A certificate obtained from a builder that built nothing is empty — the same certificate
/// anyone could construct by hand.
#[test]
fn an_empty_builder_certifies_nothing() {
    let model = GaussianModelClient::<B>::default();
    assert!(NlsStructure::certificate(&model).factors().is_empty());
    assert_eq!(NlsStructure::residual_layout(&model).unwrap().total(), 0);
}

/// The recorded nodes survive garbage collection.
///
/// A residual is a `Mapped` node with no children, so the last `Q` naming it queues it for
/// cleanup on drop — which would leave the certificate holding ids the graph no longer has.
/// The builders keep those handles alive for exactly this reason, and `cleanup()` is a
/// public method, so this is reachable rather than theoretical.
#[test]
fn cleanup_does_not_collect_the_recorded_residuals() {
    let mut model = GaussianModelClient::<B>::default();
    let mean_a = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
    let cov_a = constant_covariance_diagonal(&model, &[1.0, 2.0]);
    let a = model.add_gaussian_quantity(&mean_a, &cov_a).unwrap();

    let mean_b = model.add_constant(Real, &1.0f64);
    let cov_b = constant_covariance_diagonal(&model, &[0.5]);
    let b = model.add_gaussian_quantity(&mean_b, &cov_b).unwrap();

    let before = NlsStructure::residual_layout(&model).unwrap();
    model.cleanup();
    let after = NlsStructure::residual_layout(&model).unwrap();
    assert_eq!(
        before, after,
        "cleanup must not collect a recorded residual"
    );

    // And the residuals still evaluate: the whole subgraph beneath them survives too, not
    // only the node the certificate names.
    let values = points()
        .with_quantity(&a, point(&[0.5, -1.25]))
        .with_quantity(&b, point(&[2.25]))
        .build();
    let ids: Vec<Id> = NlsStructure::certificate(&model).residual_ids().collect();
    let evaluated = model.evaluate_ids(&values, &ids).unwrap();
    for id in ids {
        assert!(evaluated.evaluations.get(&id).is_some());
    }
}

/// Retiring the middle factor of three leaves the outer two, in their original order.
///
/// Order is load-bearing twice over: `residual_layout` is *the* row order of `J`, and the
/// `GaussianBlock` list is read positionally beside it. So it is not enough that the count drops —
/// a prune that reordered the survivors would hand an optimizer one factor's rows labelled with
/// another's, which no shape check downstream would reject.
///
/// The two caches are pruned against the same liveness question for the same reason: two lists
/// over one graph that disagreed about which factors exist would be worse than either alone.
#[test]
fn retiring_a_factor_prunes_both_caches_and_keeps_the_order() {
    let mut model = GaussianModelClient::<B>::default();

    let mut add = |dims: &[f64]| {
        let mean = model.add_constant(
            EuclideanD(dims.len()),
            &VectorD::new(&vec![0.0; dims.len()]),
        );
        let cov = constant_covariance_diagonal(&model, dims);
        model.add_gaussian_quantity(&mean, &cov).unwrap()
    };
    let a = add(&[1.0, 2.0]);
    let b = add(&[1.0, 2.0, 3.0]);
    let c = add(&[0.5]);

    let before: Vec<_> = NlsStructure::certificate(&model).factors().to_vec();
    assert_eq!(before.len(), 3);
    assert_eq!(NlsStructure::residual_layout(&model).unwrap().total(), 6);

    model.remove_closure(&[b.id()]).unwrap();
    model.cleanup();

    let after = NlsStructure::certificate(&model).factors();
    assert_eq!(
        after,
        [before[0].clone(), before[2].clone()],
        "the survivors keep their identity and their order"
    );
    assert_eq!(
        GaussianStructure::blocks(&model)
            .iter()
            .map(|block| block.residual)
            .collect::<Vec<_>>(),
        vec![before[0].residual, before[2].residual],
        "the block list is pruned in step with the certificate"
    );

    // The rows of `J` are now `a` then `c`: 2 then 1, and never 2 then 3.
    let layout = NlsStructure::residual_layout(&model).unwrap();
    assert_eq!(
        layout
            .blocks()
            .iter()
            .map(|block| block.dimension)
            .collect::<Vec<_>>(),
        vec![2, 1]
    );

    // And the model still evaluates over what is left.
    let values = points()
        .with_quantity(&a, point(&[0.5, -1.25]))
        .with_quantity(&c, point(&[2.25]))
        .build();
    let ids: Vec<Id> = NlsStructure::certificate(&model).residual_ids().collect();
    let evaluated = model.evaluate_ids(&values, &ids).unwrap();
    for id in ids {
        assert!(evaluated.evaluations.get(&id).is_some());
    }
}
