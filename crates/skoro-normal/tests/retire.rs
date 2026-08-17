//! Retiring certified factors through the client tier, not only through `remove_closure`.
//!
//! The certificate lives in the server and is pruned inside every mutating call, against
//! two questions asked of the graph: is the residual in the arena, and is the density
//! still a live likelihood term. These tests pin the three ways a factor used to outlive
//! its evidence — demotion, removal of the residual, removal of the density — each of
//! which once left `residual_layout` and the log-density disagreeing about the model.

use skoro_backend::{NdArrayBackend, Point, TensorData};
use skoro_core::error::SkoroError;
use skoro_linalg::matrix::matmul;
use skoro_model::{
    ModelBuilderOps, ModelClientOps, NlsModelClient, NlsStructure, hash_map_builders::points,
    model::ModelClient, q::Q,
};
use skoro_normal::{
    distribution::Normal,
    facade::{GaussianModelClient, GaussianModelClientOps, constant_covariance_diagonal},
    structure::GaussianStructure,
};
use skoro_operation::Input;
use skoro_riemann::{
    euclidean::{EuclideanD, VectorD},
    matrix::{EuclideanMatrixD, MatrixD},
};

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

/// Demoting a recorded quantity orphans its density: the node stays in the arena but
/// stops scoring. A liveness rule keyed on arena membership keeps the record — and hands
/// `J` rows for a term the objective no longer holds, with no error, ever.
#[test]
fn demoting_a_recorded_quantity_prunes_the_factor_everywhere() {
    let mut model = GaussianModelClient::<B>::default();
    let mean = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
    let cov = constant_covariance_diagonal(&model, &[1.0, 2.0]);
    let x = model.add_gaussian_quantity(&mean, &cov).unwrap();

    let values = points().with_quantity(&x, point(&[0.5, -1.25])).build();
    let before = scalar(model.log_density(&values).unwrap());
    assert!(before.is_finite() && before != 0.0);
    assert_eq!(NlsStructure::residual_layout(&model).unwrap().total(), 2);
    assert_eq!(GaussianStructure::blocks(&model).len(), 1);

    let certificate = NlsStructure::certificate(&model);
    let residual = certificate.factors()[0].residual;
    model.demote_to_parameter(&x).unwrap();

    // The term left the objective at the demote — and the rows of `J` and the Gaussian
    // block left with it, in the same call.
    assert_eq!(scalar(model.log_density(&values).unwrap()), 0.0);
    assert_eq!(NlsStructure::residual_layout(&model).unwrap().total(), 0);
    assert!(NlsStructure::certificate(&model).factors().is_empty());
    assert!(GaussianStructure::blocks(&model).is_empty());

    // Uncertified, the residual subgraph is collectable again.
    model.cleanup();
    assert!(model.tangent_layout(&[residual]).is_err());
}

/// Client-tier `remove` refuses a node the certificate names — either of them.
///
/// Both are childless, so nothing about the *graph* would stop this: a certified residual is
/// a leaf by construction, and a marked `ℓ` scores nothing that reads it. That is the soft
/// spot `docs/skoro.md` §1.7 used to concede, and the reason it is closed here rather than
/// papered over: `remove` sits on [`ModelClientOps`], which an estimator bounded on
/// `NlsStructure` holds, and taking either node would drop a term from the objective with no
/// error at all. `remove` is the primitive that refuses to break a reference; a record is a
/// reference.
#[test]
fn removing_a_certified_node_is_refused() {
    let mut model = NlsModelClient::<B>::default();
    let mean = model.add_constant(EuclideanD(2), &VectorD::new(&[3.0, -1.0]));
    let cov = constant_covariance_diagonal(&model, &[1.0, 2.0]);
    let x = model.add_residual_quantity(Normal::new(mean, cov)).unwrap();

    let record = NlsStructure::certificate(&model).factors()[0].clone();
    let values = points().with_quantity(&x, point(&[3.5, -0.25])).build();
    let before = scalar(model.log_density(&values).unwrap());
    assert!(before != 0.0);

    for id in [record.residual, record.density] {
        assert!(
            matches!(
                ModelClientOps::remove(&model, &[id]),
                Err(SkoroError::NodeIsCertified { .. })
            ),
            "certified node {id:?} must be refused, not silently taken"
        );
    }

    // Refused means refused: the objective, the rows of `J`, and the evidence are untouched.
    assert_eq!(scalar(model.log_density(&values).unwrap()), before);
    assert_eq!(NlsStructure::residual_layout(&model).unwrap().total(), 2);
    assert_eq!(NlsStructure::certificate(&model).factors().len(), 1);
}

/// The way a factor is meant to be retired: a closure at the quantity it scores.
///
/// One tier up from `remove`, and it takes the referents rather than refusing — which is the
/// whole distinction between the two. Everything the factor built goes with it, the record
/// leaves in the same locked call, and the objective and `J` shrink together.
#[test]
fn retiring_a_scored_quantity_takes_the_factor_and_its_record() {
    let mut model = NlsModelClient::<B>::default();
    let mean = model.add_constant(EuclideanD(2), &VectorD::new(&[3.0, -1.0]));
    let cov = constant_covariance_diagonal(&model, &[1.0, 2.0]);
    let x = model.add_residual_quantity(Normal::new(mean, cov)).unwrap();

    let record = NlsStructure::certificate(&model).factors()[0].clone();
    let values = points().with_quantity(&x, point(&[3.5, -0.25])).build();
    assert!(scalar(model.log_density(&values).unwrap()) != 0.0);

    model.remove_closure(&[x.id()]).unwrap();

    assert!(NlsStructure::certificate(&model).factors().is_empty());
    assert_eq!(NlsStructure::residual_layout(&model).unwrap().total(), 0);
    assert_eq!(scalar(model.log_density(&values).unwrap()), 0.0);
    assert!(model.tangent_layout(&[record.residual]).is_err());
    assert!(model.tangent_layout(&[record.density]).is_err());
}

/// Un-certifying a residual re-offers it to the collector.
///
/// The subtle half of the collector's contract: a candidate is *popped* when examined, so a
/// "not yet" verdict is final. Here the residual's handle dropped inside
/// `add_residual_quantity`, the next `cleanup()` declined it because the certificate named
/// it — consuming the candidate — and only the record's departure makes it collectable. If
/// pruning does not re-offer, nothing ever will.
///
/// The second factor is load-bearing: it forces a `cleanup()` between the handle's death and
/// the demote, which is what consumes the candidate. Without it the id is still queued and
/// the leak hides.
#[test]
fn uncertifying_a_residual_re_offers_it_for_collection() {
    let mut model = NlsModelClient::<B>::default();
    let m1 = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
    let c1 = constant_covariance_diagonal(&model, &[1.0, 2.0]);
    let x1 = model.add_residual_quantity(Normal::new(m1, c1)).unwrap();
    let residual = NlsStructure::certificate(&model).factors()[0].residual;

    let m2 = model.add_constant(EuclideanD(2), &VectorD::new(&[1.0, 1.0]));
    let c2 = constant_covariance_diagonal(&model, &[1.0, 1.0]);
    let _x2 = model.add_residual_quantity(Normal::new(m2, c2)).unwrap();

    model.demote_to_parameter(&x1).unwrap();
    assert!(
        NlsStructure::certificate(&model)
            .factors()
            .iter()
            .all(|factor| factor.residual != residual)
    );
    assert!(
        model.tangent_layout(&[residual]).is_err(),
        "an un-certified residual must be reclaimed, not stranded"
    );
}

/// Demotion reclaims the subgraph its orphaned density was holding up.
///
/// A density that no longer scores is garbage by the same rule as everything else, and it is
/// the *only* thing keeping this `mean` alive: `mean`'s own handle died inside `Normal::new`,
/// and after the demote its only remaining child is the old density's residual expression.
/// Treating an orphaned density as caller-decides — the rule before reference counting —
/// strands a whole factor's worth of nodes per demotion, which is once per tick of a window.
#[test]
fn demotion_reclaims_the_subgraph_its_orphaned_density_held() {
    let mut model = NlsModelClient::<B>::default();
    let f = model.add_constant(
        EuclideanMatrixD(2, 2),
        &MatrixD::new(2, 2, &[2.0, 0.0, 0.0, 2.0]),
    );
    let cov = constant_covariance_diagonal(&model, &[1.0, 1.0]);
    let m0 = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));

    let x0 = model
        .add_residual_quantity(Normal::new(m0, cov.clone()))
        .unwrap();
    let mean = matmul(&f, &x0).unwrap();
    let mean_id = mean.id();
    let x1 = model
        .add_residual_quantity(Normal::new(mean, cov.clone()))
        .unwrap();
    assert!(
        model.tangent_layout(&[mean_id]).is_ok(),
        "held up by its child"
    );

    model.demote_to_parameter(&x1).unwrap();

    assert!(
        model.tangent_layout(&[mean_id]).is_err(),
        "a severed parent must be reclaimed, not stranded"
    );
    assert!(model.tangent_layout(&[x0.id()]).is_ok());
}

/// Replacing a distribution severs the parents the new one does not keep.
///
/// The same cut as a demotion, through the other door — and the one most likely to be
/// forgotten, because `update_quantity` reads like an edit rather than a teardown.
#[test]
fn replacing_a_distribution_re_offers_the_parents_it_drops() {
    let model = ModelClient::<B>::default();
    let cov = constant_covariance_diagonal(&model, &[1.0, 1.0]);
    let base = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));

    let old_mean = matmul_identity(&model, &base);
    let old_mean_id = old_mean.id();
    let x = model
        .add_quantity(Normal::new(old_mean, cov.clone()))
        .unwrap();
    assert!(model.tangent_layout(&[old_mean_id]).is_ok());

    let new_mean = matmul_identity(&model, &base);
    model
        .update_quantity(&x, Normal::new(new_mean, cov.clone()))
        .unwrap();

    assert!(
        model.tangent_layout(&[old_mean_id]).is_err(),
        "the dropped mean must be reclaimed, not stranded"
    );
}

/// `I · v`, as a mapped node whose only handle is the one returned.
fn matmul_identity(model: &ModelClient<B>, v: &Q<B, EuclideanD>) -> Q<B, EuclideanD> {
    let identity = model.add_constant(
        EuclideanMatrixD(2, 2),
        &MatrixD::new(2, 2, &[1.0, 0.0, 0.0, 1.0]),
    );
    matmul(&identity, v).unwrap()
}
