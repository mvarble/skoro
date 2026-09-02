//! Retiring recorded factors through the client tier, not only through `remove_closure`.
//!
//! The registry lives in the server and is pruned inside every mutating call, against two
//! questions asked of the graph: is the residual in the arena, and is the density still a
//! live likelihood term. These tests pin the three ways a factor used to outlive its record
//! — demotion, removal of the residual, removal of the density — each of which once left
//! `residual_layout` and the log-density disagreeing about the model.
//!
//! There is no test here for replacing a distribution in place: `update_quantity` went with
//! `Graph::update_stochastic`, and replacing a factor is now a closure removal followed by a
//! rebuild, which `retiring_a_scored_quantity_takes_the_factor_and_its_record` covers.

use skoro_backend::{NdArrayBackend, Point, TensorData};
use skoro_core::error::SkoroError;
use skoro_linalg::matrix::matmul;
use skoro_model::{ModelBuilderOps, ModelClientOps, hash_map_builders::points, model::ModelClient};
use skoro_normal::facade::{add_gaussian_quantity, constant_covariance_diagonal, gaussian_blocks};
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

/// The registry is pruned inside the same locked call as the mutation, against two questions asked
/// of the graph: is the residual still in the arena, and is the density still a live likelihood
/// term. Demotion answers no to the second, a closure removal to both.
mod pruning {
    use super::*;

    /// Demoting a recorded quantity orphans its density: the node stays in the arena but
    /// stops scoring. A liveness rule keyed on arena membership keeps the record — and hands
    /// `J` rows for a term the objective no longer holds, with no error, ever.
    #[test]
    fn demoting_a_recorded_quantity_prunes_the_factor_everywhere() {
        let model = ModelClient::<B>::default();
        let mean = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
        let cov = constant_covariance_diagonal(&model, &[1.0, 2.0]);
        let x = add_gaussian_quantity(&model, &mean, &cov).unwrap();

        let values = points().with_quantity(&x, point(&[0.5, -1.25])).build();
        let before = scalar(model.log_density(&values).unwrap());
        assert!(before.is_finite() && before != 0.0);
        assert_eq!(model.residual_layout().unwrap().total(), 2);
        assert_eq!(gaussian_blocks(&model).len(), 1);

        let residual = model.factors()[0].residual;
        model.demote_to_parameter(&x).unwrap();

        // The term left the objective at the demote — and the rows of `J` and the Gaussian
        // block left with it, in the same call.
        assert_eq!(scalar(model.log_density(&values).unwrap()), 0.0);
        assert_eq!(model.residual_layout().unwrap().total(), 0);
        assert!(model.factors().is_empty());
        assert!(gaussian_blocks(&model).is_empty());

        // Uncertified, the residual subgraph is collectable again.
        model.cleanup();
        assert!(model.tangent_layout(&[residual]).is_err());
    }

    /// The way a factor is meant to be retired: a closure at the quantity it scores.
    ///
    /// One tier up from `remove`, and it takes the referents rather than refusing — which is the
    /// whole distinction between the two. Everything the factor built goes with it, the record
    /// leaves in the same locked call, and the objective and `J` shrink together.
    #[test]
    fn retiring_a_scored_quantity_takes_the_factor_and_its_record() {
        let model = ModelClient::<B>::default();
        let mean = model.add_constant(EuclideanD(2), &VectorD::new(&[3.0, -1.0]));
        let cov = constant_covariance_diagonal(&model, &[1.0, 2.0]);
        let x = add_gaussian_quantity(&model, &mean, &cov).unwrap();

        let record = model.factors()[0].clone();
        let values = points().with_quantity(&x, point(&[3.5, -0.25])).build();
        assert!(scalar(model.log_density(&values).unwrap()) != 0.0);

        model.remove_closure(&[x.id()]).unwrap();

        assert!(model.factors().is_empty());
        assert_eq!(model.residual_layout().unwrap().total(), 0);
        assert_eq!(scalar(model.log_density(&values).unwrap()), 0.0);
        assert!(model.tangent_layout(&[record.residual]).is_err());
        assert!(model.tangent_layout(&[record.density]).is_err());
    }
}

/// Client-tier `remove` refuses a node a record names -- either of them. Both are childless, so
/// nothing about the *graph* would stop it, and taking either would drop a term from the objective
/// with no error at all. `remove` is the primitive that refuses to break a reference.
mod rejections {
    use super::*;

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
        let model = ModelClient::<B>::default();
        let mean = model.add_constant(EuclideanD(2), &VectorD::new(&[3.0, -1.0]));
        let cov = constant_covariance_diagonal(&model, &[1.0, 2.0]);
        let x = add_gaussian_quantity(&model, &mean, &cov).unwrap();

        let record = model.factors()[0].clone();
        let values = points().with_quantity(&x, point(&[3.5, -0.25])).build();
        let before = scalar(model.log_density(&values).unwrap());
        assert!(before != 0.0);

        for id in [record.residual, record.density] {
            assert!(
                matches!(
                    ModelClientOps::remove(&model, &[id]),
                    Err(SkoroError::NodeStillBound { .. })
                ),
                "certified node {id:?} must be refused, not silently taken"
            );
        }

        // Refused means refused: the objective, the rows of `J`, and the evidence are untouched.
        assert_eq!(scalar(model.log_density(&values).unwrap()), before);
        assert_eq!(model.residual_layout().unwrap().total(), 2);
        assert_eq!(model.factors().len(), 1);
    }
}

/// What the collector gets back when a record goes. The record was the only thing keeping a
/// certified residual reachable, so un-certifying it re-offers it; and an orphaned density is
/// garbage by the same rule as everything else, which is a whole factor's worth of nodes per
/// demotion -- once per tick of a window.
mod reclamation {
    use super::*;

    /// Un-certifying a residual re-offers it to the collector.
    ///
    /// The subtle half of the collector's contract: a candidate is *popped* when examined, so a
    /// "not yet" verdict is final. Here the residual's handle dropped inside
    /// `add_gaussian_quantity`, and the record is the only thing keeping it: the moment pruning
    /// drops the record, the residual is unreachable and must go with it.
    ///
    /// The second factor is load-bearing: it forces a `cleanup()` between the handle's death and
    /// the demote, so this is a test of what the demote reclaims rather than of what an earlier
    /// collection happened to leave behind.
    #[test]
    fn un_recording_a_residual_reclaims_it() {
        let model = ModelClient::<B>::default();
        let m1 = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
        let c1 = constant_covariance_diagonal(&model, &[1.0, 2.0]);
        let x1 = add_gaussian_quantity(&model, &m1, &c1).unwrap();
        let residual = model.factors()[0].residual;

        let m2 = model.add_constant(EuclideanD(2), &VectorD::new(&[1.0, 1.0]));
        let c2 = constant_covariance_diagonal(&model, &[1.0, 1.0]);
        let _x2 = add_gaussian_quantity(&model, &m2, &c2).unwrap();
        model.cleanup();

        model.demote_to_parameter(&x1).unwrap();
        assert!(
            model.factors()
                .iter()
                .all(|factor| factor.residual != residual)
        );
        assert!(
            model.tangent_layout(&[residual]).is_err(),
            "an un-recorded residual must be reclaimed, not stranded"
        );
    }

    /// Demotion reclaims the subgraph its orphaned density was holding up.
    ///
    /// A density that no longer scores is garbage by the same rule as everything else, and after
    /// the `drop(mean)` below it is the *only* thing keeping this `mean` alive: once `x1` is
    /// demoted, `mean`'s only remaining child is the old density's residual expression. Treating
    /// an orphaned density as caller-decides strands a whole factor's worth of nodes per
    /// demotion, which is once per tick of a window.
    #[test]
    fn demotion_reclaims_the_subgraph_its_orphaned_density_held() {
        let model = ModelClient::<B>::default();
        let f = model.add_constant(
            EuclideanMatrixD(2, 2),
            &MatrixD::new(2, 2, &[2.0, 0.0, 0.0, 2.0]),
        );
        let cov = constant_covariance_diagonal(&model, &[1.0, 1.0]);
        let m0 = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));

        let x0 = add_gaussian_quantity(&model, &m0, &cov).unwrap();
        let mean = matmul(&f, &x0).unwrap();
        let mean_id = mean.id();
        let x1 = add_gaussian_quantity(&model, &mean, &cov).unwrap();
        drop(mean);
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
}
