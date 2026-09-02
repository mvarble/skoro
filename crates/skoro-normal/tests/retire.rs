//! Retiring bound factors through the client tier, not only through `remove_closure`.
//!
//! A term and its bindings live in the graph and go with it: the density stops being a live
//! likelihood term, and the bindings hanging off it leave in the same locked call. These tests
//! pin the three ways a factor used to outlive its bookkeeping — demotion, removal of the
//! residual, removal of the density — each of which once left `residual_layout` and the
//! log-density disagreeing about the model.
//!
//! There is no test here for replacing a distribution in place: `update_quantity` went with
//! `Graph::update_stochastic`, and replacing a factor is now a closure removal followed by a
//! rebuild, which `retiring_a_scored_quantity_takes_the_factor_and_its_record` covers.

use skoro_backend::{NdArrayBackend, Point, TensorData};
use skoro_core::error::SkoroError;
use skoro_linalg::matrix::matmul;
use skoro_model::{
    ModelBuilderOps, ModelClientOps,
    hash_map_builders::points,
    least_squares::residual_layout,
    model::ModelClient,
    roles::{FISHER_ROOT, RESIDUAL},
};
use skoro_normal::{
    distribution::Normal, facade::constant_covariance_diagonal, structure::GaussianBlock,
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

/// The bindings go inside the same locked call as the mutation: removing a node clears both
/// binding directions for it, and a demoted quantity's density stops being a term at all.
/// Demotion retires the density, a closure removal takes the whole factor body.
mod pruning {
    use super::*;

    /// Demoting a bound quantity orphans its density: the node stays in the arena but
    /// stops scoring. A liveness rule keyed on arena membership would keep the term — and hand
    /// `J` rows for something the objective no longer holds, with no error, ever.
    #[test]
    fn demoting_a_recorded_quantity_prunes_the_factor_everywhere() {
        let model = ModelClient::<B>::default();
        let mean = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
        let cov = constant_covariance_diagonal(&model, &[1.0, 2.0]);
        let x = model
            .add_quantity(Normal::new(mean.clone(), cov.clone()))
            .unwrap();

        let values = points().with_quantity(&x, point(&[0.5, -1.25])).build();
        let before = scalar(model.log_density(&values).unwrap());
        assert!(before.is_finite() && before != 0.0);
        assert_eq!(residual_layout(&model).unwrap().total(), 2);
        assert_eq!(GaussianBlock::all(&model).len(), 1);

        let residual = model.bound(ModelClientOps::factors(&model)[0].density, RESIDUAL)[0];
        model.demote_to_parameter(&x).unwrap();

        // The term left the objective at the demote — and the rows of `J` and the Gaussian
        // block left with it, in the same call.
        assert_eq!(scalar(model.log_density(&values).unwrap()), 0.0);
        assert_eq!(residual_layout(&model).unwrap().total(), 0);
        assert!(ModelClientOps::factors(&model).is_empty());
        assert!(GaussianBlock::all(&model).is_empty());

        // Unbound, the residual subgraph is collectable again.
        model.cleanup();
        assert!(model.tangent_layout(&[residual]).is_err());
    }

    /// The way a factor is meant to be retired: a closure at the quantity it scores.
    ///
    /// One tier up from `remove`, and it takes the referents rather than refusing — which is the
    /// whole distinction between the two. Everything the factor built goes with it, the term and
    /// its bindings leave in the same locked call, and the objective and `J` shrink together.
    #[test]
    fn retiring_a_scored_quantity_takes_the_factor_and_its_record() {
        let model = ModelClient::<B>::default();
        let mean = model.add_constant(EuclideanD(2), &VectorD::new(&[3.0, -1.0]));
        let cov = constant_covariance_diagonal(&model, &[1.0, 2.0]);
        let x = model
            .add_quantity(Normal::new(mean.clone(), cov.clone()))
            .unwrap();

        let factor = ModelClientOps::factors(&model)[0].clone();
        let residual = model.bound(factor.density, RESIDUAL)[0];
        let values = points().with_quantity(&x, point(&[3.5, -0.25])).build();
        assert!(scalar(model.log_density(&values).unwrap()) != 0.0);

        model.remove_closure(&[x.id()]).unwrap();

        assert!(ModelClientOps::factors(&model).is_empty());
        assert_eq!(residual_layout(&model).unwrap().total(), 0);
        assert_eq!(scalar(model.log_density(&values).unwrap()), 0.0);
        assert!(model.tangent_layout(&[residual]).is_err());
        assert!(model.tangent_layout(&[factor.density]).is_err());
    }
}

/// Client-tier `remove` refuses a node a binding names. A Gaussian's Fisher root is childless, so
/// nothing about the *graph* would stop it being taken, and taking it would leave the term naming
/// a node the arena no longer holds. `remove` is the primitive that refuses to break a reference.
mod rejections {
    use super::*;

    /// Client-tier `remove` refuses a node the term's bindings name.
    ///
    /// The `FISHER_ROOT` node is the discriminating case: it is childless, so nothing about the
    /// *graph* would stop this — it is a leaf by construction, read only through the binding. That
    /// is the soft spot `docs/skoro.md` §1.7 used to concede, and the reason it is closed here
    /// rather than papered over: `remove` sits on [`ModelClientOps`], which an estimator holds,
    /// and taking the node would leave a term pointing at nothing with no error at all. `remove`
    /// is the primitive that refuses to break a reference; a binding is a reference.
    ///
    /// The residual is refused too, one check earlier: `Normal`'s density reads it, so it is
    /// [`SkoroError::NodeStillReferenced`] before the binding is ever consulted. Either way it
    /// does not leave under an estimator's hands.
    #[test]
    fn removing_a_certified_node_is_refused() {
        let model = ModelClient::<B>::default();
        let mean = model.add_constant(EuclideanD(2), &VectorD::new(&[3.0, -1.0]));
        let cov = constant_covariance_diagonal(&model, &[1.0, 2.0]);
        let x = model
            .add_quantity(Normal::new(mean.clone(), cov.clone()))
            .unwrap();

        let density = ModelClientOps::factors(&model)[0].density;
        let residual = model.bound(density, RESIDUAL)[0];
        let fisher_root = model.bound(density, FISHER_ROOT)[0];
        let values = points().with_quantity(&x, point(&[3.5, -0.25])).build();
        let before = scalar(model.log_density(&values).unwrap());
        assert!(before != 0.0);

        assert!(
            matches!(
                ModelClientOps::remove(&model, &[fisher_root]),
                Err(SkoroError::NodeStillBound { id, by }) if id == fisher_root && by == density
            ),
            "a bound leaf must be refused, not silently taken"
        );
        assert!(
            matches!(
                ModelClientOps::remove(&model, &[residual]),
                Err(SkoroError::NodeStillReferenced { .. })
            ),
            "the residual the density reads must be refused too"
        );

        // Refused means refused: the objective, the rows of `J`, and the bindings are untouched.
        assert_eq!(scalar(model.log_density(&values).unwrap()), before);
        assert_eq!(residual_layout(&model).unwrap().total(), 2);
        assert_eq!(ModelClientOps::factors(&model).len(), 1);
        assert_eq!(model.bound(density, RESIDUAL), vec![residual]);
        assert_eq!(model.bound(density, FISHER_ROOT), vec![fisher_root]);
    }
}

/// What the collector gets back when a term goes. The `RESIDUAL` binding was the only thing
/// keeping the residual reachable, so unbinding it re-offers it; and an orphaned density is
/// garbage by the same rule as everything else, which is a whole factor's worth of nodes per
/// demotion -- once per tick of a window.
mod reclamation {
    use super::*;

    /// Unbinding a residual re-offers it to the collector.
    ///
    /// The subtle half of the collector's contract: a candidate is *popped* when examined, so a
    /// "not yet" verdict is final. Here the residual's handle dropped inside `Normal::factor`,
    /// and the term is the only thing keeping it: the moment the density is retired, the residual
    /// is unreachable and must go with it.
    ///
    /// The second factor is load-bearing: it forces a `cleanup()` between the handle's death and
    /// the demote, so this is a test of what the demote reclaims rather than of what an earlier
    /// collection happened to leave behind.
    #[test]
    fn un_recording_a_residual_reclaims_it() {
        let model = ModelClient::<B>::default();
        let m1 = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
        let c1 = constant_covariance_diagonal(&model, &[1.0, 2.0]);
        let x1 = model.add_quantity(Normal::new(m1, c1)).unwrap();
        let residual = model.bound(ModelClientOps::factors(&model)[0].density, RESIDUAL)[0];

        let m2 = model.add_constant(EuclideanD(2), &VectorD::new(&[1.0, 1.0]));
        let c2 = constant_covariance_diagonal(&model, &[1.0, 1.0]);
        let _x2 = model.add_quantity(Normal::new(m2, c2)).unwrap();
        model.cleanup();

        model.demote_to_parameter(&x1).unwrap();
        assert!(
            ModelClientOps::factors(&model)
                .iter()
                .all(|factor| model.bound(factor.density, RESIDUAL) != vec![residual])
        );
        assert!(
            model.tangent_layout(&[residual]).is_err(),
            "an unbound residual must be reclaimed, not stranded"
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

        let x0 = model.add_quantity(Normal::new(m0, cov.clone())).unwrap();
        let mean = matmul(&f, &x0).unwrap();
        let mean_id = mean.id();
        let x1 = model
            .add_quantity(Normal::new(mean.clone(), cov.clone()))
            .unwrap();
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
