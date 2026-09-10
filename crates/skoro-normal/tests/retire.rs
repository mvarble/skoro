//! Retiring Gaussian factors, and what their bindings do when a term goes.
//!
//! A Gaussian factor's residual and Fisher root hang off its term by binding, with no handle of
//! their own. These tests pin the ways a term leaves — demotion and closure removal — and the one
//! way it must not: a client-tier `remove` taking a bound node out from under it. Each checks that
//! the rows of `J`, the factors and the log-density agree about the model afterwards.
//!
//! There is no test here for replacing a distribution in place: `update_quantity` went with
//! `Graph::update_stochastic`, and replacing a factor is now a closure removal followed by a
//! rebuild, which `retiring_a_scored_quantity_takes_the_factor_and_its_bindings` covers.

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

/// A term leaves the objective at a demotion or a closure removal, and its rows, its Gaussian
/// block and the nodes it bound leave with it, in the same call.
mod retirement {
    use super::*;

    /// Demoting a Gaussian quantity retires its density: the node stops scoring, and the rows of
    /// `J`, the factor and the Gaussian block leave in the same call. The residual it bound is
    /// collectable from then on.
    #[test]
    fn demoting_a_gaussian_quantity_drops_the_factor_everywhere() {
        let model = ModelClient::<B>::default();
        let mean = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
        let cov = constant_covariance_diagonal(&model, &[1.0, 2.0]);
        let x = model.add_quantity(Normal::new(mean, cov)).unwrap();

        let values = points().with_quantity(&x, point(&[0.5, -1.25])).build();
        let before = scalar(model.log_density(&values).unwrap());
        assert!(before.is_finite() && before != 0.0);
        assert_eq!(residual_layout(&model).unwrap().total(), 2);
        assert_eq!(GaussianBlock::all(&model).len(), 1);

        let residual = model.bound(model.factors()[0].density, RESIDUAL)[0];
        model.demote_to_parameter(&x).unwrap();

        assert_eq!(scalar(model.log_density(&values).unwrap()), 0.0);
        assert_eq!(residual_layout(&model).unwrap().total(), 0);
        assert!(model.factors().is_empty());
        assert!(GaussianBlock::all(&model).is_empty());

        model.cleanup();
        assert!(model.tangent_layout(&[residual]).is_err());
    }

    /// The way a factor is meant to be retired: a closure at the quantity it scores.
    ///
    /// One tier up from `remove`, and it takes the referents rather than refusing — which is the
    /// whole distinction between the two. The term, its residual and its Fisher root all go, and
    /// the objective and `J` shrink together.
    #[test]
    fn retiring_a_scored_quantity_takes_the_factor_and_its_bindings() {
        let model = ModelClient::<B>::default();
        let mean = model.add_constant(EuclideanD(2), &VectorD::new(&[3.0, -1.0]));
        let cov = constant_covariance_diagonal(&model, &[1.0, 2.0]);
        let x = model.add_quantity(Normal::new(mean, cov)).unwrap();

        let density = model.factors()[0].density;
        let residual = model.bound(density, RESIDUAL)[0];
        let root = model.bound(density, FISHER_ROOT)[0];
        let values = points().with_quantity(&x, point(&[3.5, -0.25])).build();
        assert!(scalar(model.log_density(&values).unwrap()) != 0.0);

        model.remove_closure(&[x.id()]).unwrap();

        assert!(model.factors().is_empty());
        assert_eq!(residual_layout(&model).unwrap().total(), 0);
        assert_eq!(scalar(model.log_density(&values).unwrap()), 0.0);
        for id in [density, residual, root] {
            assert!(
                model.tangent_layout(&[id]).is_err(),
                "{id:?} must go with its term"
            );
        }
    }
}

/// Client-tier `remove` refuses the nodes a term binds. The Fisher root is childless, so nothing
/// about the parent edges would stop its removal, and taking it would leave the term binding a node
/// that is gone; the binding is what refuses it. `remove` is the primitive that refuses to break a
/// reference.
mod rejections {
    use super::*;

    #[test]
    fn removing_a_bound_node_is_refused() {
        let model = ModelClient::<B>::default();
        let mean = model.add_constant(EuclideanD(2), &VectorD::new(&[3.0, -1.0]));
        let cov = constant_covariance_diagonal(&model, &[1.0, 2.0]);
        let x = model.add_quantity(Normal::new(mean, cov)).unwrap();

        let density = model.factors()[0].density;
        let values = points().with_quantity(&x, point(&[3.5, -0.25])).build();
        let before = scalar(model.log_density(&values).unwrap());
        assert!(before != 0.0);

        let root = model.bound(density, FISHER_ROOT)[0];
        assert_eq!(
            ModelClientOps::remove(&model, &[root]).map(|removed| removed.len()),
            Err(SkoroError::NodeStillBound {
                id: root,
                by: density
            }),
            "a bound root must be refused, not silently taken"
        );

        // A Gaussian's residual also feeds its own density's `‖r‖²`, so the parent-edge check,
        // which runs first, refuses it before the binding is consulted.
        let residual = model.bound(density, RESIDUAL)[0];
        assert!(matches!(
            ModelClientOps::remove(&model, &[residual]),
            Err(SkoroError::NodeStillReferenced { id, .. }) if id == residual
        ));

        // Refused means refused: the objective, the rows of `J`, and the factors are untouched.
        assert_eq!(scalar(model.log_density(&values).unwrap()), before);
        assert_eq!(residual_layout(&model).unwrap().total(), 2);
        assert_eq!(model.factors().len(), 1);
    }
}

/// What the collector gets back when a term goes. The binding was the only thing keeping a residual
/// reachable, so retiring the term re-offers it; and an orphaned density is garbage by the same rule
/// as everything else, which is a whole factor's worth of nodes per demotion -- once per tick of a
/// window.
mod reclamation {
    use super::*;

    /// Retiring a term re-offers its bound residual to the collector.
    ///
    /// The second factor is load-bearing: it forces a `cleanup()` between the residual's handle
    /// dying (inside `add_quantity`) and the demotion, so this is a test of what the demotion
    /// reclaims rather than of what an earlier collection happened to leave behind.
    #[test]
    fn demoting_a_quantity_reclaims_its_bound_residual() {
        let model = ModelClient::<B>::default();
        let m1 = model.add_constant(EuclideanD(2), &VectorD::new(&[0.0, 0.0]));
        let c1 = constant_covariance_diagonal(&model, &[1.0, 2.0]);
        let x1 = model.add_quantity(Normal::new(m1, c1)).unwrap();
        let residual = model.bound(model.factors()[0].density, RESIDUAL)[0];

        let m2 = model.add_constant(EuclideanD(2), &VectorD::new(&[1.0, 1.0]));
        let c2 = constant_covariance_diagonal(&model, &[1.0, 1.0]);
        let _x2 = model.add_quantity(Normal::new(m2, c2)).unwrap();
        model.cleanup();
        assert!(
            model.tangent_layout(&[residual]).is_ok(),
            "held by its term's binding"
        );

        model.demote_to_parameter(&x1).unwrap();
        assert!(
            model
                .factors()
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
    /// A density that no longer scores is garbage by the same rule as everything else. Once `x1`
    /// is demoted, `mean`'s only remaining referents are the old density's residual expression and
    /// its `LOCATION` binding, and both belong to garbage. Treating an orphaned density as
    /// caller-decides strands a whole factor's worth of nodes per demotion.
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
        // The distribution takes the only handle to `mean`, and is consumed by `add_quantity`.
        let x1 = model.add_quantity(Normal::new(mean, cov)).unwrap();
        model.cleanup();
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
