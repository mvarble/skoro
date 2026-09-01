//! Recording a least-squares factor, and what the model checks when you do.
//!
//! The claim "this factor's negative log-density is `½‖r‖² + n`" used to be a trait bound on
//! the distribution. It is a runtime record now, and the trade is worth stating plainly:
//!
//! - what was **lost** is a compile-time guarantee that a *distribution* had a residual;
//! - what was **gained** is a check the bound could never make — that the ids handed over
//!   name the nodes the graph actually holds, and name them in the roles claimed. That is
//!   the failure that used to surface as a `NodeNotFound` at solve time, one layer away from
//!   the mistake.
//!
//! What is still unchecked, and unchecking: that the density really *is* `−½‖r‖² + n`.
//! Nothing can verify that, and the old design did not either.
//!
//! The negative half — that `remove_closure` stays off `ModelClientOps`, so an estimator
//! cannot dismantle the factors it was handed — is a `trybuild` case, since a test that
//! compiles cannot assert that another does not.

use skoro_backend::{NdArrayBackend, Point, TensorData};
use skoro_core::{discrete::Integer, error::SkoroError};
use skoro_model::{
    FactorRecord, ModelBuilderOps, ModelClientOps, hash_map_builders::points, map::register_map1,
    model::ModelClient, q::Q,
};
use skoro_operation::{Input, ops::LiteralScalarMultiplication};
use skoro_riemann::{
    euclidean::{EuclideanD, VectorD},
    real::Real,
};

type B = NdArrayBackend<f64, i32>;

fn point(values: &[f64]) -> Input<B> {
    Input::float(Point::<B>::from_data(
        TensorData::new(values.to_vec(), [1, values.len()]),
        &Default::default(),
    ))
}

/// A real-valued node marked as a free-standing likelihood term, and a residual to go with
/// it. Returns the free variable it scores too, so a caller can evaluate the model.
fn a_factor(model: &ModelClient<B>) -> (Q<B, Real>, Q<B, Real>, Q<B, EuclideanD>) {
    let theta = model.add_parameter(Real);
    let density = register_map1(Real, LiteralScalarMultiplication::new(-0.5), &theta).unwrap();
    model.mark_likelihood(&density).unwrap();
    let residual = model.add_parameter(EuclideanD(2));
    (theta, density, residual)
}

/// The happy path: a recorded factor is readable back, and its residual becomes a row block of `J`
/// in `residual_layout`. A model told about nothing has no rows rather than an error.
mod records {
    use super::*;

    /// The happy path: a recorded factor is readable back, and its residual is a row block of `J`.
    #[test]
    fn a_recorded_factor_becomes_a_row_block() {
        let model = ModelClient::<B>::default();
        let cov = model.add_constant(EuclideanD(1), &VectorD::new(&[1.0]));
        let (_theta, density, residual) = a_factor(&model);

        model
            .record_factor(FactorRecord::new(
                density.id(),
                residual.id(),
                vec![cov.id()],
            ))
            .unwrap();

        let factors = model.factors();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].density, density.id());
        assert_eq!(factors[0].residual, residual.id());
        assert_eq!(factors[0].normalizer_parents, vec![cov.id()]);
        assert_eq!(factors[0].location, None);

        // Sized by manifold dimension, in record order.
        let layout = model.residual_layout().unwrap();
        assert_eq!(layout.total(), 2);
        assert_eq!(layout.ids().collect::<Vec<_>>(), vec![residual.id()]);
    }

    /// A model told about nothing has no rows.
    #[test]
    fn an_unused_model_records_nothing() {
        let model = ModelClient::<B>::default();
        assert!(model.factors().is_empty());
        assert_eq!(model.residual_layout().unwrap().total(), 0);
    }
}

/// The checks the old trait bound could never make, each landing at the call that made the mistake
/// rather than at the next solve: a density that is not a live likelihood term, an id naming
/// nothing, a discrete residual with no tangent space, and removing a node a record still names.
mod rejections {
    use super::*;

    /// The density must be a *live likelihood term*, not merely a real-valued node.
    ///
    /// This is the check that matters most, and the one the trait bound could not make: a record
    /// whose density does not score puts rows in `J` for a term the objective does not contain.
    #[test]
    fn an_unmarked_density_is_refused() {
        let model = ModelClient::<B>::default();
        let theta = model.add_parameter(Real);
        let not_a_density =
            register_map1(Real, LiteralScalarMultiplication::new(-0.5), &theta).unwrap();
        let residual = model.add_parameter(EuclideanD(2));

        assert!(matches!(
            model.record_factor(FactorRecord::new(not_a_density.id(), residual.id(), vec![],)),
            Err(SkoroError::NotLogLikelihood { .. })
        ));
        assert!(model.factors().is_empty());
    }

    /// An id naming nothing is refused rather than stored to fail later.
    #[test]
    fn a_dangling_id_is_refused() {
        let model = ModelClient::<B>::default();
        let (_theta, density, residual) = a_factor(&model);

        // A node that existed and was then taken.
        let gone = model.add_parameter(EuclideanD(2));
        let gone_id = gone.id();
        model.remove(&[gone_id]).unwrap();

        assert!(matches!(
            model.record_factor(FactorRecord::new(gone_id, residual.id(), vec![])),
            Err(SkoroError::NodeNotFound { .. })
        ));
        assert!(matches!(
            model.record_factor(FactorRecord::new(density.id(), gone_id, vec![])),
            Err(SkoroError::NodeNotFound { .. })
        ));
        assert!(matches!(
            model.record_factor(FactorRecord::new(
                density.id(),
                residual.id(),
                vec![gone_id],
            )),
            Err(SkoroError::DanglingParent { .. })
        ));
        assert!(model.factors().is_empty());
    }

    /// A discrete residual has no tangent space, so it can contribute no rows to `J`.
    ///
    /// `residual_layout` would fail on it at solve time; refusing the record puts the error where
    /// the mistake is.
    #[test]
    fn a_discrete_residual_is_refused() {
        let model = ModelClient::<B>::default();
        let (_theta, density, _) = a_factor(&model);
        let counts = model.add_parameter(Integer);

        assert!(matches!(
            model.record_factor(FactorRecord::new(density.id(), counts.id(), vec![])),
            Err(SkoroError::NotContinuous { .. })
        ));
    }

    /// A recorded node cannot be deleted out from under the record.
    ///
    /// Both are childless, so nothing about the *graph* would stop this: a recorded residual is a
    /// leaf by construction, and a marked `ℓ` scores nothing that reads it. Taking either would
    /// drop a term from the objective with no error at all, which is why `remove` refuses.
    #[test]
    fn removing_a_recorded_node_is_refused() {
        let model = ModelClient::<B>::default();
        let (theta, density, residual) = a_factor(&model);
        model
            .record_factor(FactorRecord::new(density.id(), residual.id(), vec![]))
            .unwrap();

        for id in [density.id(), residual.id()] {
            assert!(
                matches!(
                    ModelClientOps::remove(&model, &[id]),
                    Err(SkoroError::NodeStillBound { .. })
                ),
                "recorded node {id:?} must be refused, not silently taken"
            );
        }

        // Refused means refused: the record, the rows of `J`, and the objective are untouched.
        assert_eq!(model.factors().len(), 1);
        assert_eq!(model.residual_layout().unwrap().total(), 2);
        let values = points().with_quantity(&theta, point(&[3.0])).build();
        approx::assert_abs_diff_eq!(
            model
                .log_density(&values)
                .unwrap()
                .into_data()
                .to_vec::<f64>()
                .unwrap()[0],
            -1.5,
            epsilon = 1e-12
        );
    }
}

/// Every negative claim in the design, as a `trybuild` case that must fail to compile -- a test
/// that compiles cannot assert that another does not. The `.stderr` fixtures are expected to need
/// updating when the compiler's wording changes; what must not change is that the cases fail at all.
mod compile_fail {
    /// Every negative claim in the design, as a case that must fail to compile.
    ///
    /// The `.stderr` fixtures are regenerated with `TRYBUILD=overwrite cargo test -p skoro-model
    /// --test residual`. They are expected to need updating when the compiler's wording
    /// changes; what must not change is that the cases fail to compile at all.
    #[test]
    fn remove_closure_cannot_be_reached_from_the_client_tier() {
        trybuild::TestCases::new().compile_fail("tests/compile_fail/*.rs");
    }
}
