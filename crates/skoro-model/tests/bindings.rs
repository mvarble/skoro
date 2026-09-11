//! Bindings: the labelled references a term keeps to the nodes that play a role for it.
//!
//! A distribution declares its bindings once, in the [`Factor`] it returns, and `add_quantity`
//! records them after an `ARGUMENT` binding to the quantity and to each of its parents. A
//! free-standing factor is attached the same way through `add_factor`. What a binding does to the
//! model is make the bound node a reference: the collector keeps it, `remove` refuses to take it
//! from under its term, and `remove_closure` takes the term with it.
//!
//! The negative half — that `remove_closure` stays off `ModelClientOps`, so an estimator
//! cannot dismantle the terms it was handed — is a `trybuild` case, since a test that
//! compiles cannot assert that another does not.

use skoro_backend::{NdArrayBackend, Point, TensorData};
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
    kind::Continuous,
};
use skoro_graph::graph::TermKind;
use skoro_model::{
    ModelBuilderOps, ModelClientOps,
    distribution::{Distribution, Factor, Sampler},
    hash_map_builders::points,
    least_squares::residual_layout,
    map::register_map1,
    model::ModelClient,
    q::Q,
    roles::{ARGUMENT, RESIDUAL},
};
use skoro_operation::{Input, ops::LiteralScalarMultiplication};
use skoro_riemann::{euclidean::EuclideanD, real::Real};

type B = NdArrayBackend<f64, i32>;

fn point(values: &[f64]) -> Input<B> {
    Input::float(Point::<B>::from_data(
        TensorData::new(values.to_vec(), [1, values.len()]),
        &Default::default(),
    ))
}

fn scalar(t: Point<B>) -> f64 {
    t.into_data().to_vec::<f64>().unwrap()[0]
}

#[derive(Debug, Clone)]
struct NeverSampler;

impl Sampler<B, Continuous> for NeverSampler {
    fn sample(&self, _: usize, _: Vec<Input<B>>) -> Point<B> {
        unimplemented!("these tests never generate")
    }
}

/// `log p(x | p₁, p₂) = 2x`, whose factor binds a residual built elsewhere — enough to observe
/// every binding `add_quantity` makes, with no arithmetic to get wrong.
struct BoundDistr {
    residual: Q<B, EuclideanD>,
    parents: [Id; 2],
}

impl Distribution<B, Continuous> for BoundDistr {
    type Qty = Real;
    type Sampler = NeverSampler;

    fn quantity_type(&self) -> Real {
        Real
    }
    fn sampler(&self) -> NeverSampler {
        NeverSampler
    }
    fn parent_ids(&self) -> &[Id] {
        &self.parents
    }
    fn factor(&self, x: &Q<B, Real>) -> SkoroResult<Factor<B>> {
        let density = register_map1(Real, LiteralScalarMultiplication::new(2.0), x)?;
        Ok(Factor::new(density).bind(RESIDUAL, &self.residual))
    }
}

/// `x ~ BoundDistr(p₁, p₂)`, with a mapped residual `3θ` bound to `x`'s term.
struct Bound {
    model: ModelClient<B>,
    x: Q<B, Real>,
    parents: [Q<B, Real>; 2],
    residual: Q<B, EuclideanD>,
    density: Id,
}

fn bound_model() -> Bound {
    let model = ModelClient::<B>::default();
    let parents = [model.add_parameter(Real), model.add_parameter(Real)];
    let theta = model.add_parameter(EuclideanD(2));
    let residual =
        register_map1(EuclideanD(2), LiteralScalarMultiplication::new(3.0), &theta).unwrap();
    let x = model
        .add_quantity(BoundDistr {
            residual: residual.clone(),
            parents: [parents[0].id(), parents[1].id()],
        })
        .unwrap();
    let density = model.log_density_of(&x).unwrap();
    Bound {
        model,
        x,
        parents,
        residual,
        density,
    }
}

fn contains(model: &ModelClient<B>, id: Id) -> bool {
    model.server_call(|server| server.graph().contains(id))
}

/// What `add_quantity` binds, and what those bindings do to collection, removal and demotion.
mod through_add_quantity {
    use super::*;

    #[test]
    fn add_quantity_binds_the_arguments() {
        let bound = bound_model();
        assert_eq!(
            bound.model.bound(bound.density, ARGUMENT),
            vec![bound.x.id(), bound.parents[0].id(), bound.parents[1].id()]
        );

        let factors = bound.model.factors();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].density, bound.density);
        assert_eq!(factors[0].scored, Some(bound.x.id()));
        assert_eq!(factors[0].kind, TermKind::Proposal);
        assert_eq!(
            factors[0].arguments,
            bound.model.bound(bound.density, ARGUMENT)
        );
    }

    #[test]
    fn add_quantity_binds_what_the_factor_declared() {
        let bound = bound_model();
        assert_eq!(
            bound.model.bound(bound.density, RESIDUAL),
            vec![bound.residual.id()]
        );
        // The declared roles follow the arguments.
        let roles: Vec<_> = bound
            .model
            .bindings_of(bound.density)
            .into_iter()
            .map(|(role, _)| role)
            .collect();
        assert_eq!(roles, vec![ARGUMENT, ARGUMENT, ARGUMENT, RESIDUAL]);
    }

    #[test]
    fn a_bound_node_survives_cleanup() {
        let bound = bound_model();
        let residual = bound.residual.id();

        // The same shape without a binding, to show the survival below is the binding's doing.
        let theta = bound.model.add_parameter(EuclideanD(2));
        let unbound = register_map1(EuclideanD(2), LiteralScalarMultiplication::new(3.0), &theta)
            .unwrap()
            .id();

        drop(bound.residual);
        bound.model.cleanup();

        assert!(contains(&bound.model, residual), "reachable from its term");
        assert!(!contains(&bound.model, unbound), "reachable from nothing");
    }

    #[test]
    fn remove_refuses_a_bound_node() {
        let bound = bound_model();
        let residual = bound.residual.id();
        assert_eq!(
            ModelClientOps::remove(&bound.model, &[residual]).map(|removed| removed.len()),
            Err(SkoroError::NodeStillBound {
                id: residual,
                by: bound.density
            })
        );
        assert!(contains(&bound.model, residual));
        assert_eq!(bound.model.factors().len(), 1);
    }

    #[test]
    fn remove_closure_takes_the_term_with_its_binding() {
        let bound = bound_model();
        // The stochastic step reads its parents, so they need values too.
        let values = points()
            .with_quantity(&bound.x, point(&[0.75]))
            .with_quantity(&bound.parents[0], point(&[0.0]))
            .with_quantity(&bound.parents[1], point(&[0.0]))
            .build();
        assert_eq!(scalar(bound.model.log_density(&values).unwrap()), 1.5);

        bound
            .model
            .remove_closure(&[bound.residual.id()])
            .unwrap();

        assert!(!contains(&bound.model, bound.residual.id()));
        assert!(!contains(&bound.model, bound.density));
        assert!(
            contains(&bound.model, bound.x.id()),
            "the scored quantity is upstream of neither"
        );
        assert!(bound.model.factors().is_empty());
        assert_eq!(scalar(bound.model.log_density(&values).unwrap()), 0.0);
    }

    /// A binding to a quantity of another model is refused, rather than recorded against whatever
    /// node carries the same id here — here, one of `x`'s own parents.
    #[test]
    fn add_quantity_refuses_a_binding_from_another_model() {
        let model = ModelClient::<B>::default();
        let parents = [model.add_parameter(Real), model.add_parameter(Real)];
        let elsewhere = ModelClient::<B>::default();
        let theta = elsewhere.add_parameter(EuclideanD(2));
        let residual =
            register_map1(EuclideanD(2), LiteralScalarMultiplication::new(3.0), &theta).unwrap();
        assert!(contains(&model, residual.id()), "the foreign id collides with a node here");
        let before = model.graph_node_count();

        let result = model.add_quantity(BoundDistr {
            residual,
            parents: [parents[0].id(), parents[1].id()],
        });
        assert!(matches!(
            result,
            Err(SkoroError::ModelMismatch { lhs, rhs })
                if lhs == model.server_id && rhs == elsewhere.server_id
        ));
        assert!(model.factors().is_empty());
        assert_eq!(
            model.graph_node_count(),
            before,
            "the refused quantity and its density are removed, without waiting for a cleanup"
        );
    }

    /// The demoted density and its bound residual leave in the same sweep: the density is no
    /// longer a root once retired, and nothing else reaches the residual once its handle is gone.
    #[test]
    fn demotion_sweeps_the_bindings_with_the_density() {
        let bound = bound_model();
        let residual = bound.residual.id();
        drop(bound.residual);

        bound.model.demote_to_parameter(&bound.x).unwrap();

        assert!(!contains(&bound.model, residual));
        assert!(!contains(&bound.model, bound.density));
        assert!(bound.model.factors().is_empty());
    }
}

/// Free-standing factors attached through `add_factor`, and reading the model back.
mod through_add_factor {
    use super::*;

    fn a_term(model: &ModelClient<B>) -> (Q<B, Real>, Q<B, Real>) {
        let theta = model.add_parameter(Real);
        let term = register_map1(Real, LiteralScalarMultiplication::new(-0.5), &theta).unwrap();
        (theta, term)
    }

    #[test]
    fn add_factor_marks_and_binds() {
        let model = ModelClient::<B>::default();
        let (a, term) = a_term(&model);
        let b = model.add_parameter(Real);
        let r = model.add_parameter(EuclideanD(2));

        let density = model
            .add_factor(
                Factor::new(term.clone()).bind(RESIDUAL, &r),
                &[a.id(), b.id()],
            )
            .unwrap();
        assert_eq!(density.id(), term.id());

        let factors = model.factors();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].density, term.id());
        assert_eq!(factors[0].scored, None);
        assert_eq!(factors[0].kind, TermKind::Weight);
        assert_eq!(factors[0].arguments, vec![a.id(), b.id()]);
        assert_eq!(model.bound(term.id(), RESIDUAL), vec![r.id()]);
        assert_eq!(
            model.bindings_of(term.id()),
            vec![(ARGUMENT, a.id()), (ARGUMENT, b.id()), (RESIDUAL, r.id())]
        );
    }

    #[test]
    fn add_factor_needs_an_argument() {
        let model = ModelClient::<B>::default();
        let (_theta, term) = a_term(&model);

        assert!(matches!(
            model.add_factor(Factor::new(term.clone()), &[]),
            Err(SkoroError::ShapeMismatch {
                expected: 1,
                found: 0
            })
        ));
        // Refused before marking anything.
        assert!(model.factors().is_empty());
        assert!(model.bindings_of(term.id()).is_empty());
    }

    /// A factor whose log-density or bound handles come from another model is refused before
    /// anything is marked, rather than bound to whatever node carries the same id here.
    #[test]
    fn add_factor_refuses_a_handle_from_another_model() {
        let model = ModelClient::<B>::default();
        let (theta, term) = a_term(&model);
        let elsewhere = ModelClient::<B>::default();
        let foreign = elsewhere.add_parameter(EuclideanD(2));
        assert!(contains(&model, foreign.id()), "the foreign id collides with a node here");

        assert!(matches!(
            model.add_factor(Factor::new(term.clone()).bind(RESIDUAL, &foreign), &[theta.id()]),
            Err(SkoroError::ModelMismatch { lhs, rhs })
                if lhs == model.server_id && rhs == elsewhere.server_id
        ));
        let (other_theta, foreign_term) = a_term(&elsewhere);
        assert!(matches!(
            model.add_factor(Factor::new(foreign_term), &[other_theta.id()]),
            Err(SkoroError::ModelMismatch { .. })
        ));

        // Refused before marking or binding anything.
        assert!(model.factors().is_empty());
        assert!(model.bindings_of(term.id()).is_empty());
    }

    #[test]
    fn factors_are_in_term_order() {
        let model = ModelClient::<B>::default();
        let mut densities = Vec::new();
        for _ in 0..3 {
            let (theta, term) = a_term(&model);
            model
                .add_factor(Factor::new(term.clone()), &[theta.id()])
                .unwrap();
            densities.push(term.id());
        }
        assert_eq!(
            model
                .factors()
                .iter()
                .map(|f| f.density)
                .collect::<Vec<_>>(),
            densities
        );
    }

    #[test]
    fn residual_layout_follows_term_order_and_dimension() {
        let model = ModelClient::<B>::default();
        let (a, term_a) = a_term(&model);
        let r2 = model.add_parameter(EuclideanD(2));
        model
            .add_factor(Factor::new(term_a).bind(RESIDUAL, &r2), &[a.id()])
            .unwrap();

        // A term with no residual contributes no rows.
        let (c, term_c) = a_term(&model);
        model.add_factor(Factor::new(term_c), &[c.id()]).unwrap();

        let (b, term_b) = a_term(&model);
        let r3 = model.add_parameter(EuclideanD(3));
        model
            .add_factor(Factor::new(term_b).bind(RESIDUAL, &r3), &[b.id()])
            .unwrap();

        let layout = residual_layout(&model).unwrap();
        assert_eq!(layout.total(), 5);
        assert_eq!(layout.ids().collect::<Vec<_>>(), vec![r2.id(), r3.id()]);
    }

    /// A model told about nothing has no terms and no rows.
    #[test]
    fn an_unused_model_has_no_factors() {
        let model = ModelClient::<B>::default();
        assert!(model.factors().is_empty());
        assert_eq!(residual_layout(&model).unwrap().total(), 0);
    }
}

/// Every negative claim in the design, as a `trybuild` case that must fail to compile -- a test
/// that compiles cannot assert that another does not. The `.stderr` fixtures are expected to need
/// updating when the compiler's wording changes; what must not change is that the cases fail at all.
mod compile_fail {
    /// Every negative claim in the design, as a case that must fail to compile.
    ///
    /// The `.stderr` fixtures are regenerated with `TRYBUILD=overwrite cargo test -p skoro-model
    /// --test bindings`. They are expected to need updating when the compiler's wording
    /// changes; what must not change is that the cases fail to compile at all.
    #[test]
    fn remove_closure_cannot_be_reached_from_the_client_tier() {
        trybuild::TestCases::new().compile_fail("tests/compile_fail/*.rs");
    }
}

/// A distribution whose factor fails after building part of its density from `x`.
struct FailingDistr;

impl Distribution<B, Continuous> for FailingDistr {
    type Qty = Real;
    type Sampler = NeverSampler;

    fn quantity_type(&self) -> Real {
        Real
    }
    fn sampler(&self) -> NeverSampler {
        NeverSampler
    }
    fn parent_ids(&self) -> &[Id] {
        &[]
    }
    fn factor(&self, x: &Q<B, Real>) -> SkoroResult<Factor<B>> {
        let _partial = register_map1(Real, LiteralScalarMultiplication::new(2.0), x)?;
        Err(SkoroError::ShapeMismatch {
            expected: 1,
            found: 0,
        })
    }
}

/// A quantity whose factor fails leaves nothing behind — neither the stochastic node nor the part
/// of its density built before the failure — without waiting for a `cleanup`, and the model
/// still takes quantities afterwards.
#[test]
fn add_quantity_whose_factor_fails_leaves_no_node_behind() {
    let model = ModelClient::<B>::default();
    let theta = model.add_parameter(Real);
    let before = model.graph_node_count();

    assert!(matches!(
        model.add_quantity(FailingDistr),
        Err(SkoroError::ShapeMismatch {
            expected: 1,
            found: 0
        })
    ));
    assert_eq!(model.graph_node_count(), before);
    assert!(model.factors().is_empty());

    let residual = model.add_parameter(EuclideanD(1));
    let _x = model
        .add_quantity(BoundDistr {
            residual,
            parents: [theta.id(), theta.id()],
        })
        .unwrap();
    assert_eq!(model.factors().len(), 1);
}
