//! What a binding is, and what the model checks when one is made.
//!
//! A binding is a labelled reference edge from a term's density to another node, recorded once
//! when a distribution is added (through `Distribution::factor`'s `Factor::bind`), or explicitly
//! via `bind`/`add_factor`. `ModelClientOps::{factors, bound}` read them back; `ModelClientOps`
//! itself refuses to remove a node a binding still names, and `ModelBuilderOps::remove_closure`
//! is the escape hatch that takes a bound node together with whatever binds it.
//!
//! The negative half — that `remove_closure` stays off `ModelClientOps`, so an estimator
//! cannot dismantle the factors it was handed — is a `trybuild` case, since a test that
//! compiles cannot assert that another does not.

use skoro_backend::{NdArrayBackend, Point};
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
    kind::Continuous,
};
use skoro_model::{
    ModelBuilderOps, ModelClientOps,
    distribution::{Distribution, Factor, Sampler},
    hash_map_builders::points,
    map::register_map1,
    model::ModelClient,
    q::Q,
    roles::{ARGUMENT, RESIDUAL},
};
use skoro_operation::{Input, ops::LiteralScalarMultiplication};
use skoro_riemann::{euclidean::EuclideanD, real::Real};

type B = NdArrayBackend<f64, i32>;

/// `bound_model`'s return: the model, `x`, `x`'s parent `p`, and the residual bound to `x`'s
/// term.
type BoundModel = (ModelClient<B>, Q<B, Real>, Q<B, Real>, Q<B, EuclideanD>);

#[derive(Debug, Clone)]
struct BoundSampler;

impl Sampler<B, Continuous> for BoundSampler {
    fn sample(&self, _batch_size: usize, _parameters: Vec<Input<B>>) -> Point<B> {
        unimplemented!("the tests never generate")
    }
}

/// A `Real` conditional over one parent whose factor binds a caller-supplied residual node —
/// enough to observe every binding `add_quantity` makes, with no arithmetic to get wrong.
struct BoundDistr {
    residual: Q<B, EuclideanD>,
    ids: [Id; 1],
}

impl Distribution<B, Continuous> for BoundDistr {
    type Qty = Real;
    type Sampler = BoundSampler;

    fn quantity_type(&self) -> Real {
        Real
    }
    fn sampler(&self) -> BoundSampler {
        BoundSampler
    }
    fn parent_ids(&self) -> &[Id] {
        &self.ids
    }
    fn factor(&self, x: &Q<B, Real>) -> SkoroResult<Factor<B>> {
        let density = register_map1(Real, LiteralScalarMultiplication::new(2.0), x)?;
        Ok(Factor::new(density).bind(RESIDUAL, &self.residual))
    }
}

/// `x ~ BoundDistr` over parent `p`, with `residual` bound to `x`'s term.
fn bound_model() -> BoundModel {
    let model = ModelClient::<B>::default();
    let p = model.add_parameter(Real);
    let residual = model.add_parameter(EuclideanD(1));
    let x = model
        .add_quantity(BoundDistr {
            residual: residual.clone(),
            ids: [p.id()],
        })
        .unwrap();
    (model, x, p, residual)
}

/// Attaching a factor — through `add_quantity`'s call into `Distribution::factor`, or directly
/// through `add_factor` — and what the bindings make readable: the arguments a term was marked
/// over, the roles it bound, and what happens when a caller tries to take a bound node out from
/// under the term that names it.
mod bindings {
    use super::*;

    #[test]
    fn add_quantity_binds_the_arguments() {
        let (model, x, p, _residual) = bound_model();
        let density = model.log_density_of(&x).unwrap();
        assert_eq!(model.bound(density, ARGUMENT), vec![x.id(), p.id()]);
    }

    #[test]
    fn add_quantity_binds_what_the_factor_declared() {
        let (model, x, _p, residual) = bound_model();
        let density = model.log_density_of(&x).unwrap();
        assert_eq!(model.bound(density, RESIDUAL), vec![residual.id()]);
    }

    #[test]
    fn remove_refuses_a_bound_node() {
        let (model, x, _p, residual) = bound_model();
        let density = model.log_density_of(&x).unwrap();
        assert!(matches!(
            model.remove(&[residual.id()]),
            Err(SkoroError::NodeStillBound { id, by }) if id == residual.id() && by == density
        ));
    }

    /// The demoted density and its bound residual leave in the same sweep: the density is no
    /// longer a root once retired, and nothing else references the residual once the test's own
    /// handle is gone. Both land in one `remove` batch, so `NodeStillBound` never fires.
    #[test]
    fn demotion_sweeps_the_bindings_with_the_density() {
        let (model, x, _p, residual) = bound_model();
        let residual_id = residual.id();
        drop(residual);

        model.demote_to_parameter(&x).unwrap();

        assert!(!model.server_call(|s| s.graph().contains(residual_id)));
        assert!(model.factors().is_empty());
    }

    /// `remove_closure` is the escape hatch `remove` refuses to be: naming just the bound
    /// residual still takes the density that binds it, since the closure walks referrers as
    /// well as children. With the only term gone, the model's density evaluates to zero.
    #[test]
    fn remove_closure_takes_the_term_with_its_binding() {
        let (model, x, _p, residual) = bound_model();
        let density = model.log_density_of(&x).unwrap();
        let residual_id = residual.id();

        ModelBuilderOps::remove_closure(&model, &[residual_id]).unwrap();

        assert!(!model.server_call(|s| s.graph().contains(residual_id)));
        assert!(!model.server_call(|s| s.graph().contains(density)));

        approx::assert_abs_diff_eq!(
            model
                .log_density(&points().build())
                .unwrap()
                .into_data()
                .to_vec::<f64>()
                .unwrap()[0],
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn add_factor_marks_and_binds() {
        let model = ModelClient::<B>::default();
        let a = model.add_parameter(Real);
        let b = model.add_parameter(Real);
        let term = register_map1(Real, LiteralScalarMultiplication::new(-0.5), &a).unwrap();
        let r = model.add_parameter(EuclideanD(2));

        let factor = Factor::new(term.clone()).bind(RESIDUAL, &r);
        model.add_factor(factor, &[a.id(), b.id()]).unwrap();

        let info = ModelClientOps::factors(&model);
        assert_eq!(info.len(), 1);
        assert_eq!(info[0].scored, None);
        assert_eq!(info[0].kind, skoro_graph::graph::TermKind::Weight);
        assert_eq!(info[0].arguments, vec![a.id(), b.id()]);
        assert_eq!(
            ModelClientOps::bound(&model, term.id(), RESIDUAL),
            vec![r.id()]
        );
    }

    #[test]
    fn add_factor_needs_an_argument() {
        let model = ModelClient::<B>::default();
        let a = model.add_parameter(Real);
        let term = register_map1(Real, LiteralScalarMultiplication::new(-0.5), &a).unwrap();

        assert!(matches!(
            model.add_factor(Factor::new(term), &[]),
            Err(SkoroError::ShapeMismatch {
                expected: 1,
                found: 0
            })
        ));
    }

    #[test]
    fn factors_are_in_term_order() {
        let model = ModelClient::<B>::default();
        let mut densities = Vec::new();
        for i in 0..3 {
            let a = model.add_parameter(Real);
            let term =
                register_map1(Real, LiteralScalarMultiplication::new(-0.5 - i as f64), &a).unwrap();
            model
                .add_factor(Factor::new(term.clone()), &[a.id()])
                .unwrap();
            densities.push(term.id());
        }
        let info = ModelClientOps::factors(&model);
        assert_eq!(
            info.iter().map(|f| f.density).collect::<Vec<_>>(),
            densities
        );
    }

    #[test]
    fn a_bound_node_survives_cleanup() {
        let model = ModelClient::<B>::default();
        let a = model.add_parameter(Real);
        let term = register_map1(Real, LiteralScalarMultiplication::new(-0.5), &a).unwrap();
        let r = model.add_parameter(EuclideanD(2));
        let r_id = r.id();

        let factor = Factor::new(term).bind(RESIDUAL, &r);
        model.add_factor(factor, &[a.id()]).unwrap();
        drop(r);

        model.cleanup();
        assert!(model.server_call(|s| s.graph().contains(r_id)));
    }

    #[test]
    fn residual_layout_follows_term_order_and_dimension() {
        let model = ModelClient::<B>::default();
        let a = model.add_parameter(Real);
        let term_a = register_map1(Real, LiteralScalarMultiplication::new(-0.5), &a).unwrap();
        let r2 = model.add_parameter(EuclideanD(2));
        model
            .add_factor(Factor::new(term_a).bind(RESIDUAL, &r2), &[a.id()])
            .unwrap();

        let b = model.add_parameter(Real);
        let term_b = register_map1(Real, LiteralScalarMultiplication::new(-0.5), &b).unwrap();
        let r3 = model.add_parameter(EuclideanD(3));
        model
            .add_factor(Factor::new(term_b).bind(RESIDUAL, &r3), &[b.id()])
            .unwrap();

        let layout = skoro_model::least_squares::residual_layout(&model).unwrap();
        assert_eq!(layout.total(), 5);
        assert_eq!(layout.ids().collect::<Vec<_>>(), vec![r2.id(), r3.id()]);
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
