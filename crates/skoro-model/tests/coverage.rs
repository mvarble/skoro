//! Whether the readers that assemble an objective cover the whole log-density.
//!
//! `information_blocks` and `residual_layout` skip a term that bound nothing under the role they
//! want, and neither the blocks nor the layout says which terms those were. `terms_missing` names
//! them and `require_role` refuses over them; the readers themselves are unchanged, which is what
//! most of these tests pin.

use skoro_backend::{Flex, Point, Tensor, TensorData};
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
    kind::Continuous,
};
use skoro_model::{
    ModelBuilderOps, ModelReaderOps,
    coverage::{require_role, terms_missing},
    distribution::{Distribution, Factor, Sampler},
    hash_map_builders::points,
    information::{information_blocks, information_root},
    least_squares::residual_layout,
    map::register_map1,
    model::ModelClient,
    q::Q,
    roles::{FISHER_ROOT, RESIDUAL},
};
use skoro_operation::{Input, ValueMap, ops::LiteralScalarMultiplication};
use skoro_riemann::{
    euclidean::EuclideanD,
    matrix::{EuclideanMatrixD, MatrixD},
    real::Real,
};

type B = Flex;
skoro_backend::f64_test_binary!(Flex);

fn row(value: f64) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(vec![value], [1, 1]), &Default::default())
}

/// A scalar `θ`, a point for it, and its free layout: the setting for every model below.
fn theta(model: &ModelClient<B>) -> (Q<B, EuclideanD>, ValueMap<B>) {
    let theta = model.add_parameter(EuclideanD(1));
    let values = points().with_quantity(&theta, row(0.3)).build();
    (theta, values)
}

/// A `1 × 1` root: `C = [c]`, so a term over `θ` has information `c²`.
fn root(model: &ModelClient<B>, c: f64) -> Q<B, EuclideanMatrixD> {
    model.add_constant(EuclideanMatrixD(1, 1), &MatrixD::new(1, 1, &[c]))
}

/// A term over `θ` binding `bindings`, added to the log-density.
fn term(
    model: &ModelClient<B>,
    theta: &Q<B, EuclideanD>,
    bindings: impl FnOnce(Factor<B>) -> Factor<B>,
) -> Id {
    let density = register_map1(Real, LiteralScalarMultiplication::new(-0.5), theta).unwrap();
    model
        .add_factor(bindings(Factor::new(density)), &[theta.id()])
        .unwrap()
        .id()
}

// --------------------------------------------------------------------------------------------
// The gap list, and its agreement with what the reader actually assembled.
// --------------------------------------------------------------------------------------------

/// The gaps are exactly the terms `information_blocks` left out.
///
/// Asserted as a set complement rather than against a literal, because that equality is the real
/// contract: both functions filter on `in_density` and decide on the arity of the same `bound`
/// call, and if the two ever drift apart this is what catches it.
#[test]
fn the_gaps_are_the_complement_of_the_blocks() {
    let model = ModelClient::<B>::default();
    let (theta, values) = theta(&model);
    let c1 = root(&model, 2.0);
    let c2 = root(&model, 0.5);

    let first = term(&model, &theta, |f| f.bind(FISHER_ROOT, &c1));
    let middle = term(&model, &theta, |f| f);
    let last = term(&model, &theta, |f| f.bind(FISHER_ROOT, &c2));

    assert_eq!(terms_missing(&model, FISHER_ROOT), vec![middle]);

    let free = model.tangent_layout(&[theta.id()]).unwrap();
    let assembled: Vec<Id> = information_blocks(&model, &values, &free)
        .unwrap()
        .iter()
        .map(|block| block.factor)
        .collect();
    assert_eq!(assembled, vec![first, last]);

    // Every scored term is either assembled or reported, and never both.
    let mut accounted = assembled;
    accounted.extend(terms_missing(&model, FISHER_ROOT));
    accounted.sort();
    let mut scored: Vec<Id> = model
        .factors()
        .iter()
        .filter(|factor| factor.in_density)
        .map(|factor| factor.density)
        .collect();
    scored.sort();
    assert_eq!(accounted, scored);
}

/// The guard refuses that same model, and the reader still does not.
#[test]
fn the_guard_refuses_what_the_reader_skips() {
    let model = ModelClient::<B>::default();
    let (theta, values) = theta(&model);
    let c = root(&model, 2.0);
    term(&model, &theta, |f| f.bind(FISHER_ROOT, &c));
    let middle = term(&model, &theta, |f| f);

    assert_eq!(
        require_role(&model, FISHER_ROOT),
        Err(SkoroError::MissingBinding {
            id: middle,
            role: FISHER_ROOT,
        })
    );

    // The permissive path is untouched: it still assembles the rooted term alone.
    let free = model.tangent_layout(&[theta.id()]).unwrap();
    let a = information_root(&model, &values, &free).unwrap();
    assert_eq!(a.dims(), [1, 1, 1]);
}

/// The message names the term and the role, since that is all the caller gets.
#[test]
fn the_error_displays_the_role() {
    let model = ModelClient::<B>::default();
    let (theta, _) = theta(&model);
    term(&model, &theta, |f| f);

    let message = require_role(&model, FISHER_ROOT).unwrap_err().to_string();
    assert!(
        message.contains("fisher_root"),
        "the role must be named: {message}"
    );
}

/// A fully rooted model has no gaps and passes the guard.
#[test]
fn a_fully_rooted_model_has_no_gaps() {
    let model = ModelClient::<B>::default();
    let (theta, values) = theta(&model);
    let c1 = root(&model, 2.0);
    let c2 = root(&model, 0.5);
    term(&model, &theta, |f| f.bind(FISHER_ROOT, &c1));
    term(&model, &theta, |f| f.bind(FISHER_ROOT, &c2));

    assert_eq!(terms_missing(&model, FISHER_ROOT), Vec::<Id>::new());
    assert_eq!(require_role(&model, FISHER_ROOT), Ok(()));

    let free = model.tangent_layout(&[theta.id()]).unwrap();
    assert_eq!(
        information_root(&model, &values, &free).unwrap().dims(),
        [1, 2, 1],
        "both terms contributed a row"
    );
}

/// An empty model has no gaps: there is no term to be missing anything.
///
/// `require_role` passing here is not the same claim as `information_root` succeeding — that
/// still fails, because a root of no blocks has no rows. The guard's contract is about coverage,
/// not about there being anything to cover.
#[test]
fn an_empty_model_has_no_gaps() {
    let model = ModelClient::<B>::default();
    let (theta, values) = theta(&model);

    assert_eq!(terms_missing(&model, FISHER_ROOT), Vec::<Id>::new());
    assert_eq!(require_role(&model, FISHER_ROOT), Ok(()));

    let free = model.tangent_layout(&[theta.id()]).unwrap();
    assert!(information_root(&model, &values, &free).is_err());
}

// --------------------------------------------------------------------------------------------
// Proposal-only terms.
// --------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct NeverSampler;

impl Sampler<B, Continuous> for NeverSampler {
    fn sample(&self, _: usize, _: Vec<Input<B>>) -> Point<B> {
        unimplemented!("these tests never generate")
    }
}

/// `log p(x) = 2x`, over no parents, binding nothing — a bespoke distribution of the kind a
/// non-Gaussian model must write, and the case that makes the gap reachable.
struct Bespoke;

impl Distribution<B, Continuous> for Bespoke {
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
        Ok(Factor::new(register_map1(
            Real,
            LiteralScalarMultiplication::new(2.0),
            x,
        )?))
    }
}

/// The same, binding a root it was handed — a distribution can only bind nodes it holds.
struct BespokeRooted(Q<B, EuclideanMatrixD>);

impl Distribution<B, Continuous> for BespokeRooted {
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
        let density = register_map1(Real, LiteralScalarMultiplication::new(2.0), x)?;
        Ok(Factor::new(density).bind(FISHER_ROOT, &self.0))
    }
}

/// A proposal-only term is no part of `p`, so it is neither a gap nor something to satisfy —
/// whether or not it bound a root.
///
/// The rootless half is the one that matters: without the `in_density` filter it would be
/// reported, and a caller would refuse a model that is in fact fully covered.
#[test]
fn a_proposal_only_term_is_never_a_gap() {
    let model = ModelClient::<B>::default();
    let (theta, _) = theta(&model);
    let c = root(&model, 2.0);
    term(&model, &theta, |f| f.bind(FISHER_ROOT, &c));

    model.add_proposal(Bespoke).unwrap();
    model.add_proposal(BespokeRooted(c.clone())).unwrap();

    assert_eq!(terms_missing(&model, FISHER_ROOT), Vec::<Id>::new());
    assert_eq!(require_role(&model, FISHER_ROOT), Ok(()));
}

/// A bespoke distribution added to the log-density *is* reported.
///
/// This is the case a future Fisher-scoring loop must not silently mis-solve: the gradient it
/// takes covers this term, and the information it assembles does not.
#[test]
fn a_bespoke_scored_term_is_reported() {
    let model = ModelClient::<B>::default();
    let (theta, _) = theta(&model);
    let c = root(&model, 2.0);
    let rooted = term(&model, &theta, |f| f.bind(FISHER_ROOT, &c));

    let x = model.add_quantity(Bespoke).unwrap();
    let bespoke = model.log_density_of(&x).unwrap();

    assert_eq!(terms_missing(&model, FISHER_ROOT), vec![bespoke]);
    assert_eq!(
        require_role(&model, FISHER_ROOT),
        Err(SkoroError::MissingBinding {
            id: bespoke,
            role: FISHER_ROOT,
        })
    );
    assert_ne!(bespoke, rooted);
}

// --------------------------------------------------------------------------------------------
// The same question for another role.
// --------------------------------------------------------------------------------------------

/// The check is role-general, not Fisher-specific: `residual_layout` skips a term with no
/// `RESIDUAL` exactly as `information_blocks` skips one with no root, and the same call finds it.
#[test]
fn the_check_is_role_general() {
    let model = ModelClient::<B>::default();
    let (theta, _) = theta(&model);
    // A mapped node, as a real residual always is: `Normal` builds `r = L⁻¹(x ⊖ μ)`.
    let residual =
        register_map1(EuclideanD(1), LiteralScalarMultiplication::new(2.0), &theta).unwrap();
    let with = term(&model, &theta, |f| f.bind(RESIDUAL, &residual));
    let without = term(&model, &theta, |f| f);

    assert_eq!(terms_missing(&model, RESIDUAL), vec![without]);
    assert_eq!(
        require_role(&model, RESIDUAL),
        Err(SkoroError::MissingBinding {
            id: without,
            role: RESIDUAL,
        })
    );

    // And the reader is still permissive, laying out the one residual it found.
    assert_eq!(residual_layout(&model).unwrap().total(), 1);
    assert_eq!(terms_missing(&model, FISHER_ROOT), vec![with, without]);
}
