//! Summing log-likelihood terms with signs, in one compiled pass.
//!
//! `log_weight = Σ in_density − Σ in_proposal` used to be two calls to `log_density_terms`,
//! each compiling its own program over its own term set. The two sets overlap almost entirely
//! -- a term in both cancels to within rounding -- so the second compile walked most of the
//! same graph to produce numbers that were about to be subtracted away.

mod common;

use common::{B, ConstantSampler, SumOperation, into_vec, points};
use skoro_core::error::SkoroError;
use skoro_core::id::Id;
use skoro_graph::graph::Graph;
use skoro_graph::operation::DiffGraphOperation;
use skoro_riemann::real::Real;

/// Three independent density terms over one parameter each, so any subset sums cleanly.
struct Terms {
    graph: Graph<B>,
    p: [Id; 3],
    term: [Id; 3],
}

impl Terms {
    fn new() -> Self {
        let mut graph = Graph::<B>::new();
        let mut p = Vec::new();
        let mut term = Vec::new();
        for _ in 0..3 {
            let z = graph
                .add_stochastic(Real, ConstantSampler(0.0), vec![])
                .unwrap();
            let param = graph.add_parameter(Real);
            let density = graph
                .add_mapped(Real, DiffGraphOperation(SumOperation), vec![param])
                .unwrap();
            graph.mark_log_density(density, z).unwrap();
            p.push(param);
            term.push(density);
        }
        Self {
            graph,
            p: p.try_into().unwrap(),
            term: term.try_into().unwrap(),
        }
    }

    fn at(&self, values: [f64; 3]) -> skoro_operation::ValueMap<B> {
        points([
            (self.p[0], values[0]),
            (self.p[1], values[1]),
            (self.p[2], values[2]),
        ])
    }
}

/// The signed sum is the difference of the two one-sided sums.
#[test]
fn a_signed_sum_agrees_with_subtracting_two_sums() {
    let m = Terms::new();
    let pts = m.at([2.0, 3.0, 5.0]);
    let plus = [m.term[0], m.term[1]];
    let minus = [m.term[1], m.term[2]];

    let fused = m
        .graph
        .log_density_terms_signed(&pts, &plus, &minus)
        .unwrap();
    let separate = m.graph.log_density_terms(&pts, &plus).unwrap()
        - m.graph.log_density_terms(&pts, &minus).unwrap();

    assert_eq!(into_vec(fused.clone()), into_vec(separate));
    // (2 + 3) - (3 + 5) = -3
    assert_eq!(into_vec(fused), vec![-3.0]);
}

/// It compiles once, not once per side.
///
/// The whole point of the fusion. Counted rather than inferred: the cache holds one program per
/// distinct target list, so two compiles would leave two entries behind.
#[test]
fn a_signed_sum_compiles_one_program() {
    let m = Terms::new();
    let pts = m.at([2.0, 3.0, 5.0]);
    assert_eq!(m.graph.cached_program_count(), 0);

    m.graph
        .log_density_terms_signed(&pts, &[m.term[0], m.term[1]], &[m.term[1], m.term[2]])
        .unwrap();

    assert_eq!(m.graph.cached_program_count(), 1);
}

/// A term on both sides cancels, and is compiled once.
#[test]
fn a_term_on_both_sides_cancels() {
    let m = Terms::new();
    let pts = m.at([2.0, 3.0, 5.0]);

    let out = m
        .graph
        .log_density_terms_signed(&pts, &[m.term[0]], &[m.term[0]])
        .unwrap();

    assert_eq!(into_vec(out), vec![0.0]);
    assert_eq!(m.graph.cached_program_count(), 1);
}

/// Both sides are validated, and the minus side is not an afterthought.
#[test]
fn a_non_term_on_either_side_is_rejected() {
    let m = Terms::new();
    let pts = m.at([1.0, 1.0, 1.0]);
    let not_a_term = m.p[0];

    for (plus, minus) in [
        (vec![not_a_term], vec![m.term[0]]),
        (vec![m.term[0]], vec![not_a_term]),
    ] {
        let err = m
            .graph
            .log_density_terms_signed(&pts, &plus, &minus)
            .unwrap_err();
        assert!(matches!(err, SkoroError::NotLogLikelihood { id } if id == not_a_term));
    }
}

/// Two empty sides are zero, at the batch the points imply.
#[test]
fn an_empty_signed_sum_is_zero() {
    let m = Terms::new();

    let out = m
        .graph
        .log_density_terms_signed(&m.at([1.0, 1.0, 1.0]), &[], &[])
        .unwrap();

    assert_eq!(into_vec(out), vec![0.0]);
}
