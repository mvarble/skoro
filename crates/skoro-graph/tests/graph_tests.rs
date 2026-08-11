use skoro_backend::{CotangentFrame, NdArrayBackend, Point};
use skoro_core::kind::Continuous;
use skoro_core::sampler::Sampler;
use skoro_graph::{
    graph::Graph,
    node::{Node, ObservedNode},
};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};
use skoro_riemann::real::Real;

/// A dummy backend to use for testing
type B = NdArrayBackend<f32, i32>;

#[derive(Debug, Clone)]
struct DummySampler;

impl Sampler<B, Continuous> for DummySampler {
    fn sample(&self, _batch_size: usize, _parameters: Vec<Input<B>>) -> Point<B> {
        unimplemented!()
    }
}

#[derive(Debug, Clone)]
struct DummyOperation;

impl DiffOperation<B> for DummyOperation {
    type VjpState = ();
    type HvpState = ();
    fn forward(&self, _: Vec<Input<B>>, _: DiffModeState<B>) -> DiffOperationForward<B, (), ()> {
        unimplemented!()
    }
    fn vjp(&self, _: &(), _: CotangentFrame<B>) -> Vec<Option<CotangentFrame<B>>> {
        unimplemented!()
    }
    fn hvp(&self, _: &(), _: &(), _: CotangentFrame<B>) -> Vec<Option<CotangentFrame<B>>> {
        unimplemented!()
    }
}

#[test]
fn graph_topology() {
    let mut graph = Graph::<B>::new();

    // Add some base parameters/constants
    let c1 = graph.add_parameter(Real);
    let c2 = graph.add_constant(Real, &2.0);

    // Add a stochastic node with c1 and c2 as parents
    let stoch_id = graph
        .add_stochastic(Real, DummySampler, vec![c1, c2])
        .unwrap();

    // Verify it added both nodes
    assert!(graph.contains(stoch_id));

    // Verify the log density has the stochastic variable as a parent
    if let Node::Stochastic(stoch) = graph.node(stoch_id).unwrap() {
        assert!(stoch.parent_ids.contains(&c1));
        assert!(stoch.parent_ids.contains(&c2));
    } else {
        panic!();
    }

    // Add a mapped node depending on the stochastic output
    let m1 = graph
        .add_mapped(Real, DummyOperation, vec![stoch_id])
        .unwrap();
    assert!(graph.contains(m1));
}

#[test]
fn add_stochastic_dangling_parent_errors() {
    use skoro_core::error::SkoroError;
    use skoro_core::id::IdGenerator;

    let mut graph = Graph::<B>::new();
    // an Id that was never issued to this graph
    let bogus = IdGenerator::new().issue_id();
    assert!(matches!(
        graph.add_stochastic(Real, DummySampler, vec![bogus]),
        Err(SkoroError::DanglingParent { .. })
    ));
}

#[test]
fn remove_with_children_errors() {
    use skoro_core::error::SkoroError;

    let mut graph = Graph::<B>::new();
    let p1 = graph.add_parameter(Real);
    let _m1 = graph.add_mapped(Real, DummyOperation, vec![p1]).unwrap();

    // This should error because m1 depends on p1
    assert!(matches!(
        graph.remove(&[p1]),
        Err(SkoroError::NodeStillReferenced { .. })
    ));
}

#[test]
fn observe() {
    let mut graph = Graph::<B>::new();
    let stoch_id = graph.add_stochastic(Real, DummySampler, vec![]).unwrap();

    assert!(matches!(graph.node(stoch_id).unwrap(), Node::Stochastic(_)));

    graph.observe::<_, Real, _>(stoch_id, &5.0).unwrap();

    match graph.node(stoch_id).unwrap() {
        Node::Observed(ObservedNode { value, .. }) => {
            let d = value
                .clone()
                .unwrap_float()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(d.len(), 1);
            assert_eq!(d[0], 5.0);
        }
        _ => panic!("observation did not work"),
    }
}
