use skoro_core::error::SkoroResult;
use skoro_model::ModelBuilderOps;
use std::collections::HashMap;

use skoro_backend::{CotangentFrame, NdArrayBackend, Point, TensorData};
use skoro_core::{
    discrete::{Boolean, Categorical},
    id::Id,
    kind::Continuous,
    sampler::Sampler,
};
use skoro_graph::node::{Node, ObservedNode};
use skoro_model::{distribution::Distribution, map::register_map1, model::ModelClient, q::Q};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};
use skoro_riemann::real::Real;

type B = NdArrayBackend<f32, i32>;

/// A `DiffOperation` whose only purpose is to let us construct the density of a `Stochastic` node.
#[derive(Debug, Clone)]
struct DummyOperation;

impl DiffOperation<B> for DummyOperation {
    type VjpState = ();
    type HvpState = ();

    fn forward(
        &self,
        points: Vec<Input<B>>,
        _: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let points: Vec<Point<B>> = points.into_iter().map(Input::into_float).collect();
        DiffOperationForward::forward_only(2.0 * points.into_iter().next().unwrap())
    }

    fn vjp(&self, _: &Self::VjpState, _: CotangentFrame<B>) -> Vec<Option<CotangentFrame<B>>> {
        unimplemented!()
    }

    fn hvp(
        &self,
        _: &Self::VjpState,
        _: &Self::HvpState,
        _: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        unimplemented!()
    }
}

/// A `Distribution` whose only purpose is to let us construct a `Stochastic` node.
#[derive(Debug, Clone)]
struct DummyDistribution;

impl Sampler<B, Continuous> for DummyDistribution {
    fn sample(&self, _: usize, _: Vec<Input<B>>) -> Point<B> {
        unimplemented!()
    }
}

impl Distribution<B, Continuous> for DummyDistribution {
    type Qty = Real;
    type Sampler = Self;
    fn quantity_type(&self) -> Real {
        Real
    }
    fn parent_ids(&self) -> &[Id] {
        &[]
    }
    fn sampler(&self) -> Self::Sampler {
        self.clone()
    }
    fn log_density(&self, x: &Q<B, Real>) -> SkoroResult<Q<B, Real>> {
        register_map1(Real, DummyOperation, x)
    }
}

#[test]
fn construct_and_drop_releases_server() {
    let client = ModelClient::<B>::new(Default::default());
    let saved_id: Id = client.server_id;

    let clone = client.clone();
    drop(clone);
    // Original still alive — `from_id` must succeed.
    assert!(ModelClient::<B>::from_id(saved_id).is_some());

    drop(client);
    // All strong refs gone — `from_id` must now fail.
    assert!(ModelClient::<B>::from_id(saved_id).is_none());
}

#[test]
fn add_constant_and_parameter_round_trip() {
    let client = ModelClient::<B>::new(Default::default());

    let c: Q<B, Real> = client.add_constant(Real, &1.0);
    let p: Q<B, Real> = client.add_parameter(Real);

    assert_ne!(c.id(), p.id());
}

fn assert_observed(client: &ModelClient<B>, id: Id, expected: f32) {
    client.server_call(|server| match server.graph().node(id).unwrap() {
        Node::Observed(ObservedNode { value, .. }) => {
            let d = value
                .as_unbatched()
                .expect("observe stores an unbatched value")
                .clone()
                .unwrap_float()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(d, vec![expected]);
        }
        _ => panic!("expected Observed node after observe()"),
    })
}

#[test]
fn q_observe_transitions_node_to_observed() {
    let client = ModelClient::<B>::new(Default::default());
    let q = client.add_quantity(DummyDistribution).unwrap();
    // `Q::observe` — compiler infers `M` from `self: &Q<B, M>`.
    q.observe(&5.0).unwrap();
    assert_observed(&client, q.id(), 5.0);
    q.observe(&10.0).unwrap();
    assert_observed(&client, q.id(), 10.0);
}

#[test]
fn q_unobserve_returns_node_to_stochastic() {
    let client = ModelClient::<B>::new(Default::default());
    let q = client.add_quantity(DummyDistribution).unwrap();
    q.observe(&5.0).unwrap();
    assert_observed(&client, q.id(), 5.0);

    q.unobserve().unwrap();
    client.server_call(|server| {
        assert!(
            matches!(server.graph().node(q.id()), Some(Node::Stochastic(_))),
            "unobserve must return the node to Stochastic"
        );
    });

    // Unobserving an already-stochastic quantity is a no-op, not a panic.
    q.unobserve().unwrap();
    client.server_call(|server| {
        assert!(matches!(
            server.graph().node(q.id()),
            Some(Node::Stochastic(_))
        ));
    });
}

#[test]
fn discrete_quantities_coexist_with_continuous() {
    let client = ModelClient::<B>::new(Default::default());

    // Discrete leaves (non-differentiable) alongside a continuous parameter.
    let _cat = client.add_constant(Categorical(3), &1usize);
    let _flag = client.add_parameter(Boolean);
    let theta: Q<B, Real> = client.add_parameter(Real);

    // A continuous mapped quantity over the continuous parameter (y = 2 * theta),
    // built through the operator sugar (which has a real VJP).
    let y = (theta.clone() * 2.0f32).unwrap();

    // First-order evaluation of the continuous sub-graph must still work and
    // produce a gradient for the continuous parameter, undisturbed by the
    // presence of discrete nodes in the model.
    let mut points = HashMap::new();
    points.insert(
        theta.id(),
        Input::float(Point::<B>::from_data(
            TensorData::new(vec![4.0f32], [1, 1]),
            &Default::default(),
        )),
    );

    let eval = y.evaluate_ord1(&points).unwrap();
    let value = eval.value.into_data().to_vec::<f32>().unwrap();
    assert_eq!(value, vec![8.0]); // 2 * 4

    let grad = eval.cotangents[&theta.id()]
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(grad, vec![2.0]); // d(2*theta)/d(theta)
}
