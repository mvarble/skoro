use skoro_backend::NdArrayBackend;
use skoro_core::discrete::{Boolean, Categorical};
use skoro_graph::{graph::Graph, node::Node};

type B = NdArrayBackend<f32, i32>;

#[test]
fn add_discrete_constant_and_parameter() {
    let mut graph = Graph::<B>::new();

    // A categorical with 5 categories, fixed at index 3.
    let c = graph.add_constant(Categorical(5), &3usize);
    // A free boolean parameter.
    let p = graph.add_parameter(Boolean);

    assert!(graph.contains(c));
    assert!(graph.contains(p));

    // Both nodes report as discrete.
    assert!(graph.node(c).unwrap().is_discrete());
    assert!(graph.node(p).unwrap().is_discrete());

    // The stored variants are the unified constant/parameter ones.
    assert!(matches!(graph.node(c).unwrap(), Node::Constant(_)));
    assert!(matches!(graph.node(p).unwrap(), Node::Parameter(_)));
}

#[test]
fn discrete_constant_surfaces_in_forward_output() {
    let mut graph = Graph::<B>::new();
    let c = graph.add_constant(Categorical(4), &2usize);

    // Generating the model should surface the discrete value (and not panic in
    // the differentiation-free forward path).
    let out = graph.generate(1).unwrap();
    let (value, _) = out
        .constants
        .get(&c)
        .expect("discrete constant should be surfaced in forward output");
    let data = value
        .clone()
        .into_int()
        .into_data()
        .to_vec::<i32>()
        .unwrap();
    assert_eq!(data, vec![2]);
}
