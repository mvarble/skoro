//! Layout mechanics: block placement, the flatten/unflatten round trip, zero-filling,
//! identity seeding, and the error surface.
//!
//! The dimension-versus-storage rule these rest on is exercised where the two actually
//! diverge, on `SO3`/`SE3`, in `skoro-se3/tests/common/mod.rs` — no manifold reachable
//! from this crate has `dimension() != num_components()`.

use std::collections::HashMap;

use skoro_backend::{NdArrayBackend, Tensor, TensorData};
use skoro_core::{error::SkoroError, id::Id};
use skoro_graph::{graph::Graph, tangent_layout::TangentLayout};
use skoro_riemann::{euclidean::EuclideanD, real::Real};

type B = NdArrayBackend<f64, i32>;

/// A graph with a `Real`, a `EuclideanD(3)` and a `EuclideanD(2)` parameter.
fn fixture() -> (Graph<B>, [Id; 3]) {
    let mut graph = Graph::<B>::new();
    let a = graph.add_parameter(Real);
    let b = graph.add_parameter(EuclideanD(3));
    let c = graph.add_parameter(EuclideanD(2));
    (graph, [a, b, c])
}

fn vector(data: Vec<f64>, dim: usize) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(data, [1, dim]), &Default::default())
}

#[test]
fn blocks_are_contiguous_and_in_caller_order() {
    let (graph, [a, b, c]) = fixture();
    let layout = TangentLayout::from_ids(&graph, &[b, a, c]).unwrap();

    assert_eq!(layout.total(), 6);
    assert_eq!(layout.ids().collect::<Vec<_>>(), vec![b, a, c]);

    let blocks = layout.blocks();
    assert_eq!((blocks[0].offset, blocks[0].dimension), (0, 3));
    assert_eq!((blocks[1].offset, blocks[1].dimension), (3, 1));
    assert_eq!((blocks[2].offset, blocks[2].dimension), (4, 2));

    assert!(layout.contains(a));
    assert_eq!(layout.block(a).unwrap().offset, 3);
}

#[test]
fn from_ids_sorted_ignores_caller_order() {
    let (graph, [a, b, c]) = fixture();
    let sorted = TangentLayout::from_ids_sorted(&graph, &[c, a, b]).unwrap();
    assert_eq!(sorted.ids().collect::<Vec<_>>(), vec![a, b, c]);
}

#[test]
fn round_trip_preserves_every_block() {
    let (graph, [a, b, c]) = fixture();
    let layout = TangentLayout::from_ids(&graph, &[a, b, c]).unwrap();

    let mut map = HashMap::new();
    map.insert(a, vector(vec![1.0], 1));
    map.insert(b, vector(vec![2.0, 3.0, 4.0], 3));
    map.insert(c, vector(vec![5.0, 6.0], 2));

    let flat = layout
        .flatten_vectors(&map, 1, &Default::default())
        .unwrap();
    assert_eq!(flat.dims(), [1, 6]);
    assert_eq!(
        flat.clone().into_data().to_vec::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    let back = layout.unflatten_vector(flat).unwrap();
    for id in [a, b, c] {
        assert_eq!(
            back[&id].clone().into_data().to_vec::<f64>().unwrap(),
            map[&id].clone().into_data().to_vec::<f64>().unwrap(),
            "block {id:?} did not survive the round trip"
        );
    }
}

#[test]
fn missing_ids_flatten_to_zeros() {
    // A VJP legitimately returns no cotangent for a leaf that does not reach the
    // target. The right answer is a zero column, not an error.
    let (graph, [a, b, c]) = fixture();
    let layout = TangentLayout::from_ids(&graph, &[a, b, c]).unwrap();

    let mut map = HashMap::new();
    map.insert(b, vector(vec![7.0, 8.0, 9.0], 3));

    let flat = layout
        .flatten_vectors(&map, 1, &Default::default())
        .unwrap();
    assert_eq!(
        flat.into_data().to_vec::<f64>().unwrap(),
        vec![0.0, 7.0, 8.0, 9.0, 0.0, 0.0]
    );
}

#[test]
fn frames_round_trip() {
    let (graph, [a, b, _]) = fixture();
    let layout = TangentLayout::from_ids(&graph, &[a, b]).unwrap();

    let frames = 2;
    let mut map = HashMap::new();
    map.insert(
        a,
        Tensor::<B, 3>::from_data(
            TensorData::new(vec![1.0, 2.0], [1, 1, 2]),
            &Default::default(),
        ),
    );
    map.insert(b, Tensor::<B, 3>::ones([1, 3, 2], &Default::default()));

    let flat = layout
        .flatten_frames(&map, 1, frames, &Default::default())
        .unwrap();
    assert_eq!(flat.dims(), [1, 4, 2]);

    let back = layout.unflatten_frame(flat).unwrap();
    assert_eq!(back[&a].dims(), [1, 1, 2]);
    assert_eq!(back[&b].dims(), [1, 3, 2]);
    assert_eq!(
        back[&a].clone().into_data().to_vec::<f64>().unwrap(),
        vec![1.0, 2.0]
    );
}

#[test]
fn identity_frames_are_the_rows_of_the_identity() {
    let (graph, [a, b, _]) = fixture();
    let layout = TangentLayout::from_ids(&graph, &[a, b]).unwrap();
    let frames = layout.identity_frames::<B>(1, &Default::default());

    // `a` is 1-dimensional at offset 0 -> the first row of I_4.
    assert_eq!(frames[&a].dims(), [1, 1, 4]);
    assert_eq!(
        frames[&a].clone().into_data().to_vec::<f64>().unwrap(),
        vec![1.0, 0.0, 0.0, 0.0]
    );

    // `b` is 3-dimensional at offset 1 -> rows 1..4 of I_4.
    assert_eq!(frames[&b].dims(), [1, 3, 4]);
    assert_eq!(
        frames[&b].clone().into_data().to_vec::<f64>().unwrap(),
        vec![
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]
    );
}

#[test]
fn identity_frames_repeat_across_the_batch() {
    let (graph, [a, _, _]) = fixture();
    let layout = TangentLayout::from_ids(&graph, &[a]).unwrap();
    let frames = layout.identity_frames::<B>(3, &Default::default());
    assert_eq!(frames[&a].dims(), [3, 1, 1]);
}

#[test]
fn duplicate_ids_are_rejected() {
    let (graph, [a, b, _]) = fixture();
    assert!(matches!(
        TangentLayout::from_ids(&graph, &[a, b, a]),
        Err(SkoroError::DuplicateLayoutId { id }) if id == a
    ));
}

#[test]
fn unknown_ids_are_rejected() {
    let (graph, [a, _, _]) = fixture();
    let mut other = Graph::<B>::new();
    let stranger = other.add_parameter(Real);
    // Distinct graphs issue ids from zero, so pick one this graph cannot hold.
    let unknown = if graph.contains(stranger) {
        let mut g = Graph::<B>::new();
        for _ in 0..64 {
            g.add_parameter(Real);
        }
        g.add_parameter(Real)
    } else {
        stranger
    };
    assert!(matches!(
        TangentLayout::from_ids(&graph, &[a, unknown]),
        Err(SkoroError::NodeNotFound { .. })
    ));
}

#[test]
fn discrete_quantities_are_rejected() {
    use skoro_core::discrete::Integer;
    let mut graph = Graph::<B>::new();
    let n = graph.add_parameter(Integer);
    assert!(matches!(
        TangentLayout::from_ids(&graph, &[n]),
        Err(SkoroError::NotContinuous { id }) if id == n
    ));
}

#[test]
fn a_wrong_width_is_rejected_on_unflatten() {
    let (graph, [a, b, c]) = fixture();
    let layout = TangentLayout::from_ids(&graph, &[a, b, c]).unwrap();
    let wrong = Tensor::<B, 2>::zeros([1, 5], &Default::default());
    assert!(matches!(
        layout.unflatten_vector(wrong),
        Err(SkoroError::LayoutDimensionMismatch {
            expected: 6,
            found: 5
        })
    ));
}

#[test]
fn same_shape_compares_dimensions_not_ids() {
    let (graph, [a, b, _]) = fixture();
    let mut graph = graph;
    let a2 = graph.add_parameter(Real);
    let b2 = graph.add_parameter(EuclideanD(3));

    let state = TangentLayout::from_ids(&graph, &[a, b]).unwrap();
    let state_next = TangentLayout::from_ids(&graph, &[a2, b2]).unwrap();
    assert!(state.same_shape(&state_next));

    // Same total width, different block structure -> not the same shape.
    let regrouped = TangentLayout::from_ids(&graph, &[b, a]).unwrap();
    assert_eq!(regrouped.total(), state.total());
    assert!(!state.same_shape(&regrouped));
}

#[test]
fn an_empty_layout_is_legal() {
    let (graph, _) = fixture();
    let layout = TangentLayout::from_ids(&graph, &[]).unwrap();
    assert_eq!(layout.total(), 0);
    assert_eq!(layout.blocks().len(), 0);
}
