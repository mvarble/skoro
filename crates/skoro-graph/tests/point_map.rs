//! `PointMap`: the type-tagged store of node values, and the manifold operations over it.
//!
//! The map half is ordinary bookkeeping. The manifold half is not: `retract` and
//! `inverse_retract` are where a discrete quantity has to be refused rather than silently
//! treated as a vector, and where a flat manifold's broadcasting `Addition` would otherwise
//! turn a batch disagreement into a plausible wrong answer.

mod common;

use std::collections::HashMap;

use common::*;
use skoro_core::{
    discrete::Integer,
    error::SkoroError,
    id::Id,
    kind::{Continuous, Discrete},
};
use skoro_graph::{
    graph::Graph, point_map::PointMap, quantity::DynQuantityType, quantity::QuantityType,
    tangent_layout::TangentLayout,
};
use skoro_operation::Input;
use skoro_riemann::{euclidean::EuclideanD, real::Real};

fn integer_dyn() -> std::sync::Arc<dyn DynQuantityType<B>> {
    <Integer as QuantityType<B, Discrete>>::into_dyn(Integer)
}

fn euclidean_dyn(d: usize) -> std::sync::Arc<dyn DynQuantityType<B>> {
    <EuclideanD as QuantityType<B, Continuous>>::into_dyn(EuclideanD(d))
}

/// A graph of a `Real` and a `EuclideanD(3)` parameter, with a map holding both.
fn fixture() -> (Graph<B>, Id, Id, PointMap<B>) {
    let mut graph = Graph::<B>::new();
    let r = graph.add_parameter(Real);
    let v = graph.add_parameter(EuclideanD(3));

    let mut map = PointMap::<B>::default();
    map.insert(r, Input::Float(scalar(1.0)), real_dyn());
    map.insert(
        v,
        Input::Float(tensor2(vec![10.0, 20.0, 30.0], 1, 3)),
        euclidean_dyn(3),
    );
    (graph, r, v, map)
}

/// `PointMap` as a container: insertion, lookup, iteration and the accessors a caller reaches for
/// before any geometry is involved. It behaves like a `HashMap` in these regards, and the tests
/// say so explicitly so that a change to the storage cannot quietly alter them.
mod map_mechanics {
    use super::*;

    #[test]
    fn insert_reports_what_it_displaced() {
        let mut map = PointMap::<B>::default();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);

        let id = Graph::<B>::new().add_parameter(Real);
        assert!(
            map.insert(id, Input::Float(scalar(1.0)), real_dyn())
                .is_none()
        );
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());

        // A second insert at the same id hands back the old value *and* its quantity type.
        let (old, _) = map
            .insert(id, Input::Float(scalar(2.0)), real_dyn())
            .expect("the displaced entry");
        assert_eq!(into_vec(old.unwrap_float()), vec![1.0]);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn get_and_remove_agree_about_membership() {
        let (_, r, v, mut map) = fixture();
        assert!(map.get(&r).is_some());
        assert_eq!(map.keys().count(), 2);

        let (point, qty) = map.remove(&r).expect("present");
        assert_eq!(into_vec(point.unwrap_float()), vec![1.0]);
        assert_eq!(qty.num_components(), 1);
        assert!(map.get(&r).is_none());
        assert!(map.remove(&r).is_none());
        assert!(map.get(&v).is_some());
    }

    /// The iterator pairs each point with its quantity type, and its overridden adapters agree
    /// with the plain traversal.
    #[test]
    fn iteration_carries_the_quantity_types() {
        let (_, _, _, map) = fixture();

        assert_eq!(map.iter().count(), 2);
        assert_eq!(map.iter().size_hint(), (2, Some(2)));
        assert!(map.iter().last().is_some());
        assert!(map.iter().nth(1).is_some());
        assert!(map.iter().nth(2).is_none());

        // The overrides all index `quantity_types` by the id the inner iterator yields, so an
        // empty map is the case where that index must not be reached at all.
        let empty = skoro_graph::point_map::PointMap::<B>::default();
        assert!(empty.iter().last().is_none());
        assert_eq!(empty.iter().count(), 0);

        for (id, point, quantity_type) in map.iter() {
            // The pairing is the invariant: the type must describe the point beside it.
            assert_eq!(point.dims()[1], quantity_type.num_components(), "{id:?}");
        }
    }

    #[test]
    fn converts_to_the_bare_point_map() {
        let (_, _, _, map) = fixture();
        assert_eq!(map.as_ref().len(), 2);
        let bare: HashMap<Id, Input<B>> = map.into();
        assert_eq!(bare.len(), 2);
    }
}

/// Stepping a whole map by a map of tangent vectors, each quantity through its own manifold's
/// retraction. `retract_mut` demands a tangent for every point it holds, which is why an optimizer
/// cannot use it directly -- its map also holds constants it is not stepping.
mod retraction {
    use super::*;

    #[test]
    fn retracts_each_quantity_along_its_own_manifold() {
        let (_, r, v, map) = fixture();
        let mut tangents = HashMap::new();
        tangents.insert(r, scalar(0.5));
        tangents.insert(v, tensor2(vec![1.0, 2.0, 3.0], 1, 3));

        let moved = map.retract(&tangents).unwrap();
        assert_eq!(
            into_vec(moved.get(&r).unwrap().0.clone().unwrap_float()),
            vec![1.5]
        );
        assert_eq!(
            into_vec(moved.get(&v).unwrap().0.clone().unwrap_float()),
            vec![11.0, 22.0, 33.0]
        );
        // `retract` works on a clone; the original is untouched.
        assert_eq!(
            into_vec(map.get(&r).unwrap().0.clone().unwrap_float()),
            vec![1.0]
        );
    }

    /// Not every point needs a tangent -- the ones without simply stay put.
    #[test]
    fn quantities_without_a_tangent_are_left_alone() {
        let (_, r, v, mut map) = fixture();
        let mut tangents = HashMap::new();
        tangents.insert(r, scalar(0.5));

        map.retract_mut(&tangents).unwrap();
        assert_eq!(
            into_vec(map.get(&r).unwrap().0.clone().unwrap_float()),
            vec![1.5]
        );
        assert_eq!(
            into_vec(map.get(&v).unwrap().0.clone().unwrap_float()),
            vec![10.0, 20.0, 30.0]
        );
    }

    /// Every tangent, on the other hand, does need a point.
    #[test]
    fn a_tangent_without_a_point_is_rejected() {
        let (mut graph, r, _, map) = fixture();
        let stranger = graph.add_parameter(Real);
        let mut tangents = HashMap::new();
        tangents.insert(r, scalar(0.5));
        tangents.insert(stranger, scalar(0.5));

        assert!(matches!(
            map.retract(&tangents),
            Err(SkoroError::MissingPoint { id }) if id == stranger
        ));
    }

    /// A discrete quantity has no tangent space, so retracting one is refused rather than
    /// reinterpreted as integer arithmetic.
    #[test]
    fn a_discrete_quantity_cannot_be_retracted() {
        let mut graph = Graph::<B>::new();
        let n = graph.add_parameter(Integer);
        let mut map = PointMap::<B>::default();
        map.insert(n, Input::Int(int2(vec![3], 1, 1)), integer_dyn());

        let mut tangents = HashMap::new();
        tangents.insert(n, scalar(0.5));
        assert!(matches!(
            map.retract(&tangents),
            Err(SkoroError::NotContinuous { id }) if id == n
        ));
    }
}

/// Stepping by one flat vector and a `TangentLayout` instead: this is the form an optimizer
/// actually has, and it is *partial* by design, touching only the quantities the layout names and
/// leaving the rest of the map alone.
mod flat_retraction {
    use super::*;

    #[test]
    fn steps_only_the_quantities_the_layout_names() {
        let (graph, r, v, map) = fixture();
        // A third quantity, in the map but not in the layout.
        let layout = TangentLayout::from_ids(&graph, &[r]).unwrap();

        let moved = map
            .retract_flat(&layout, tensor2(vec![0.25], 1, 1))
            .unwrap();
        assert_eq!(
            into_vec(moved.get(&r).unwrap().0.clone().unwrap_float()),
            vec![1.25]
        );
        assert_eq!(
            into_vec(moved.get(&v).unwrap().0.clone().unwrap_float()),
            vec![10.0, 20.0, 30.0],
            "a quantity outside the layout must not move"
        );
    }

    #[test]
    fn unflattens_each_block_to_its_own_manifold() {
        let (graph, r, v, mut map) = fixture();
        let layout = TangentLayout::from_ids(&graph, &[r, v]).unwrap();
        map.retract_flat_mut(&layout, tensor2(vec![1.0, 1.0, 2.0, 3.0], 1, 4))
            .unwrap();

        assert_eq!(
            into_vec(map.get(&r).unwrap().0.clone().unwrap_float()),
            vec![2.0]
        );
        assert_eq!(
            into_vec(map.get(&v).unwrap().0.clone().unwrap_float()),
            vec![11.0, 22.0, 33.0]
        );
    }

    #[test]
    fn a_wrong_width_is_rejected() {
        let (graph, r, v, map) = fixture();
        let layout = TangentLayout::from_ids(&graph, &[r, v]).unwrap();
        assert!(matches!(
            map.retract_flat(&layout, tensor2(vec![1.0, 2.0], 1, 2)),
            Err(SkoroError::LayoutDimensionMismatch {
                expected: 4,
                found: 2
            })
        ));
    }

    /// A batch-1 point met with a batch-2 step is an error, not a broadcast.
    ///
    /// This is the guard the type's own docs single out: a flat manifold's `Addition` broadcasts
    /// happily, so without the check the caller gets a confidently wrong answer instead of a
    /// complaint.
    #[test]
    fn a_disagreeing_batch_is_rejected_rather_than_broadcast() {
        let (graph, r, _, map) = fixture();
        let layout = TangentLayout::from_ids(&graph, &[r]).unwrap();
        assert!(matches!(
            map.retract_flat(&layout, tensor2(vec![1.0, 2.0], 2, 1)),
            Err(SkoroError::BatchSizeMismatch {
                expected: 2,
                found: 1
            })
        ));
    }

    #[test]
    fn a_layout_id_absent_from_the_map_is_rejected() {
        let (mut graph, r, _, map) = fixture();
        let stranger = graph.add_parameter(Real);
        let layout = TangentLayout::from_ids(&graph, &[r, stranger]).unwrap();
        assert!(matches!(
            map.retract_flat(&layout, tensor2(vec![1.0, 1.0], 1, 2)),
            Err(SkoroError::MissingPoint { id }) if id == stranger
        ));
    }
}

/// The other direction: two maps in, one tangent map out, `b ⊖ a` per quantity. The sign
/// convention is fixed here -- `a.inverse_retract(b)` is the displacement *from* `a` -- and the
/// opposite one negates every residual downstream while still minimizing to the same point.
mod inverse_retraction {
    use super::*;

    /// `inverse_retract` inverts `retract`, which is the defining relation of the pair.
    #[test]
    fn recovers_the_tangent_that_produced_a_point() {
        let (_, r, v, map) = fixture();
        let mut tangents = HashMap::new();
        tangents.insert(r, scalar(0.5));
        tangents.insert(v, tensor2(vec![1.0, 2.0, 3.0], 1, 3));

        let moved = map.retract(&tangents).unwrap();
        let recovered = map.inverse_retract(&moved).unwrap();

        assert_eq!(into_vec(recovered[&r].clone()), vec![0.5]);
        assert_eq!(into_vec(recovered[&v].clone()), vec![1.0, 2.0, 3.0]);
    }

    /// Only the ids the *other* map holds are paired; a wider source is not an error.
    #[test]
    fn pairs_only_the_ids_the_other_map_holds() {
        let (_, r, _, map) = fixture();
        let mut other = PointMap::<B>::default();
        other.insert(r, Input::Float(scalar(4.0)), real_dyn());

        let recovered = map.inverse_retract(&other).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(into_vec(recovered[&r].clone()), vec![3.0]);
    }

    #[test]
    fn a_point_missing_from_the_source_is_rejected() {
        let (mut graph, _, _, map) = fixture();
        let stranger = graph.add_parameter(Real);
        let mut other = PointMap::<B>::default();
        other.insert(stranger, Input::Float(scalar(4.0)), real_dyn());

        assert!(matches!(
            map.inverse_retract(&other),
            Err(SkoroError::MissingPoint { id }) if id == stranger
        ));
    }

    /// Two points can only be subtracted if they live on the same manifold. `EuclideanD(1)` and
    /// `Real` have the same storage and the same dimension, and are still not interchangeable.
    #[test]
    fn mismatched_manifolds_are_rejected() {
        let (_, r, _, map) = fixture();
        let mut other = PointMap::<B>::default();
        other.insert(r, Input::Float(scalar(4.0)), euclidean_dyn(1));

        assert!(matches!(
            map.inverse_retract(&other),
            Err(SkoroError::ManifoldMismatch { id }) if id == r
        ));
    }

    #[test]
    fn a_discrete_source_is_rejected() {
        let mut graph = Graph::<B>::new();
        let n = graph.add_parameter(Integer);
        let mut map = PointMap::<B>::default();
        map.insert(n, Input::Int(int2(vec![3], 1, 1)), integer_dyn());

        assert!(matches!(
            map.inverse_retract(&map.clone()),
            Err(SkoroError::NotContinuous { id }) if id == n
        ));
    }

    /// A continuous *type* whose stored point is nonetheless discrete: the second
    /// `NotContinuous` guard, on the other map's tensor rather than on this map's type.
    #[test]
    fn a_discrete_tensor_under_a_continuous_type_is_rejected() {
        let (_, r, _, map) = fixture();
        let mut other = PointMap::<B>::default();
        other.insert(r, Input::Int(int2(vec![4], 1, 1)), real_dyn());

        assert!(matches!(
            map.inverse_retract(&other),
            Err(SkoroError::NotContinuous { id }) if id == r
        ));
    }
}
