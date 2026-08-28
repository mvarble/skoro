//! The type-level kind tags and their runtime discriminator.
//!
//! `Kind` is what lets one generic body serve both element types: the same `Sampler<B, K>` or
//! `QuantityType<B, K>` code wraps a float tensor for `Continuous` and an integer one for
//! `Discrete`. The methods here are the wrap/unwrap boundary between the two, and the property
//! worth checking is that each refuses the other's tensors rather than reinterpreting them.

use skoro_backend::{Int, NdArrayBackend, Tensor, TensorData};
use skoro_core::{
    any_tensor::AnyTensor,
    kind::{Continuous, Discrete, Kind, QuantityKind},
};

type B = NdArrayBackend<f64, i32>;

fn floats(data: Vec<f64>, batch: usize, components: usize) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(data, [batch, components]),
        &Default::default(),
    )
}

fn ints(data: Vec<i32>, batch: usize, components: usize) -> Tensor<B, 2, Int> {
    Tensor::from_data(
        TensorData::new(data, [batch, components]),
        &Default::default(),
    )
}

#[test]
fn each_kind_reports_its_own_discriminator() {
    assert_eq!(<Continuous as Kind<B>>::KIND, QuantityKind::Continuous);
    assert_eq!(<Discrete as Kind<B>>::KIND, QuantityKind::Discrete);

    // The discriminator is a plain enum, compared by value.
    assert_eq!(QuantityKind::Continuous, QuantityKind::Continuous);
    assert_ne!(QuantityKind::Continuous, QuantityKind::Discrete);
    assert!(format!("{:?}", QuantityKind::Discrete).contains("Discrete"));
    let copy = QuantityKind::Continuous;
    assert_eq!(copy, QuantityKind::Continuous);

    assert!(format!("{Continuous:?}").contains("Continuous"));
    assert!(format!("{Discrete:?}").contains("Discrete"));
}

#[test]
fn continuous_wraps_and_unwraps_float_tensors() {
    let point = floats(vec![1.5, -2.5], 1, 2);

    let wrapped = <Continuous as Kind<B>>::wrap_point(point.clone());
    assert!(!wrapped.is_discrete());
    assert_eq!(
        <Continuous as Kind<B>>::as_point_ref(&wrapped)
            .expect("a float tensor is a continuous point")
            .dims(),
        [1, 2]
    );
    assert_eq!(
        <Continuous as Kind<B>>::unwrap_point(wrapped)
            .expect("a float tensor is a continuous point")
            .into_data()
            .to_vec::<f64>()
            .unwrap(),
        vec![1.5, -2.5]
    );

    // Row-major elements, which is what packing round-trips through.
    assert_eq!(
        <Continuous as Kind<B>>::to_elements(&floats(vec![1.0, 2.0, 3.0, 4.0], 2, 2)),
        vec![1.0, 2.0, 3.0, 4.0]
    );

    let unbatched = <Continuous as Kind<B>>::wrap_unbatched_point(Tensor::<B, 1>::from_data(
        TensorData::new(vec![7.0], [1]),
        &Default::default(),
    ));
    assert!(matches!(unbatched, AnyTensor::Float(_)));
}

#[test]
fn discrete_wraps_and_unwraps_integer_tensors() {
    let point = ints(vec![3, -4], 1, 2);

    let wrapped = <Discrete as Kind<B>>::wrap_point(point.clone());
    assert!(wrapped.is_discrete());
    assert_eq!(
        <Discrete as Kind<B>>::as_point_ref(&wrapped)
            .expect("an int tensor is a discrete point")
            .dims(),
        [1, 2]
    );
    assert_eq!(
        <Discrete as Kind<B>>::unwrap_point(wrapped)
            .expect("an int tensor is a discrete point")
            .into_data()
            .to_vec::<i32>()
            .unwrap(),
        vec![3, -4]
    );

    assert_eq!(
        <Discrete as Kind<B>>::to_elements(&ints(vec![1, 2, 3, 4], 2, 2)),
        vec![1, 2, 3, 4]
    );

    let unbatched = <Discrete as Kind<B>>::wrap_unbatched_point(Tensor::<B, 1, Int>::from_data(
        TensorData::new(vec![9], [1]),
        &Default::default(),
    ));
    assert!(matches!(unbatched, AnyTensor::Int(_)));
}

/// The point of the kind parameter: neither accessor will reinterpret the other's tensor.
///
/// Both accessors return `Option` rather than panicking, because a kind mismatch here is a
/// *question* being asked -- "is this mine?" -- not an invariant being broken.
#[test]
fn each_kind_refuses_the_others_tensor() {
    let float = AnyTensor::<B, 2>::Float(floats(vec![1.0], 1, 1));
    let int = AnyTensor::<B, 2>::Int(ints(vec![1], 1, 1));

    assert!(<Discrete as Kind<B>>::as_point_ref(&float).is_none());
    assert!(<Discrete as Kind<B>>::unwrap_point(float.clone()).is_none());

    assert!(<Continuous as Kind<B>>::as_point_ref(&int).is_none());
    assert!(<Continuous as Kind<B>>::unwrap_point(int).is_none());

    // And the matching direction still works, so the test above is not vacuous.
    assert!(<Continuous as Kind<B>>::as_point_ref(&float).is_some());
}
