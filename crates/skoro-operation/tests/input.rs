use skoro_backend::{Int, NdArrayBackend, Tensor};
use skoro_operation::Input;

type B = NdArrayBackend<f32, i32>;

fn float(v: f32) -> Input<B> {
    Input::float(Tensor::<B, 2>::from_floats([[v]], &Default::default()))
}

fn int(v: i32) -> Input<B> {
    Input::int(Tensor::<B, 2, Int>::from_data(
        skoro_backend::TensorData::new(vec![v], [1, 1]),
        &Default::default(),
    ))
}

#[test]
fn transparent_float_arithmetic() {
    let a = float(3.0);
    let b = float(4.0);
    let sum = (a + b).into_data().to_vec::<f32>().unwrap();
    assert_eq!(sum, vec![7.0]);

    let prod = (float(3.0) * float(4.0))
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(prod, vec![12.0]);
}

#[test]
fn accessors_and_dims() {
    let f = float(2.5);
    assert!(!f.is_discrete());
    assert_eq!(f.dims(), [1, 1]);
    assert_eq!(
        f.as_float().clone().into_data().to_vec::<f32>().unwrap(),
        vec![2.5]
    );

    let i = int(7);
    assert!(i.is_discrete());
    assert_eq!(i.dims(), [1, 1]);
    assert_eq!(
        i.as_int().clone().into_data().to_vec::<i32>().unwrap(),
        vec![7]
    );
}

#[test]
#[should_panic(
    expected = "arithmetic on a discrete operation input is not supported; discrete operations belong to the discrete track (no implicit coercion)"
)]
fn arithmetic_on_int_panics() {
    let _ = int(1) + int(2);
}

#[test]
#[should_panic(expected = "expected a continuous")]
fn as_float_on_int_panics() {
    let _ = int(1).as_float().clone();
}

#[test]
#[should_panic(expected = "expected a discrete")]
fn as_int_on_float_panics() {
    let _ = float(1.0).as_int().clone();
}
