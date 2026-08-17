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
fn accessors_and_dims() {
    let f = float(2.5);
    assert!(!f.is_discrete());
    assert_eq!(f.dims(), [1, 1]);
    assert_eq!(
        f.unwrap_float_ref()
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        vec![2.5]
    );

    let i = int(7);
    assert!(i.is_discrete());
    assert_eq!(i.dims(), [1, 1]);
    assert_eq!(
        i.unwrap_int().clone().into_data().to_vec::<i32>().unwrap(),
        vec![7]
    );
}
