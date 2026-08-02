use approx::assert_abs_diff_eq;

use skoro_backend::{NdArrayBackend, Tensor};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    ops::{Addition, LiteralPointwiseMul, PointwiseMul, Subtraction},
};

type B = NdArrayBackend<f32, i32>;

fn into_f32_vec<const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

#[test]
fn pointwise_mul() {
    let kernel = PointwiseMul;
    let x = Tensor::<B, 2>::from_floats([[1.0, 2.0]], &Default::default());
    let y = Tensor::<B, 2>::from_floats([[3.0, 4.0]], &Default::default());

    // Forward
    let DiffOperationForward::ForwardOnly { output } = kernel.forward(
        vec![Input::Float(x.clone()), Input::Float(y.clone())],
        DiffModeState::ForwardOnly,
    ) else {
        panic!("Expected ForwardOnly");
    };
    assert_abs_diff_eq!(into_f32_vec(output).as_slice(), [3.0, 8.0].as_slice());

    // JVP: F(x,y) = x * y => dF = dx * y + x * dy
    let dx = Tensor::<B, 3>::from_floats([[[0.1, 0.2]]], &Default::default());
    let dy = Tensor::<B, 3>::from_floats([[[0.3, 0.4]]], &Default::default());
    let DiffOperationForward::Jvp { tangent, .. } = kernel.forward(
        vec![Input::Float(x.clone()), Input::Float(y.clone())],
        DiffModeState::Jvp(vec![Some(dx.clone()), Some(dy.clone())]),
    ) else {
        panic!("Expected Jvp");
    };
    // [0.1 * 3.0 + 1.0 * 0.3, 0.2 * 4.0 + 2.0 * 0.4] = [0.6, 1.6]
    assert_abs_diff_eq!(
        into_f32_vec(tangent.unwrap()).as_slice(),
        [0.6, 1.6].as_slice()
    );

    // VJP
    let DiffOperationForward::Vjp { vjp_state, .. } = kernel.forward(
        vec![Input::Float(x.clone()), Input::Float(y.clone())],
        DiffModeState::Vjp(vec![true, true]),
    ) else {
        panic!("Expected Vjp");
    };
    let omega = Tensor::<B, 3>::from_floats([[[1.0, 1.0]]], &Default::default());
    let cotangents = kernel.vjp(&vjp_state, omega.clone());
    assert_abs_diff_eq!(
        into_f32_vec(cotangents[0].clone().unwrap()).as_slice(),
        [3.0, 4.0].as_slice()
    );
    assert_abs_diff_eq!(
        into_f32_vec(cotangents[1].clone().unwrap()).as_slice(),
        [1.0, 2.0].as_slice()
    );

    // HVP
    let dx_vec = Tensor::<B, 2>::from_floats([[0.1, 0.2]], &Default::default());
    let dy_vec = Tensor::<B, 2>::from_floats([[0.3, 0.4]], &Default::default());
    let DiffOperationForward::Hvp {
        vjp_state,
        hvp_state,
        ..
    } = kernel.forward(
        vec![Input::Float(x), Input::Float(y)],
        DiffModeState::Hvp(vec![Some(dx_vec), Some(dy_vec)]),
    )
    else {
        panic!("Expected Hvp");
    };
    let hvp_cotangents = kernel.hvp(&vjp_state, &hvp_state, omega);
    assert_abs_diff_eq!(
        into_f32_vec(hvp_cotangents[0].clone().unwrap()).as_slice(),
        [0.3, 0.4].as_slice()
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp_cotangents[1].clone().unwrap()).as_slice(),
        [0.1, 0.2].as_slice()
    );
}

#[test]
fn addition() {
    let kernel = Addition;
    let x = Tensor::<B, 2>::from_floats([[1.0, 2.0]], &Default::default());
    let y = Tensor::<B, 2>::from_floats([[3.0, 4.0]], &Default::default());

    // Forward
    let DiffOperationForward::ForwardOnly { output } = kernel.forward(
        vec![Input::Float(x.clone()), Input::Float(y.clone())],
        DiffModeState::ForwardOnly,
    ) else {
        panic!("Expected ForwardOnly");
    };
    assert_abs_diff_eq!(into_f32_vec(output).as_slice(), [4.0, 6.0].as_slice());

    // JVP
    let dx = Tensor::<B, 3>::from_floats([[[0.1, 0.2]]], &Default::default());
    let dy = Tensor::<B, 3>::from_floats([[[0.3, 0.4]]], &Default::default());
    let DiffOperationForward::Jvp { tangent, .. } = kernel.forward(
        vec![Input::Float(x.clone()), Input::Float(y.clone())],
        DiffModeState::Jvp(vec![Some(dx.clone()), Some(dy.clone())]),
    ) else {
        panic!("Expected Jvp");
    };
    assert_abs_diff_eq!(
        into_f32_vec(tangent.unwrap()).as_slice(),
        [0.4, 0.6].as_slice()
    );

    // VJP
    let DiffOperationForward::Vjp { vjp_state, .. } = kernel.forward(
        vec![Input::Float(x.clone()), Input::Float(y.clone())],
        DiffModeState::Vjp(vec![true, true]),
    ) else {
        panic!("Expected Vjp");
    };
    let omega = Tensor::<B, 3>::from_floats([[[5.0, 6.0]]], &Default::default());
    let cotangents = kernel.vjp(&vjp_state, omega.clone());
    assert_abs_diff_eq!(
        into_f32_vec(cotangents[0].clone().unwrap()).as_slice(),
        [5.0, 6.0].as_slice()
    );
    assert_abs_diff_eq!(
        into_f32_vec(cotangents[1].clone().unwrap()).as_slice(),
        [5.0, 6.0].as_slice()
    );

    // HVP
    let dx_vec = Tensor::<B, 2>::from_floats([[0.1, 0.2]], &Default::default());
    let dy_vec = Tensor::<B, 2>::from_floats([[0.3, 0.4]], &Default::default());
    let DiffOperationForward::Hvp {
        vjp_state,
        hvp_state,
        ..
    } = kernel.forward(
        vec![Input::Float(x), Input::Float(y)],
        DiffModeState::Hvp(vec![Some(dx_vec), Some(dy_vec)]),
    )
    else {
        panic!("Expected Hvp");
    };
    let hvp_cotangents = kernel.hvp(&vjp_state, &hvp_state, omega);
    assert_abs_diff_eq!(
        into_f32_vec(hvp_cotangents[0].clone().unwrap()).as_slice(),
        [0.0, 0.0].as_slice()
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp_cotangents[1].clone().unwrap()).as_slice(),
        [0.0, 0.0].as_slice()
    );
}

#[test]
fn subtraction() {
    let kernel = Subtraction;
    let x = Tensor::<B, 2>::from_floats([[5.0, 6.0]], &Default::default());
    let y = Tensor::<B, 2>::from_floats([[1.0, 2.0]], &Default::default());

    // Forward
    let DiffOperationForward::ForwardOnly { output } = kernel.forward(
        vec![Input::Float(x.clone()), Input::Float(y.clone())],
        DiffModeState::ForwardOnly,
    ) else {
        panic!("Expected ForwardOnly");
    };
    assert_abs_diff_eq!(into_f32_vec(output).as_slice(), [4.0, 4.0].as_slice());

    // JVP
    let dx = Tensor::<B, 3>::from_floats([[[0.5, 0.6]]], &Default::default());
    let dy = Tensor::<B, 3>::from_floats([[[0.1, 0.2]]], &Default::default());
    let DiffOperationForward::Jvp { tangent, .. } = kernel.forward(
        vec![Input::Float(x.clone()), Input::Float(y.clone())],
        DiffModeState::Jvp(vec![Some(dx.clone()), Some(dy.clone())]),
    ) else {
        panic!("Expected Jvp");
    };
    assert_abs_diff_eq!(
        into_f32_vec(tangent.unwrap()).as_slice(),
        [0.4, 0.4].as_slice()
    );

    // VJP
    let DiffOperationForward::Vjp { vjp_state, .. } = kernel.forward(
        vec![Input::Float(x.clone()), Input::Float(y.clone())],
        DiffModeState::Vjp(vec![true, true]),
    ) else {
        panic!("Expected Vjp");
    };
    let omega = Tensor::<B, 3>::from_floats([[[5.0, 6.0]]], &Default::default());
    let cotangents = kernel.vjp(&vjp_state, omega.clone());
    assert_abs_diff_eq!(
        into_f32_vec(cotangents[0].clone().unwrap()).as_slice(),
        [5.0, 6.0].as_slice()
    );
    assert_abs_diff_eq!(
        into_f32_vec(cotangents[1].clone().unwrap()).as_slice(),
        [-5.0, -6.0].as_slice()
    );

    // HVP
    let dx_vec = Tensor::<B, 2>::from_floats([[0.1, 0.2]], &Default::default());
    let dy_vec = Tensor::<B, 2>::from_floats([[0.3, 0.4]], &Default::default());
    let DiffOperationForward::Hvp {
        vjp_state,
        hvp_state,
        output,
        ..
    } = kernel.forward(
        vec![Input::Float(x), Input::Float(y)],
        DiffModeState::Hvp(vec![Some(dx_vec), Some(dy_vec)]),
    )
    else {
        panic!("Expected Hvp");
    };
    assert_abs_diff_eq!(into_f32_vec(output).as_slice(), [4.0, 4.0].as_slice()); // Validates the bug fix applied previously
    let hvp_cotangents = kernel.hvp(&vjp_state, &hvp_state, omega);
    assert_abs_diff_eq!(
        into_f32_vec(hvp_cotangents[0].clone().unwrap()).as_slice(),
        [0.0, 0.0].as_slice()
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp_cotangents[1].clone().unwrap()).as_slice(),
        [0.0, 0.0].as_slice()
    );
}

#[test]
fn literal_pointwise_mul() {
    let kernel = LiteralPointwiseMul::<B>::new(2.0);
    let x = Tensor::<B, 2>::from_floats([[3.0, 4.0]], &Default::default());

    // Forward
    let DiffOperationForward::ForwardOnly { output } =
        kernel.forward(vec![Input::Float(x.clone())], DiffModeState::ForwardOnly)
    else {
        panic!("Expected ForwardOnly");
    };
    assert_abs_diff_eq!(into_f32_vec(output).as_slice(), [6.0, 8.0].as_slice());

    // JVP
    let dx = Tensor::<B, 3>::from_floats([[[0.1, 0.2]]], &Default::default());
    let DiffOperationForward::Jvp { tangent, .. } = kernel.forward(
        vec![Input::Float(x.clone())],
        DiffModeState::Jvp(vec![Some(dx.clone())]),
    ) else {
        panic!("Expected Jvp");
    };
    assert_abs_diff_eq!(
        into_f32_vec(tangent.unwrap()).as_slice(),
        [0.2, 0.4].as_slice()
    );

    // VJP
    let DiffOperationForward::Vjp { vjp_state, .. } = kernel.forward(
        vec![Input::Float(x.clone())],
        DiffModeState::Vjp(vec![true]),
    ) else {
        panic!("Expected Vjp");
    };
    let omega = Tensor::<B, 3>::from_floats([[[5.0, 6.0]]], &Default::default());
    let cotangents = kernel.vjp(&vjp_state, omega.clone());
    assert_abs_diff_eq!(
        into_f32_vec(cotangents[0].clone().unwrap()).as_slice(),
        [10.0, 12.0].as_slice()
    );

    // HVP
    let dx_vec = Tensor::<B, 2>::from_floats([[0.1, 0.2]], &Default::default());
    let DiffOperationForward::Hvp {
        vjp_state,
        hvp_state,
        ..
    } = kernel.forward(
        vec![Input::Float(x)],
        DiffModeState::Hvp(vec![Some(dx_vec)]),
    )
    else {
        panic!("Expected Hvp");
    };
    let hvp_cotangents = kernel.hvp(&vjp_state, &hvp_state, omega);
    assert_abs_diff_eq!(
        into_f32_vec(hvp_cotangents[0].clone().unwrap()).as_slice(),
        [0.0, 0.0].as_slice()
    );
}
