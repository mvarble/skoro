mod common;
type B = skoro_backend::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32>;
tests!(crate::B, 1.0e-3, 1.0e-2);

/// `FixedLinearMap` follows its input's device (review follow-up 3.2): the matrix is built on the
/// default adapter, the input lives on the CPU adapter, and the value and both Jacobians still
/// match the host product, on the input's device.
///
/// Ignored because it needs a second wgpu adapter (a software one such as lavapipe, next to the
/// default GPU), which CI does not have. Run it with
/// `cargo test -p skoro-linalg --test wgpu_f32 -- --ignored`.
#[test]
#[ignore = "needs a second wgpu adapter (WgpuDevice::Cpu)"]
fn fixed_linear_map_follows_its_input_device() {
    use burn_wgpu::WgpuDevice;
    use skoro_backend::{Tensor, TensorData};
    use skoro_linalg::matrix::{FixedLinearMap, linear_map};
    use skoro_model::{
        ModelClientOps,
        hash_map_builders::points,
        jacobian::{jacobian_jvp, jacobian_vjp},
        model::ModelClient,
    };
    use skoro_riemann::euclidean::EuclideanD;

    const K: [f64; 6] = [0.7, -1.3, 2.1, 0.4, -0.25, 1.9];
    const X: [f32; 4] = [1.5, -0.5, -2.0, 0.75];
    let other = WgpuDevice::Cpu;
    let map = FixedLinearMap::<B>::from_row_major(3, 2, &K, &WgpuDevice::DefaultDevice);

    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    let y = linear_map(&x, EuclideanD(3), map).unwrap();
    let input = Tensor::<B, 2>::from_data(TensorData::new(X.to_vec(), [2, 2]), &other);
    let values = points().with_quantity(&x, input).build();

    let to_host = |data: TensorData| data.convert::<f64>().to_vec::<f64>().unwrap();
    let got = y.evaluate(&values).unwrap();
    assert_eq!(got.device(), other);
    let got = to_host(got.into_data());
    for batch in 0..2 {
        for row in 0..3 {
            let want: f64 = (0..2)
                .map(|col| K[row * 2 + col] * X[batch * 2 + col] as f64)
                .sum();
            assert!((got[batch * 3 + row] - want).abs() < 1e-4);
        }
    }

    let inputs = model.tangent_layout(&[x.id()]).unwrap();
    let outputs = model.tangent_layout(&[y.id()]).unwrap();
    for jacobian in [
        jacobian_vjp(&model, &values, &inputs, &outputs).unwrap(),
        jacobian_jvp(&model, &values, &inputs, &outputs).unwrap(),
    ] {
        assert_eq!(jacobian.device(), other);
        let jacobian = to_host(jacobian.into_data());
        for batch in 0..2 {
            for (entry, want) in K.iter().enumerate() {
                assert!((jacobian[batch * 6 + entry] - want).abs() < 1e-5);
            }
        }
    }
}
