use skoro_backend::ToCubeTensor;

mod common;

type B = skoro_backend::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32>;

exercise_api!(B);
unary_kernels_on_non_standard_layouts!(B);

#[test]
fn to_cube_tensor() {
    skoro_backend::Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &Default::default())
        .to_cube_tensor();
}

#[test]
#[should_panic(expected = "Skoro should never quantize tensors.")]
fn to_cube_tensor_quantized() {
    skoro_backend::Tensor::<B, 1>::zeros([4], &Default::default())
        .quantize_dynamic(&Default::default())
        .to_cube_tensor();
}
