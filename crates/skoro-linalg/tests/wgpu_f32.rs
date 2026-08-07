mod common;
type B = skoro_backend::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32>;
tests!(crate::B, 1.0e-3, 1.0e-2);
