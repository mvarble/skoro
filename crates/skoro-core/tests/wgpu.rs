mod common;

type B = skoro_backend::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32>;

tests!(B);
