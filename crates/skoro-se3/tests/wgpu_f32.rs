mod common;

tests!(skoro_backend::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32>, 3e-3, 1e-1);
