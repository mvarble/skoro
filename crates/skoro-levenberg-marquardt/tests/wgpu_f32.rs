mod common;

tests!(
    skoro_backend::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32>,
    1e-3,
    1e-2
);
