mod common;

tests!(skoro_backend::NdArrayBackend<f32, i32>, 1e-3, 1e-2);
