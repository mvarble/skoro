mod common;

tests!(skoro_backend::NdArrayBackend<f64, i32>, 1e-6, 1e-5);
