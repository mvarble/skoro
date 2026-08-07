mod common;

tests!(skoro_backend::NdArrayBackend<f64, i32>, 1e-7, 1e-4);
