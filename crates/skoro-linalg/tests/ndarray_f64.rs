mod common;
type B = skoro_backend::NdArrayBackend<f64, i32>;
tests!(crate::B, 1.0e-6, 1.0e-5);
