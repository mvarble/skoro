mod common;

type B = skoro_backend::NdArrayBackend<f32, i32>;

exercise_api!(B);
unary_kernels_on_non_standard_layouts!(B);
