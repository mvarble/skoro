mod common;
type B = skoro_backend::Flex;
skoro_backend::f64_test_binary!(skoro_backend::Flex);
tests!(crate::B, 1.0e-6, 1.0e-5);
