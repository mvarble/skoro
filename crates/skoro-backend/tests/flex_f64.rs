mod common;

type B = skoro_backend::Flex;
skoro_backend::f64_test_binary!(skoro_backend::Flex);

exercise_api!(B);
float_policy_is!(B, F64);
