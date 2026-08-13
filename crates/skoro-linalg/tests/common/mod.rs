pub mod matrix;
pub mod tril;
pub mod vector;

#[macro_export]
macro_rules! tests {
    ($B:ty, $delta:literal, $epsilon:literal) => {
        mod vector {
            #[test]
            fn norm_squared() {
                super::common::vector::norm_squared_impl::<$B>();
            }

            #[test]
            fn norm() {
                super::common::vector::norm_impl::<$B>();
            }

            #[test]
            fn sum() {
                super::common::vector::sum_impl::<$B>();
            }

            #[test]
            fn cross_product() {
                super::common::vector::cross_product_impl::<$B>();
            }
        }

        mod tril {
            #[test]
            fn vector_product() {
                super::common::tril::vector_product_impl::<$B>();
            }

            #[test]
            fn invert() {
                super::common::tril::invert_impl::<$B>();
            }

            #[test]
            fn invert_agrees_with_dense() {
                super::common::tril::invert_agrees_with_dense_impl::<$B>();
            }

            #[test]
            fn diag() {
                super::common::tril::diag_impl::<$B>();
            }
        }

        mod matrix {
            #[test]
            fn left_broadcast_matmul() {
                super::common::matrix::left_broadcast_matmul_impl::<$B>();
            }

            #[test]
            fn right_broadcast_matmul() {
                super::common::matrix::right_broadcast_matmul_impl::<$B>();
            }

            #[test]
            fn matmul() {
                super::common::matrix::matmul_impl::<$B>($delta, $epsilon);
            }
        }
    };
}
