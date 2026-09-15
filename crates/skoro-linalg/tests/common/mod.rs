pub mod matrix;
pub mod tril;
pub mod vector;

#[macro_export]
macro_rules! tests {
    ($B:ty, $delta:literal, $epsilon:literal) => {
        /// Vector reductions and products: `norm_squared`, `norm`, `sum`, and the cross product. Each is a
        /// graph operation, so what is checked is the value *and* its derivatives against finite
        /// differences at the backend's own precision.
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

        /// The lower-triangular manifold `Tril`: its inverse-retraction round trip, its action on a vector,
        /// its inverse, and its diagonal. `invert_agrees_with_dense` pins the backend kernel against the
        /// host `faer` factorization, which is the only independent check available for it.
        mod tril {
            #[test]
            fn inverse_retraction_round_trip() {
                super::common::tril::inverse_retraction_round_trip_impl::<$B>();
            }

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
            fn invert_derivatives() {
                super::common::tril::invert_derivatives_impl::<$B>($delta, $epsilon);
            }

            #[test]
            fn diag() {
                super::common::tril::diag_impl::<$B>();
            }

            #[test]
            fn diag_survives_abs() {
                super::common::tril::diag_survives_abs_impl::<$B>();
            }
        }

        /// `Matmul` and its two broadcast forms, where one operand carries a batch and the other does not.
        /// A broadcast that expanded the wrong operand still produces a well-shaped product, so the values
        /// are asserted rather than the dims alone.
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

            #[test]
            fn sparse_linear_map() {
                super::common::matrix::sparse_linear_map_impl::<$B>($delta, $epsilon);
            }
        }
    };
}
