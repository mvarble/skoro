//! The tensor/host boundary, which is the new code and therefore the first thing tested.
//!
//! Every fixture here is non-square and asymmetric on purpose. A transposed conversion is
//! the classic silent failure of a row-major/column-major boundary, and it is invisible on
//! a square matrix of repeated values.

use skoro_backend::{Backend, Tensor, TensorData};
use skoro_core::error::SkoroError;
use skoro_dense::dense::Dense;

fn tensor<B: Backend, const D: usize>(data: Vec<f64>, shape: [usize; D]) -> Tensor<B, D> {
    Tensor::from_data(
        TensorData::new(data, shape).convert::<B::FloatElem>(),
        &B::Device::default(),
    )
}

fn into_vec<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f64> {
    t.into_data().convert::<f64>().to_vec::<f64>().unwrap()
}

/// `[1, 2, 3]` in, the same `[1, 2, 3]` out. The values are distinct and the shape is not
/// square, so a transposed conversion cannot pass.
pub fn matrix_round_trip_impl<B: Backend>(epsilon: f64) {
    let values = vec![1.5, -2.25, 3.0, -0.75, 4.5, 6.25];
    let t = tensor::<B, 3>(values.clone(), [1, 2, 3]);
    let dense = Dense::from_tensor(t).unwrap();

    assert_eq!((dense.rows(), dense.cols()), (2, 3));
    // Row-major: `(0, 2)` is the third value, not the second row.
    assert_eq!(dense.get(0, 2), 3.0);
    assert_eq!(dense.get(1, 0), -0.75);

    let back = into_vec(dense.to_tensor::<B>(&B::Device::default()));
    for (a, e) in back.iter().zip(&values) {
        assert!((a - e).abs() <= epsilon, "round trip changed {e} into {a}");
    }
}

/// `[1, n]` in, an `n × 1` column out, `[1, n]` back.
pub fn vector_round_trip_impl<B: Backend>(epsilon: f64) {
    let values = vec![0.5, -1.25, 2.75, 8.0];
    let dense = Dense::from_vector(tensor::<B, 2>(values.clone(), [1, 4])).unwrap();

    assert_eq!((dense.rows(), dense.cols()), (4, 1));

    let back = into_vec(dense.to_vector::<B>(&B::Device::default()));
    assert_eq!(back.len(), 4);
    for (a, e) in back.iter().zip(&values) {
        assert!((a - e).abs() <= epsilon, "round trip changed {e} into {a}");
    }
}

/// The single-batch restriction is load-bearing and must fail loudly. Taking slice 0 would
/// silently discard the rest of an estimator's batch.
pub fn a_batched_input_is_rejected_impl<B: Backend>() {
    assert_eq!(
        Dense::from_tensor(tensor::<B, 3>(vec![0.0; 2 * 3 * 4], [2, 3, 4])).map(|d| d.rows()),
        Err(SkoroError::BatchNotSingleton { found: 2 })
    );
    assert_eq!(
        Dense::from_vector(tensor::<B, 2>(vec![0.0; 3 * 4], [3, 4])).map(|d| d.rows()),
        Err(SkoroError::BatchNotSingleton { found: 3 })
    );
}

#[macro_export]
macro_rules! tests {
    ($B:ty, $epsilon:literal) => {
        #[test]
        fn matrix_round_trip() {
            common::matrix_round_trip_impl::<$B>($epsilon);
        }
        #[test]
        fn vector_round_trip() {
            common::vector_round_trip_impl::<$B>($epsilon);
        }
        #[test]
        fn a_batched_input_is_rejected() {
            common::a_batched_input_is_rejected_impl::<$B>();
        }
        #[test]
        fn split_and_stack_round_trip() {
            common::split_and_stack_round_trip_impl::<$B>($epsilon);
        }
        #[test]
        fn splitting_a_single_row_agrees_with_from_tensor() {
            common::splitting_a_single_row_agrees_with_from_tensor_impl::<$B>();
        }
        #[test]
        fn stack_vector_rejects_a_ragged_batch() {
            common::stack_vector_rejects_a_ragged_batch_impl::<$B>();
        }
    };
}

/// `split_tensor` gives one matrix per batch row, each already row-major and in order.
///
/// The fixture is `[2, 2, 3]` with every value distinct: a split that strided wrongly, or
/// that transposed within a row, lands somewhere visible.
pub fn split_and_stack_round_trip_impl<B: Backend>(epsilon: f64) {
    let values: Vec<f64> = (0..12).map(|i| 0.5 + i as f64 * 1.25).collect();
    let split = Dense::split_tensor(tensor::<B, 3>(values.clone(), [2, 2, 3])).unwrap();

    assert_eq!(split.len(), 2);
    for (row, block) in split.iter().enumerate() {
        assert_eq!((block.rows(), block.cols()), (2, 3));
        for r in 0..2 {
            for c in 0..3 {
                let expected = values[row * 6 + r * 3 + c];
                assert!(
                    (block.get(r, c) - expected).abs() <= epsilon,
                    "row {row} element ({r}, {c}): got {}, want {expected}",
                    block.get(r, c)
                );
            }
        }
    }

    // Vectors round-trip through split/stack unchanged.
    let vector_values = vec![1.5, -2.25, 3.0, -0.75, 4.5, 6.25];
    let columns = Dense::split_vector(tensor::<B, 2>(vector_values.clone(), [3, 2])).unwrap();
    assert_eq!(columns.len(), 3);
    assert_eq!((columns[0].rows(), columns[0].cols()), (2, 1));

    let restacked = into_vec(Dense::stack_vector::<B>(&columns, &B::Device::default()).unwrap());
    for (a, e) in restacked.iter().zip(&vector_values) {
        assert!((a - e).abs() <= epsilon, "round trip changed {e} into {a}");
    }
}

/// A batch of one is still a batch: `split_*` accepts it where `from_*` would too, and the
/// two agree.
pub fn splitting_a_single_row_agrees_with_from_tensor_impl<B: Backend>() {
    let values = vec![1.5, -2.25, 3.0, -0.75, 4.5, 6.25];
    let split = Dense::split_tensor(tensor::<B, 3>(values.clone(), [1, 2, 3])).unwrap();
    let single = Dense::from_tensor(tensor::<B, 3>(values, [1, 2, 3])).unwrap();

    assert_eq!(split.len(), 1);
    assert_eq!(split[0], single);
}

/// `stack_vector` refuses what it cannot reassemble, rather than guessing.
pub fn stack_vector_rejects_a_ragged_batch_impl<B: Backend>() {
    assert_eq!(
        Dense::stack_vector::<B>(&[], &B::Device::default()).map(|t| t.dims()),
        Err(SkoroError::ShapeMismatch {
            expected: 1,
            found: 0
        }),
        "an empty batch has no recoverable width"
    );
    assert_eq!(
        Dense::stack_vector::<B>(
            &[Dense::zeros(3, 1), Dense::zeros(2, 1)],
            &B::Device::default()
        )
        .map(|t| t.dims()),
        Err(SkoroError::ShapeMismatch {
            expected: 3,
            found: 2
        })
    );
    assert_eq!(
        Dense::stack_vector::<B>(&[Dense::zeros(3, 2)], &B::Device::default()).map(|t| t.dims()),
        Err(SkoroError::ShapeMismatch {
            expected: 1,
            found: 2
        }),
        "a matrix is not a column"
    );
}
