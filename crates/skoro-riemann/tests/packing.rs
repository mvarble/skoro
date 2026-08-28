//! `Packable` for the flat manifolds, and the container types a caller packs from.
//!
//! A manifold's `pack`/`unpack` pair is an offset calculation over a flat element slice, so the
//! round trip is the whole claim: what comes back out at an offset is what went in, for every
//! element type the backends use. The dynamic manifolds carry their dimension at runtime, which
//! is the one place a shape can disagree with the data -- `shape_checks` is where that disagrees
//! loudly instead of writing past the block.

use skoro_backend::Element;
use skoro_core::pack::Packable;
use skoro_riemann::{
    euclidean::{Euclidean, EuclideanD, Vector, VectorD},
    matrix::{EuclideanMatrix, EuclideanMatrixD, Matrix, MatrixD},
    real::Real,
};

/// `pack`/`unpack` for each flat manifold, at an offset into a shared element slice. What comes
/// back out is what went in, for every element type the backends use -- the packing is generic in
/// it, and only running all four catches a conversion that is lossless in one and not the others.
mod round_trips {
    use super::*;

    fn real_packing_impl<F>()
    where
        F: Element + PartialEq,
        Real: Packable<F, F>,
    {
        let mut packed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(F::from_elem);
        let ty = Real;
        assert_eq!(ty.unpack(&packed, 0), F::from_elem(1.0));
        assert_eq!(ty.unpack(&packed, 1), F::from_elem(2.0));
        assert_eq!(ty.unpack(&packed, 2), F::from_elem(3.0));
        assert_eq!(ty.unpack(&packed, 3), F::from_elem(4.0));
        assert_eq!(ty.unpack(&packed, 4), F::from_elem(5.0));
        assert_eq!(ty.unpack(&packed, 5), F::from_elem(6.0));
        ty.pack(&F::from_elem(3.0), 0, &mut packed);
        ty.pack(&F::from_elem(10.0), 1, &mut packed);
        ty.pack(&F::from_elem(0.3), 2, &mut packed);
        ty.pack(&F::from_elem(-8.1), 3, &mut packed);
        ty.pack(&F::from_elem(0.0), 4, &mut packed);
        ty.pack(&F::from_elem(-3.0), 5, &mut packed);
        assert_eq!(
            packed,
            [
                F::from_elem(3.0),
                F::from_elem(10.0),
                F::from_elem(0.3),
                F::from_elem(-8.1),
                F::from_elem(0.0),
                F::from_elem(-3.0)
            ]
        );
    }

    fn dyn_euclidean_packing_impl<F>()
    where
        F: Element + PartialEq,
        EuclideanD: Packable<F, VectorD<F>>,
    {
        let mut packed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(F::from_elem);
        let ty = EuclideanD(2);
        assert_eq!(
            ty.unpack(&packed, 0),
            VectorD::<F>::new(&[F::from_elem(1.0), F::from_elem(2.0)])
        );
        assert_eq!(
            ty.unpack(&packed, 2),
            VectorD::<F>::new(&[F::from_elem(3.0), F::from_elem(4.0)])
        );
        assert_eq!(
            ty.unpack(&packed, 4),
            VectorD::<F>::new(&[F::from_elem(5.0), F::from_elem(6.0)])
        );
        ty.pack(
            &VectorD::<F>::new(&[F::from_elem(-5.0), F::from_elem(6.0)]),
            0,
            &mut packed,
        );
        ty.pack(
            &VectorD::<F>::new(&[F::from_elem(0.0), F::from_elem(3.0)]),
            2,
            &mut packed,
        );
        ty.pack(
            &VectorD::<F>::new(&[F::from_elem(1.0), F::from_elem(1.0)]),
            4,
            &mut packed,
        );
        assert_eq!(
            packed,
            [
                F::from_elem(-5.0),
                F::from_elem(6.0),
                F::from_elem(0.0),
                F::from_elem(3.0),
                F::from_elem(1.0),
                F::from_elem(1.0)
            ]
        );
    }

    fn euclidean_packing_impl<F>()
    where
        F: Element + PartialEq,
        Euclidean<2>: Packable<F, Vector<2, F>>,
    {
        let mut packed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(F::from_elem);
        let ty = Euclidean::<2>;
        assert_eq!(
            ty.unpack(&packed, 0),
            Vector::<2, F>::new([F::from_elem(1.0), F::from_elem(2.0)])
        );
        assert_eq!(
            ty.unpack(&packed, 2),
            Vector::<2, F>::new([F::from_elem(3.0), F::from_elem(4.0)])
        );
        assert_eq!(
            ty.unpack(&packed, 4),
            Vector::<2, F>::new([F::from_elem(5.0), F::from_elem(6.0)])
        );
        ty.pack(
            &Vector::<2, F>::new([F::from_elem(-5.0), F::from_elem(6.0)]),
            0,
            &mut packed,
        );
        ty.pack(
            &Vector::<2, F>::new([F::from_elem(0.0), F::from_elem(3.0)]),
            2,
            &mut packed,
        );
        ty.pack(
            &Vector::<2, F>::new([F::from_elem(1.0), F::from_elem(1.0)]),
            4,
            &mut packed,
        );
        assert_eq!(
            packed,
            [
                F::from_elem(-5.0),
                F::from_elem(6.0),
                F::from_elem(0.0),
                F::from_elem(3.0),
                F::from_elem(1.0),
                F::from_elem(1.0)
            ]
        );
    }

    fn dyn_euclidean_matrix_packing_impl<F>()
    where
        F: Element + PartialEq,
        EuclideanMatrixD: Packable<F, MatrixD<F>>,
    {
        let mut packed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].map(F::from_elem);
        let ty = EuclideanMatrixD(2, 2);
        assert_eq!(
            ty.unpack(&packed, 0),
            MatrixD::<F>::new(2, 2, &[1.0, 2.0, 3.0, 4.0].map(F::from_elem))
        );
        assert_eq!(
            ty.unpack(&packed, 4),
            MatrixD::<F>::new(2, 2, &[5.0, 6.0, 7.0, 8.0].map(F::from_elem))
        );
        ty.pack(
            &MatrixD::<F>::new(2, 2, &[-1.0, 3.0, 4.0, 5.0].map(F::from_elem)),
            0,
            &mut packed,
        );
        ty.pack(
            &MatrixD::<F>::new(2, 2, &[1.0, -3.0, 0.0, 5.1].map(F::from_elem)),
            4,
            &mut packed,
        );
        assert_eq!(
            packed,
            [-1.0, 3.0, 4.0, 5.0, 1.0, -3.0, 0.0, 5.1].map(F::from_elem)
        );
    }

    fn euclidean_matrix_packing_impl<F>()
    where
        F: Element + PartialEq,
        EuclideanMatrix<2, 2>: Packable<F, Matrix<2, 2, F>>,
    {
        let mut packed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].map(F::from_elem);
        let ty = EuclideanMatrix::<2, 2>;
        assert_eq!(
            ty.unpack(&packed, 0),
            Matrix::<2, 2, F>::new(&[1.0, 2.0, 3.0, 4.0].map(F::from_elem))
        );
        assert_eq!(
            ty.unpack(&packed, 4),
            Matrix::<2, 2, F>::new(&[5.0, 6.0, 7.0, 8.0].map(F::from_elem))
        );
        ty.pack(
            &Matrix::<2, 2, F>::new(&[-1.0, 3.0, 4.0, 5.0].map(F::from_elem)),
            0,
            &mut packed,
        );
        ty.pack(
            &Matrix::<2, 2, F>::new(&[1.0, -3.0, 0.0, 5.1].map(F::from_elem)),
            4,
            &mut packed,
        );
        assert_eq!(
            packed,
            [-1.0, 3.0, 4.0, 5.0, 1.0, -3.0, 0.0, 5.1].map(F::from_elem)
        );
    }

    #[test]
    fn real_packing() {
        real_packing_impl::<f32>();
        real_packing_impl::<f64>();
    }

    #[test]
    fn dyn_euclidean_packing() {
        dyn_euclidean_packing_impl::<f32>();
        dyn_euclidean_packing_impl::<f64>();
    }

    #[test]
    fn euclidean_packing() {
        euclidean_packing_impl::<f32>();
        euclidean_packing_impl::<f64>();
    }

    #[test]
    fn dyn_euclidean_matrix_packing() {
        dyn_euclidean_matrix_packing_impl::<f32>();
        dyn_euclidean_matrix_packing_impl::<f64>();
    }

    #[test]
    fn euclidean_matrix_packing() {
        euclidean_matrix_packing_impl::<f32>();
        euclidean_matrix_packing_impl::<f64>();
    }
}

/// The dynamically dimensioned manifolds carry their dimension at runtime, which is the one place
/// the shape can disagree with the data handed over. It panics naming both shapes rather than
/// writing past the block it was given.
mod shape_checks {
    use super::*;

    #[test]
    #[should_panic(
        expected = "Cannot pack vector data of length 2 into a Euclidean space of dimension 4"
    )]
    fn failed_dyn_euclidean_packing() {
        let mut packed = [0.0; 4];
        EuclideanD(4).pack(&VectorD::<f32>::new(&[1.0, 2.0]), 0, &mut packed);
    }

    #[test]
    #[should_panic(
        expected = "Cannot pack matrix data of shape 2×1 into a Euclidean matrix space of shape 2×2"
    )]
    fn failed_dyn_euclidean_matrix_packing() {
        let mut packed = [0.0; 4];
        EuclideanMatrixD(2, 2).pack(&MatrixD::<f32>::new(2, 1, &[1.0, 2.0]), 0, &mut packed);
    }
}

/// `VectorD`/`Vector` and `MatrixD`/`Matrix`, the concrete data types a caller packs *from*.
/// Their fallible constructors are the boundary where a shape mistake becomes a `SkoroError`
/// rather than a panic; the infallible ones are for callers that already know the shape and would
/// rather have a stack trace than a `Result`.
mod containers {
    use skoro_core::error::SkoroError;
    use skoro_riemann::{
        euclidean::{Vector, Vector3, VectorD},
        matrix::{Matrix, MatrixD},
    };

    #[test]
    fn runtime_vectors_expose_their_contents() {
        let mut v = VectorD::<f64>::new(&[1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert!(!v.is_empty());
        assert_eq!(v.slice(), &[1.0, 2.0, 3.0]);

        v.slice_mut()[1] = 20.0;
        assert_eq!(v.slice(), &[1.0, 20.0, 3.0]);

        assert_eq!(VectorD::<f64>::from_vec(vec![7.0]).slice(), &[7.0]);
        assert!(VectorD::<f64>::from_vec(vec![]).is_empty());

        // Buildable from an iterator, which is how a packed row is turned back into data.
        let collected: VectorD<f64> = (0..3).map(|i| i as f64).collect();
        assert_eq!(collected.slice(), &[0.0, 1.0, 2.0]);

        assert_eq!(v, VectorD::<f64>::new(&[1.0, 20.0, 3.0]));
        assert_ne!(v, VectorD::<f64>::new(&[1.0, 2.0, 3.0]));
        assert_eq!(v.clone().slice(), v.slice());
        assert!(format!("{v:?}").contains("VectorD(3)"));
    }

    #[test]
    fn static_vectors_expose_their_contents() {
        let mut v = Vector3::<f64>::new([1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        // A statically-sized vector of dimension 3 is never empty, by construction.
        assert!(!v.is_empty());
        assert_eq!(v.slice(), &[1.0, 2.0, 3.0]);

        v.slice_mut()[0] = -1.0;
        assert_eq!(v.slice(), &[-1.0, 2.0, 3.0]);

        assert_eq!(v, Vector::<3, f64>::new([-1.0, 2.0, 3.0]));
        assert_ne!(v, Vector3::<f64>::new([0.0, 0.0, 0.0]));
        let copy = v;
        assert_eq!(copy.slice(), v.slice());
        assert!(format!("{v:?}").contains("Vector<3>"));
    }

    #[test]
    fn runtime_matrices_expose_their_shape_and_contents() {
        let mut m = MatrixD::<f64>::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!((m.rows(), m.cols()), (2, 3));
        assert_eq!(m.len(), 6);
        assert!(!m.is_empty());
        assert_eq!(m.slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        m.slice_mut()[5] = 60.0;
        assert_eq!(m.slice()[5], 60.0);

        assert_eq!(
            MatrixD::<f64>::from_vec(2, 1, vec![1.0, 2.0]).slice(),
            &[1.0, 2.0]
        );
        assert_eq!(
            m,
            MatrixD::<f64>::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 60.0])
        );
        assert_ne!(m, MatrixD::<f64>::new(2, 3, &[0.0; 6]));
        assert_eq!(m.clone().slice(), m.slice());

        // Row-major `Debug`, one field per row.
        let text = format!("{m:?}");
        assert!(text.contains("MatrixD(2, 3)"));
        assert!(text.contains("1.0"));
    }

    /// A degenerate shape prints without indexing into nothing.
    #[test]
    fn an_empty_matrix_prints_without_panicking() {
        let m = MatrixD::<f64>::new(0, 0, &[]);
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
        assert_eq!(format!("{m:?}"), "MatrixD(0, 0)([[]])");

        let m = Matrix::<0, 0, f64>::new(&[]);
        assert_eq!(format!("{m:?}"), "MatrixD(0, 0)([[]])");
    }

    #[test]
    fn static_matrices_expose_their_contents() {
        let mut m = Matrix::<2, 2, f64>::new(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.len(), 4);
        assert!(!m.is_empty());
        assert_eq!(m.slice(), &[1.0, 2.0, 3.0, 4.0]);

        m.slice_mut()[0] = 10.0;
        assert_eq!(m.slice()[0], 10.0);

        assert_eq!(Matrix::<2, 2, f64>::from_vec(vec![10.0, 2.0, 3.0, 4.0]), m);
        assert_ne!(m, Matrix::<2, 2, f64>::new(&[0.0; 4]));
        assert_eq!(m.clone().slice(), m.slice());
        assert!(format!("{m:?}").contains("MatrixD(2, 2)"));
    }

    /// The fallible constructors report a shape mismatch instead of unwinding.
    ///
    /// Which of the two a caller wants depends on where the data came from: a literal in the
    /// source is a programming error and should panic, while a length read from a file is a
    /// recoverable one.
    #[test]
    fn fallible_constructors_report_shape_mismatches() {
        assert_eq!(
            MatrixD::<f64>::try_new(2, 3, &[1.0, 2.0]),
            Err(SkoroError::ShapeMismatch {
                expected: 6,
                found: 2
            })
        );
        assert_eq!(
            MatrixD::<f64>::try_from_vec(2, 3, vec![1.0, 2.0]),
            Err(SkoroError::ShapeMismatch {
                expected: 6,
                found: 2
            })
        );
        assert_eq!(
            Matrix::<2, 2, f64>::try_new(&[1.0]),
            Err(SkoroError::ShapeMismatch {
                expected: 4,
                found: 1
            })
        );
        assert_eq!(
            Matrix::<2, 2, f64>::try_from_vec(vec![1.0]),
            Err(SkoroError::ShapeMismatch {
                expected: 4,
                found: 1
            })
        );

        // And they succeed on the right shape, so the test is not vacuous.
        assert!(MatrixD::<f64>::try_new(2, 1, &[1.0, 2.0]).is_ok());
        assert!(Matrix::<2, 2, f64>::try_new(&[1.0, 2.0, 3.0, 4.0]).is_ok());
    }

    /// The other half of that trade, one case per infallible constructor: a length that disagrees with
    /// the shape panics naming the constructor, so the stack trace points at the call site rather than
    /// at the first operation to read a short buffer.
    mod infallible_constructors_panic {
        use super::*;

        #[test]
        #[should_panic(expected = "MatrixD::new: data length does not match shape")]
        fn matrix_d_new() {
            let _ = MatrixD::<f64>::new(2, 3, &[1.0]);
        }

        #[test]
        #[should_panic(expected = "MatrixD::from_vec: data length does not match shape")]
        fn matrix_d_from_vec() {
            let _ = MatrixD::<f64>::from_vec(2, 3, vec![1.0]);
        }

        #[test]
        #[should_panic(expected = "Matrix::new: data length does not match shape")]
        fn matrix_new() {
            let _ = Matrix::<2, 2, f64>::new(&[1.0]);
        }

        #[test]
        #[should_panic(expected = "Matrix::from_vec: data length does not match shape")]
        fn matrix_from_vec() {
            let _ = Matrix::<2, 2, f64>::from_vec(vec![1.0]);
        }
    }
}
