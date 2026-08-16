use skoro_backend::Element;
use skoro_core::pack::Packable;
use skoro_riemann::{
    euclidean::{Euclidean, EuclideanD, Vector, VectorD},
    matrix::{EuclideanMatrix, EuclideanMatrixD, Matrix, MatrixD},
    real::Real,
};

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
