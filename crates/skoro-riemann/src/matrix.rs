use skoro_backend::{Backend, Element};
use skoro_core::error::{SkoroError, SkoroResult};
use skoro_core::pack::{NumComponents, Packable};
use skoro_operation::{diff::DiffOperation, ops::Addition};

use crate::Manifold;
use crate::euclidean::EuclideanInverseRetraction;

/// A matrix with a runtime shape.
#[derive(Clone, PartialEq)]
pub struct MatrixD<F: Element>(ndarray::Array2<F>);

impl<F: Element> std::fmt::Debug for MatrixD<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (nrows, ncols) = self.0.dim();
        if nrows == 0 || ncols == 0 {
            write!(f, "MatrixD(0, 0)([[]])")
        } else {
            let mut out = f.debug_tuple(&format!("MatrixD({nrows}, {ncols})"));
            for i in 0..nrows {
                out.field(&self.0.row(i).to_vec());
            }
            out.finish_non_exhaustive()
        }
    }
}

impl<F: Element> MatrixD<F> {
    /// Construct a matrix, panicking if `elems.len() != nrows * ncols`.
    ///
    /// Use [`try_new`](Self::try_new) for a recoverable variant.
    #[inline]
    pub fn new(nrows: usize, ncols: usize, elems: &[F]) -> Self {
        Self::try_new(nrows, ncols, elems).expect("MatrixD::new: data length does not match shape")
    }

    /// Construct a matrix, returning [`SkoroError::ShapeMismatch`] if
    /// `elems.len() != nrows * ncols`.
    #[inline]
    pub fn try_new(nrows: usize, ncols: usize, elems: &[F]) -> SkoroResult<Self> {
        Self::try_from_vec(nrows, ncols, elems.to_vec())
    }

    /// Construct a matrix, panicking if `elems.len() != nrows * ncols`.
    ///
    /// Use [`try_from_vec`](Self::try_from_vec) for a recoverable variant.
    #[inline]
    pub fn from_vec(nrows: usize, ncols: usize, elems: Vec<F>) -> Self {
        Self::try_from_vec(nrows, ncols, elems)
            .expect("MatrixD::from_vec: data length does not match shape")
    }

    /// Construct a matrix, returning [`SkoroError::ShapeMismatch`] if
    /// `elems.len() != nrows * ncols`.
    #[inline]
    pub fn try_from_vec(nrows: usize, ncols: usize, elems: Vec<F>) -> SkoroResult<Self> {
        let found = elems.len();
        ndarray::Array2::from_shape_vec([nrows, ncols], elems)
            .map(Self)
            .map_err(|_| SkoroError::ShapeMismatch {
                expected: nrows * ncols,
                found,
            })
    }

    #[inline]
    pub fn slice(&self) -> &[F] {
        self.0.as_slice().unwrap()
    }

    #[inline]
    pub fn slice_mut(&mut self) -> &mut [F] {
        self.0.as_slice_mut().unwrap()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

/// A matrix with a static shape.
#[derive(Clone, PartialEq)]
pub struct Matrix<const M: usize, const N: usize, F: Element>(ndarray::Array2<F>);

impl<const M: usize, const N: usize, F: Element> std::fmt::Debug for Matrix<M, N, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if M == 0 || N == 0 {
            write!(f, "MatrixD(0, 0)([[]])")
        } else {
            let mut out = f.debug_tuple(&format!("MatrixD({M}, {N})"));
            for i in 0..M {
                out.field(&self.0.row(i).to_vec());
            }
            out.finish_non_exhaustive()
        }
    }
}

impl<const M: usize, const N: usize, F: Element> Matrix<M, N, F> {
    /// Construct a matrix, panicking if `elems.len() != M * N`.
    ///
    /// Use [`try_new`](Self::try_new) for a recoverable variant.
    #[inline]
    pub fn new(elems: &[F]) -> Self {
        Self::try_new(elems).expect("Matrix::new: data length does not match shape")
    }

    /// Construct a matrix, returning [`SkoroError::ShapeMismatch`] if
    /// `elems.len() != M * N`.
    #[inline]
    pub fn try_new(elems: &[F]) -> SkoroResult<Self> {
        Self::try_from_vec(elems.to_vec())
    }

    /// Construct a matrix, panicking if `elems.len() != M * N`.
    ///
    /// Use [`try_from_vec`](Self::try_from_vec) for a recoverable variant.
    #[inline]
    pub fn from_vec(elems: Vec<F>) -> Self {
        Self::try_from_vec(elems).expect("Matrix::from_vec: data length does not match shape")
    }

    /// Construct a matrix, returning [`SkoroError::ShapeMismatch`] if
    /// `elems.len() != M * N`.
    #[inline]
    pub fn try_from_vec(elems: Vec<F>) -> SkoroResult<Self> {
        let found = elems.len();
        ndarray::Array2::from_shape_vec([M, N], elems)
            .map(Self)
            .map_err(|_| SkoroError::ShapeMismatch {
                expected: M * N,
                found,
            })
    }

    #[inline]
    pub fn slice(&self) -> &[F] {
        self.0.as_slice().unwrap()
    }

    #[inline]
    pub fn slice_mut(&mut self) -> &mut [F] {
        self.0.as_slice_mut().unwrap()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

/// Euclidean manifold operations of matrix of a specific runtime shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EuclideanMatrixD(pub usize, pub usize);

impl<B: Backend> Manifold<B> for EuclideanMatrixD {
    fn dimension(&self) -> usize {
        self.0 * self.1
    }

    fn retraction(&self) -> impl DiffOperation<B> {
        Addition
    }

    fn inverse_retraction(&self) -> impl DiffOperation<B> {
        EuclideanInverseRetraction
    }
}

impl NumComponents for EuclideanMatrixD {
    fn num_components(&self) -> usize {
        self.0 * self.1
    }
}

impl<F: Element> Packable<F, MatrixD<F>> for EuclideanMatrixD {
    fn pack(&self, data: &MatrixD<F>, offset: usize, target: &mut [F]) {
        assert_eq!(data.len(), self.0);
        target[offset..(offset + self.0)].copy_from_slice(data.slice());
    }
    fn unpack(&self, source: &[F], offset: usize) -> MatrixD<F> {
        MatrixD::<F>::new(
            self.0,
            self.1,
            &source[offset..(offset + (self.0 * self.1))],
        )
    }
}

/// Euclidean manifold operations of matrix of a specific static shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EuclideanMatrix<const M: usize, const N: usize>;

pub type EuclideanMatrix11 = EuclideanMatrix<1, 1>;
pub type EuclideanMatrix12 = EuclideanMatrix<1, 2>;
pub type EuclideanMatrix13 = EuclideanMatrix<1, 3>;
pub type EuclideanMatrix14 = EuclideanMatrix<1, 4>;
pub type EuclideanMatrix15 = EuclideanMatrix<1, 5>;
pub type EuclideanMatrix16 = EuclideanMatrix<1, 6>;

pub type EuclideanMatrix21 = EuclideanMatrix<2, 1>;
pub type EuclideanMatrix22 = EuclideanMatrix<2, 2>;
pub type EuclideanMatrix23 = EuclideanMatrix<2, 3>;
pub type EuclideanMatrix24 = EuclideanMatrix<2, 4>;
pub type EuclideanMatrix25 = EuclideanMatrix<2, 5>;
pub type EuclideanMatrix26 = EuclideanMatrix<2, 6>;

pub type EuclideanMatrix31 = EuclideanMatrix<3, 1>;
pub type EuclideanMatrix32 = EuclideanMatrix<3, 2>;
pub type EuclideanMatrix33 = EuclideanMatrix<3, 3>;
pub type EuclideanMatrix34 = EuclideanMatrix<3, 4>;
pub type EuclideanMatrix35 = EuclideanMatrix<3, 5>;
pub type EuclideanMatrix36 = EuclideanMatrix<3, 6>;

pub type EuclideanMatrix41 = EuclideanMatrix<4, 1>;
pub type EuclideanMatrix42 = EuclideanMatrix<4, 2>;
pub type EuclideanMatrix43 = EuclideanMatrix<4, 3>;
pub type EuclideanMatrix44 = EuclideanMatrix<4, 4>;
pub type EuclideanMatrix45 = EuclideanMatrix<4, 5>;
pub type EuclideanMatrix46 = EuclideanMatrix<4, 6>;

pub type EuclideanMatrix51 = EuclideanMatrix<5, 1>;
pub type EuclideanMatrix52 = EuclideanMatrix<5, 2>;
pub type EuclideanMatrix53 = EuclideanMatrix<5, 3>;
pub type EuclideanMatrix54 = EuclideanMatrix<5, 4>;
pub type EuclideanMatrix55 = EuclideanMatrix<5, 5>;
pub type EuclideanMatrix56 = EuclideanMatrix<5, 6>;

pub type EuclideanMatrix61 = EuclideanMatrix<6, 1>;
pub type EuclideanMatrix62 = EuclideanMatrix<6, 2>;
pub type EuclideanMatrix63 = EuclideanMatrix<6, 3>;
pub type EuclideanMatrix64 = EuclideanMatrix<6, 4>;
pub type EuclideanMatrix65 = EuclideanMatrix<6, 5>;
pub type EuclideanMatrix66 = EuclideanMatrix<6, 6>;

impl<const M: usize, const N: usize, B: Backend> Manifold<B> for EuclideanMatrix<M, N> {
    fn dimension(&self) -> usize {
        M * N
    }

    fn retraction(&self) -> impl DiffOperation<B> {
        Addition
    }

    fn inverse_retraction(&self) -> impl DiffOperation<B> {
        EuclideanInverseRetraction
    }
}

impl<const M: usize, const N: usize> NumComponents for EuclideanMatrix<M, N> {
    fn num_components(&self) -> usize {
        M * N
    }
}

impl<const M: usize, const N: usize, F: Element> Packable<F, Matrix<M, N, F>>
    for EuclideanMatrix<M, N>
{
    fn pack(&self, data: &Matrix<M, N, F>, offset: usize, target: &mut [F]) {
        target[offset..(offset + M * N)].copy_from_slice(data.slice());
    }
    fn unpack(&self, tensor: &[F], offset: usize) -> Matrix<M, N, F> {
        let mut out = Matrix::<M, N, F>::new(&vec![F::default(); M * N]);
        out.slice_mut()
            .copy_from_slice(&tensor[offset..(offset + (M * N))]);
        out
    }
}
